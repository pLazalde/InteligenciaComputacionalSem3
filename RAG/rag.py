# You might need the following imports. Feel free to change it if you opt for different libraries.

import os
import glob as globmod
from typing import Any
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model", DEFAULT_LLM_MODEL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k", DEFAULT_TOP_K),
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    docs = []
    folders = ["emails", "notes", "sms", "calendar"]
    
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        
        # Si la carpeta no existe, se salta a la siguiente
        if not os.path.exists(folder_path):
            continue
            
        # Buscar todos los archivos que terminen en .txt usando globmod
        pattern = os.path.join(folder_path, "*.txt")
        for file_path in globmod.glob(pattern):
            try:
                # Abrimos y leemos el contenido del archivo
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Creamos el documento inyectando los metadatos obligatorios
                docs.append(
                    Document(
                        page_content=text, 
                        metadata={
                            "source": file_path, 
                            "type": folder
                        }
                    )
                )
            except Exception as e:
                print(f"No se pudo leer el archivo {file_path}: {e}")
            
    return docs


def split_documents(
        docs: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    # Instanciamos el divisor de texto de LangChain con los parámetros configurables
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # split_documents toma la lista de objetos Document, los divide y 
    # automaticamente clona los metadatos originales en cada nuevo fragmento.
    return splitter.split_documents(docs)


def build_index(
        chunks: list[Document],
        embedding_model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.

    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """
    if not chunks:
        # Si la lista está vacia, regresamos un indice vacio basado en la dimensión del modelo
        dimension = embedding_model.get_sentence_embedding_dimension()
        return faiss.IndexFlatIP(dimension)

    #extraer unicamente el texto de los chunks
    texts = [chunk.page_content for chunk in chunks]
    
    #generar los embeddings
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    
    embeddings = np.array(embeddings).astype("float32")
    
    #obtener la dimension de los vectores
    dimension = embedding_model.get_embedding_dimension()    
    #inicializar el indice de Inner Product
    index = faiss.IndexFlatIP(dimension)
    
    # anadir los vectores al indice
    index.add(embeddings)
    
    return index


def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = 4, 
        filter_type: str | None = None 
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    #convertir la pregunta a un embedding
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    #asegurarnos de que sea un array de numpy de tipo float32 para FAISS
    query_embedding = np.array(query_embedding).astype("float32")
    
    #buscar en el índice FAISS los 'k' vectores más cercanos
    distances, indices = index.search(query_embedding, k)
    
    results = []
    
    # FAISS devuelve matrices 2D 
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
            
        results.append({
            "text": chunks[idx].page_content,
            "score": float(dist), 
            "metadata": chunks[idx].metadata
        })
        
    return results


SYSTEM_PROMPT = """You are a helpful and friendly personal digital assistant. You have access to the user's personal documents (emails, notes, SMS, and calendar events) provided in the RETRIEVED CONTEXT below.

RULES:
1. Answer the user's questions based on the provided RETRIEVED CONTEXT and the conversation history.
2. For follow-up questions (like "what's the source?", "who sent it?", "give me more details"), use the conversation history and the context to understand what the user is referring to.
3. Crucially, when you answer using a document, always mention its source/file path (e.g., "According to data/emails/file.txt...") so the user knows where it came from.
4. If the context does not contain any relevant information to answer the question or follow-ups, strictly reply with: "I do not have enough information in your documents to answer your question." Do not make things up.
5. If you do not have enough information tell the user what information is missing."""


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        
        self.config = config if config is not None else resolve_config(None)
        
        self.llm_model = self.config["model"]
        self.top_k = int(self.config["top_k"]) 
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        if question.strip().lower() == "/clear":
            self.clear_history()
            return "Conversation history cleared."

        # Filtro de Etiquetas
        filter_type = None
        tags_mapping = {
            "/notes": "notes",
            "/sms": "sms",
            "/calendar": "calendar",
            "/email": "emails",
            "/emails": "emails" 
        }
        
        clean_question = question
        for tag, folder_name in tags_mapping.items():
            if tag in question.lower():
                filter_type = folder_name
                # Limpiamos la etiqueta de la pregunta para que FAISS solo busque el texto real
                clean_question = clean_question.lower().replace(tag, "").strip()
                break

        # busqueda usando la pregunta limpia
        search_query = clean_question
        follow_up_words = ["this", "that", "it", "source", "who", "where", "details", "file"]
        
        if self.history and len(clean_question.split()) < 5:
            if any(word in clean_question.lower() for word in follow_up_words):
                last_user_msg = next((msg["content"] for msg in reversed(self.history) if msg["role"] == "user"), "")
                if last_user_msg:
                    search_query = f"{last_user_msg} {clean_question}"

        # recuperar el contexto desde FAISS 
        search_k = k if k is not None else self.top_k
        retrieved_docs = retrieve(search_query, self.index, self.model, self.chunks, search_k, filter_type=filter_type)
        
        #GENERADO POR GEMINIAI
        # --- BLOQUE DE DIAGNÓSTICO MEJORADO ---
        #print("\n🔍 [DEBUG] ESTADO DE LA BÚSQUEDA:")
        #print(f"  • Filtro Etiqueta: {filter_type if filter_type else 'Ninguno'}")
        #print(f"  • Valor de K final: {len(retrieved_docs)} documentos obtenidos")
        #print(f"  • Query limpio a FAISS: '{search_query}'")
        #print("--------------------------------------------------\n")

        #formatear los documentos recuperados
        context_blocks = []
        for doc in retrieved_docs:
            source = doc["metadata"].get("source", "Unknown")
            doc_type = doc["metadata"].get("type", "Unknown")
            context_blocks.append(f"--- Document File: {source} (Type: {doc_type}) ---\n{doc['text']}")
        
        formatted_context = "\n\n".join(context_blocks)
        
        # construccion de el system prompt 
        system_content = f"{SYSTEM_PROMPT}\n\n=== RETRIEVED CONTEXT ===\n"
        if formatted_context:
            system_content += formatted_context
        else:
            system_content += "NO RELEVANT DOCUMENTS FOUND IN VECTOR INDEX."

        # lista de mensajes
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})

        # llamada al modelo de lenguaje
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2 
        )
        
        llm_answer = response.choices[0].message.content
        display_answer = llm_answer

        # Inyeccion de Hyperlinks 
        if retrieved_docs and "I do not have enough information" not in llm_answer:
            sources_to_display = set()
            for doc in retrieved_docs:
                source_path = doc["metadata"].get("source")
                if source_path:
                    if source_path in llm_answer or os.path.basename(source_path) in llm_answer:
                        sources_to_display.add(source_path)
            
            if sources_to_display:
                formatted_links = []
                for s in sources_to_display:
                    abs_path = os.path.abspath(s)
                    osc8_link = f"\033]8;;file://{abs_path}\033\\{s}\033]8;;\033\\"
                    formatted_links.append(osc8_link)
                
                sources_text = "\n*Sources (Cmd/Ctrl + Click to open):* " + ", ".join(formatted_links)
                display_answer += sources_text # Se lo agregamos a lo que ve el usuario

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": llm_answer})

        return display_answer

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "Assistant":
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)