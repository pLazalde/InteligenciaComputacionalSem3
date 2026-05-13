from openai import OpenAI
import os
from dotenv import load_dotenv

SYSTEM_MESSAGE = "You are a chatbot. You will have a conversation with a user. Be friendly and concise"

if __name__ == "__main__":
    load_dotenv()
    URL = os.environ.get('OPENAI_BASE_URL')
    KEY = os.environ.get('OPENAI_KEY')
    MODEL = os.environ.get('MODEL')

    client = OpenAI(
        base_url=URL,
        api_key=KEY,
    )

    print(f"Chatting with {MODEL} model at {URL}\n")

    historial_conversacion = [
        {'role': 'system', 'content': SYSTEM_MESSAGE}
    ]

    while True:
        message = input("> ")
        
        if message.lower() in ['exit', 'quit']:
            print("Conversación finalizada.")
            break

        historial_conversacion.append({'role': 'user', 'content': message})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=historial_conversacion
        )
        
        bot_response = response.choices[0].message.content
        print(bot_response)
        
        historial_conversacion.append({'role': 'assistant', 'content': bot_response})