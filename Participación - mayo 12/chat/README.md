# LLM Chat 

This is a simple console script to chat with an LLM. 

## Dependencies

Install the required dependencies with:

```bash
pip install openai python-dotenv
```

## Configuration

This app relies on the following environment variables to be set:

- OPENAI_BASE_URL
- OPENAI_API_KEY
- MODEL

Either set them in your terminal before running the application, or create a .env file with the corresponding values.

## Run the app

To start chatting with the LLM, run:

```bash
python chat.py
```

This will start a conversation; write your message and press enter to send it, then wait for the LLM generated response.

The conversation will continue endlessly until you manually stop the process using Ctrl+C.