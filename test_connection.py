import os
from dotenv import load_dotenv
from src.core.llm import build_chat_model

load_dotenv()

def main():
    print("Testing connection to MIMO API...")
    try:
        model = build_chat_model(provider="openai")
        response = model.invoke("Hello, are you there?")
        print("Connection successful! Response from model:")
        print("--------------------------------------------------")
        print(response.content)
        print("--------------------------------------------------")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
