import sys
import os
sys.path.append(os.getcwd())
from agent_executor import OllamaClient

def test():
    client = OllamaClient(model="qwen2.5-coder:3b")
    ok, msg = client.check_connection()
    print(f"Check connection: {ok}, {msg}")
    if ok:
        print("Testing generation...")
        try:
            res = client.generate("Write a one-line python script that prints 'Ollama test successful'", temperature=0.1)
            print("Response:")
            print(res)
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    test()
