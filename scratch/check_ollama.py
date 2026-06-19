import urllib.request
import json

def check_ollama():
    try:
        response = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(response.read().decode('utf-8'))
        print("Ollama is running!")
        print("Available models:")
        for model in data.get('models', []):
            print(f"- {model['name']}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")

if __name__ == "__main__":
    check_ollama()
