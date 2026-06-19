import urllib.request
import json

def check_status():
    try:
        response = urllib.request.urlopen("http://127.0.0.1:8002/api/status", timeout=2)
        data = json.loads(response.read().decode('utf-8'))
        print("Server is up!")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    check_status()
