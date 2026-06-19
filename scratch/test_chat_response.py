# scratch/test_chat_response.py
import urllib.request
import json
import time
import sys

# Configure stdout to use utf-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

url = "http://localhost:8095/api/chat"
payload = {
    "question": "Does high temperature rise cause cracking or buckling?",
    "domain": "cat_v3_moe"
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)
        print("STATUS: SUCCESS")
        ans = data.get("answer", "")
        # Replace non-ascii chars to be safe on all shells
        ans_safe = ans.replace("→", "->").encode('ascii', errors='replace').decode('ascii')
        print("ANSWER:")
        print(ans_safe)
        print("\nREASONING PATH:")
        path = data.get("reasoning_path", [])
        path_str = " -> ".join(path)
        print(path_str.encode('ascii', errors='replace').decode('ascii'))
        print("\nSOURCE:")
        print(data.get("source"))
except Exception as e:
    print("FAILED:", e)
