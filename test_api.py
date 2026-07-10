import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
QUESTIONS = [
    "요즘 많이 외롭습니다.",
    "오늘 날씨가 참 좋네요.",
    "몸이 좀 안 좋은 것 같아요.",
    "자식들이 바빠서 통 연락이 없어요.",
    "오늘 점심으로 뭘 먹을까요?",
]

results = []
for q in QUESTIONS:
    body = json.dumps({"message": q, "max_new_tokens": 60}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    results.append(f"Q: {q}\nA: {data['response']}\nlatency: {elapsed:.2f}s\n---")

with open(HERE / "logs" / "chat_test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results) + "\n")

print("done")
