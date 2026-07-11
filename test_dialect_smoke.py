"""2차 파인튜닝(사투리 시드 + 멀티턴 보강) 이후 사투리/구어체 입력 대응을 확인하는 스모크 테스트."""
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
API_KEY = (Path(__file__).parent / "api_key.txt").read_text(encoding="utf-8").strip()

QUESTIONS = [
    "오늘 아칙에 밥 묵었어라.",
    "근육이 따르지라, 아 내가 젊어서.",
    "그래갖고 저쪽에 공원 있잖아요.",
    "몸이 영 개안찮아서 죽겄어예.",
    "혼차 있으니께 적적혀 죽겄네.",
]


def post_chat(message):
    body = json.dumps({"message": message, "max_new_tokens": 60}).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    req = urllib.request.Request(BASE + "/chat", data=body, headers=headers, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", ""), time.time() - t0


def main():
    out = ["=" * 60, "사투리/구어체 입력 스모크 테스트", "=" * 60]
    for q in QUESTIONS:
        resp, dt = post_chat(q)
        out.append(f"Q: {q}")
        out.append(f"A: {resp}   ({dt:.2f}s)")
        out.append("")
    text = "\n".join(out)
    print(text)
    (Path(__file__).parent / "logs" / "dialect_test_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
