"""확장된 /chat의 history 파라미터로 대화 문맥이 실제로 이어지는지 확인하는 스모크 테스트."""
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
API_KEY = (Path(__file__).parent / "api_key.txt").read_text(encoding="utf-8").strip()


def post_chat(message, history=None):
    body = json.dumps({
        "message": message,
        "history": history or [],
        "max_new_tokens": 60,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    req = urllib.request.Request(BASE + "/chat", data=body, headers=headers, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", ""), time.time() - t0


def main():
    out = ["=" * 60, "멀티턴 히스토리 문맥 유지 테스트", "=" * 60]

    # 시나리오: 처음에 손주 얘기를 꺼내고, 다음 턴에서 지시대명사("걔")로 되짚어 물어봄.
    # 문맥을 정말 반영한다면 "손주"에 대한 이야기로 자연스럽게 이어져야 한다.
    history = []

    q1 = "손주가 다음 주에 놀러 온대요."
    a1, dt1 = post_chat(q1, history=history)
    out.append(f"[1턴] Q: {q1}")
    out.append(f"[1턴] A: {a1}   ({dt1:.2f}s)")
    history.append({"role": "user", "text": q1})
    history.append({"role": "assistant", "text": a1})

    q2 = "몇 살인지 안 물어보셨네요, 이번에 초등학교 들어가요."
    a2, dt2 = post_chat(q2, history=history)
    out.append(f"\n[2턴] Q: {q2}")
    out.append(f"[2턴] A: {a2}   ({dt2:.2f}s)")
    history.append({"role": "user", "text": q2})
    history.append({"role": "assistant", "text": a2})

    q3 = "걔가 오면 뭘 해주면 좋아할까요?"
    a3, dt3 = post_chat(q3, history=history)
    out.append(f"\n[3턴] Q: {q3}")
    out.append(f"[3턴] A: {a3}   ({dt3:.2f}s)")

    out.append("\n" + "-" * 60)
    out.append("비교용: history 없이 3턴 질문만 단독으로 보냈을 때")
    a3_nohistory, dt3b = post_chat(q3, history=[])
    out.append(f"Q: {q3}")
    out.append(f"A(문맥 없음): {a3_nohistory}   ({dt3b:.2f}s)")

    text = "\n".join(out)
    print(text)
    (Path(__file__).parent / "logs" / "multiturn_test_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
