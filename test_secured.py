"""서버 준비를 기다렸다가 API 키 인증 + 응답을 검증하는 테스트 클라이언트."""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://127.0.0.1:8000"
API_KEY = (Path(__file__).parent / "api_key.txt").read_text(encoding="utf-8").strip()
QUESTIONS = [
    "요즘 많이 외롭습니다.",
    "오늘 날씨가 참 좋네요.",
    "손주가 다음 주에 놀러 온대요.",
]


def post_chat(message, api_key=None):
    body = json.dumps({"message": message, "max_new_tokens": 60}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(BASE + "/chat", data=body, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data.get("response", ""), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), time.time() - t0


def wait_ready(timeout=1800):
    print("서버 준비 대기 중(모델 다운로드/로딩)...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                h = json.loads(r.read().decode("utf-8"))
                if h.get("model_loaded"):
                    print("→ 모델 로딩 완료!", flush=True)
                    return True
                print("  ...서버는 떴으나 모델 로딩 중", flush=True)
        except Exception:
            pass
        time.sleep(10)
    return False


def main():
    out = []
    if not wait_ready():
        print("시간 초과: 서버가 준비되지 않음")
        return

    out.append("=" * 60)
    out.append(f"API 키: {API_KEY}")
    out.append("=" * 60)

    # 1) 키 없이 요청 → 401 이어야 정상
    code, msg, _ = post_chat("테스트", api_key=None)
    out.append(f"\n[인증 테스트] 키 없이 요청 → HTTP {code}  ({'차단됨 ✓' if code == 401 else '예상과 다름 ✗'})")

    # 2) 틀린 키 → 401
    code, msg, _ = post_chat("테스트", api_key="sk-wrong-key")
    out.append(f"[인증 테스트] 틀린 키 → HTTP {code}  ({'차단됨 ✓' if code == 401 else '예상과 다름 ✗'})")

    # 3) 올바른 키 → 200 + 응답
    out.append("\n[정상 호출] 올바른 키로 대화:")
    for q in QUESTIONS:
        code, resp, dt = post_chat(q, api_key=API_KEY)
        out.append(f"  Q: {q}")
        out.append(f"  A: {resp}   (HTTP {code}, {dt:.2f}s)")
        out.append("")

    text = "\n".join(out)
    print(text)
    (Path(__file__).parent / "logs" / "secured_test_results.txt").write_text(text, encoding="utf-8")
    print("\n결과 저장 → logs/secured_test_results.txt")


if __name__ == "__main__":
    main()
