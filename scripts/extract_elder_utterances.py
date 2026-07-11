"""
elder_dataset.jsonl(원본 인터뷰 전사)에서 '어르신 쪽 발화'만 추출한다.
------------------------------------------------------------------
elder_dataset.jsonl은 하나의 연속 전사를 슬라이딩 윈도우로 잘라놓은 것이라
(각 줄의 user == 이전 줄의 assistant) 그대로는 학습에 못 쓴다.
대신 assistant 필드만 순서대로 이어붙이면 원본 발화 순서를 중복 없이 복원할 수 있다.

이후 인터뷰어(질문자) 발화·추임새·조각 문장을 걸러내
말동무 학습에 '진짜 어르신 말투/사투리' 시드로 쓸 수 있는 문장만 남긴다.

출력: dataset/elder_seed_utterances.jsonl
  {"id": 1, "text": "..."}   # 필터 통과한 순서 그대로

사용:
  python scripts/extract_elder_utterances.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
SRC = DATA_DIR / "elder_dataset.jsonl"
OUT = DATA_DIR / "elder_seed_utterances.jsonl"

MIN_LEN = 12  # 이보다 짧으면 추임새/조각 문장으로 간주해 제외

# 인터뷰어(질문자) 발화로 보이는 패턴 — 질문형 어미, 요청 표현
QUESTION_PAT = re.compile(
    r"(요\?|까요\?|나요\?|가요\?|용\?|죠\?|\?$|"
    r"부탁드|여쭤|말씀해\s*주|알려\s*주|소개.*해)"
)


def reconstruct_transcript(path):
    """슬라이딩 윈도우로 잘린 파일에서 원본 발화 순서를 중복 없이 복원."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return []
    first_user = rows[0]["messages"][0]["content"].strip()
    seq = [first_user] + [r["messages"][1]["content"].strip() for r in rows]
    return seq


def is_elder_utterance(text):
    if len(text) < MIN_LEN:
        return False
    if QUESTION_PAT.search(text):
        return False
    return True


def main():
    seq = reconstruct_transcript(SRC)
    print(f"원본 전사 복원: {len(seq)}턴")

    kept = []
    seen = set()
    for text in seq:
        if not is_elder_utterance(text):
            continue
        key = re.sub(r"\s+", "", text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)

    with open(OUT, "w", encoding="utf-8") as f:
        for i, text in enumerate(kept, start=1):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")

    print(f"필터 통과: {len(kept)}개 (제외: {len(seq) - len(kept)}개)")
    print(f"저장 → {OUT}")
    print("\n샘플:")
    for text in kept[:10]:
        print(f"  - {text}")


if __name__ == "__main__":
    main()
