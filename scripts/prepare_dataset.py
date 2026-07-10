"""dataset/dataset_answer_*.jsonl 전부 병합 -> id 중복 제거 -> shuffle -> train/val 분할.

사용:
    python scripts/prepare_dataset.py
출력:
    dataset/train.jsonl, dataset/val.jsonl  (각 줄: {"prompt": ..., "response": ...})
"""
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트
DATA_DIR = ROOT / "dataset"

SEED = 42
VAL_RATIO = 0.05  # 10,000개 중 500개 검증용

PROMPT_TEMPLATE = "### 어르신: {utterance_1}\n### 말동무:"
RESPONSE_TEMPLATE = " {utterance_2}"


def answer_files():
    return sorted(glob.glob(str(DATA_DIR / "dataset_answer_*.jsonl")))


def load_all_rows():
    rows = {}
    for path in answer_files():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[row["id"]] = row  # id 기준 중복 제거
    return list(rows.values())


def main():
    rows = load_all_rows()
    print(f"총 {len(rows)}개 로드 (파일: {len(answer_files())}개)")

    random.seed(SEED)
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(utterance_1=r["utterance_1"]),
                "response": RESPONSE_TEMPLATE.format(utterance_2=r["utterance_2"]),
            }, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(utterance_1=r["utterance_1"]),
                "response": RESPONSE_TEMPLATE.format(utterance_2=r["utterance_2"]),
            }, ensure_ascii=False) + "\n")

    print(f"dataset/train.jsonl: {len(train_rows)}개, dataset/val.jsonl: {len(val_rows)}개")


if __name__ == "__main__":
    main()
