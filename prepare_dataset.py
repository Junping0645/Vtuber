"""dataset_answer_*.jsonl 전부 병합 -> id 중복 제거 -> shuffle -> train/val 분할.

사용:
    python prepare_dataset.py
출력:
    train.jsonl, val.jsonl  (각 줄: {"prompt": ..., "response": ...})
"""
import glob
import json
import random

SEED = 42
VAL_RATIO = 0.05  # 10,000개 중 500개 검증용

PROMPT_TEMPLATE = "### 어르신: {utterance_1}\n### 말동무:"
RESPONSE_TEMPLATE = " {utterance_2}"


def load_all_rows():
    rows = {}
    for path in sorted(glob.glob("dataset_answer_*.jsonl")):
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
    print(f"총 {len(rows)}개 로드 (파일: {len(glob.glob('dataset_answer_*.jsonl'))}개)")

    random.seed(SEED)
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(utterance_1=r["utterance_1"]),
                "response": RESPONSE_TEMPLATE.format(utterance_2=r["utterance_2"]),
            }, ensure_ascii=False) + "\n")

    with open("val.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(utterance_1=r["utterance_1"]),
                "response": RESPONSE_TEMPLATE.format(utterance_2=r["utterance_2"]),
            }, ensure_ascii=False) + "\n")

    print(f"train.jsonl: {len(train_rows)}개, val.jsonl: {len(val_rows)}개")


if __name__ == "__main__":
    main()
