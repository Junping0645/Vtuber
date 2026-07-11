"""dataset/dataset_answer_*.jsonl(단발) + dataset_multiturn_*.jsonl(멀티턴) 전부 병합 ->
shuffle -> train/val 분할.

멀티턴 대화는 턴마다 슬라이딩 윈도우로 잘라 여러 개의 prompt/response 쌍으로 확장한다
(대화 전체를 한 방에 학습시키는 게 아니라, 각 말동무 답변마다 그 앞 대화 전체를 문맥으로
준 prompt/response 샘플을 만든다 -> 단발과 동일한 스키마라 학습 스크립트 수정이 필요 없음).

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
VAL_RATIO = 0.05

PROMPT_TEMPLATE = "### 어르신: {utterance_1}\n### 말동무:"
RESPONSE_TEMPLATE = " {utterance_2}"


def answer_files():
    return sorted(glob.glob(str(DATA_DIR / "dataset_answer_*.jsonl")))


def multiturn_files():
    return sorted(glob.glob(str(DATA_DIR / "dataset_multiturn_*.jsonl")))


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


def expand_multiturn(turns):
    """대화 한 편을 말동무 턴마다 (그 앞 문맥 전체 prompt, 그 답변 response) 쌍으로 확장."""
    examples = []
    history = ""
    for t in turns:
        if t["role"] == "user":
            history += f"### 어르신: {t['text']}\n### 말동무:"
        else:  # assistant
            examples.append({"prompt": history, "response": f" {t['text']}"})
            history += f" {t['text']}\n"
    return examples


def load_multiturn_examples():
    examples = []
    for path in multiturn_files():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.extend(expand_multiturn(row["turns"]))
    return examples


def main():
    rows = load_all_rows()
    single_examples = [
        {
            "prompt": PROMPT_TEMPLATE.format(utterance_1=r["utterance_1"]),
            "response": RESPONSE_TEMPLATE.format(utterance_2=r["utterance_2"]),
        }
        for r in rows
    ]
    print(f"단발 {len(single_examples)}개 로드 (파일: {len(answer_files())}개)")

    multiturn_examples = load_multiturn_examples()
    print(f"멀티턴 {len(multiturn_examples)}개로 확장 (대화 파일: {len(multiturn_files())}개)")

    all_examples = single_examples + multiturn_examples
    random.seed(SEED)
    random.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * VAL_RATIO))
    val_rows, train_rows = all_examples[:n_val], all_examples[n_val:]

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"총 {len(all_examples)}개 -> dataset/train.jsonl: {len(train_rows)}개, dataset/val.jsonl: {len(val_rows)}개")


if __name__ == "__main__":
    main()
