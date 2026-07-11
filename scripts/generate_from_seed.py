"""
실제 어르신 발화(dataset/elder_seed_utterances.jsonl)를 utterance_1로 그대로 쓰고,
말동무 답변(utterance_2)만 새로 생성한다.
------------------------------------------------------------------
generate_dataset_batch.py와 달리 utterance_1은 GPT가 지어내지 않는다 —
elder_dataset.jsonl에서 추출한 진짜 사투리/구어체 발화를 그대로 쓴다.
목적: 모델이 실제 어르신의 사투리·조각 문장 입력을 만나도 페르소나를
유지하며 자연스럽게 반응하도록(입력 쪽 다양성 보강, 출력 페르소나는 그대로).

출력은 dataset_answer_*.jsonl과 동일 스키마라 prepare_dataset.py에 바로 합류된다.
  {"id": ..., "category": "사투리_시드", "utterance_1": "...(원문 그대로)...",
   "utterance_2": "...", "emotion": "..."}

사용:
  python scripts/generate_from_seed.py --out dataset_answer_dialect_01.jsonl --total 300

주의: 루트 .env 에 OPENAI_API_KEY 필요.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
SEED_PATH = DATA_DIR / "elder_seed_utterances.jsonl"
load_dotenv(ROOT / ".env")
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
BATCH_SIZE = 15
CATEGORY = "사투리_시드"
EMOTIONS = ["neutral", "happy", "caring", "sad", "worried",
            "surprised", "playful", "thoughtful"]

SYSTEM = (
    "너는 한국어 대화 데이터셋 생성기다. 독거 어르신을 위한 AI 말동무 학습용 데이터를 만든다.\n"
    "입력으로 실제 어르신이 한 말(utterance_1)이 여러 개 주어진다. 이 문장은 실제 인터뷰 녹취를 "
    "그대로 옮긴 것이라 사투리·구어체·문장이 끊긴 형태가 섞여 있을 수 있다. 절대 고치거나 표준어로 "
    "바꾸지 말고 원문 그대로 받아들여라.\n"
    "각 문장에 대해 말동무 AI의 답변(utterance_2)만 생성하라.\n"
    "[AI 페르소나] 따뜻한 존댓말, 짧고 또렷한 문장, 먼저 공감한 뒤 가볍게 안부나 질문을 건넨다. "
    "잔소리·가르치려는 말투 금지. 사투리를 따라 하지 말고 AI는 항상 표준어로 답한다.\n"
    "입력 문장이 조각나 있거나 맥락이 부족해도 자연스럽게 넘어가는 답을 만들어라 "
    "(예: 짧은 추임새면 편하게 맞장구, 문장이 끊겼으면 자연스럽게 이어받기).\n"
    "건강 악화·외로움·상실 신호가 보이면 다정히 챙기고, 필요하면 가까운 사람이나 전문기관 연결을 "
    "부드럽게 권한다(위협적·단정적 어투 금지).\n"
    "emotion = utterance_2의 감정. 다음 중 하나만: "
    "neutral, happy, caring, sad, worried, surprised, playful, thoughtful.\n"
    "반드시 아래 JSON 객체 하나만 출력한다. 입력 순서를 유지하고 개수를 정확히 맞춰라. "
    "설명·마크다운·주석 금지:\n"
    '{"data":[{"utterance_1":"...(입력 그대로)...","utterance_2":"...","emotion":"..."}]}'
)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def load_seeds():
    seeds = []
    with open(SEED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line)["text"])
    return seeds


def load_global_state(out_path):
    seen, used_u1, max_id = set(), set(), 0
    out_name = os.path.basename(out_path)
    for fp in glob.glob(str(DATA_DIR / "dataset_answer_*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(norm(r.get("utterance_1")))
                max_id = max(max_id, r.get("id", 0))
                if os.path.basename(fp) == out_name:
                    used_u1.add(norm(r.get("utterance_1")))
    return seen, used_u1, max_id


def call_api(batch, retries=5):
    payload = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
    user_msg = f"다음 {len(batch)}개 문장 각각에 답변을 생성하라 (입력 순서 유지):\n{payload}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            obj = json.loads(resp.choices[0].message.content)
            return obj.get("data") or obj.get("pairs") or []
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [재시도 {attempt+1}/{retries}] {e} → {wait}s 대기")
            time.sleep(wait)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="출력 파일명 (예: dataset_answer_dialect_01.jsonl)")
    ap.add_argument("--total", type=int, default=0,
                     help="생성할 개수 (0이면 시드 전체)")
    args = ap.parse_args()

    out_path = str(DATA_DIR / args.out)
    seeds = load_seeds()
    seen, used_u1, max_id = load_global_state(out_path)
    next_id = max_id + 1

    # 전역에서 이미 utterance_1로 쓰인 시드는 건너뜀 (재실행 시 이어하기)
    pending = [s for s in seeds if norm(s) not in used_u1]
    target = len(pending) if args.total <= 0 else min(args.total, len(pending))
    pending = pending[:target]

    print(f"→ {args.out}: 시드 {len(seeds)}개 중 처리 대상 {target}개 | 시작 id {next_id}")

    done = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            items = call_api(batch)
            by_u1 = {norm(it.get("utterance_1")): it for it in items if isinstance(it, dict)}
            for text in batch:
                it = by_u1.get(norm(text))
                if it is None:
                    continue
                u2 = (it.get("utterance_2") or "").strip()
                emo = (it.get("emotion") or "neutral").strip()
                if not u2:
                    continue
                if emo not in EMOTIONS:
                    emo = "neutral"
                row = {"id": next_id, "category": CATEGORY,
                       "utterance_1": text, "utterance_2": u2, "emotion": emo}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                next_id += 1
                done += 1
            print(f"진행: {done}/{target}")
            time.sleep(0.5)

    print(f"\n완료 → dataset/{args.out} ({done}개 생성, 마지막 id {next_id - 1})")


if __name__ == "__main__":
    main()
