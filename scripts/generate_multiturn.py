"""
말동무 멀티턴 대화 데이터 생성기
------------------------------------------------
단발 생성기(generate_dataset_batch.py)의 멀티턴 버전.
- 대화 '한 편'(3~4번 주고받기)을 turns 배열로 생성한다.
- dataset/ 안 모든 dataset_multiturn_*.jsonl을 읽어 중복 방지 + id 연속 + resume 지원.

출력 형식 (JSONL 한 줄 = 대화 한 편):
  {"id": 11, "category": "건강/통증",
   "turns": [{"role":"user","text":"..."},
             {"role":"assistant","text":"...","emotion":"caring"}, ...]}

사용:
  python scripts/generate_multiturn.py --out dataset_multiturn_01.jsonl --total 500

주의: 루트 .env 에 OPENAI_API_KEY 필요.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 윈도우 콘솔에서 한글이 깨지지 않도록 UTF-8로 맞춘다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
load_dotenv(ROOT / ".env")
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
# 한 호출에 여러 편을 요구하면 모델이 각 편을 2턴으로 줄여버린다.
# 대화 길이를 확보하려면 반드시 호출당 1편만 생성시킬 것.
CONVS_PER_CALL = 1
MIN_TURNS, MAX_TURNS = 4, 8   # user로 시작해 assistant로 끝나므로 항상 짝수
EMOTIONS = ["neutral", "happy", "caring", "sad", "worried",
            "surprised", "playful", "thoughtful"]

# 카테고리 가중치(단발 데이터와 동일 비율). 목표 total을 이 비율대로 배분.
WEIGHTS = {
    "인사/안부": 50, "건강/통증": 90, "수면·약·병원": 70, "식사/입맛": 70,
    "손주": 60, "자식·며느리": 60, "외로움": 90, "회상·고향": 60,
    "날씨·계절": 50, "일상(TV·화초·산책)": 70, "걱정·불안": 80, "소소한 기쁨": 60,
    "시장·동네": 50, "상실·슬픔": 50, "명절·행사": 40, "끼니·마무리": 50,
}

SYSTEM = (
    "너는 한국어 멀티턴 대화 데이터셋 생성기다. 독거 어르신과 AI 말동무의 '이어지는 대화'를 만든다.\n"
    "- 각 대화는 4~8턴(짝수). 반드시 어르신(user)으로 시작해 말동무(assistant)로 끝낸다.\n"
    "- 뒤 턴은 앞 내용을 실제로 이어받아라(후속 질문·지시대명사 등). 턴이 겉돌면 안 된다.\n"
    "- 어르신의 지역(표준/경상/전라/충청 사투리)·성별·성격·기분을 대화마다 다르게 설정해 반영하라.\n"
    "- 말동무 페르소나: 따뜻한 존댓말, 짧고 또렷한 문장, 먼저 공감. 잔소리·가르치려는 말투 금지.\n"
    "- 중요: 말동무의 답을 '매번 질문'으로 끝내지 마라. 위로·공감·마무리로 끝나는 턴을 반드시 섞어라.\n"
    "- 건강 악화·외로움·상실 신호엔 다정히 챙기고, 필요시 가까운 사람/전문기관을 부드럽게 권한다"
    "(위협적·단정적 어투 금지).\n"
    "- 각 assistant 턴에 emotion 1개: neutral, happy, caring, sad, worried, surprised, playful, thoughtful.\n"
    "- 아래 JSON 객체 하나만 출력한다. 설명·마크다운 금지:\n"
    '{"conversations":[{"turns":[{"role":"user","text":"..."},'
    '{"role":"assistant","text":"...","emotion":"..."}]}]}'
)


def distribute(total):
    """total편을 WEIGHTS 비율대로 정수 배분(최대잉여법)."""
    wsum = sum(WEIGHTS.values())
    raw = {k: total * w / wsum for k, w in WEIGHTS.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(WEIGHTS, key=lambda k: raw[k] - base[k], reverse=True)[: total - sum(base.values())]:
        base[k] += 1
    return base


def norm(s):
    return re.sub(r"\s+", "", s or "")


def build_user_prompt(category, n):
    return (f'카테고리는 "{category}"이다. 이 주제로 자연스러운 멀티턴 대화 {n}편을 생성하라.\n'
            f'각 대화는 최소 {MIN_TURNS}턴, 최대 {MAX_TURNS}턴이어야 한다. '
            f'즉 어르신과 말동무가 최소 {MIN_TURNS // 2}번씩 번갈아 주고받아야 한다. '
            '2턴짜리 단발 대화는 절대 안 된다.\n'
            '어르신의 지역·성별·성격·기분을 구체적으로 설정해 반영하라. '
            'JSON 객체 형식으로만 출력하라.')


def call_api(category, n, retries=5):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_prompt(category, n)},
                ],
            )
            obj = json.loads(resp.choices[0].message.content)
            convs = obj.get("conversations") or obj.get("data") or []
            return [c.get("turns", []) for c in convs if isinstance(c, dict)]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [재시도 {attempt+1}/{retries}] {e} → {wait}s 대기")
            time.sleep(wait)
    return []


def sanitize(turns):
    """
    턴을 정리·검증한다. 규칙에 못 맞추면 None 반환.
      - user로 시작, user/assistant 교대
      - assistant로 끝나도록 뒤를 잘라냄(짝수 길이)
      - MIN_TURNS..MAX_TURNS 범위
      - assistant 턴에만 유효한 emotion
    """
    if not isinstance(turns, list) or len(turns) < MIN_TURNS:
        return None

    cleaned = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            return None
        expected = "user" if i % 2 == 0 else "assistant"
        if t.get("role") != expected:
            return None
        text = (t.get("text") or "").strip()
        if not text:
            return None
        turn = {"role": expected, "text": text}
        if expected == "assistant":
            emo = (t.get("emotion") or "").strip()
            turn["emotion"] = emo if emo in EMOTIONS else "caring"
        cleaned.append(turn)

    cleaned = cleaned[:MAX_TURNS]
    if len(cleaned) % 2 == 1:      # assistant로 끝나도록 마지막 user 제거
        cleaned.pop()
    if len(cleaned) < MIN_TURNS:
        return None
    return cleaned


def load_global_state(out_path):
    """dataset/ 내 모든 멀티턴 파일에서 중복방지 set, OUT의 카테고리별 개수, 전역 max id 복원."""
    seen, out_counts, max_id = set(), {}, 0
    out_name = os.path.basename(out_path)
    for fp in glob.glob(str(DATA_DIR / "dataset_multiturn_*.jsonl")):
        is_out = os.path.basename(fp) == out_name
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = r.get("turns") or []
                if turns:
                    seen.add(norm(turns[0].get("text")))   # 첫 발화로 중복 판단
                max_id = max(max_id, r.get("id", 0))
                if is_out:
                    out_counts[r["category"]] = out_counts.get(r["category"], 0) + 1
    return seen, out_counts, max_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="출력 파일명 (예: dataset_multiturn_01.jsonl)")
    ap.add_argument("--total", type=int, required=True, help="생성할 대화 편수")
    args = ap.parse_args()

    out_path = str(DATA_DIR / args.out)
    targets = distribute(args.total)
    seen, counts, max_id = load_global_state(out_path)
    next_id = max_id + 1

    print(f"→ {args.out}: 목표 {args.total}편 | 기존 {len(seen)}편(중복방지) | 시작 id {next_id}")

    MAX_FAIL_STREAK = 8  # 연속 실패가 이만큼 쌓여야 카테고리를 포기한다 (단발 실패로 조기 중단 방지)

    with open(out_path, "a", encoding="utf-8") as fout:
        for category, target in targets.items():
            have = counts.get(category, 0)
            fail_streak = 0
            while have < target:
                need = min(CONVS_PER_CALL, target - have)
                added = 0
                for turns in call_api(category, need):
                    cleaned = sanitize(turns)
                    if cleaned is None:
                        continue
                    key = norm(cleaned[0]["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(
                        {"id": next_id, "category": category, "turns": cleaned},
                        ensure_ascii=False) + "\n")
                    fout.flush()
                    next_id += 1
                    have += 1
                    added += 1
                    if have >= target:
                        break
                print(f"[{category}] {have}/{target} (+{added})")
                if added == 0:
                    fail_streak += 1
                    if fail_streak >= MAX_FAIL_STREAK:
                        print(f"  ↳ 연속 {MAX_FAIL_STREAK}회 추가 없음(중복/형식오류 과다). 다음 카테고리로.")
                        break
                else:
                    fail_streak = 0
                time.sleep(0.5)

    print(f"\n완료 → dataset/{args.out} (마지막 id {next_id - 1})")


if __name__ == "__main__":
    main()
