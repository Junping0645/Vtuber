"""
말동무 대화 데이터 1,000쌍 생성기 (OpenAI API)
------------------------------------------------
- data_find.py 기반. 목표를 1,000개로 축소하고 .env를 자동 로드한다.
- 출력: dataset_answer_01.jsonl  (JSONL 한 줄 = 대화 쌍 1개)
  {"id","category","utterance_1","utterance_2","emotion"}
- 중간에 끊겨도 다시 실행하면 이어서 채운다(resume).

  python generate_dataset_1000.py
"""
import os, re, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")   # OPENAI_API_KEY 로드
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"
OUT = str(Path(__file__).parent / "dataset_answer_01.jsonl")
BATCH_SIZE = 15
TEMPERATURE = 1.0
EMOTIONS = ["neutral", "happy", "caring", "sad", "worried", "surprised", "playful", "thoughtful"]

# 목표 총 1,000개 (원본 1만 개 배분을 1/10로 축소, 합계 = 1000)
CATEGORIES = {
    "인사/안부": 50, "건강/통증": 90, "수면·약·병원": 70, "식사/입맛": 70,
    "손주": 60, "자식·며느리": 60, "외로움": 90, "회상·고향": 60,
    "날씨·계절": 50, "일상(TV·화초·산책)": 70, "걱정·불안": 80, "소소한 기쁨": 60,
    "시장·동네": 50, "상실·슬픔": 50, "명절·행사": 40, "끼니·마무리": 50,
}
TARGET_TOTAL = sum(CATEGORIES.values())

SYSTEM = (
    "너는 한국어 대화 데이터셋 생성기다. 독거 어르신을 위한 AI 말동무 학습용 대화 쌍을 만든다.\n"
    "- utterance_1 = 어르신의 말. 주제·지역(표준/경상/전라/충청 사투리)·성별·성격·기분을 매번 다르게 하라.\n"
    "- utterance_2 = 말동무 AI의 답변. [AI 페르소나] 따뜻한 존댓말, 짧고 또렷한 문장, "
    "먼저 공감한 뒤 가볍게 안부나 질문을 건넨다. 잔소리·가르치려는 말투 금지.\n"
    "- 건강 악화·외로움·상실 신호가 보이면 가볍게 넘기지 말고 다정히 챙기고, "
    "필요하면 가까운 사람이나 전문기관 연결을 부드럽게 권한다(위협적·단정적 어투 금지).\n"
    "- emotion = utterance_2의 감정. 다음 중 하나만: "
    "neutral, happy, caring, sad, worried, surprised, playful, thoughtful.\n"
    "- 반드시 아래 JSON 객체 하나만 출력한다. 설명·마크다운·주석 금지:\n"
    '{"data":[{"utterance_1":"...","utterance_2":"...","emotion":"..."}]}'
)


def build_user_prompt(category, n):
    return (f'카테고리는 "{category}"이다. 이 주제로 서로 겹치지 않는 대화 쌍 {n}개를 생성하라. '
            '어르신의 지역·성별·성격·기분을 골고루 섞어라. '
            'JSON 객체 형식으로만 출력하라.')


def norm(s):
    return re.sub(r"\s+", "", s or "")


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
            return obj.get("data") or obj.get("pairs") or []
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [재시도 {attempt+1}/{retries}] {e} → {wait}s 대기")
            time.sleep(wait)
    return []


def load_existing():
    """이어하기: 기존 파일에서 중복 방지용 set과 카테고리별 개수 복원."""
    seen, counts, max_id = set(), {}, 0
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(norm(r.get("utterance_1")))
                counts[r["category"]] = counts.get(r["category"], 0) + 1
                max_id = max(max_id, r.get("id", 0))
    return seen, counts, max_id


def main():
    seen, counts, max_id = load_existing()
    next_id = max_id + 1
    if seen:
        print(f"이어하기: 기존 {len(seen)}개 발견. 이어서 채웁니다.")

    with open(OUT, "a", encoding="utf-8") as fout:
        for category, target in CATEGORIES.items():
            have = counts.get(category, 0)
            while have < target:
                need = min(BATCH_SIZE, target - have)
                items = call_api(category, need)
                added = 0
                for it in items:
                    u1 = (it.get("utterance_1") or "").strip()
                    u2 = (it.get("utterance_2") or "").strip()
                    emo = (it.get("emotion") or "neutral").strip()
                    if not u1 or not u2:
                        continue
                    if emo not in EMOTIONS:
                        emo = "neutral"
                    key = norm(u1)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {"id": next_id, "category": category,
                           "utterance_1": u1, "utterance_2": u2, "emotion": emo}
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    next_id += 1
                    have += 1
                    added += 1
                    if have >= target:
                        break
                print(f"[{category}] {have}/{target} (+{added})  |  전체 {next_id-1}/{TARGET_TOTAL}")
                if added == 0:
                    print("  ↳ 새로 추가 없음(중복 과다). 다음 카테고리로 넘어감.")
                    break
                time.sleep(0.5)

    print(f"\n완료 → {OUT}  (총 {next_id-1}개)")


if __name__ == "__main__":
    main()
