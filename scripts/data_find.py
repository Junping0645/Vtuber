"""
말동무 대화 데이터 대량 생성기 (OpenAI API)
------------------------------------------------
- 노인(utterance_1) → AI 말동무(utterance_2) 쌍을 카테고리별로 생성
- 출력 형식(JSONL 한 줄): {"id","category","utterance_1","utterance_2","emotion"}
- 특징: 카테고리 균형 / 중복 제거 / 재시도(백오프) / 이어하기(resume)

[사용법]
  pip install openai
  export OPENAI_API_KEY="sk-..."      # (윈도우: set OPENAI_API_KEY=...)
  python generate_dataset.py
  * 중간에 끊겨도 다시 실행하면 이어서 채움.

[비용] gpt-4o-mini는 매우 저렴. 1만 개(짧은 쌍) 생성이면 대략 몇 달러 안쪽.
       정확한 단가는 현재 OpenAI 요금표를 확인할 것.
"""
import os, re, json, time
from pathlib import Path
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수 자동 사용

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트
MODEL = "gpt-4o-mini"          # 저렴. 품질 더 원하면 상위 모델로 교체
OUT = str(ROOT / "dataset" / "maldongmu_generated.jsonl")
BATCH_SIZE = 15                # 한 번 호출에 생성할 쌍 수 (품질/비용 균형)
TEMPERATURE = 1.0             # 다양성 확보
EMOTIONS = ["neutral","happy","caring","sad","worried","surprised","playful","thoughtful"]

# 카테고리별 목표 개수 (합계가 전체 목표). 편향 방지를 위해 고르게 배분.
CATEGORIES = {
    "인사/안부": 500, "건강/통증": 900, "수면·약·병원": 700, "식사/입맛": 700,
    "손주": 600, "자식·며느리": 600, "외로움": 900, "회상·고향": 600,
    "날씨·계절": 500, "일상(TV·화초·산책)": 700, "걱정·불안": 800, "소소한 기쁨": 600,
    "시장·동네": 500, "상실·슬픔": 500, "명절·행사": 400, "끼니·마무리": 500,
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
                    if key in seen:      # 중복 스킵
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
                # 중복이 심해 더 안 늘면 다음 카테고리로 (무한루프 방지)
                if added == 0:
                    print("  ↳ 새로 추가 없음(중복 과다). 다음 카테고리로 넘어감.")
                    break
                time.sleep(0.5)   # 레이트리밋 여유

    print(f"\n완료 → {OUT}  (총 {next_id-1}개)")

if __name__ == "__main__":
    main()