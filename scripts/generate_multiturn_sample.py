"""멀티턴 대화 '샘플 10개' 생성기 — 데이터 수집 템플릿용.
애들에게 "이 형식 그대로 모아줘" 하고 보여주기 위한 예시 파일을 만든다.

출력: dataset/dataset_multiturn_sample.jsonl  (한 줄 = 대화 1편)
형식:
  {"id", "category", "turns":[{"role":"user","text":...},
                              {"role":"assistant","text":...,"emotion":...}, ...]}
"""
import json, re, time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트
load_dotenv(ROOT / ".env")
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"
OUT = ROOT / "dataset" / "dataset_multiturn_sample.jsonl"
EMOTIONS = ["neutral","happy","caring","sad","worried","surprised","playful","thoughtful"]

# 다양성을 위해 10개 서로 다른 카테고리
CATEGORIES = ["인사/안부","건강/통증","수면·약·병원","외로움","손주",
              "자식·며느리","회상·고향","날씨·계절","걱정·불안","끼니·마무리"]

SYSTEM = (
    "너는 한국어 멀티턴 대화 데이터셋 생성기다. 독거 어르신과 AI 말동무의 '이어지는 대화 한 편'을 만든다.\n"
    "- 3~5턴(어르신→말동무→어르신→말동무…)으로 구성. 반드시 어르신(user)으로 시작, 말동무(assistant)로 끝낸다.\n"
    "- 뒤 턴은 앞 내용을 실제로 이어받아라(후속 질문·지시대명사 등). 매 턴이 겉돌면 안 된다.\n"
    "- 말동무 페르소나: 따뜻한 존댓말, 짧고 또렷한 문장, 먼저 공감. 잔소리·가르치려는 말투 금지.\n"
    "- 중요: 말동무의 답을 '매번 질문'으로 끝내지 마라. 위로·공감·마무리로 끝나는 턴도 섞어라(되묻기 습관 방지).\n"
    "- 건강 악화·외로움·상실 신호엔 다정히 챙기고, 필요시 가까운 사람/전문기관을 부드럽게 권한다.\n"
    "- 각 assistant 턴에 emotion 1개: neutral, happy, caring, sad, worried, surprised, playful, thoughtful.\n"
    "- 아래 JSON 객체 하나만 출력한다. 설명·마크다운 금지:\n"
    '{"turns":[{"role":"user","text":"..."},{"role":"assistant","text":"...","emotion":"..."}]}'
)

def gen_one(category):
    resp = client.chat.completions.create(
        model=MODEL, temperature=1.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'카테고리 "{category}"로 자연스러운 멀티턴 대화 한 편을 생성하라. '
                                        '어르신의 지역·성별·성격·기분을 구체적으로 설정해 반영하라.'},
        ],
    )
    return json.loads(resp.choices[0].message.content).get("turns", [])

def valid(turns):
    if len(turns) < 3: return False
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant": return False
    for i, t in enumerate(turns):  # user/assistant 교대 확인
        if t["role"] != ("user" if i % 2 == 0 else "assistant"): return False
    return True

def main():
    rows = []
    for i, cat in enumerate(CATEGORIES, 1):
        for attempt in range(3):
            turns = gen_one(cat)
            for t in turns:  # emotion 정리
                if t["role"] == "assistant" and t.get("emotion") not in EMOTIONS:
                    t["emotion"] = "caring"
                t["text"] = (t.get("text") or "").strip()
            if valid(turns):
                rows.append({"id": i, "category": cat, "turns": turns})
                print(f"[{i:2d}] {cat} — {len(turns)}턴 ✓")
                break
            time.sleep(1)
        else:
            print(f"[{i:2d}] {cat} — 생성 실패(스킵)")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n완료 → {OUT}  ({len(rows)}편)")

if __name__ == "__main__":
    main()
