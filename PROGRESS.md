# 말동무 프로젝트 진행상황

> 독거 어르신용 AI 말동무 챗봇 — 데이터셋 생성 & 파인튜닝
> 최종 업데이트: 2026-07-09

---

## ✅ 완료: QLoRA 파인튜닝 (Polyglot-Ko 3.8B)

- **환경:** RTX 3060 Ti (8GB), `vtuber/.venv` (PyTorch 2.11 cu128, transformers 5.13, peft 0.19, bitsandbytes 0.49, 4bit nf4 QLoRA).
- **설정:** LoRA rank 8, `query_key_value`+`dense`만 타깃(과적합 억제), 프롬프트 구간 loss 마스킹, train 9,500 / val 500 분할, eval every 100 steps, EarlyStoppingCallback(patience 3).
- **결과:** epoch 3.37에서 early stopping 발동 (최고 성능은 epoch 2.86, **eval_loss 1.376**), 총 학습 시간 **75.7분**(4541초), train_loss 1.396.
  - eval_loss 추이: 1.596(ep0.17) → 1.402(ep1.68) → **1.376(ep2.86, best)** → 1.388(ep3.2, 반등 감지 후 중단)
  - 즉, 과적합 시작 지점을 정확히 잡아서 그 직전 체크포인트가 저장됨.
- **VRAM:** 학습 중 최대 4.1GB/8GB, 추론 시 3.4GB/8GB — 여유 충분.
- **산출물:** `vtuber/qlora-out/final_adapter/` (adapter_model.safetensors, 약 18.9MB).
- **스크립트:** `prepare_dataset.py`(병합/분할), `train_qlora.py`(학습, `--max-steps`로 디버그 가능).

## ✅ 완료: 추론 API 서버 + 스모크 테스트

- **파일:** `serve_api.py` (FastAPI, `POST /chat`, `GET /health`), 베이스 모델+어댑터 4bit 로드.
- **테스트 방법:** PowerShell `Invoke-RestMethod`는 응답 한글이 인코딩 버그로 깨짐(Windows PowerShell 5.1 UTF-8 디코딩 문제) → `test_api.py`(Python urllib)로 재검증, 정상 출력 확인.
- **샘플 응답 (5문항, `chat_test_results.txt`):**
  | 어르신 발화 | 말동무 응답 | 지연시간 |
  |---|---|---|
  | 요즘 많이 외롭습니다. | 그런 기분 드시는 건 정말 힘드시겠어요. 혹시 가까운 분과 이야기해보신 적 있으신가요? | 1.61s |
  | 오늘 날씨가 참 좋네요. | 네, 정말 상쾌한 날이에요! 밖에 나가서 산책하실 건가요? | 1.08s |
  | 몸이 좀 안 좋은 것 같아요. | 건강에 대한 걱정, 정말 무겁죠. 꼭 필요한 경우 가까운 병원에서 상담해보시는 건 어떠세요? | 1.43s |
  | 자식들이 바빠서 통 연락이 없어요. | 그러시군요, 소통이 잘 안되는 것 같아 아쉬워요. 자주 전화라도 해보세요! | 1.40s |
  | 오늘 점심으로 뭘 먹을까요? | 점심은 가볍게 드시는 게 좋죠. 무엇을 드실지 궁금하네요! | 1.11s |
- **평균 응답 지연:** 약 1.3초 (max_new_tokens=60 기준). 반복/횡설수설 없이 자연스러운 공감형 답변 생성 확인.

### 🔜 다음으로 고려할 것
- [ ] 데이터 한계(전부 single-turn, 답변 64% 질문형)로 인해 실제 대화에서 계속 되묻는 패턴이 나올 수 있음 — 필요하면 멀티턴 보강 데이터로 2차 파인튜닝 고려.
- [ ] 서버는 현재 백그라운드로 계속 켜져 있음(127.0.0.1:8000). 안 쓸 땐 종료해서 VRAM(3.4GB) 반환 권장.
- [ ] 정량 평가(perplexity 외 휴먼 평가/체크리스트) 아직 없음.

---

## ✅ 완료: 데이터셋 10,000개

- **파일:** `dataset_answer_01.jsonl` ~ `dataset_answer_10.jsonl` (각 1,000개, 총 10,000개)
- **형식 (JSONL 한 줄):**
  ```json
  {"id": 1, "category": "인사/안부", "utterance_1": "어르신 말", "utterance_2": "말동무 답변", "emotion": "neutral"}
  ```
- **필드:** `id`(1~10000 전역 고유), `category`(16종), `utterance_1`(어르신), `utterance_2`(말동무 AI), `emotion`(8종: neutral/happy/caring/sad/worried/surprised/playful/thoughtful)
- **검증 완료:** id 고유 10000/10000, 발화 중복 0건, 스키마 일관 ✔

### 데이터 품질 분석
| 지표 | 값 | 해석 |
|---|---|---|
| 답변 distinct-3gram | 0.81 | 다양성 양호 |
| 답변 첫 어절 최다 | "그런" 8.2% | 획일적이지 않음 |
| 완전 중복 답변 | 0개 | 깨끗함 |
| 어르신/답변 평균 길이 | 26자 / 48자 | 짧은 단발성 |

### ⚠️ 데이터 한계 (파인튜닝 전 인지할 것)
1. **전부 single-turn** — 멀티턴 대화 없음 → 모델이 "문맥 이어가기"를 못 배움. 고성능 원하면 3~5턴 대화 보강 권장.
2. **답변 64%가 질문(?)으로 끝남** — 되묻는 습관 우려. 위로만 하는 답변도 섞으면 자연스러워짐.
3. **emotion 편중** — caring/thoughtful 많고 surprised 극소.

---

## 🔜 다음 단계: LoRA/QLoRA 파인튜닝

- **환경:** RTX 3060 Ti (8GB) 데스크톱. 8GB라 풀 파인튜닝 불가 → **QLoRA 필수.**
- **데이터 양:** 충분함 (LoRA는 500~5k면 되고 지금 10k는 넉넉).
- **결정 남은 것:**
  - [ ] 베이스 모델 크기: Polyglot-Ko **1.3B**(추천) / 3.8B / 5.8B
  - [ ] 멀티턴 데이터 보강 여부
- **예상 학습 시간:** KoGPT2 125M이면 몇 분 / Polyglot-Ko 1.3B QLoRA면 대략 30~90분.

### 학습 스크립트 만들 때 반영할 것
- 포맷 변환: `utterance_1` → 프롬프트, `utterance_2` → 응답 (chat template)
- QLoRA: 4bit 양자화(bitsandbytes) + LoRA 어댑터, `fp16=True`
- 10개 jsonl 파일 병합 + train/val 분할 필요

---

## 🛠 생성 스크립트 (데이터 추가 시)

```bash
# vtuber 폴더에서. .env에 OPENAI_API_KEY(sk-proj) 필요
PYTHONUTF8=1 python generate_dataset_batch.py --out dataset_answer_11.jsonl --total 1000
```
- `generate_dataset_batch.py`: 폴더 내 모든 `dataset_answer_*.jsonl`을 읽어 **중복 방지 + id 연속 + resume** 자동 처리.
- `data_find.py`: 원본(1만 목표 단일파일), `generate_dataset_1000.py`: 1000개 단일파일 버전.
- 콘솔 한글 깨짐 방지: 실행 시 `PYTHONUTF8=1` 붙일 것.

---

## 📁 파일 목록
```
vtuber/
├── .env                          # OPENAI_API_KEY (커밋 금지)
├── data_find.py                  # 원본 생성기 (1만 목표)
├── generate_dataset_1000.py      # 1000개 단일파일 생성기
├── generate_dataset_batch.py     # ★ 범용 배치 생성기 (중복방지+resume)
├── dataset_answer_01~10.jsonl    # ★ 완성 데이터 10,000개
└── PROGRESS.md                   # 이 파일
```
