# 말동무 (Maldongmu) — 독거 어르신용 AI 말동무 챗봇

`EleutherAI/polyglot-ko-3.8b`를 QLoRA로 파인튜닝해 만든, 독거 어르신을 위한 공감형 대화 챗봇입니다.
로컬 GPU에서 4bit로 구동되며, API 키 인증이 붙은 자체 호스팅 API로 서빙합니다.

> 진행 상황·실험 결과 상세는 [PROGRESS.md](PROGRESS.md) 참고.

## 폴더 구조

```
vtuber/
├── dataset/          데이터셋 (jsonl) — 저장소에는 미포함
├── models/           학습 산출물 (qlora-out/final_adapter)
├── logs/             학습 로그 · 테스트 결과
├── scripts/          데이터 생성 · 전처리
├── .venv/            가상환경
├── demo.py           one-shot 대화 데모 (중간점검용)
├── train_qlora.py    QLoRA 파인튜닝
├── serve_api.py      추론 API 서버 (인증 없음)
├── serve_api_secured.py   API 키 인증 붙은 추론 서버
├── test_api.py / test_secured.py   응답 · 인증 검증 클라이언트
└── requirements.txt
```

| 파일 | 설명 |
|---|---|
| `scripts/generate_dataset_batch.py` | 단발 대화 데이터셋 생성기 (OpenAI API, 중복방지·resume) |
| `scripts/generate_multiturn.py` | **멀티턴 대화 생성기** (턴 교대 검증·중복방지·resume) |
| `scripts/generate_multiturn_sample.py` | 멀티턴 샘플 10편 생성기 (형식 안내용) |
| `scripts/prepare_dataset.py` | 데이터 병합 + train/val 분할 |
| `train_qlora.py` | QLoRA 파인튜닝 (4bit, bitsandbytes + peft) |
| `serve_api_secured.py` | **API 키 인증** 붙은 추론 서버 |
| `models/qlora-out/final_adapter/` | 학습된 LoRA 어댑터 (약 21MB) |

## 사용법

모든 명령은 **프로젝트 루트**에서 실행합니다.

```bash
# 1) 환경 구성
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2) 한 번만 대화해보기 (중간점검) — 답변 1회 출력 후 종료
python demo.py "요즘 잠이 안 와요"

# 3) API 서버 실행 (최초 실행 시 베이스 모델 ~8GB 다운로드)
PYTHONUTF8=1 python serve_api_secured.py
#  → 최초 실행 시 api_key.txt 에 API 키가 자동 발급됨

# 4) 호출
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: <api_key.txt의 키>" \
  -d '{"message": "오늘 날씨가 참 좋네요"}'
```

### 데이터셋 재생성 / 학습

```bash
# 단발 대화 (utterance_1 / utterance_2)
python scripts/generate_dataset_batch.py --out dataset_answer_01.jsonl --total 1000

# 멀티턴 대화 (turns 배열) — 호출당 1편씩 생성하므로 500편에 10~15분
python scripts/generate_multiturn.py --out dataset_multiturn_01.jsonl --total 500

python scripts/prepare_dataset.py     # dataset/train.jsonl, val.jsonl 생성
python train_qlora.py                 # models/qlora-out 에 저장
```
데이터 생성에는 루트의 `.env`에 `OPENAI_API_KEY`가 필요합니다.

> Windows PowerShell에서는 `PYTHONUTF8=1 python ...` 같은 bash식 환경변수 접두사를 쓸 수 없습니다.
> `demo.py`와 `generate_multiturn.py`는 스스로 UTF-8을 설정하므로 그냥 실행하면 됩니다.
> 다른 스크립트에서 한글이 깨지면 `$env:PYTHONUTF8=1` 을 먼저 실행하세요.

## 데이터셋 안내

학습 데이터(`dataset/*.jsonl`)와 비밀 값(`.env`, `api_key.txt`)은 저장소에 포함하지 않습니다.
데이터는 `scripts/generate_dataset_batch.py`로 재생성할 수 있습니다.
생성은 OPENAI API로 10000문장 생성했습니다.

## 요구 환경

- NVIDIA GPU (VRAM 6GB+ — 4bit 추론 시 약 2.4GB 사용)
- Python 3.10, CUDA 12.8 지원 PyTorch (`requirements.txt` 상단 주의사항 참고)


Supported by Claude Code
