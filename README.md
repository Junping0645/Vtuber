# 말동무 (Maldongmu) — 독거 어르신용 AI 말동무 챗봇

`EleutherAI/polyglot-ko-3.8b`를 QLoRA로 파인튜닝해 만든, 독거 어르신을 위한 공감형 대화 챗봇입니다.
로컬 GPU에서 4bit로 구동되며, API 키 인증이 붙은 자체 호스팅 API로 서빙합니다.

> 진행 상황·실험 결과 상세는 [PROGRESS.md](PROGRESS.md) 참고.

## 구성

| 파일 | 설명 |
|---|---|
| `generate_dataset_batch.py` | 대화 데이터셋 생성기 (OpenAI API, 중복방지·resume) |
| `prepare_dataset.py` | 데이터 병합 + train/val 분할 |
| `train_qlora.py` | QLoRA 파인튜닝 (4bit, bitsandbytes + peft) |
| `serve_api.py` | 추론 API 서버 (인증 없음) |
| `serve_api_secured.py` | **API 키 인증** 붙은 추론 서버 |
| `test_secured.py` | 키 인증 + 응답 검증 클라이언트 |
| `qlora-out/final_adapter/` | 학습된 LoRA 어댑터 (약 21MB) |

## 사용법

```bash
# 1) 환경
pip install torch transformers peft bitsandbytes accelerate fastapi "uvicorn[standard]"

# 2) 서버 실행 (최초 실행 시 베이스 모델 ~8GB 다운로드)
PYTHONUTF8=1 python serve_api_secured.py
#  → 최초 실행 시 api_key.txt 에 API 키가 자동 발급됨

# 3) 호출
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: <api_key.txt의 키>" \
  -d '{"message": "오늘 날씨가 참 좋네요"}'
```

## 데이터셋 안내

학습 데이터(`dataset_answer_*.jsonl`, `train/val.jsonl`)와 비밀 값(`.env`, `api_key.txt`)은
저장소에 포함하지 않습니다. 데이터는 `generate_dataset_batch.py`로 재생성할 수 있습니다.

## 요구 환경

- NVIDIA GPU (VRAM 6GB+ — 4bit 추론 시 약 2.4GB 사용)
- Python 3.10, CUDA 지원 PyTorch
