---
base_model: EleutherAI/polyglot-ko-3.8b
library_name: peft
pipeline_tag: text-generation
language:
- ko
tags:
- base_model:adapter:EleutherAI/polyglot-ko-3.8b
- lora
- transformers
- korean
- chatbot
---

# 말동무 — 독거 어르신용 AI 대화 챗봇 (QLoRA 어댑터)

`EleutherAI/polyglot-ko-3.8b`를 QLoRA(4bit NF4)로 파인튜닝한 LoRA 어댑터입니다.
독거 어르신과 짧고 따뜻한 존댓말로 대화하며, 먼저 공감하고 필요시 안부를 챙기는 페르소나로 학습했습니다.

## 학습 데이터
- 단발 대화 11,266개: 16개 카테고리(건강/외로움/손주/명절 등) × 8개 감정, GPT-4o-mini로 생성 + 실제 어르신 인터뷰 전사에서 추출한 사투리/구어체 발화 1,266개(사투리 입력 이해 보강용)
- 멀티턴 대화 1,000편(대화당 평균 7.1턴) → 턴마다 슬라이딩 윈도우로 확장해 3,546개 학습 샘플로 변환
- 최종 train 14,072 / val 740

## 학습 설정
- LoRA: r=8, alpha=16, dropout=0.1, target_modules=["query_key_value", "dense"] (GPT-NeoX 어텐션 위주로 좁게 타깃 — 과적합 억제)
- 4bit NF4 양자화(bitsandbytes), MAX_LENGTH=768
- batch_size=2, gradient_accumulation_steps=8 (유효 배치 16), 5 epoch 캡 + EarlyStoppingCallback(patience=3)
- 결과: epoch 3.296에서 early stopping, **eval_loss 1.366**, RTX 3060 Ti(8GB)에서 141.9분 소요, peak VRAM 3.4GB

## 사용법

베이스 모델(7.6GB)을 4bit로 로드한 뒤 이 어댑터를 얹습니다.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "EleutherAI/polyglot-ko-3.8b"
ADAPTER = "Junping0645/Vtuber"  # 이 리포

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

prompt = "### 어르신: 요즘 많이 외롭습니다.\n### 말동무:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8,
                      top_p=0.9, repetition_penalty=1.15,
                      pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 멀티턴(대화 이력 유지)

이전 턴들을 같은 포맷으로 이어붙이면 문맥을 반영한 답변이 나옵니다(학습 시 사용한 포맷과 동일해야 함):

```
### 어르신: 손주가 다음 주에 놀러 온대요.
### 말동무: 정말요? 기대되시겠어요! 즐거운 시간 보내시길 바래요.
### 어르신: 몇 살인지 안 물어보셨네요, 이번에 초등학교 들어가요.
### 말동무: 그렇군요! 요즘 아이들은 빠르죠.
### 어르신: 걔가 오면 뭘 해주면 좋아할까요?
### 말동무:
```

## 참고
- VRAM: 학습 3.4GB / 추론 약 3.3GB (RTX 3060 Ti 8GB 기준)
- Framework: PEFT 0.19.1, transformers 5.13.0, bitsandbytes 0.49.2, torch 2.11.0+cu128
- 개인 프로젝트용 비공개 모델입니다.
