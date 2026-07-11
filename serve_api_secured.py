"""말동무 SLM을 'API 키로 호출하는 LLM'처럼 서빙하는 보안 API 서버.

- OpenAI처럼: 요청에 API 키 헤더가 있어야만 응답한다.
- 키는 최초 실행 시 자동 생성되어 api_key.txt 에 저장된다 (환경변수 MALDONGMU_API_KEY로 덮어쓰기 가능).
- 모델은 로컬 GPU에서 4bit(QLoRA 어댑터)로 구동 → 클라우드/과금 없음.

실행:
    PYTHONUTF8=1 python serve_api_secured.py
호출 예:
    curl -X POST http://127.0.0.1:8000/chat ^
      -H "Content-Type: application/json" ^
      -H "x-api-key: <발급된 키>" ^
      -d "{\"message\": \"오늘 날씨가 참 좋네요\"}"
"""
import os
import secrets
from pathlib import Path

import torch
from fastapi import FastAPI, Depends, Header, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).parent
BASE_MODEL = "EleutherAI/polyglot-ko-3.8b"
ADAPTER_DIR = str(HERE / "models" / "qlora-out" / "final_adapter")
KEY_FILE = HERE / "api_key.txt"


def get_or_create_api_key() -> str:
    """환경변수 > 파일 순으로 키를 읽고, 없으면 새로 발급해 파일에 저장."""
    env_key = os.getenv("MALDONGMU_API_KEY")
    if env_key:
        return env_key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    new_key = "sk-maldongmu-" + secrets.token_urlsafe(24)
    KEY_FILE.write_text(new_key, encoding="utf-8")
    return new_key


API_KEY = get_or_create_api_key()

app = FastAPI(title="말동무 챗봇 API (키 인증)")
tokenizer = None
model = None


class Turn(BaseModel):
    role: str  # "user"(어르신) 또는 "assistant"(말동무)
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []  # 이전 대화 턴들(과거→현재 순). 비우면 단발 대화와 동일하게 동작.
    max_new_tokens: int = 80
    temperature: float = 0.8


class ChatResponse(BaseModel):
    response: str


def build_prompt(history: list[Turn], message: str) -> str:
    """학습 시 사용한 것과 동일한 방식으로 대화 이력을 이어 붙인다
    (scripts/prepare_dataset.py의 expand_multiturn과 동일 포맷)."""
    text = ""
    for t in history:
        if t.role == "user":
            text += f"### 어르신: {t.text}\n### 말동무:"
        else:
            text += f" {t.text}\n"
    text += f"### 어르신: {message}\n### 말동무:"
    return text


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    """x-api-key 헤더 또는 Authorization: Bearer <key> 를 검증."""
    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(f"[key] 이 서버의 API 키: {API_KEY}")
    print(f"[model] 로딩 시작: {BASE_MODEL} (+ QLoRA 어댑터). 첫 실행 시 ~8GB 다운로드.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("[model] 로딩 완료. 준비됨.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "auth": "x-api-key"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    prompt = build_prompt(req.history, req.message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response_text = full_text[len(prompt_text):].strip()
    return ChatResponse(response=response_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
