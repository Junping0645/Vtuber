"""학습된 QLoRA 어댑터로 말동무 챗봇 추론 API 서버.

사용:
    PYTHONUTF8=1 python serve_api.py
    (기본 http://127.0.0.1:8000, 문서는 /docs)

요청 예:
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" ^
         -d "{\"message\": \"오늘 날씨가 참 좋네요\"}"
"""
import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "EleutherAI/polyglot-ko-3.8b"
ADAPTER_DIR = "qlora-out/final_adapter"
PROMPT_TEMPLATE = "### 어르신: {message}\n### 말동무:"

app = FastAPI(title="말동무 챗봇 API")
tokenizer = None
model = None


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 80
    temperature: float = 0.8


class ChatResponse(BaseModel):
    response: str


@app.on_event("startup")
def load_model():
    global tokenizer, model
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = PROMPT_TEMPLATE.format(message=req.message)
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
    response_text = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    return ChatResponse(response=response_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
