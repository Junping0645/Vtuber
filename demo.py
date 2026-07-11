"""말동무 챗봇 데모 — 인자 없이 실행하면 대화 이력을 유지하며 계속 주고받는다.

실행 (PowerShell / cmd / bash 어디서든):
    python demo.py                      # 대화형(멀티턴), 빈 입력이면 종료
    python demo.py "요즘 잠이 안 와요"   # 인자로 주면 단발 질문 1개만 하고 종료

중간점검용. serve_api_secured.py를 켜지 않고도 로컬에서 바로 모델을 호출해 테스트할 수 있다.
대화 이력 이어붙이는 방식은 serve_api_secured.py의 build_prompt와 동일하다.
"""
import sys
import time
from pathlib import Path

# 윈도우 콘솔(cp949)에서 한글이 깨지지 않도록 스스로 UTF-8로 맞춘다.
# (PYTHONUTF8=1 을 매번 붙이지 않아도 되게 하려는 것)
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # 파이프로 연결된 경우 등은 무시

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).parent
BASE_MODEL = "EleutherAI/polyglot-ko-3.8b"
ADAPTER_DIR = str(HERE / "models" / "qlora-out" / "final_adapter")


def load():
    """베이스 모델(4bit) + QLoRA 어댑터 로드."""
    print("모델 로딩 중... (4bit)", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print(f"로딩 완료 ({time.time() - t0:.1f}초)\n", flush=True)
    return tokenizer, model


def clean(text):
    """모델이 답변 뒤에 가짜 대화(### 어르신: ...)를 이어 붙이면 거기서 잘라낸다."""
    for stop in ("### 어르신", "###", "\n\n"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def build_prompt(history, message):
    """serve_api_secured.py의 build_prompt와 동일한 포맷으로 대화 이력을 이어붙인다."""
    text = ""
    for role, turn_text in history:
        if role == "user":
            text += f"### 어르신: {turn_text}\n### 말동무:"
        else:
            text += f" {turn_text}\n"
    text += f"### 어르신: {message}\n### 말동무:"
    return text


@torch.no_grad()
def reply(tokenizer, model, history, message, max_new_tokens=80, temperature=0.8):
    prompt = build_prompt(history, message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return clean(full[len(prompt_text):])


def main():
    # 인자로 질문을 주면 단발 1회만 하고 종료. 인자가 없으면 대화형(멀티턴)으로 계속 이어간다.
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:]).strip()
        if not message:
            print("입력이 비어 있어 종료합니다.")
            return
        tokenizer, model = load()
        t0 = time.time()
        answer = reply(tokenizer, model, [], message)
        print(f"어르신 > {message}")
        print(f"말동무 > {answer}")
        print(f"        ({time.time() - t0:.2f}s)")
        return

    tokenizer, model = load()
    print("대화형 모드입니다. 빈 입력을 누르면 종료합니다.\n")
    history = []
    while True:
        try:
            message = input("어르신 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return
        if not message:
            print("종료합니다.")
            return

        t0 = time.time()
        answer = reply(tokenizer, model, history, message)
        print(f"말동무 > {answer}")
        print(f"        ({time.time() - t0:.2f}s)")
        history.append(("user", message))
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
