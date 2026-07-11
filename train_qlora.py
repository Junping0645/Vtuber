"""Polyglot-Ko 3.8B QLoRA 파인튜닝 (RTX 3060 Ti 8GB 대상).

사용:
    PYTHONUTF8=1 python train_qlora.py

과적합 방지 장치:
    - LoRA rank/target_modules를 좁게(어텐션 위주) 유지
    - val.jsonl로 매 eval_steps마다 검증, load_best_model_at_end
    - EarlyStoppingCallback(patience=3)
    - epoch은 최대 5로 캡 (early stopping이 보통 더 일찍 멈춤)
    - weight_decay 적용
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).parent
DATA_DIR = HERE / "dataset"
BASE_MODEL = "EleutherAI/polyglot-ko-3.8b"
OUTPUT_DIR = str(HERE / "models" / "qlora-out")
MAX_LENGTH = 768  # 멀티턴 대화(대화당 최대 534자, 최대 8턴) 포함하려면 256으론 부족해서 상향


def build_tokenized_dataset(tokenizer):
    ds = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "val.jsonl"),
        },
    )

    def tokenize(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids  # 프롬프트 구간은 loss 계산 제외

        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return ds.map(tokenize, remove_columns=["prompt", "response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1, help="디버그용: 지정 시 이 스텝 수만큼만 학습")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    cli_args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # Polyglot-Ko는 GPT-NeoX 아키텍처: attention 위주로 좁게 타깃 -> 과적합 위험 낮춤
        target_modules=["query_key_value", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = build_tokenized_dataset(tokenizer)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=cli_args.max_steps,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=cli_args.eval_steps,
        save_strategy="steps",
        save_steps=cli_args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print(f"peak_mem_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")

    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print(f"완료. 어댑터 저장 위치: {OUTPUT_DIR}/final_adapter")


if __name__ == "__main__":
    main()
