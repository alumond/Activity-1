"""Colab-first AfriMedQA QLoRA fine-tuning pipeline for Llama 3.2 Instruct.

This script is intentionally organized so lightweight data-formatting checks can
run locally, while model loading and training remain Colab/GPU operations.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DATASET_ID = "intronhealth/afrimedqa_v2"
DEFAULT_OUTPUT_DIR = "outputs/afrimed_llama3_sft"
DEFAULT_FINAL_ADAPTER_DIR = "outputs/final_adapter"


@dataclass(frozen=True)
class PipelineConfig:
    model_id: str = DEFAULT_MODEL_ID
    dataset_id: str = DEFAULT_DATASET_ID
    output_dir: str = DEFAULT_OUTPUT_DIR
    final_adapter_dir: str = DEFAULT_FINAL_ADAPTER_DIR
    max_length: int = 2048
    max_train_samples: int | None = None
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    logging_steps: int = 25
    save_steps: int = 200


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_data_for_llm(example: dict[str, Any]) -> dict[str, str]:
    """Format one AfriMedQA row into supervised fine-tuning text."""
    question = _clean_text(example.get("question"))
    response = _clean_text(example.get("answer_rationale"))
    question_type = _clean_text(example.get("question_type")) or "question"
    specialty = _clean_text(example.get("specialty")) or "general healthcare"

    instruction = (
        f"You are an expert African health AI. Answer the following {question_type} "
        f"in the context of African healthcare, focusing on the {specialty} "
        "specialty. Provide a comprehensive, accurate, and regionally appropriate "
        "response."
    )
    instruction += f"\n\nQuestion: {question}"

    answer_options = example.get("answer_options")
    if question_type.upper() == "MCQ" and answer_options:
        options = _clean_text(answer_options).strip("[]").replace("'", "")
        instruction += f"\nOptions: {options}"

    return {
        "text": (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
    }


def is_usable_example(example: dict[str, Any]) -> bool:
    """Keep examples with a question and an answer rationale."""
    text = _clean_text(example.get("text"))
    if not text or text.endswith("### Response:") or text.endswith("### Response:\nNone"):
        return False
    return "Question:" in text and "### Response:" in text


def load_and_prepare_dataset(config: PipelineConfig):
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_id)
    train_dataset = dataset["train"]
    if config.max_train_samples:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))

    formatted = train_dataset.map(
        format_data_for_llm,
        remove_columns=train_dataset.column_names,
        desc="Formatting AfriMedQA examples",
    )
    cleaned = formatted.filter(is_usable_example, desc="Removing unusable examples")
    return cleaned


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing training examples",
    )


def build_model_for_qlora(model_id: str):
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model)


def apply_lora(model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(config: PipelineConfig) -> None:
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.final_adapter_dir, exist_ok=True)

    dataset = load_and_prepare_dataset(config)
    tokenizer = load_tokenizer(config.model_id)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config.max_length)
    model = apply_lora(build_model_for_qlora(config.model_id))

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        bf16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.model.save_pretrained(config.final_adapter_dir)
    tokenizer.save_pretrained(config.final_adapter_dir)
    print(f"Final LoRA adapter saved to: {config.final_adapter_dir}")


def preview(config: PipelineConfig, rows: int) -> None:
    dataset = load_and_prepare_dataset(config)
    for index in range(min(rows, len(dataset))):
        print(f"\n--- Example {index + 1} ---")
        print(dataset[index]["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AfriMedQA Llama 3.2 QLoRA pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        command_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
        command_parser.add_argument("--max-train-samples", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Run Colab/GPU QLoRA fine-tuning")
    add_common_options(train_parser)
    train_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--final-adapter-dir", default=DEFAULT_FINAL_ADAPTER_DIR)
    train_parser.add_argument("--max-length", type=int, default=2048)
    train_parser.add_argument("--num-train-epochs", type=float, default=3)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--logging-steps", type=int, default=25)
    train_parser.add_argument("--save-steps", type=int, default=200)

    preview_parser = subparsers.add_parser("preview", help="Preview formatted training rows")
    add_common_options(preview_parser)
    preview_parser.add_argument("--rows", type=int, default=2)

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        output_dir=getattr(args, "output_dir", DEFAULT_OUTPUT_DIR),
        final_adapter_dir=getattr(args, "final_adapter_dir", DEFAULT_FINAL_ADAPTER_DIR),
        max_length=getattr(args, "max_length", 2048),
        max_train_samples=args.max_train_samples,
        num_train_epochs=getattr(args, "num_train_epochs", 3),
        per_device_train_batch_size=getattr(args, "per_device_train_batch_size", 2),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 8),
        learning_rate=getattr(args, "learning_rate", 2e-4),
        logging_steps=getattr(args, "logging_steps", 25),
        save_steps=getattr(args, "save_steps", 200),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.command == "preview":
        preview(config, args.rows)
    elif args.command == "train":
        train(config)


if __name__ == "__main__":
    main()
