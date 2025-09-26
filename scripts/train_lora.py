import os
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a JSONL dataset with fields 'input' and 'output'."""
    return load_dataset("json", data_files=path, split="train")


def preprocess_function(tokenizer, max_source_length: int, max_target_length: int):
    def fn(batch: Dict[str, Any]):
        inputs = tokenizer(
            batch["input"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["output"],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    return fn


def maybe_enable_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 with LoRA for Manas Mitra")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--train_file", type=str, default=os.path.join("data", "dataset.jsonl"))
    parser.add_argument("--output_dir", type=str, default="outputs/lora-manas-mitra")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Total number of training epochs to perform")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of training steps for warmup")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--max_source_length", type=int, default=384,
                        help="Maximum input sequence length after tokenization")
    parser.add_argument("--max_target_length", type=int, default=192,
                        help="Maximum output sequence length after tokenization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA into base and save final model")
    parser.add_argument("--final_dir", type=str, default="outputs/merged-manas-mitra")

    args = parser.parse_args()

    device = maybe_enable_mps()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Enhanced LoRA configuration for conversational ability
    lora_config = LoraConfig(
        r=16,  # Increased rank for better adaptation to conversational patterns
        lora_alpha=32,  # Alpha scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v", "o", "wi", "wo"],  # Include feed-forward layers
    )

    model = get_peft_model(model, lora_config)

    raw_ds = load_jsonl_dataset(args.train_file)
    processed = raw_ds.map(
        preprocess_function(tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

    # Training arguments compatible with current Transformers version
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        seed=args.seed,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        remove_unused_columns=True,
        optim="adamw_torch",
        # Platform-specific settings
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        load_best_model_at_end=False,  # Disabled as it requires evaluation setup
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter weights
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_and_save:
        # Reload base model and merge adapters
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        from peft import PeftModel

        merged = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = merged.merge_and_unload()  # merge LoRA weights into base
        os.makedirs(args.final_dir, exist_ok=True)
        merged.save_pretrained(args.final_dir)
        tokenizer.save_pretrained(args.final_dir)
        print(f"Merged model saved to {args.final_dir}")
    else:
        print("Training complete. Adapter weights saved. Use --merge_and_save to create a merged model.")


if __name__ == "__main__":
    main()
