#!/usr/bin/env python3
"""Single-GPU QLoRA candidate trainer; preflight is safe without model weights.

The independent evaluator owns the sealed holdout. This script accepts only a
development/training export made by prepare_preschooler_training.py and never
reports a promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DOMAINS = (
    "mathematics_logic", "physics", "chemistry", "software",
    "financial_data", "language_culture", "gamedev_ui", "agent_planning",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preflight(data_dir: Path, domain: str) -> tuple[list[dict], list[dict], dict]:
    if domain not in DOMAINS:
        raise ValueError("choose one of the eight registered domains")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "preschooler-sft-manifest/v1" or manifest.get("promotion_eligible") is not False:
        raise ValueError("only a non-promotion Preschooler evidence export is accepted")
    train_file, development_file = data_dir / "train.jsonl", data_dir / "development.jsonl"
    if manifest.get("train_sha256") != sha(train_file) or manifest.get("development_sha256") != sha(development_file):
        raise ValueError("evidence split differs from the manifest")
    train, development = read_jsonl(train_file), read_jsonl(development_file)
    selected = []
    checked = []
    for split, rows, target in (("train", train, selected), ("development", development, checked)):
        for row in rows:
            meta = row.get("metadata", {})
            if meta.get("split") != split or not isinstance(row.get("messages"), list):
                raise ValueError("tampered or invalid evidence row")
            if meta.get("domain") == domain:
                target.append(row)
    if not selected or not checked:
        raise ValueError(f"{domain} requires nonempty train and development splits")
    train_families = {row["metadata"]["source_family"] for row in selected}
    development_families = {row["metadata"]["source_family"] for row in checked}
    if train_families & development_families:
        raise ValueError("source family leak across splits")
    return selected, checked, manifest


def train(data_dir: Path, out: Path, domain: str, max_steps: int) -> None:
    if max_steps < 0:
        raise ValueError("max_steps must be zero (two epochs) or positive")
    examples, development, manifest = preflight(data_dir, domain)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise ValueError("set CUDA_VISIBLE_DEVICES to an explicitly leased single Ada GPU")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    if torch.cuda.device_count() != 1 or "RTX 4000 Ada" not in torch.cuda.get_device_name(0):
        raise ValueError("this first QLoRA recipe requires one visible RTX 4000 Ada")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    class Messages(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            messages = self.rows[index]["messages"]
            if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                raise ValueError("only one user/assistant exchange is supported")
            prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True,
                                                   add_generation_prompt=True)["input_ids"]
            full = tokenizer.apply_chat_template(messages, tokenize=True,
                                                 add_generation_prompt=False)["input_ids"]
            if len(full) > 2048 or len(prefix) >= len(full):
                raise ValueError("example exceeds 2048 tokens or has no answer tokens")
            return {"input_ids": full, "attention_mask": [1] * len(full),
                    "labels": [-100] * len(prefix) + full[len(prefix):]}

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        ids = [row["input_ids"] + [tokenizer.pad_token_id] * (width - len(row["input_ids"])) for row in batch]
        mask = [row["attention_mask"] + [0] * (width - len(row["attention_mask"])) for row in batch]
        labels = [row["labels"] + [-100] * (width - len(row["labels"])) for row in batch]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labels)}

    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=2, max_steps=max_steps if max_steps else -1,
        learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=10, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2,
        bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
        report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Messages(examples),
                      eval_dataset=Messages(development), data_collator=collate)
    trainer.train()
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    result = {"schema": "preschooler-training-run/v1", "status": "candidate_only",
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "domain": domain, "train_examples": len(examples),
              "development_examples": len(development), "max_steps": max_steps,
              "holdout_commit_sha256": manifest["holdout_commit_sha256"],
              "data_manifest_sha256": sha(data_dir / "manifest.json"),
              "adapter_path": str((out / "adapter").resolve()),
              "promotion_eligible": False}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means two full epochs; positive values cap a canary")
    parser.add_argument("--train", action="store_true", help="download base weights and use one explicitly leased Ada GPU")
    args = parser.parse_args()
    examples, development, manifest = preflight(args.data, args.domain)
    if not args.train:
        print(json.dumps({"status": "preflight_only", "base_model": BASE_MODEL,
                          "base_revision": BASE_REVISION,
                          "domain": args.domain, "train": len(examples), "development": len(development),
                          "holdout_commit_sha256": manifest["holdout_commit_sha256"]}, indent=2))
        return
    if args.out is None:
        parser.error("--out is required with --train")
    train(args.data, args.out, args.domain, args.max_steps)


if __name__ == "__main__":
    main()
