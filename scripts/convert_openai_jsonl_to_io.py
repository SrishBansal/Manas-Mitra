#!/usr/bin/env python3
"""
Convert OpenAI messages-style JSONL (training_data.jsonl) into local seq2seq IO format
expected by scripts/train_lora.py, with fields: {"input": ..., "output": ...}

Usage:
  python scripts/convert_openai_jsonl_to_io.py \
    --in training_data.jsonl \
    --out data/dataset_from_openai.jsonl

Each output line:
  {"input": "Student: <user text>\nAssistant:", "output": "<assistant text>"}
"""
import argparse
import json
from typing import List, Dict


def extract_pair(messages: List[Dict]) -> tuple[str, str] | None:
    # Use the last contiguous user->assistant pair
    last_user = None
    last_assistant = None
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            last_user = content
            last_assistant = None
        elif role == "assistant" and last_user is not None:
            last_assistant = content
    if last_user and last_assistant:
        return last_user, last_assistant
    return None


def main():
    ap = argparse.ArgumentParser(description="Convert OpenAI messages JSONL to IO jsonl for local fine-tuning")
    ap.add_argument("--in", dest="inp", required=True, help="Path to OpenAI messages jsonl (e.g., training_data.jsonl)")
    ap.add_argument("--out", dest="out", default="data/dataset_from_openai.jsonl", help="Output IO jsonl path")
    args = ap.parse_args()

    total = 0
    kept = 0
    with open(args.inp, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                continue
            pair = extract_pair(msgs)
            if not pair:
                continue
            user, assistant = pair
            inp_text = f"Student: {user}\nAssistant:"
            out_text = assistant.strip()
            if not user or not out_text:
                continue
            f_out.write(json.dumps({"input": inp_text, "output": out_text}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Converted {kept} pairs (from {total} lines) to {args.out}")


if __name__ == "__main__":
    main()
