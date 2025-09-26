#!/usr/bin/env python3
"""
End-to-end: CSV -> JSONL -> OpenAI fine-tune -> save model id

Usage:
  python scripts/openai_finetune.py \
    --csv lrec_translated_400.csv \
    --out_json training_data.jsonl \
    --model_base gpt-3.5-turbo \
    --model_id_file model_id.txt

Requirements:
  - OPENAI_API_KEY must be set in environment
  - OpenAI CLI installed (used to start fine-tune job as per user instruction)
"""
import argparse
import os
import re
import subprocess
import sys

from pathlib import Path

from convert_csv_to_jsonl import load_rows, pairs_from_rows


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def extract_model_id(cli_output: str) -> str | None:
    # Try to find a model id like ft:gpt-3.5-turbo-xxx or ftjob-... lines and subsequent model
    # Newer CLI prints JSON; fallback to regex.
    m = re.search(r"\bft:\S+", cli_output)
    if m:
        return m.group(0)
    # Some outputs include "id": "ft:..."
    m = re.search(r'"id"\s*:\s*"(ft:[^"]+)"', cli_output)
    if m:
        return m.group(1)
    # Try model in JSON: "fine_tuned_model": "ft:..."
    m = re.search(r'"fine_tuned_model"\s*:\s*"(ft:[^"]+)"', cli_output)
    if m:
        return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser(description="Convert CSV to JSONL and start OpenAI fine-tune")
    ap.add_argument("--csv", required=True, help="Path to IREC/LREC CSV (e.g., lrec_translated_400.csv)")
    ap.add_argument("--out_json", default="training_data.jsonl", help="Output JSONL file path")
    ap.add_argument("--model_base", default="gpt-3.5-turbo", help="Base model for fine-tune")
    ap.add_argument("--model_id_file", default="model_id.txt", help="Where to save fine-tuned model id")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    # Convert CSV -> JSONL
    rows = load_rows(args.csv)
    items = pairs_from_rows(rows)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write((str(obj).replace("'", '"')) + "\n")
    print(f"Wrote {len(items)} conversation pairs to {out_path}")

    if len(items) == 0:
        print("ERROR: No valid user->assistant pairs found. Aborting fine-tune.", file=sys.stderr)
        sys.exit(2)

    # Start fine-tune via CLI as requested
    cmd = [
        "openai", "api", "fine_tunes.create",
        "-t", str(out_path),
        "-m", args.model_base,
    ]
    print("Starting fine-tune:", " ".join(cmd))
    code, out, err = run(cmd)
    if code != 0:
        print("OpenAI CLI error:\n" + err, file=sys.stderr)
        sys.exit(code)
    print(out)

    model_id = extract_model_id(out) or extract_model_id(err or "")
    if not model_id:
        print("WARNING: Could not parse fine-tuned model id from CLI output. Please check the CLI response above.")
    else:
        Path(args.model_id_file).write_text(model_id, encoding="utf-8")
        print(f"Saved fine-tuned model id to {args.model_id_file}: {model_id}")


if __name__ == "__main__":
    main()
