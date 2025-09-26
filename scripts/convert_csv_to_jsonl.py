#!/usr/bin/env python3
# Convert IREC/LREC translated CSV into OpenAI fine-tuning JSONL with messages format
# Usage:
#   python scripts/convert_csv_to_jsonl.py --csv path/to/lrec_translated_400.csv --out training_data.jsonl

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

META_PATTERNS = [
    r"\byou are .*empathetic\b",
    r"\brespond (briefly|as|with)\b",
    r"\bprovide (a )?(brief|short) (response|reply)\b",
    r"\bsimulated (emotion|label)\b",
    r"\bassistant instruction\b",
    r"^system[:\-]",
    r"^assistant[:\-]",
    r"^bot[:\-]",
    r"^note[:\-]",
]


def clean_text(s: str) -> str:
    if s is None:
        return ""
    text = str(s).strip().strip('"').strip("'")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove bracketed artifacts
    text = re.sub(r"\[(?:/?)[^\]]+\]", "", text)
    low = text.lower()
    for pat in META_PATTERNS:
        if re.search(pat, low):
            return ""
    return text


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    required = {"dialogueId", "utterance_no", "authorRole", "utterance"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"CSV must have columns: {sorted(required)}; got {reader.fieldnames}")
    # Cast numeric where applicable and sort
    for r in rows:
        try:
            r["dialogueId"] = int(r["dialogueId"]) if r["dialogueId"] != "" else -1
        except Exception:
            pass
        try:
            r["utterance_no"] = int(r["utterance_no"]) if r["utterance_no"] != "" else -1
        except Exception:
            pass
    rows.sort(key=lambda x: (x.get("dialogueId", 0), x.get("utterance_no", 0)))
    return rows


def pairs_from_rows(rows: List[Dict[str, str]]) -> List[Dict]:
    by_dialogue: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in rows:
        role = str(r.get("authorRole", "")).strip().lower()
        utt = r.get("utterance", "")
        by_dialogue[str(r.get("dialogueId", ""))].append((role, utt))

    items: List[Dict] = []
    for dial_id, turns in by_dialogue.items():
        for i in range(len(turns) - 1):
            role1, u1 = turns[i]
            role2, u2 = turns[i + 1]
            if role1 in {"user", "student"} and role2 in {"bot", "assistant"}:
                u_text = clean_text(u1)
                a_text = clean_text(u2)
                if not u_text or not a_text:
                    continue
                # Drop assistant lines that still look like instructions
                if re.search(r"\b(you are|respond|provide|instruction|simulated|as an ai|as a bot)\b", a_text.lower()):
                    continue
                items.append({
                    "messages": [
                        {"role": "user", "content": u_text},
                        {"role": "assistant", "content": a_text},
                    ]
                })
    return items


def main():
    ap = argparse.ArgumentParser(description="Convert IREC/LREC CSV to OpenAI messages JSONL (user->assistant pairs)")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., lrec_translated_400.csv)")
    ap.add_argument("--out", default="training_data.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    items = pairs_from_rows(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for obj in items:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} conversation pairs to {args.out}")


if __name__ == "__main__":
    main()
