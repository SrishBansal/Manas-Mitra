#!/usr/bin/env python3
"""
Terminal chat using a fine-tuned OpenAI model id stored in model_id.txt

Usage:
  export OPENAI_API_KEY=...
  python scripts/openai_chat_cli.py --model_id_file model_id.txt
"""
import argparse
import os
import sys

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = (
    "You are Manas Mitra, a warm, empathetic peer supporter for students. "
    "Always respond in simple, supportive, and natural English. "
    "Keep replies short, kind, and human-like."
)


def main():
    ap = argparse.ArgumentParser(description="Chat with fine-tuned OpenAI model")
    ap.add_argument("--model_id_file", default="model_id.txt", help="File containing fine-tuned model id")
    args = ap.parse_args()

    if OpenAI is None:
        print("ERROR: openai package not installed. Try: pip install openai", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model_id_file):
        print(f"ERROR: model id file not found: {args.model_id_file}", file=sys.stderr)
        sys.exit(1)

    model_id = open(args.model_id_file, "r", encoding="utf-8").read().strip()
    if not model_id:
        print("ERROR: model id file is empty.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    print("Manas Mitra is ready. Type 'exit' to quit.\n")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit", "bye"}:
            print("Manas Mitra: Take care. If you want to talk again, I’m here anytime.")
            break

        # Build messages with system prompt and short context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in history[-3:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user})

        # Create response
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.9,
                top_p=0.9,
                max_tokens=120,
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Manas Mitra: Sorry, I had trouble responding ({e}).")
            continue

        print("Manas Mitra:", reply)
        history.append((user, reply))


if __name__ == "__main__":
    main()
