import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def maybe_enable_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_path: str, base_model: str | None = None):
    """
    Load either a merged model (plain transformers) from model_path,
    or if base_model is provided, load base + LoRA adapter from model_path.
    """
    if base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer


def generate_reply(model, tokenizer, text: str, max_length: int = 64):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference for Manas Mitra FLAN-T5")
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model or adapter dir")
    parser.add_argument("--base_model", type=str, default=None, help="Base model id (if loading adapter)")
    parser.add_argument("--prompt", type=str, default="Have you been enjoying your hobbies?")
    parser.add_argument("--locale", type=str, default="en-IN", help="Locale hint for safety messages")
    args = parser.parse_args()

    device = maybe_enable_mps()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    model.eval()

    # Safety check before generation
    try:
        from safety import check_crisis
    except Exception:
        check_crisis = None

    if check_crisis is not None:
        crisis = check_crisis(args.prompt, locale=args.locale)
        if crisis is not None:
            print(crisis)
            return

    reply = generate_reply(model, tokenizer, args.prompt)
    print(reply)


if __name__ == "__main__":
    main()
