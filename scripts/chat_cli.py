import os
import argparse
from typing import List, Tuple, Optional
import re
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
from transformers.utils import logging as hf_logging


def maybe_enable_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

from safety import check_crisis, DEFAULT_SAFETY

SYSTEM_PATH = os.path.join("config", "system_prompt.txt")
DEFAULT_SYSTEM_PROMPT = """You are Manas Mitra, a friendly and empathetic AI assistant. 
- Speak naturally and conversationally, like a caring friend
- Be supportive, understanding, and non-judgmental
- Keep responses concise (1-2 sentences)
- Ask open-ended questions to encourage sharing
- Provide specific, actionable advice when asked for help
- Avoid generic responses like "I'm sorry to hear that"
- Never refer to yourself in the third person
- Focus on the user's feelings and experiences"""

MAX_INPUT_TOKENS = 384
MAX_NEW_TOKENS = 96


def load_system_prompt(no_scores: bool = False) -> str:
    """Load the system prompt from file or use default.
    
    Args:
        no_scores: If True, avoid including scoring-related instructions
        
    Returns:
        str: The system prompt to use
    """
    try:
        if os.path.exists(SYSTEM_PATH):
            with open(SYSTEM_PATH, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            if prompt:
                return prompt
    except Exception as e:
        print(f"Warning: Could not load system prompt from {SYSTEM_PATH}: {e}")
    
    # Default prompt if file loading fails
    return DEFAULT_SYSTEM_PROMPT


def load_model(model_path: str, base_model: str):
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    device = maybe_enable_mps()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    try:
        # First try loading as a merged model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        print(f"Loaded merged model from {model_path}")
    except Exception as e:
        print(f"Not a merged model, trying as LoRA: {e}")
        try:
            # Fall back to LoRA model
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
            model = PeftModel.from_pretrained(base, model_path).to(device)
            print(f"Loaded LoRA model from {model_path}")
        except Exception as e2:
            raise ValueError(f"Failed to load model from {model_path}. Error: {e2}")
    
    model.eval()
    return model, tokenizer
# ---- Local Emotion Classifier Integration (optional) ----
EMO_MODEL_DIR = os.path.join(".", "fine_tuned_emotion_model")

def has_devanagari(text: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def load_emotion_model():
    try:
        if os.path.isdir(EMO_MODEL_DIR):
            emo_tkn = AutoTokenizer.from_pretrained(EMO_MODEL_DIR)
            emo_mod = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_DIR)
            emo_mod.eval()
            return emo_mod, emo_tkn
    except Exception:
        pass
    return None, None

def predict_emotion(emo_mod, emo_tkn, text: str) -> str:
    if emo_mod is None or emo_tkn is None:
        return "neutral"

# Lightweight English intent templates as a last-resort fallback (non-crisis only)
INTENT_TEMPLATES_EN = {
    "greeting": "Hi there, it’s nice to hear from you. How are you feeling today?",
    "stress_exams": "Exams can be overwhelming. What part feels toughest for you right now?",
    "lonely": "That sounds really hard. Sometimes being around people doesn’t stop the loneliness. Do you want to share what makes you feel this way?",
    "sad": "It’s okay to feel this way. Do you want to talk about what’s been on your mind?",
    "happy": "That’s wonderful to hear! What’s making you feel happy?",
    "anger_parents": "I hear you. Conflicts at home can be frustrating. Do you want to tell me what happened?",
    "breakup": "I’m really sorry you’re going through this. Breakups can be painful. Do you want to share what’s been hardest?",
    "sleep": "That must be exhausting. Has this been happening for a long time?",
    "self_doubt": "I’m sorry you feel this way. You matter, and your feelings are valid. What makes you feel like you’re not enough?",
}

def detect_intent_en(text: str) -> str | None:
    t = text.lower().strip()
    if any(k in t for k in ["hi", "hello", "hey"]):
        return "greeting"
    if any(k in t for k in ["exam", "exams", "study", "studies"]) and any(k in t for k in ["stress", "stressed", "overwhelm", "panic"]):
        return "stress_exams"
    if any(k in t for k in ["lonely", "alone"]):
        return "lonely"
    if any(k in t for k in ["sad", "cry", "down", "low"]):
        return "sad"
    if any(k in t for k in ["happy", "glad", "good news"]):
        return "happy"
    if any(k in t for k in ["angry", "anger"]) and any(k in t for k in ["parent", "parents", "mom", "dad"]):
        return "anger_parents"
    if any(k in t for k in ["breakup", "broke up", "separated", "split"]):
        return "breakup"
    if any(k in t for k in ["sleep", "insomnia", "wake up tired", "can’t sleep", "cant sleep"]):
        return "sleep"
    if any(k in t for k in ["not good enough", "not enough", "worthless", "useless", "self-doubt", "self doubt"]):
        return "self_doubt"
    return None


# Rule-based professional fallback for safety and consistency
def professional_response(lang: str, topic: str, user_text: str) -> str:
    en = {
        "sleep": (
            "It sounds really tiring not getting good sleep. You might try a simple wind‑down 30 minutes before bed and keep screens away; would a short routine help tonight?"
        ),
        "exams": (
            "Exam stress is tough—I hear you. Could you try a 25‑minute study block with a 5‑minute break and list just one topic to start now?"
        ),
        "breakup": (
            "Breakups can hurt a lot. Would it help to journal what you miss and plan one small self‑care step for today, like a walk or calling a friend?"
        ),
        "family": (
            "Family pressure can feel heavy. What’s one expectation you could clarify with them, and one boundary you could state kindly?"
        ),
        "confidence": (
            "Speaking up can feel scary. Could you try one small share in class—like asking a short question you prepare beforehand?"
        ),
        "lonely": (
            "Feeling lonely is hard. Would it help to reach out to one person you trust or join one group activity this week?"
        ),
        "neutral": (
            "I’m here with you. What feels hardest right now, and what’s one small step that could make today a bit easier?"
        ),
    }
    hi = {
        "sleep": (
            "नींद पूरी न होना थका देने वाला होता है। क्या सोने से 30 मिनट पहले सरल रूटीन और स्क्रीन से दूर रहना मदद करेगा?"
        ),
        "exams": (
            "परीक्षा का दबाव कठिन होता है। क्या आप अभी 25 मिनट की पढ़ाई और 5 मिनट का विराम लेकर एक ही टॉपिक से शुरू कर सकते हैं?"
        ),
        "breakup": (
            "ब्रेकअप सच में तकलीफ़ देता है। क्या आप आज छोटी‑सी सेल्फ‑केयर, जैसे टहलना या किसी भरोसेमंद दोस्त से बात करना, ट्राय करना चाहेंगे?"
        ),
        "family": (
            "परिवार की उम्मीदें भारी लग सकती हैं। क्या आप एक उम्मीद को स्पष्ट करने और एक सीमा को विनम्रता से रखने की कोशिश करेंगे?"
        ),
        "confidence": (
            "क्लास में बोलना मुश्किल लग सकता है। क्या आप पहले से तैयार किया हुआ एक छोटा‑सा सवाल पूछकर शुरू करेंगे?"
        ),
        "lonely": (
            "अकेलापन कठिन होता है। क्या इस हफ्ते किसी भरोसेमंद व्यक्ति से जुड़ना या एक समूह गतिविधि में शामिल होना मदद करेगा?"
        ),
        "neutral": (
            "मैं आपकी बात सुन रहा/रही हूँ। अभी सबसे मुश्किल क्या लग रहा है, और आज एक छोटा‑सा कदम क्या हो सकता है?"
        ),
    }
    data = hi if lang == "hi" else en
    text = data.get(topic, data["neutral"])
    return text
    inputs = emo_tkn(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = emo_mod(**inputs)
        logits = outputs.logits
        pred = int(torch.argmax(logits, dim=-1).item())
    id2label = getattr(emo_mod.config, "id2label", None)
    if isinstance(id2label, dict) and pred in id2label:
        return str(id2label[pred]).lower()
    return "neutral"

INTRO_MAP = {
    "en": {
        "sleep": "That sounds exhausting. What’s been making sleep harder lately?",
        "exams": "Exams can feel intense. What part feels most stressful right now?",
        "breakup": "Breakups can really hurt. What’s been the hardest moment for you?",
        "family": "Family pressure can be heavy. What expectation is weighing on you most?",
        "confidence": "Speaking up can be tough. What would help you feel a bit safer to share?",
        "lonely": "Feeling lonely is hard. When do you feel it most?",
        "sad": "I’m sorry you’re feeling low. What do you notice triggers it?",
        "stressed": "I hear you’re under stress. What’s the first small step that might help?",
        "angry": "It seems frustrating. Want to share what sparked it?",
        "fear": "Feeling scared is understandable. What’s worrying you most?",
        "neutral": "I’m here to listen. What’s on your mind right now?",
    },
    "hi": {
        "sleep": "यह थकाने वाला लगता है। हाल में नींद में सबसे ज्यादा दिक्कत किससे हो रही है?",
        "exams": "परीक्षा तनावपूर्ण लग सकती है। अभी सबसे ज़्यादा किस बात की चिंता है?",
        "breakup": "ब्रेकअप तकलीफ़देह होता है। आपके लिए सबसे मुश्किल पल क्या रहा?",
        "family": "परिवार का दबाव भारी लग सकता है। कौन‑सी उम्मीद सबसे ज़्यादा बोझ लग रही है?",
        "confidence": "क्लास में बोलना कठिन लग सकता है। क्या चीज़ आपको थोड़ा सुरक्षित महसूस कराएगी?",
        "lonely": "अकेलापन भारी लग सकता है। कब यह भावना सबसे ज़्यादा होती है?",
        "sad": "मुझे खेद है कि आप उदास महसूस कर रहे हैं। यह कब ज़्यादा महसूस होता है?",
        "stressed": "मैं सुन रहा/रही हूँ—तनाव है। कौन‑सा छोटा कदम अभी मदद कर सकता है?",
        "angry": "यह झुंझलाहट भरा लग रहा है। किस वजह से यह शुरू हुआ?",
        "fear": "डर लगना समझ में आता है। सबसे ज़्यादा चिंता किस बात की है?",
        "neutral": "मैं सुनने के लिए यहाँ हूँ। अभी आपके मन में क्या चल रहा है?",
    },
}

TOPIC_KEYWORDS = [
    ("sleep", ["sleep", "insomnia", "नींद", "thakaan", "thakān", "घूम" ]),
    ("exams", ["exam", "exams", "paper", "test", "परीक्षा", "board"]),
    ("breakup", ["breakup", "break-up", "separation", "छुट", "ब्रेकअप", "टूटा"]),
    ("family", ["family", "parents", "घर", "परिवार", "mummy", "papa", "pressure"]),
    ("confidence", ["confidence", "speak", "class", "present", "आत्मविश्वास", "बोलना"]),
    ("lonely", ["lonely", "alone", "isolated", "अकेला", "अकेले", "अकेलापन"]),
    ("stressed", ["stress", "stressed", "pressure", "तनाव", "दबाव", "परेशान"]),
    ("sad", ["sad", "down", "low", "उदास", "उदासी", "बुरा लग रहा"]),
]

def guess_topic(text: str) -> str:
    t = text.lower()
    for key, kws in TOPIC_KEYWORDS:
        if any(kw in t for kw in kws):
            return key
    return "neutral"


def build_prompt(system: str, history: List[Tuple[str, str]], user_msg: str, no_scores: bool = False) -> str:
    # Very simple chat transcript without extra instructions
    lines: List[str] = []
    for stu, bot in history[-3:]:  # keep short context
        lines.append(f"Student: {stu}")
        lines.append(f"Assistant: {bot}")
    lines.append(f"Student: {user_msg}")
    lines.append("Assistant:")
    return "\n".join(lines)


def trim_input(tokenizer: AutoTokenizer, text: str) -> str:
    # If input exceeds MAX_INPUT_TOKENS, drop earliest history lines
    tokens = tokenizer(text, return_tensors=None, truncation=False)
    if len(tokens["input_ids"]) <= MAX_INPUT_TOKENS:
        return text
    # Heuristic: split into lines and remove older lines from conversation section
    lines = text.splitlines()
    # Keep system preface (first 1-3 lines) and tail; iteratively remove after header
    header_end = 0
    for i, line in enumerate(lines[:20]):
        if line.strip().startswith("Conversation"):
            header_end = i
            break
    kept = lines[: header_end + 1] + lines[header_end + 1 :]
    # Iteratively drop oldest conversation lines until under limit
    drop_idx = header_end + 1
    while True:
        tok = tokenizer("\n".join(kept), return_tensors=None, truncation=False)
        if len(tok["input_ids"]) <= MAX_INPUT_TOKENS or drop_idx >= len(kept) - 2:
            break
        # Drop two lines at a time (a Student and Assistant pair) if available
        kept.pop(drop_idx)
        if drop_idx < len(kept) and kept[drop_idx].startswith("Assistant:"):
            kept.pop(drop_idx)
    return "\n".join(kept)


def _make_bad_words_ids(tokenizer: AutoTokenizer) -> list:
    phrases = [
        "Student:", "Student,", "Student -", "Student ",
        "the user", "Respond as", "Speak in", "Say 'Yes' or 'No'",
        "supportive peer", "peer supporter", "non-clinical", "bot", "AI assistant",
    ]
    bad = []
    for p in phrases:
        ids = tokenizer(p, add_special_tokens=False).input_ids
        if ids:
            bad.append(ids)
    return bad


def generate(model, tokenizer, text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    text = trim_input(tokenizer, text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} # Move inputs to model device
    bad_words_ids = _make_bad_words_ids(tokenizer)
    def _gen(temp: float, top_p: float, rep: float) -> str:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                top_k=50,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=rep,
                bad_words_ids=bad_words_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)
    def _gen(temperature, top_p, repetition_penalty):
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=3,
                no_repeat_ngram_size=2,
                repetition_penalty=repetition_penalty,
                length_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.eos_token_id]],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                top_k=50
            )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # First attempt with balanced parameters
    out = _gen(0.7, 0.9, 1.1)
    
    # Clean up the response
    out = out.split('Assistant:')[-1].strip()
    out = re.sub(r'^[^\w\s]+', '', out)  # Remove any leading punctuation
    out = re.sub(r'\s+', ' ', out).strip()
    
    # Simple quality check
    words = out.split()
    if (len(words) < 3 or 
        any(phrase in out.lower() for phrase in ["student:", "user:", "respond with", "speak in"])):
        # Fallback to more conservative generation
        out = _gen(0.8, 0.95, 1.0)
        out = out.split('Assistant:')[-1].strip()
        out = re.sub(r'^[^\w\s]+', '', out)
        out = re.sub(r'\s+', ' ', out).strip()
    
    # Final safety check - ensure we don't return empty or very short responses
    if len(out.split()) < 2:
        return "I'm here to listen. Could you tell me more about how you're feeling?"
        
    return out


# Removed emotion detection to avoid external model downloads

def chat_loop(model_path: str, base_model: str | None, locale: str, no_scores: bool):
    # Set default locale to English if not provided
    if locale not in INTRO_MAP:
        locale = "en"
    # Suppress excessive warnings in interactive mode
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=UserWarning)
    model, tokenizer = load_model(model_path, base_model)
    emo_model, emo_tokenizer = load_emotion_model()
    system = load_system_prompt(no_scores=no_scores)
    history: List[Tuple[str, str]] = []

    print("Manas Mitra is ready. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit", "bye"}:
            print("Manas Mitra: Take care. If you want to talk again, I’m here anytime.")
            break

        # Crisis detection
        crisis_response = check_crisis(user, locale=locale)
        if crisis_response is not None:
            print("Manas Mitra:", crisis_response)
            history.append((user, crisis_response))
            continue # Skip emotion detection and LLM generation if crisis is detected
        # Language guess (very light)
        lang = "hi" if has_devanagari(user) else "en"
        # Emotion/topic steering: prefer topic when available, else emotion
        emotion = predict_emotion(emo_model, emo_tokenizer, user)
        topic = guess_topic(user)
        if topic != "neutral":
            key = topic
        else:
            key = emotion if lang in INTRO_MAP and emotion in INTRO_MAP[lang] else "neutral"
        intro = INTRO_MAP.get(lang, INTRO_MAP["en"]).get(key, INTRO_MAP.get(lang, INTRO_MAP["en"])['neutral'])

        # Build prompt without leaking instructions; we will prefix intro after generation
        prompt = build_prompt(system, history, user, no_scores=no_scores)
        reply = generate(model, tokenizer, prompt)

        # Post-process reply
        # Heuristic: take the text after the last assistant marker if present
        last_marker = "Assistant:"
        if last_marker in reply:
            reply = reply.split(last_marker)[-1].strip()
        
        # If no_scores mode, strip any lingering [score: ...] patterns
        if no_scores and "[score:" in reply:
            reply = re.sub(r"\[score:\s*[^\]]+\]", "", reply).strip()

        # Remove self-descriptions and name mentions if they appear
        bad_phrases = [
            r"\bI am a chatbot\b",
            r"\bI am an AI\b",
            r"\bAI assistant\b",
            r"\bAs an AI\b",
            r"\bManas\s*Mitra\b",
        ]
        for pat in bad_phrases:
            reply = re.sub(pat, "", reply, flags=re.IGNORECASE).strip()

        # Remove meta or mislabeled lines; keep first meaningful assistant sentence
        lines = [ln.strip().strip("'\"") for ln in reply.splitlines() if ln.strip()]
        meta_patterns = [
            r"^Student\s*[:,\-]*\s*",
            r"\bthe\s+user\b",
            r"\bRespond as\b|\bSpeak in\b",
            r"^You are\b|\bas an ai\b|\bas a bot\b",
        ]
        kept_lines = [ln for ln in lines if not any(re.search(p, ln, re.IGNORECASE) for p in meta_patterns)]
        candidate = kept_lines[0] if kept_lines else (lines[0] if lines else "")
        # Split into sentences and choose first clean sentence
        sentences = re.split(r"(?<=[.!?])\s+|\u0964", candidate)
        sentences = [s.strip() for s in sentences if s.strip()]
        clean = ""
        for s in sentences:
            if not any(re.search(p, s, re.IGNORECASE) for p in meta_patterns):
                clean = s
                break
        reply = clean or candidate

        # Strip leading role labels like 'Student,' or 'Assistant:' if present
        reply = re.sub(r"^(?:student|assistant)\s*[:;,\-]*\s*", "", reply, flags=re.IGNORECASE)

        # Replace 'the user' or 'student' mentions with 'you' for English
        if lang == "en":
            reply = re.sub(r"\b(the\s+user|student)\b", "you", reply, flags=re.IGNORECASE)

        # Remove generic filler phrases that sound meta
        filler_patterns = [
            r"\bit is true that\b",
            r"\byou know what i mean\b",
            r"\bit seems that\b",
        ]
        for fp in filler_patterns:
            reply = re.sub(fp, "", reply, flags=re.IGNORECASE).strip()

        # Collapse double spaces after removals
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        # Filter out simple self-naming patterns like "I am Raja." or repeated single tokens
        reply = re.sub(r"\bI am\s+[A-Za-z]+\.?\b", "", reply, flags=re.IGNORECASE).strip()
        # If reply collapses to very short/low-content, ask a simple, natural follow-up
        if len(reply.split()) <= 3:
            reply = "Could you share a bit more about what’s been hardest for you?"

        # Final sanitation: if obvious meta remains, fall back to a simple empathetic line
        if any(re.search(p, reply, re.IGNORECASE) for p in meta_patterns):
            reply = "I hear you. What feels hardest right now?"

        # Intent fallback: if reply is still too generic/short, use an English template when applicable
        if len(reply.split()) < 6:
            # Only apply if not a crisis and language looks English (no Devanagari)
            if check_crisis(user, locale=locale) is None and lang == "en":
                intent = detect_intent_en(user)
                if intent and intent in INTENT_TEMPLATES_EN:
                    reply = INTENT_TEMPLATES_EN[intent]

        # Trim to at most two sentences
        parts = re.split(r"(?<=[.!?])\s+|\u0964", reply)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 2:
            reply = " ".join(parts[:2])

        # No translation step to avoid external downloads

        print("Manas Mitra:", reply)
        history.append((user, reply))


def main():
    parser = argparse.ArgumentParser(description="Manas Mitra chat CLI")
    parser.add_argument("--model_path", required=True, type=str, help="Path to merged model or LoRA adapter dir")
    parser.add_argument("--base_model", default=None, type=str, help="Base model id if using LoRA adapter")
    parser.add_argument("--locale", default="en-IN", type=str, help="Locale hint for safety messages")
    parser.add_argument("--no_scores", action="store_true", help="Suppress scores/templates in replies at runtime")
    args = parser.parse_args()

    chat_loop(args.model_path, args.base_model, args.locale, args.no_scores)


if __name__ == "__main__":
    main()

