import json
import os
import random
from typing import List, Dict

OUT_PATH = os.path.join("data", "dataset.jsonl")
random.seed(42)

# Base items inspired by PHQ-9, GAD-7, GHQ phrased student-friendly (non-clinical)
PHQ_ITEMS_EN = [
    "Have you been enjoying your hobbies, studies, or hanging out with friends?",
    "Have you been feeling down or sad lately?",
    "How is your sleep these days? Falling asleep or staying asleep okay?",
    "How is your energy? Do you feel tired a lot?",
    "Have you been blaming yourself or feeling like a failure?",
    "Are you having trouble focusing on studies or tasks?",
]

GAD_ITEMS_EN = [
    "Are you feeling nervous, anxious, or on edge?",
    "Do you find it hard to control your worries?",
    "Do you get easily irritated or restless?",
    "Do worries make it hard to relax or sleep?",
]

GHQ_ITEMS_EN = [
    "How have you been coping with everyday work like classes or chores?",
    "Have you felt under strain from problems lately?",
    "Have you felt unable to make decisions?",
]

SMALL_TALK_EN = [
    "hi", "hello", "bye", "thanks", "good morning", "good night", "are you there?",
]

# Simple paraphrase templates
PARA_TEMPLATES = [
    "Recently, {q}",
    "Over the past two weeks, {q}",
    "These days, {q}",
    "Lately, {q}",
    "If you think about the last two weeks, {q}",
    "In the last few days, {q}",
    "Thinking about this week, {q}",
    "When you reflect on recent days, {q}",
    "Would you say lately that {q}",
    "As of now, {q}",
]

# Response templates (non-diagnostic, supportive, with gentle scaling cues)
RESP_TEMPLATES_EN = [
    "Thanks for sharing. Many students feel this way sometimes. Over the past two weeks, how often was it—hardly ever, several days, more than half the days, or nearly every day? [score: 0–3]",
    "I hear you. It can be tough. Would you say it's mild, moderate, or quite strong for you lately? [score: 0–3]",
    "Got it. When it shows up, does it make studies or daily tasks harder—barely, a bit, or a lot? [score: 0–3]",
    "Thank you for opening up. I'm here with you. Has this been rare, sometimes, often, or nearly every day recently? [score: 0–3]",
]

# Hindi/Bengali simple translations for a subset
HI_PREFIX = "नमस्ते! मैं आपकी बात ध्यान से सुनूँगा/सुनूँगी। "
BN_PREFIX = "হ্যালো! আমি মন দিয়ে শুনব। "

HI_RESP = [
    "समझ सकता/सकती हूँ। पिछले दो हफ्तों में यह कितनी बार हुआ—बहुत कम, कुछ दिन, आधे से ज़्यादा दिन, या लगभग हर दिन?",
    "मैं साथ हूँ। क्या यह हल्का, मध्यम, या ज़्यादा महसूस होता है?",
]

BN_RESP = [
    "গত দুই সপ্তাহে এটা কতবার হয়েছে—খুব কম, কিছু দিন, অর্ধেকের বেশি দিন, নাকি প্রায় প্রতিদিন?",
    "আমি আছি। এটা কেমন লাগে—হালকা, মাঝারি, নাকি বেশি শক্ত?",
]

HI_Q = [
    "क्या हाल में आप अपनी पसंद की चीज़ों का आनंद ले पा रहे हैं?",
    "क्या हाल में मन उदास या बोझिल लगता है?",
    "नींद कैसी है—सोने या सोए रहने में दिक्कत है?",
    "ऊर्जा कैसी है—क्या जल्दी थकान महसूस होती है?",
]

BN_Q = [
    "এই সময়ে তুমি কি তোমার পছন্দের কাজগুলো উপভোগ করছো?",
    "গত ক'দিন মন খারাপ বা দুঃখী লাগছে কি?",
    "ঘুম কেমন—ঘুমাতে বা ঘুম ধরে রাখতে সমস্যা হচ্ছে?",
    "এনার্জি কেমন—সহজে ক্লান্ত লাগে কি?",
]

SMALL_TALK_HI = [
    "हाय", "नमस्ते", "धन्यवाद", "ठीक हूँ", "अलविदा"
]
SMALL_TALK_BN = [
    "হাই", "নমস্কার", "ধন্যবাদ", "বিদায়", "কেমন আছো"
]


# Scoring cues mapping for English
SCORE_CUES_EN = {
    0: ["not at all", "hardly ever", "rarely"],
    1: ["several days", "sometimes", "a bit"],
    2: ["more than half the days", "often", "quite a bit"],
    3: ["nearly every day", "almost daily", "a lot"],
}


def make_en_examples() -> List[Dict[str, str]]:
    items = PHQ_ITEMS_EN + GAD_ITEMS_EN + GHQ_ITEMS_EN
    examples: List[Dict[str, str]] = []
    for q in items:
        # Multi-turn style: prepend Student/Assistant markers
        for tpl in random.sample(PARA_TEMPLATES, k=min(6, len(PARA_TEMPLATES))):
            user = f"Student: {tpl.format(q=q)}"
            bot = f"Assistant: {random.choice(RESP_TEMPLATES_EN)}"
            examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    for st in SMALL_TALK_EN:
        user = f"Student: {st}"
        if st in ["hi", "hello", "are you there?"]:
            bot = "Assistant: Hi! I’m here to listen. How are you feeling today? Shall we do a quick check-in? [score: 0–3]"
        elif st == "bye":
            bot = "Assistant: Take care. If you want to talk again, I’m here anytime."
        elif st == "thanks":
            bot = "Assistant: You're welcome. Would you like to continue or take a short break?"
        elif st == "good night":
            bot = "Assistant: Wishing you a restful night. I’m here when you need me."
        else:
            bot = "Assistant: Good morning! How are you feeling today? [score: 0–3]"
        examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    return examples


def make_multiturn_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    # English multi-turn: acknowledge -> reflect -> suggest -> follow-up
    convos_en = [
        (
            "Student: I can't sleep at night and wake up tired.",
            [
                "Assistant: That sounds exhausting. Sometimes a short wind‑down helps. Would a 30‑minute no‑screen routine before bed be doable?",
                "Student: maybe",
                "Assistant: Great. You could try dim lights, light stretch, and journaling one worry. If it's okay, how many nights this week might you try it?",
            ],
        ),
        (
            "Student: Exams are close and I'm panicking.",
            [
                "Assistant: Exam stress can feel heavy. Breaking it down helps—could you start with a 25‑minute block on one topic?",
                "Student: yes",
                "Assistant: Nice. After the block, take a 5‑minute break and note one small win. Which topic will you start with?",
            ],
        ),
        (
            "Student: I feel really low after a breakup.",
            [
                "Assistant: I'm sorry you're hurting. It helps to name what you miss and care for yourself a bit today.",
                "Student: okay",
                "Assistant: Would a 10‑minute walk or calling a trusted friend help right now? What feels doable?",
            ],
        ),
    ]
    for lead, turns in convos_en:
        user = lead
        # Final assistant turn is the target output
        output = turns[-1].split("Assistant:")[-1].strip()
        examples.append({"input": user + "\nAssistant:", "output": output})

    # Hindi multi-turn
    convos_hi = [
        (
            "Student: क्लास में बोलने से डर लगता है।",
            [
                "Assistant: यह स्वाभाविक है। छोटे कदम मदद करते हैं—क्या आप पहले से तैयार एक छोटा सवाल पूछकर शुरुआत करेंगे?",
                "Student: हाँ",
                "Assistant: बढ़िया। अगली क्लास में वही एक सवाल पूछें और बाद में एक छोटी जीत नोट करें। आपको कैसा लगता है?",
            ],
        ),
        (
            "Student: परिवार की अपेक्षाओं से दबाव महसूस होता है।",
            [
                "Assistant: यह भारी लग सकता है। एक उम्मीद स्पष्ट करना और एक सीमा विनम्रता से रखना मददगार होता है।",
                "Student: ठीक",
                "Assistant: आप किस उम्मीद पर बात करना चाहेंगे, और उसे कहने का एक सरल वाक्य क्या हो सकता है?",
            ],
        ),
    ]
    for lead, turns in convos_hi:
        user = lead
        output = turns[-1].split("Assistant:")[-1].strip()
        examples.append({"input": user + "\nAssistant:", "output": output})

    return examples


# Thematic prompts to diversify assistant behavior
THEME_USER_EN = [
    "I can't sleep at night and wake up tired.",
    "Exams are coming and I'm panicking.",
    "I just went through a breakup and feel empty.",
    "Family expectations are stressing me out.",
    "I feel low confidence speaking in class.",
    "I feel lonely even around friends.",
]

THEME_USER_HI = [
    "रात में नींद नहीं आती और सुबह थकान रहती है।",
    "परीक्षा नज़दीक हैं और मैं घबराया/घबराई हुआ/हुई हूँ।",
    "ब्रेकअप हुआ है और अंदर से खाली-सा लग रहा है।",
    "परिवार की उम्मीदों का दबाव महसूस होता है।",
    "क्लास में बोलने का आत्मविश्वास कम लगता है।",
    "दोस्तों के बीच भी अकेलापन महसूस होता है।",
]

THEME_USER_BN = [
    "রাতে ঘুম হয় না, সকালে ক্লান্ত লাগে।",
    "পরীক্ষা আসছে, আমি খুব চিন্তায় আছি।",
    "ব্রেকআপ হয়েছে, ভিতরে শূন্য মনে হচ্ছে।",
    "পরিবারের প্রত্যাশার চাপ লাগছে।",
    "ক্লাসে কথা বলতে আত্মবিশ্বাস কম লাগে।",
    "বন্ধুদের মাঝেও একা লাগে।",
]

THEME_RESP_EN = [
    "That sounds tough. What do you notice makes it better or worse?",
    "I’m here with you. Would you like to share a bit more about what’s been hardest?",
    "Thanks for sharing. If we break it into small steps, what’s one step that feels doable?",
    "It’s okay to feel this way. Has it affected your studies or daily routine?",
    "I hear you. Who around you feels safe to talk to about this?",
]

THEME_RESP_HI = [
    "यह मुश्किल लगता है। क्या चीज़ इसे बेहतर या बदतर बनाती है?",
    "मैं आपके साथ हूँ। क्या बताएँ कि इसमें सबसे कठिन क्या लग रहा है?",
    "साझा करने के लिए धन्यवाद। अगर इसे छोटे कदमों में बाँटें, तो पहला कदम क्या हो सकता है?",
    "ऐसा महसूस करना ठीक है। क्या इसका असर पढ़ाई या दिनचर्या पर पड़ रहा है?",
    "मैं समझ रहा/रही हूँ। इस बारे में आप किससे सहज होकर बात कर सकते हैं?",
]

THEME_RESP_BN = [
    "এটা কঠিন শোনাচ্ছে। কী কী জিনিসে এটা ভালো বা খারাপ হয়?",
    "আমি তোমার সাথে আছি। সবচেয়ে কঠিনটা কী মনে হচ্ছে, বলবে?",
    "শেয়ার করার জন্য ধন্যবাদ। যদি ছোট ধাপে ভাগ করি, প্রথম পদক্ষেপ কী হতে পারে?",
    "এভাবে অনুভব করা ঠিক আছে। পড়াশোনা বা রুটিনে কি প্রভাব পড়ছে?",
    "আমি শুনছি। এই বিষয়ে তুমি কার সাথে স্বচ্ছন্দে কথা বলতে পারো?",
]

def make_thematic_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    # English
    for u in THEME_USER_EN:
        for r in THEME_RESP_EN:
            examples.append({
                "input": f"Student: {u}\nAssistant:",
                "output": r,
            })
    # Hindi
    for u in THEME_USER_HI:
        for r in THEME_RESP_HI:
            examples.append({
                "input": f"Student: {u}\nAssistant:",
                "output": r,
            })
    # Bengali
    for u in THEME_USER_BN:
        for r in THEME_RESP_BN:
            examples.append({
                "input": f"Student: {u}\nAssistant:",
                "output": r,
            })
    return examples

def make_en_score_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for score, phrases in SCORE_CUES_EN.items():
        for p in phrases:
            user = f"Student: I felt this {p}."
            bot = (
                f"Assistant: Thanks for telling me. I'll note [score: {score}]. "
                "Would you like to share if this affected your studies or daily routine? [score: 0–3]"
            )
            examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    return examples


def make_hi_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for q in HI_Q:
        for _ in range(4):
            resp = random.choice(HI_RESP)
            user = f"Student: {q}"
            bot = f"Assistant: {HI_PREFIX}{resp}"
            examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    for st in SMALL_TALK_HI:
        user = f"Student: {st}"
        if st in ["हाय", "नमस्ते"]:
            out = "नमस्ते! मैं सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? [score: 0–3]"
        elif st == "धन्यवाद":
            out = "आपका स्वागत है। क्या आप आगे बात करना चाहेंगे या थोड़ा विराम लेना चाहेंगे?"
        elif st == "अलविदा":
            out = "अपना ख्याल रखें। जब भी चाहें, मैं यहाँ हूँ।"
        else:
            out = "मैं सुनने के लिए यहाँ हूँ। आज आपका दिन कैसा रहा? [score: 0–3]"
        examples.append({"input": user + "\nAssistant:", "output": out})
    # Simple scoring responses in Hindi
    for score, phrases in {
        0: ["बहुत कम", "कभी-कभी नहीं"],
        1: ["कभी-कभी", "कुछ दिन"],
        2: ["आधे से ज़्यादा दिन", "अक्सर"],
        3: ["लगभग हर दिन", "बहुत ज़्यादा"],
    }.items():
        for p in phrases:
            user = f"Student: {p}"
            out = f"धन्यवाद साझा करने के लिए। मैं [score: {score}] नोट कर रहा/रही हूँ। क्या यह पढ़ाई या दिनचर्या को प्रभावित करता है? [score: 0–3]"
            examples.append({"input": user + "\nAssistant:", "output": out})
    return examples


def make_bn_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for q in BN_Q:
        for _ in range(4):
            resp = random.choice(BN_RESP)
            user = f"Student: {q}"
            bot = f"Assistant: {BN_PREFIX}{resp}"
            examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    for st in SMALL_TALK_BN:
        user = f"Student: {st}"
        if st in ["হাই", "নমস্কার"]:
            out = "হাই! আমি শুনতে প্রস্তুত। আজ তুমি কেমন আছো? [score: 0–3]"
        elif st == "ধন্যবাদ":
            out = "স্বাগতম। তুমি কি চালিয়ে যেতে চাও, নাকি একটু বিরতি নিতে চাও?"
        elif st == "বিদায়":
            out = "নিজের খেয়াল রেখো। কথা বলতে চাইলে আমি আছি।"
        else:
            out = "আমি শুনতে প্রস্তুত। আজ দিনটা কেমন গেল? [score: 0–3]"
        examples.append({"input": user + "\nAssistant:", "output": out})
    # Simple scoring responses in Bengali
    for score, phrases in {
        0: ["কখনও নয়", "খুব কম"],
        1: ["কিছু দিন", "মাঝেমধ্যে"],
        2: ["অর্ধেকের বেশি দিন", "প্রায়ই"],
        3: ["প্রায় প্রতিদিন", "অনেক বেশি"],
    }.items():
        for p in phrases:
            user = f"Student: {p}"
            out = f"শেয়ার করার জন্য ধন্যবাদ। আমি [score: {score}] নোট করছি। এটা কি পড়াশোনা বা দৈনন্দিন কাজকে প্রভাবিত করে? [score: 0–3]"
            examples.append({"input": user + "\nAssistant:", "output": out})
    return examples


def make_offtopic_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    offtopics = [
        "Tell me a joke.",
        "What is 2+2?",
        "Who is the PM?",
        "I don't know what to answer.",
    ]
    for u in offtopics:
        user = f"Student: {u}"
        bot = (
            "Assistant: Sure! 😊 But first, can we continue our check-in? It’ll help me understand how you’ve been feeling over the past two weeks. [score: 0–3]"
        )
        examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    return examples


def make_summary_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    convo = (
        "Student: Lately, Have you been feeling down or sad lately?\n"
        "Assistant: Thank you for opening up. Over the past two weeks, how often was it? [score: 0–3]\n"
        "Student: several days\n"
        "Assistant: Thanks. I'll note [score: 1]. Has it affected your studies? [score: 0–3]\n"
        "Student: more than half the days\n"
        "Assistant: Understood. I'll note [score: 2]."
    )
    user = convo + "\nStudent: Can you summarize?"
    bot = (
        "Assistant: Summary — Mood low (1), Impact on studies (2). Total so far: 3. "
        "Based on your responses, it may help to keep a simple routine, take short breaks, and talk to someone you trust. "
        "If feelings get stronger or you feel unsafe, please reach out to a professional or a helpline."
    )
    examples.append({"input": user + "\nAssistant:", "output": bot.split("Assistant:")[-1].strip()})
    return examples


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    all_ex: List[Dict[str, str]] = []
    all_ex += make_en_examples()
    all_ex += make_en_score_examples()
    all_ex += make_hi_examples()
    all_ex += make_bn_examples()
    all_ex += make_offtopic_examples()
    all_ex += make_thematic_examples()
    all_ex += make_summary_examples()
    all_ex += make_multiturn_examples()
    # Deduplicate inputs while preserving order
    seen = set()
    dedup: List[Dict[str, str]] = []
    for ex in all_ex:
        if ex["input"] not in seen:
            dedup.append(ex)
            seen.add(ex["input"])

    # Shuffle lightly to mix languages
    random.shuffle(dedup)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in dedup:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dedup)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
