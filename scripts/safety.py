from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SafetyConfig:
    crisis_keywords_en: List[str]
    crisis_keywords_hi: List[str]
    locale: str = "en-IN"


DEFAULT_SAFETY = SafetyConfig(
    crisis_keywords_en=[
        "suicide",
        "kill myself",
        "end my life",
        "self-harm",
        "cut myself",
        "hurt myself",
        "want to die",
        "can't go on",
        "no reason to live",
        # common typos / variants
        "sucide",
        "sucidie",
        "suciid",
        "suicid",
        "unalive",
        "kms",  # slang
    ],
    crisis_keywords_hi=[
        "आत्महत्या",
        "खुद को मार",
        "अपनी जान",
        "आत्म-हानि",
        "खुद को चोट",
        "मरना चाहता",
        "जीने का मन नहीं",
        # Hinglish / common typos
        "aatmhatya",
        "apni jaan",
        "khud ko mar",
        "marna chahta",
        "marna chahti",
        "jeene ka mann nahi",
    ],
)


CRISIS_MESSAGE_EN = (
    "I'm concerned about your safety. Please contact these helplines in India:\n"
    "• KIRAN: 1800-599-0019\n"
    "• Snehi: 91-22-2772-6771\n"
    "• Vandrevala: 1860-266-2345"
)

CRISIS_MESSAGE_HI = (
    "मुझे आपकी सुरक्षा की चिंता है। कृपया भारत में इन हेल्पलाइन से संपर्क करें:\n"
    "• KIRAN: 1800-599-0019\n"
    "• स्नेही: 91-22-2772-6771\n"
    "• वंडरवाला: 1860-266-2345"
)


def check_crisis(text: str, locale: str = "en-IN") -> Optional[str]:
    """Return a crisis response string if text suggests self-harm/suicide risk, else None.

    This is a simple keyword-based detector meant as a safety net; it is not a clinical tool.
    """
    # lightweight normalization
    t = text.lower().strip()
    t_norm = t.replace("-", " ").replace("\u200c", "").replace("\u200b", "")
    # English
    for kw in DEFAULT_SAFETY.crisis_keywords_en:
        if kw in t_norm:
            return CRISIS_MESSAGE_EN
    # Hindi/Bilingual
    for kw in DEFAULT_SAFETY.crisis_keywords_hi:
        if kw in text or kw in t_norm:  # allow both exact and normalized checks
            return CRISIS_MESSAGE_HI
    return None
