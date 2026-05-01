from collections import defaultdict
from typing import Dict, List


LEVEL_BANDS = [
    ("Beginner", 0.0, 0.55),
    ("Intermediate", 0.55, 0.75),
    ("Advanced", 0.75, 0.9),
    ("Near Native", 0.9, 1.01),
]


PHONE_TIPS: Dict[str, str] = {
    "TH": "For TH sounds, place the tongue lightly between the teeth and release air.",
    "DH": "For voiced TH (like 'this'), keep the tongue between teeth and add voicing.",
    "R": "For R, curl the tongue slightly back without touching the roof of the mouth.",
    "L": "For L, touch the tongue tip to the ridge just behind your top teeth.",
    "W": "For W, round your lips before releasing into the next vowel.",
    "V": "For V, touch upper teeth to lower lip and keep voicing on.",
    "B": "For B/P contrast, ensure B is voiced and P has stronger air burst.",
    "P": "For P, use a clear burst of air and avoid adding voicing.",
    "AE": "Open your jaw more for AE (as in 'cat') and keep it short.",
    "AH": "Relax your tongue for AH and avoid over-rounding lips.",
    "IH": "Keep IH short and relaxed (as in 'sit').",
    "IY": "For IY (as in 'see'), stretch lips slightly and keep the vowel long enough.",
    "ER": "For ER, keep the tongue centered and add slight R coloring.",
}


def classify_level(accuracy_like: float) -> str:
    for label, low, high in LEVEL_BANDS:
        if low <= accuracy_like < high:
            return label
    return "Beginner"


def _base_phone(phone: str) -> str:
    # ARPABET symbols from CMU may end with stress digits (e.g. AH0, IY1)
    return "".join(ch for ch in phone if not ch.isdigit()).upper()


def build_feedback(
    alignment_phone_timestamps: List[Dict],
    max_items: int = 5,
    weak_threshold: float = 85.0,
) -> List[Dict]:
    if not alignment_phone_timestamps:
        return []

    grouped = defaultdict(list)
    for entry in alignment_phone_timestamps:
        base = _base_phone(entry.get("phone", ""))
        if not base:
            continue
        grouped[base].append(entry)

    weak = []
    for phone, items in grouped.items():
        avg = sum(float(i.get("score", 0.0)) for i in items) / max(len(items), 1)
        weak.append((avg, phone, items))

    weak.sort(key=lambda x: x[0])
    output: List[Dict] = []
    for avg_score, phone, items in weak[:max_items]:
        if avg_score >= weak_threshold:
            continue
        tip = PHONE_TIPS.get(phone, f"Practice clearer articulation for {phone} in slow repetitions.")
        output.append(
            {
                "phone": phone,
                "avg_score": round(avg_score, 2),
                "occurrences": len(items),
                "tip": tip,
                "examples": [
                    {
                        "word": i.get("word"),
                        "start_sec": i.get("start_sec"),
                        "end_sec": i.get("end_sec"),
                    }
                    for i in items[:3]
                ],
            }
        )
    return output
