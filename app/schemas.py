from typing import Any, Dict, List

from pydantic import BaseModel, Field


class WordScoreOut(BaseModel):
    word: str
    score: float


class PhonemeScoreOut(BaseModel):
    word: str
    reference_phones: str
    hypothesis_phones: str
    score: float


class AlignmentWordOut(BaseModel):
    word: str
    start_sec: float
    end_sec: float
    score: float


class AlignmentPhoneOut(BaseModel):
    word: str
    phone: str
    start_sec: float
    end_sec: float
    score: float


class FeedbackExampleOut(BaseModel):
    word: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class FeedbackOut(BaseModel):
    phone: str
    avg_score: float
    occurrences: int
    tip: str
    examples: List[FeedbackExampleOut]


class ScoreResponse(BaseModel):
    model_key: str
    model_path: str
    dual_mode: bool = False
    content_asr_model_key: str = ""
    content_asr_model_path: str = ""
    prompt: str
    transcript: str
    wer: float
    accuracy_like: float
    level: str
    confidence: float
    content_score: float
    accent_score: float
    accent_source: str = "centroid"
    accent_score_centroid: float | None = None
    accent_similarity: float
    accent_distance: float
    alignment_confidence: float
    phoneme_score: float
    overall_score: float
    word_scores: List[WordScoreOut]
    phoneme_scores: List[PhonemeScoreOut]
    alignment_word_timestamps: List[AlignmentWordOut]
    alignment_phone_timestamps: List[AlignmentPhoneOut]
    feedback: List[FeedbackOut]
    accent_ml_active: bool = False
    content_mismatch_gate: bool = False


def compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_key": payload.get("model_key"),
        "dual_mode": payload.get("dual_mode", False),
        "content_asr_model_key": payload.get("content_asr_model_key", ""),
        "prompt": payload.get("prompt"),
        "transcript": payload.get("transcript"),
        "overall_score": payload.get("overall_score"),
        "level": payload.get("level"),
        "wer": payload.get("wer"),
        "content_score": payload.get("content_score"),
        "accent_score": payload.get("accent_score"),
        "accent_source": payload.get("accent_source", "centroid"),
        "phoneme_score": payload.get("phoneme_score"),
        "content_mismatch_gate": payload.get("content_mismatch_gate", False),
        "feedback": payload.get("feedback", []),
    }
