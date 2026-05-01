import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List

import pronouncing


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / len(ref_words)


@dataclass
class WordScore:
    word: str
    score: float


@dataclass
class PhonemeScore:
    word: str
    reference_phones: str
    hypothesis_phones: str
    score: float


def word_level_scores(reference: str, hypothesis: str) -> List[WordScore]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    scores: List[WordScore] = []

    for i, word in enumerate(ref_words):
        hyp = hyp_words[i] if i < len(hyp_words) else ""
        ratio = SequenceMatcher(None, word, hyp).ratio()
        scores.append(WordScore(word=word, score=ratio))

    return scores


def _word_to_phones(word: str) -> str:
    cleaned = re.sub(r"[^A-Z']", "", word.upper())
    if not cleaned:
        return ""
    phones = pronouncing.phones_for_word(cleaned.lower())
    if not phones:
        return ""
    return phones[0]


def phoneme_level_scores(reference: str, hypothesis: str) -> List[PhonemeScore]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    scores: List[PhonemeScore] = []

    for i, ref_word in enumerate(ref_words):
        hyp_word = hyp_words[i] if i < len(hyp_words) else ""
        ref_phones = _word_to_phones(ref_word)
        hyp_phones = _word_to_phones(hyp_word)
        ratio = SequenceMatcher(None, ref_phones, hyp_phones).ratio() if ref_phones else 0.0
        scores.append(
            PhonemeScore(
                word=ref_word,
                reference_phones=ref_phones,
                hypothesis_phones=hyp_phones,
                score=ratio,
            )
        )

    return scores
