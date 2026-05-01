import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchaudio

from app.scoring import _word_to_phones


@dataclass
class WordAlignment:
    word: str
    start_sec: float
    end_sec: float
    confidence: float


def _build_target_text(prompt: str) -> str:
    return prompt.replace(" ", "|")


def ctc_align_prompt(
    logits: torch.Tensor,
    prompt: str,
    tokenizer,
    waveform_len_samples: int,
    sample_rate: int = 16000,
) -> Tuple[List[Dict], List[Dict], float]:
    target_text = _build_target_text(prompt)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    if not target_ids:
        return [], [], 0.0

    log_probs = torch.log_softmax(logits.detach().cpu(), dim=-1)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
    targets = torch.tensor([target_ids], dtype=torch.long)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)

    # CTC forced alignment requires enough frame length for target tokens.
    if len(target_ids) > int(input_lengths[0]):
        return [], [], 0.0

    try:
        aligned_tokens, aligned_scores = torchaudio.functional.forced_align(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=tokenizer.pad_token_id,
        )
    except RuntimeError:
        # Return empty alignments for hard failures (e.g., token/frame mismatch edge cases).
        return [], [], 0.0

    token_spans = torchaudio.functional.merge_tokens(
        aligned_tokens[0],
        aligned_scores[0],
        blank=tokenizer.pad_token_id,
    )

    if len(token_spans) != len(target_ids):
        return [], [], 0.0

    total_duration = waveform_len_samples / float(sample_rate)
    num_frames = log_probs.shape[1]
    sec_per_frame = total_duration / float(max(num_frames, 1))

    words = prompt.split()
    words_align: List[WordAlignment] = []
    phones_align: List[Dict] = []

    cursor = 0
    for word in words:
        token_count = len(word)
        if token_count <= 0 or cursor + token_count > len(token_spans):
            continue

        w_spans = token_spans[cursor : cursor + token_count]
        cursor += token_count
        if cursor < len(token_spans):
            # Skip the "|" separator after each word when present.
            cursor += 1

        start_frame = int(w_spans[0].start)
        end_frame = int(w_spans[-1].end)
        mean_log = float(sum(float(s.score) for s in w_spans) / max(len(w_spans), 1))
        # TokenSpan scores are log-domain; convert to bounded confidence.
        conf = float(max(0.0, min(1.0, math.exp(mean_log))))
        start_sec = start_frame * sec_per_frame
        end_sec = end_frame * sec_per_frame

        words_align.append(
            WordAlignment(
                word=word,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=conf,
            )
        )

        phone_str = _word_to_phones(word)
        phones = [p for p in phone_str.split() if p]
        if not phones:
            continue

        span = max(end_sec - start_sec, 0.0)
        per_phone = span / len(phones) if phones else 0.0
        for i, phone in enumerate(phones):
            p_start = start_sec + i * per_phone
            p_end = p_start + per_phone
            phones_align.append(
                {
                    "word": word,
                    "phone": phone,
                    "start_sec": round(p_start, 4),
                    "end_sec": round(p_end, 4),
                    "score": round(conf * 100.0, 2),
                }
            )

    avg_conf = (
        sum(float(w.confidence) for w in words_align) / len(words_align) if words_align else 0.0
    )
    words_payload = [
        {
            "word": w.word,
            "start_sec": round(w.start_sec, 4),
            "end_sec": round(w.end_sec, 4),
            "score": round(w.confidence * 100.0, 2),
        }
        for w in words_align
    ]
    return words_payload, phones_align, avg_conf
