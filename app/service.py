import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor

from app.accent import AccentScorer
from app.accent_head import AccentHead
from app.alignment import ctc_align_prompt
from app.config import (
    load_model_registry,
    load_scoring_calib,
    resolve_active_model,
    resolve_calib_path,
    resolve_content_asr_model,
)
from app.feedback import build_feedback, classify_level
from app.scoring import normalize_text, phoneme_level_scores, simple_wer, word_level_scores


class PronunciationService:
    """
    Dual-model (optional):
    - Content ASR (Wav2Vec2): transcript, WER, confidence, CTC alignment.
    - Accent (HuBERT): hidden-state embedding -> accent_score vs native centroid.
    """

    def __init__(self, model_dir: Path | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        registry = load_model_registry()
        self.active_model_key, resolved_dir, self.model_info = resolve_active_model()
        self.accent_model_dir = model_dir or resolved_dir
        self.model_dir = self.accent_model_dir

        self.accent_processor = AutoProcessor.from_pretrained(str(self.accent_model_dir))
        self.accent_model = AutoModelForCTC.from_pretrained(str(self.accent_model_dir)).to(
            self.device
        ).eval()

        self.dual_enabled, self.content_asr_key, content_path = resolve_content_asr_model(registry)
        self.content_processor = None
        self.content_model = None
        if self.dual_enabled and content_path is not None:
            try:
                self.content_processor = AutoProcessor.from_pretrained(str(content_path))
                self.content_model = AutoModelForCTC.from_pretrained(str(content_path)).to(
                    self.device
                ).eval()
            except Exception as exc:
                print(f"dual_model_disabled_load_failed: {exc}", flush=True)
                self.dual_enabled = False
                self.content_processor = None
                self.content_model = None

        self.accent_scorer = AccentScorer(
            model=self.accent_model,
            processor=self.accent_processor,
            device=self.device,
        )
        self._calib = load_scoring_calib()
        self.accent_ml_head = None
        self.accent_ml_calib: Dict[str, float] | None = None
        self.accent_ml_active = False
        self._try_load_accent_ml()

    def _try_load_accent_ml(self) -> None:
        aml = self._calib.get("accent_ml") or {}
        if not aml.get("enabled", False):
            return
        head_path = resolve_calib_path(str(aml.get("head_path", "")))
        if not head_path.exists():
            print(f"accent_ml_skip_missing_checkpoint path={head_path}", flush=True)
            return
        try:
            ckpt = torch.load(head_path, map_location=self.device)
            hidden_size = int(ckpt["hidden_size"])
            num_classes = int(ckpt["num_classes"])
            head = AccentHead(hidden_size=hidden_size, num_classes=num_classes).to(self.device)
            head.load_state_dict(ckpt["head_state_dict"])
            head.eval()
            self.accent_ml_head = head
            calib_path = resolve_calib_path(str(aml.get("calibration_path", "")))
            if calib_path.exists():
                self.accent_ml_calib = json.loads(calib_path.read_text(encoding="utf-8"))
            else:
                self.accent_ml_calib = None
            self.accent_ml_active = True
            print(
                f"accent_ml_loaded path={head_path} calibrated={self.accent_ml_calib is not None}",
                flush=True,
            )
        except Exception as exc:
            print(f"accent_ml_load_failed: {exc}", flush=True)
            self.accent_ml_head = None
            self.accent_ml_calib = None
            self.accent_ml_active = False

    def _asr_forward(
        self, processor: AutoProcessor, model: AutoModelForCTC, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, str, float]:
        inputs = processor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcript = normalize_text(processor.batch_decode(pred_ids)[0])
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs.max(dim=-1).values.mean().item())
        return logits, transcript, confidence

    def _pooled_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        inputs = self.accent_processor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.accent_model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        return hidden.mean(dim=1).squeeze(0)

    def _score_accent_from_embedding(self, emb: torch.Tensor) -> Dict[str, Any]:
        legacy = self.accent_scorer.score_embedding(emb)
        out: Dict[str, Any] = {
            "accent_score": legacy["accent_score"],
            "accent_similarity": legacy["accent_similarity"],
            "accent_distance": legacy["accent_distance"],
            "accent_source": "centroid",
        }
        if self.accent_ml_head is None:
            return out
        with torch.inference_mode():
            emb_b = emb.unsqueeze(0).to(self.device)
            raw, _ = self.accent_ml_head(emb_b)
            raw_val = float(raw.squeeze(0).item())
        if self.accent_ml_calib:
            a = float(self.accent_ml_calib.get("affine_slope", 1.0))
            b = float(self.accent_ml_calib.get("affine_bias", 0.0))
            score_ml = max(0.0, min(100.0, a * raw_val + b))
        else:
            score_ml = max(0.0, min(100.0, raw_val))
        out["accent_score"] = score_ml
        out["accent_score_centroid"] = legacy["accent_score"]
        out["accent_source"] = "ml"
        return out

    def _accent_forward(self, waveform: torch.Tensor) -> Dict[str, Any]:
        emb = self._pooled_embedding(waveform)
        return self._score_accent_from_embedding(emb)

    def _load_audio(self, path: Path) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception as exc:
            raise ValueError("Could not decode uploaded audio. Please upload a valid wav/webm file.") from exc
        waveform = waveform.mean(dim=0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.numel() == 0:
            raise ValueError("Audio is empty after decoding.")
        return waveform

    def score_file(self, audio_path: Path, prompt_text: str) -> Dict[str, Any]:
        prompt = normalize_text(prompt_text)
        waveform = self._load_audio(audio_path)

        if self.dual_enabled and self.content_model is not None and self.content_processor is not None:
            logits, transcript, confidence = self._asr_forward(
                self.content_processor, self.content_model, waveform
            )
            if self.device == "cuda":
                torch.cuda.empty_cache()
            accent = self._accent_forward(waveform)
            align_tokenizer = self.content_processor.tokenizer
        else:
            inputs = self.accent_processor(
                waveform.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.accent_model(**inputs, output_hidden_states=True)
                logits = out.logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcript = normalize_text(self.accent_processor.batch_decode(pred_ids)[0])
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs.max(dim=-1).values.mean().item())
            hidden = out.hidden_states[-1]
            emb = hidden.mean(dim=1).squeeze(0)
            accent = self._score_accent_from_embedding(emb)
            align_tokenizer = self.accent_processor.tokenizer

        wer = simple_wer(prompt, transcript)
        accuracy_like = max(0.0, 1.0 - wer)
        words = word_level_scores(prompt, transcript)
        phones = phoneme_level_scores(prompt, transcript)
        phoneme_avg = sum(p.score for p in phones) / len(phones) if phones else 0.0
        aligned_words, aligned_phones, alignment_conf = ctc_align_prompt(
            logits=logits,
            prompt=prompt,
            tokenizer=align_tokenizer,
            waveform_len_samples=int(waveform.shape[0]),
            sample_rate=16000,
        )
        fb_thr = float(self._calib["feedback"].get("weak_phone_threshold", 85.0))
        feedback_items = build_feedback(aligned_phones, weak_threshold=fb_thr)

        content_score = accuracy_like * 100.0
        accent_score = float(accent["accent_score"])
        accent_source = str(accent.get("accent_source", "centroid"))
        accent_centroid = accent.get("accent_score_centroid")
        phoneme_score = round(phoneme_avg * 100, 2)
        w = self._calib["overall"]
        wc = float(w.get("w_content", 0.2))
        wa = float(w.get("w_accent", 0.6))
        wp = float(w.get("w_phoneme", 0.2))
        s = wc + wa + wp
        if s <= 0:
            wc, wa, wp, s = 0.2, 0.6, 0.2, 1.0
        overall_score = round(
            (wc / s) * content_score + (wa / s) * accent_score + (wp / s) * phoneme_score,
            2,
        )
        content_mismatch_gate = content_score < 30.0
        if content_mismatch_gate:
            overall_score = 0.0
        level = classify_level(overall_score / 100.0)

        content_path_str = ""
        if self.dual_enabled and self.content_model is not None:
            # best-effort path for response (from registry)
            dm = load_model_registry().get("dual_model") or {}
            content_path_str = str(dm.get("content_model_path", ""))

        return {
            "model_key": self.active_model_key,
            "model_path": str(self.accent_model_dir),
            "dual_mode": bool(self.dual_enabled),
            "content_asr_model_key": self.content_asr_key if self.dual_enabled else "",
            "content_asr_model_path": content_path_str,
            "prompt": prompt,
            "transcript": transcript,
            "wer": wer,
            "accuracy_like": accuracy_like,
            "level": level,
            "confidence": confidence,
            "content_score": round(content_score, 2),
            "accent_score": round(accent_score, 2),
            "accent_source": accent_source,
            "accent_score_centroid": round(float(accent_centroid), 2)
            if accent_centroid is not None
            else None,
            "accent_similarity": round(accent["accent_similarity"], 6),
            "accent_distance": round(accent["accent_distance"], 6),
            "accent_ml_active": self.accent_ml_active,
            "alignment_confidence": round(alignment_conf * 100, 2),
            "phoneme_score": phoneme_score,
            "overall_score": overall_score,
            "content_mismatch_gate": content_mismatch_gate,
            "word_scores": [{"word": w.word, "score": round(w.score * 100, 2)} for w in words],
            "phoneme_scores": [
                {
                    "word": p.word,
                    "reference_phones": p.reference_phones,
                    "hypothesis_phones": p.hypothesis_phones,
                    "score": round(p.score * 100, 2),
                }
                for p in phones
            ],
            "alignment_word_timestamps": aligned_words,
            "alignment_phone_timestamps": aligned_phones,
            "feedback": feedback_items,
        }
