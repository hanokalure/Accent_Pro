from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

from app.config import ACCENT_REF_CACHE, DATA_ROOT, load_scoring_calib


class AccentScorer:
    def __init__(
        self,
        model,
        processor,
        device: str,
        cache_path: Path = ACCENT_REF_CACHE,
        data_root: Path = DATA_ROOT,
        split: str = "train-clean-100",
        max_samples: int = 120,
        per_speaker_cap: int = 1,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.cache_path = cache_path
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        self.per_speaker_cap = per_speaker_cap
        self._calib = load_scoring_calib()["accent"]
        self._stats = self._load_or_build()

    def _embed_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        inputs = self.processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [B, T, H]
        return hidden.mean(dim=1).squeeze(0).detach().cpu()

    def _load_or_build(self) -> Dict[str, torch.Tensor | float]:
        if self.cache_path.exists():
            return torch.load(self.cache_path, map_location="cpu")

        libri_root = self.data_root / "LibriSpeech"
        allow_download = os.getenv("ACCENT_BUILD_DOWNLOAD_LIBRISPEECH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not libri_root.exists() and not allow_download:
            raise RuntimeError(
                "Accent reference stats cache is missing and LibriSpeech is not available. "
                "Include accent_reference_stats.pt in your model bundle (bootstrap copies it to "
                f"{self.cache_path}), set ACCENT_REF_CACHE to an existing .pt file, or place "
                f"LibriSpeech under {self.data_root} and set ACCENT_BUILD_DOWNLOAD_LIBRISPEECH=1."
            )

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds = torchaudio.datasets.LIBRISPEECH(
            root=str(self.data_root),
            url=self.split,
            download=allow_download,
        )
        embeds: List[torch.Tensor] = []
        speaker_counts: Dict[int, int] = defaultdict(int)
        selected_indices = self._select_balanced_indices(ds)

        for i in selected_indices:
            waveform, sr, _txt, speaker_id, *_rest = ds[i]
            speaker_id = int(speaker_id)
            w = waveform.mean(dim=0)
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            embeds.append(self._embed_waveform(w))
            speaker_counts[speaker_id] += 1

        if not embeds:
            raise RuntimeError("Failed to build accent reference: no embeddings collected.")

        mat = torch.stack(embeds, dim=0)
        centroid = mat.mean(dim=0)
        centroid = centroid / centroid.norm(p=2).clamp_min(1e-9)
        sims = torch.nn.functional.cosine_similarity(mat, centroid.unsqueeze(0), dim=1)
        stats = {
            "centroid": centroid,
            "sim_mean": float(sims.mean().item()),
            "sim_std": float(sims.std(unbiased=False).item()),
            "num_samples": len(embeds),
            "num_speakers": len(speaker_counts),
        }
        torch.save(stats, self.cache_path)
        return stats

    def _select_balanced_indices(self, ds) -> List[int]:
        """
        Select sample indices with per-speaker cap without decoding every waveform.
        LibriSpeech walker items are file ids like 'speaker-chapter-utterance'.
        """
        selected: List[int] = []
        speaker_counts: Dict[int, int] = defaultdict(int)
        walker = getattr(ds, "_walker", None)
        if not walker:
            return list(range(min(self.max_samples, len(ds))))

        for idx, fileid in enumerate(walker):
            if len(selected) >= self.max_samples:
                break
            try:
                speaker_id = int(str(fileid).split("-")[0])
            except Exception:
                continue
            if speaker_counts[speaker_id] >= self.per_speaker_cap:
                continue
            selected.append(idx)
            speaker_counts[speaker_id] += 1
        return selected

    def score_embedding(self, embedding: torch.Tensor) -> Dict[str, float]:
        emb = embedding.detach().cpu()
        emb = emb / emb.norm(p=2).clamp_min(1e-9)
        centroid = self._stats["centroid"]
        sim = float(torch.dot(emb, centroid).item())
        dist = float(1.0 - sim)

        mean = float(self._stats["sim_mean"])
        std = float(self._stats["sim_std"]) if float(self._stats["sim_std"]) > 1e-6 else 1e-6
        z = (sim - mean) / std
        base = float(self._calib.get("z_baseline", 90.0))
        slope = float(self._calib.get("z_slope", 8.0))
        lo = float(self._calib.get("min_score", 20.0))
        hi = float(self._calib.get("max_score", 99.0))
        accent_score = max(lo, min(hi, base + slope * z))
        return {
            "accent_similarity": sim,
            "accent_distance": dist,
            "accent_score": accent_score,
        }
