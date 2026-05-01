import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor

from app.accent import AccentScorer
from app.config import DATA_ROOT


def label_from_score(score: float) -> str:
    if score >= 82.0:
        return "native_like"
    if score >= 68.0:
        return "mixed"
    return "non_native_like"


def split_name(i: int, n: int) -> str:
    if i < int(0.8 * n):
        return "train"
    if i < int(0.9 * n):
        return "val"
    return "test"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=r"c:\Accent_Cursor\runs\hubert_stage3_large_ft\final")
    p.add_argument("--out-manifest", default=r"c:\Accent_Cursor\data\labeled_accent\manifest_bootstrap.csv")
    p.add_argument("--split-url", default="dev-clean")
    p.add_argument("--max-samples", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    root = Path(DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    ds = torchaudio.datasets.LIBRISPEECH(
        root=str(root),
        url=args.split_url,
        download=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir).to(device).eval()
    scorer = AccentScorer(model=model, processor=processor, device=device)

    walker = list(getattr(ds, "_walker", []))
    if not walker:
        raise RuntimeError("LibriSpeech walker not available.")

    # Shuffle deterministically and keep a compact subset for 4GB GPUs.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(walker), generator=g).tolist()
    selected = perm[: min(args.max_samples, len(perm))]

    rows: List[Tuple[str, str, float, str, str, str]] = []
    n = len(selected)
    for j, idx in enumerate(selected):
        waveform, sr, transcript, speaker_id, _chapter_id, _utt_id = ds[idx]
        wav = waveform.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        emb = scorer._embed_waveform(wav.to(device))
        acc = scorer.score_embedding(emb)
        score = float(acc["accent_score"])
        label = label_from_score(score)

        fileid = str(walker[idx])
        parts = fileid.split("-")
        if len(parts) < 3:
            continue
        speaker = parts[0]
        chapter = parts[1]
        rel = Path(args.split_url) / speaker / chapter / f"{fileid}.flac"
        audio_path = str(root / "LibriSpeech" / rel)
        rows.append(
            (
                audio_path,
                transcript,
                round(score, 2),
                label,
                f"spk_{speaker_id}",
                split_name(j, n),
            )
        )
        if (j + 1) % 25 == 0:
            print(f"processed={j+1}/{n}", flush=True)

    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "prompt_text", "accent_score", "accent_label", "speaker_id", "split"])
        w.writerows(rows)

    print(f"wrote_manifest={out_manifest} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

