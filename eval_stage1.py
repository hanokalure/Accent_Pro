import argparse
import re
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCTC, AutoProcessor


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
    return d[len(r)][len(h)] / len(r)


def collate(batch, processor):
    waves = []
    texts = []
    for w, sr, t, *_ in batch:
        if sr != 16000:
            w = torchaudio.functional.resample(w.squeeze(0), sr, 16000).unsqueeze(0)
        waves.append(w.squeeze(0).numpy())
        texts.append(normalize_text(t))
    inputs = processor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=r"c:\Accent_Cursor\runs\stage1_wav2vec2_ctc\final")
    p.add_argument("--data-root", default=r"c:\Accent_Cursor\data\librispeech")
    p.add_argument("--split", default="dev-clean")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir).to(device).eval()
    ds = torchaudio.datasets.LIBRISPEECH(root=args.data_root, url=args.split, download=False)
    n = min(args.samples, len(ds))
    subset = Subset(ds, list(range(n)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, processor),
    )

    wers = []
    with torch.no_grad():
        for i, (inputs, refs) in enumerate(loader, start=1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyps = processor.batch_decode(pred_ids)
            for r, h in zip(refs, hyps):
                h = normalize_text(h)
                wers.append(wer(r, h))
            if i % 20 == 0:
                print(f"eval_batches={i}", flush=True)

    avg_wer = sum(wers) / len(wers) if wers else 1.0
    approx_accuracy = max(0.0, 1.0 - avg_wer)
    print(f"samples={len(wers)}")
    print(f"wer={avg_wer:.4f}")
    print(f"approx_accuracy={approx_accuracy:.4f}")


if __name__ == "__main__":
    main()
