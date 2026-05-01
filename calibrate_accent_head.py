import argparse
import csv
import json
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor

from app.accent_head import AccentHead
from train_accent_labeled import BatchCollator, LabeledAccentDataset, load_rows, pooled_hidden


def fit_affine(pred: torch.Tensor, target: torch.Tensor):
    # y = a*x + b (least squares)
    x = pred
    y = target
    xm = torch.mean(x)
    ym = torch.mean(y)
    var = torch.mean((x - xm) ** 2).clamp_min(1e-9)
    cov = torch.mean((x - xm) * (y - ym))
    a = cov / var
    b = ym - a * xm
    return float(a.item()), float(b.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=r"c:\Accent_Cursor\data\labeled_accent\manifest_template.csv")
    p.add_argument("--model-dir", default=r"c:\Accent_Cursor\runs\hubert_stage3_large_ft\final")
    p.add_argument("--head-path", default=r"c:\Accent_Cursor\runs\accent_head\accent_head_best.pt")
    p.add_argument("--out", default=r"c:\Accent_Cursor\runs\accent_head\calibration.json")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    val_rows = load_rows(Path(args.manifest), "val")
    if not val_rows:
        raise RuntimeError("No val rows found in manifest.")

    ckpt = torch.load(args.head_path, map_location="cpu")
    label_to_id = ckpt.get("label_to_id", {})
    hidden_size = int(ckpt["hidden_size"])
    num_classes = int(ckpt["num_classes"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    ctc_model = AutoModelForCTC.from_pretrained(args.model_dir).to(device).eval()
    ctc_model.requires_grad_(False)

    head = AccentHead(hidden_size=hidden_size, num_classes=num_classes).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    loader = torch.utils.data.DataLoader(
        LabeledAccentDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=BatchCollator(processor=processor, label_to_id=label_to_id),
    )

    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pooled = pooled_hidden(ctc_model, batch)
            pred_score, _ = head(pooled)
            preds.append(pred_score.detach().cpu())
            trues.append(batch["accent_scores"].detach().cpu())

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    a, b = fit_affine(pred, true)
    mae_before = float(torch.mean(torch.abs(pred - true)).item())
    calibrated = torch.clamp(a * pred + b, 0.0, 100.0)
    mae_after = float(torch.mean(torch.abs(calibrated - true)).item())

    out = {
        "affine_slope": a,
        "affine_bias": b,
        "mae_before": mae_before,
        "mae_after": mae_after,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
