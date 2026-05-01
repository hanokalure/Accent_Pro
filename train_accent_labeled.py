import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCTC, AutoProcessor

from app.accent_head import AccentHead


def clamp_score(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def load_rows(manifest_path: Path, split: str) -> List[Dict]:
    rows: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "").strip().lower() != split.lower():
                continue
            if not row.get("audio_path"):
                continue
            row["accent_score"] = clamp_score(float(row.get("accent_score", "0")))
            row["accent_label"] = (row.get("accent_label") or "").strip().lower()
            rows.append(row)
    return rows


class LabeledAccentDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        p = Path(row["audio_path"])
        waveform, sr = torchaudio.load(str(p))
        waveform = waveform.mean(dim=0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return {
            "waveform": waveform,
            "accent_score": float(row["accent_score"]),
            "accent_label": row["accent_label"],
            "audio_path": str(p),
        }


@dataclass
class BatchCollator:
    processor: AutoProcessor
    label_to_id: Dict[str, int]

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        waves = [x["waveform"].numpy() for x in batch]
        scores = torch.tensor([x["accent_score"] for x in batch], dtype=torch.float32)
        class_ids = []
        for x in batch:
            lab = x["accent_label"]
            class_ids.append(self.label_to_id.get(lab, -1))
        class_ids_t = torch.tensor(class_ids, dtype=torch.long)

        inputs = self.processor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs["accent_scores"] = scores
        inputs["accent_class_ids"] = class_ids_t
        return inputs


def pooled_hidden(ctc_model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        out = ctc_model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )
    hidden = out.hidden_states[-1]
    return hidden.mean(dim=1)


def evaluate(ctc_model, head, loader, device, num_classes: int):
    head.eval()
    mae_vals = []
    total_cls = 0
    hit_cls = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pooled = pooled_hidden(ctc_model, batch)
        pred_score, pred_cls = head(pooled)
        true_score = batch["accent_scores"]
        mae_vals.append(torch.mean(torch.abs(pred_score - true_score)).item())

        if num_classes > 0 and pred_cls is not None:
            true_cls = batch["accent_class_ids"]
            mask = true_cls >= 0
            if torch.any(mask):
                pred_ids = torch.argmax(pred_cls[mask], dim=-1)
                hit_cls += int((pred_ids == true_cls[mask]).sum().item())
                total_cls += int(mask.sum().item())

    mae = float(sum(mae_vals) / max(len(mae_vals), 1))
    cls_acc = float(hit_cls / total_cls) if total_cls > 0 else 0.0
    return mae, cls_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=r"c:\Accent_Cursor\data\labeled_accent\manifest_template.csv")
    p.add_argument("--model-dir", default=r"c:\Accent_Cursor\runs\hubert_stage3_large_ft\final")
    p.add_argument("--outdir", default=r"c:\Accent_Cursor\runs\accent_head")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg-weight", type=float, default=1.0)
    p.add_argument("--cls-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    manifest_path = Path(args.manifest)
    train_rows = load_rows(manifest_path, "train")
    val_rows = load_rows(manifest_path, "val")
    if not train_rows or not val_rows:
        raise RuntimeError("Need both train and val rows in manifest.")

    label_set = sorted({r["accent_label"] for r in train_rows + val_rows if r["accent_label"]})
    label_to_id = {lab: i for i, lab in enumerate(label_set)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    ctc_model = AutoModelForCTC.from_pretrained(args.model_dir).to(device).eval()
    ctc_model.requires_grad_(False)

    hidden_size = int(ctc_model.config.hidden_size)
    head = AccentHead(hidden_size=hidden_size, num_classes=len(label_to_id)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    collator = BatchCollator(processor=processor, label_to_id=label_to_id)
    train_loader = DataLoader(
        LabeledAccentDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        LabeledAccentDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    best_mae = 1e9
    best_path = Path(args.outdir) / "accent_head_best.pt"
    metrics_path = Path(args.outdir) / "metrics.json"
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                pooled = pooled_hidden(ctc_model, batch)
            pred_score, pred_cls = head(pooled)
            reg_loss = mse(pred_score, batch["accent_scores"])

            cls_loss = torch.tensor(0.0, device=device)
            if len(label_to_id) > 0 and pred_cls is not None:
                mask = batch["accent_class_ids"] >= 0
                if torch.any(mask):
                    cls_loss = ce(pred_cls[mask], batch["accent_class_ids"][mask])

            loss = args.reg_weight * reg_loss + args.cls_weight * cls_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_mae, val_cls_acc = evaluate(ctc_model, head, val_loader, device, len(label_to_id))
        train_loss = float(sum(losses) / max(len(losses), 1))
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae, 4),
            "val_cls_acc": round(val_cls_acc, 4),
        }
        history.append(row)
        print(row, flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "label_to_id": label_to_id,
                    "hidden_size": hidden_size,
                    "num_classes": len(label_to_id),
                },
                best_path,
            )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (Path(args.outdir) / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)

    print(f"training_complete best_mae={best_mae:.4f} best_path={best_path}", flush=True)


if __name__ == "__main__":
    main()
