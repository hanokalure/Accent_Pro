import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LibriSpeechForCTC(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str):
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        waveform, sample_rate, transcript, *_ = self.ds[idx]
        transcript = normalize_text(transcript)
        return waveform.squeeze(0), sample_rate, transcript


@dataclass
class Collator:
    processor: AutoProcessor

    def __call__(self, batch):
        waves: List[torch.Tensor] = []
        texts: List[str] = []
        for w, sr, t in batch:
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            waves.append(w.numpy())
            texts.append(t)

        inputs = self.processor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        labels = self.processor(text=texts, return_tensors="pt", padding=True).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"c:\Accent_Cursor\data\librispeech")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--model-id", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--processor-id", default="")
    parser.add_argument("--outdir", default=r"c:\Accent_Cursor\runs\stage1_wav2vec2_ctc")
    parser.add_argument("--subset-size", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--init-lm-head-from", default="")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this training setup.")

    processor_source = args.processor_id if args.processor_id else args.model_id
    processor = AutoProcessor.from_pretrained(processor_source)
    model_source = args.resume_from if args.resume_from else args.model_id
    model = AutoModelForCTC.from_pretrained(model_source).to(device)
    if args.init_lm_head_from and not args.resume_from:
        donor = AutoModelForCTC.from_pretrained(args.init_lm_head_from)
        if (
            model.lm_head.weight.shape == donor.lm_head.weight.shape
            and model.lm_head.bias.shape == donor.lm_head.bias.shape
        ):
            model.lm_head.weight.data.copy_(donor.lm_head.weight.data)
            model.lm_head.bias.data.copy_(donor.lm_head.bias.data)
            print(f"initialized_lm_head_from={args.init_lm_head_from}", flush=True)
        else:
            print("lm_head_shape_mismatch_skipping_init", flush=True)
    model.freeze_feature_encoder()
    model.train()

    dataset = LibriSpeechForCTC(args.data_root, args.split)
    subset_size = min(args.subset_size, len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset = Subset(dataset, indices[:subset_size])

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=Collator(processor),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    opt_step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            step += 1

            if step % 10 == 0:
                avg = running / 10.0
                running = 0.0
                print(f"step={step} loss={avg:.4f}", flush=True)

            if step % args.save_every == 0:
                ckpt_dir = os.path.join(args.outdir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                print(f"saved={ckpt_dir}", flush=True)

    # Flush remaining accumulated gradients when max_steps is not divisible by grad_accum.
    if step % args.grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        print(f"flushed_final_gradients optimizer_steps={opt_step}", flush=True)

    final_dir = os.path.join(args.outdir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"training_complete final={final_dir}", flush=True)


if __name__ == "__main__":
    main()
