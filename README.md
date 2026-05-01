# Accent Scoring Project

End-to-end pronunciation scoring prototype with:
- HuBERT-based ASR model inference
- word/phoneme scoring
- CTC alignment timestamps
- coaching feedback tips
- FastAPI backend + browser demo frontend

## Active Production Model

Configured in `model_registry.json`:
- key: `hubert_stage3_large_ft_v1`
- path: `c:\Accent_Cursor\runs\hubert_stage3_large_ft\final`

## Checkpoints

Periodic backups are stored under `c:\Accent_Cursor_checkpoints\` (folder + zip).  
Example: `checkpoint_good_work_YYYYMMDD_HHMMSS`.

## Tunable scoring

Edit `scoring_calib.json` to adjust accent curve and overall weights without code changes. Restart the server after edits.

## Dual-model (Wav2Vec2 + HuBERT)

When `dual_model.enabled` is true in `model_registry.json`:

- **Wav2Vec2** (`content_model_path`): transcript, WER, confidence, CTC alignment.
- **HuBERT** (active model): accent embedding vs native centroid.

Disable dual mode (`"enabled": false`) if GPU memory is tight (4 GB laptops).

Default overall blend (see `scoring_calib.json`): roughly **42% content / 35% accent / 23% phoneme**.

## Labeled accent training (new)

Use labeled manifest template:
- `data/labeled_accent/manifest_template.csv`

Train accent head (regression + optional class):

```bash
python train_accent_labeled.py --manifest "c:\Accent_Cursor\data\labeled_accent\manifest_template.csv"
```

Calibrate head output on validation split:

```bash
python calibrate_accent_head.py --manifest "c:\Accent_Cursor\data\labeled_accent\manifest_template.csv"
```

Artifacts:
- `runs/accent_head/accent_head_best.pt`
- `runs/accent_head/metrics.json`
- `runs/accent_head/calibration.json`

## Run

```bash
python -m pip install -r requirements.txt
python run_api.py
```

Open:
- frontend: `http://localhost:8000`
- health: `http://localhost:8000/health`
- model info: `http://localhost:8000/model-info`

## API Endpoints

- `POST /score`  
  Full response (typed schema) with transcript, scores, timestamps, feedback.

- `POST /score-compact`  
  Smaller payload for mobile/lightweight clients.

Form fields:
- `prompt_text` (string)
- `audio_file` (file, wav/webm recommended)

## Notes

- Current overall accuracy is ASR-derived (`1 - WER`) baseline.
- Pronunciation coaching layer uses alignment + phoneme heuristics.
- For strict pronunciation benchmarking, add human-rated pronunciation labels.
