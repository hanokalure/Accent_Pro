# 🎙️ Accent Scoring Project

An end-to-end pronunciation scoring system powered by **HuBERT** and **Wav2Vec2**, designed to evaluate spoken English with detailed pronunciation analysis and coaching feedback.

## ✨ Features

- 🎤 Automatic Speech Recognition (ASR)
- 🗣️ Accent & pronunciation scoring
- 🔤 Word and phoneme-level scores
- ⏱️ CTC word timestamps
- 💡 Personalized pronunciation tips
- 🌐 FastAPI backend with browser demo

---

## 🤖 Active Model

The production model is configured in `model_registry.json`.

**Current Model**
- **Key:** `hubert_stage3_large_ft_v1`
- **Path:** `c:\Accent_Cursor\runs\hubert_stage3_large_ft\final`

---

## ⚙️ Dual Model Mode

Enable `dual_model.enabled = true` in `model_registry.json`.

| Model | Purpose |
|-------|---------|
| 📝 **Wav2Vec2** | Transcript, WER, confidence & CTC alignment |
| 🎯 **HuBERT** | Accent embedding & pronunciation scoring |

> 💻 For systems with **4 GB GPU memory**, disable dual mode for smoother performance.

---

## 📊 Scoring Weights

Default overall score:

- 📝 **Content:** 42%
- 🎯 **Accent:** 35%
- 🔤 **Phoneme:** 23%

You can customize these weights in **`scoring_calib.json`** without changing the code.

> 🔄 Restart the API after editing the calibration file.

---

## 🏋️ Training the Accent Head

Train using a labeled pronunciation dataset:

```bash
python train_accent_labeled.py --manifest data/labeled_accent/manifest_template.csv
```

Calibrate the trained model:

```bash
python calibrate_accent_head.py --manifest data/labeled_accent/manifest_template.csv
```

Generated artifacts:

- ✅ `runs/accent_head/accent_head_best.pt`
- ✅ `runs/accent_head/calibration.json`
- ✅ `runs/accent_head/metrics.json`

---

## 🚀 Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python run_api.py
```

---

## 🌍 Local URLs

| Service | URL |
|---------|-----|
| 🖥️ Frontend | http://localhost:8000 |
| ❤️ Health Check | http://localhost:8000/health |
| ℹ️ Model Info | http://localhost:8000/model-info |

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /score` | Full pronunciation analysis |
| `POST /score-compact` | Lightweight response for mobile |

### Required Form Fields

- `prompt_text`
- `audio_file`

---

## 💾 Backups

Automatic checkpoints are stored in:

```text
c:\Accent_Cursor_checkpoints\
```

Example:

```text
checkpoint_good_work_YYYYMMDD_HHMMSS
```

---

## 📝 Notes

- 📈 Overall accuracy currently uses **1 − WER** as the baseline.
- 🎯 Pronunciation coaching combines phoneme heuristics with CTC alignment.
- ⭐ For production-quality pronunciation scoring, training with **human-rated pronunciation labels** is highly recommended.
