"""
Bootstrap models + calibration files before uvicorn starts.

Modes:
- USE_DRIVE_ZIP=true: download one zip from Google Drive, unzip under deployed_models/
- USE_DRIVE_WEIGHTS=true: download model.safetensors from Drive + fetch tokenizer/config skeletons
- else: snapshot_download from HF_MODEL_REPO_ID

Writes runtime model_registry.json + scoring_calib.json under ACCENT_BASE_DIR (/app).
"""

import json
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gdown
from huggingface_hub import snapshot_download


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _extract_drive_file_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if not s:
        return None
    # Already looks like a bare id
    if "/" not in s and len(s) >= 20:
        return s
    parsed = urlparse(s)
    q = parse_qs(parsed.query)
    if "id" in q and q["id"]:
        return q["id"][0]
    parts = [p for p in parsed.path.split("/") if p]
    if "d" in parts:
        try:
            return parts[parts.index("d") + 1]
        except Exception:
            return None
    return None


def _is_zip_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def _download_drive_zip(drive_url_or_id: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = _extract_drive_file_id(drive_url_or_id)
    url = drive_url_or_id.strip()
    print(f"drive_zip_download_start id={file_id}", flush=True)

    # Prefer gdown (handles large-file confirm flows reliably).
    try:
        if file_id:
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
        else:
            gdown.download(url=url, output=str(zip_path), quiet=False)
    except Exception as exc:
        raise RuntimeError(f"gdown zip download failed: {exc}") from exc

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise RuntimeError("Zip download produced empty file.")

    if not _is_zip_file(zip_path):
        head = zip_path.read_bytes()[:200]
        raise RuntimeError(
            "Downloaded file is not a zip (likely HTML from Drive). "
            f"First bytes: {head!r}"
        )


def _unzip(zip_path: Path, out_dir: Path) -> None:
    print("extracting_models_zip...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(out_dir))


def _ensure_accent_ref_cache(base_dir: Path, deployed_models: Path) -> None:
    """Place accent centroid stats where AccentScorer expects them (/app/runs/...)."""
    target = base_dir / "runs" / "accent_reference_stats.pt"
    if target.exists():
        return
    candidates = [
        deployed_models / "hf_models" / "accent_reference_stats.pt",
        deployed_models / "hf_models" / "runs" / "accent_reference_stats.pt",
        deployed_models / "accent_reference_stats.pt",
    ]
    for src in candidates:
        if src.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            print(f"accent_cache_installed from={src}", flush=True)
            return


def _ensure_configs_from_base(model_dir: Path, base_model_id: str) -> None:
    if not base_model_id:
        return
    snapshot_download(
        repo_id=base_model_id,
        local_dir=str(model_dir),
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "special_tokens_map.json",
        ],
    )


def _download_drive_weights(url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    if not url.strip():
        raise RuntimeError("Missing drive URL for weights download.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = _extract_drive_file_id(url)
    print(f"drive_weights_download target={target_path.name}", flush=True)
    try:
        if file_id:
            gdown.download(id=file_id, output=str(target_path), quiet=False)
        else:
            gdown.download(url=url.strip(), output=str(target_path), quiet=False)
    except Exception as exc:
        raise RuntimeError(f"gdown weights download failed: {exc}") from exc

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise RuntimeError("Weights download produced empty file.")


def main() -> None:
    base_dir = Path(os.getenv("ACCENT_BASE_DIR", "/app"))
    enable_dual = _env_bool("ENABLE_DUAL_MODEL", False)

    use_drive_zip = _env_bool("USE_DRIVE_ZIP", False)
    drive_zip_url = os.getenv("DRIVE_ZIP_URL", "").strip()

    use_drive_weights = _env_bool("USE_DRIVE_WEIGHTS", False)
    hubert_drive_url = os.getenv("HUBERT_DRIVE_URL", "")
    wav2vec_drive_url = os.getenv("WAV2VEC_DRIVE_URL", "")
    hubert_base_id = os.getenv("HUBERT_BASE_ID", "facebook/hubert-large-ls960-ft")
    wav2vec_base_id = os.getenv("WAV2VEC_BASE_ID", "facebook/wav2vec2-base-960h")

    repo_id = os.getenv("HF_MODEL_REPO_ID", "").strip()
    revision = os.getenv("HF_MODEL_REPO_REVISION", "main").strip() or "main"

    models_root = base_dir / "deployed_models"
    models_root.mkdir(parents=True, exist_ok=True)

    snapshot_path: Path
    repo_meta = repo_id
    revision_meta = revision

    if use_drive_zip:
        if not drive_zip_url:
            raise RuntimeError("USE_DRIVE_ZIP=true but DRIVE_ZIP_URL is missing.")
        zip_path = models_root / "accent_models_bundle.zip"
        _download_drive_zip(drive_zip_url, zip_path)
        _unzip(zip_path, models_root)
        zip_path.unlink(missing_ok=True)
        snapshot_path = models_root
        repo_meta = "drive_zip"
        revision_meta = "n/a"
    elif use_drive_weights:
        # Expect hf_models tree already present OR created after downloading weights.
        snapshot_path = models_root
        root_prefix = snapshot_path / "hf_models"
        hubert_dir = root_prefix / "hubert_stage3_large_ft" / "final"
        wav2vec2_dir = root_prefix / "stage2_wav2vec2_ctc" / "final"

        _download_drive_weights(hubert_drive_url, hubert_dir / "model.safetensors")
        if enable_dual:
            _download_drive_weights(wav2vec_drive_url, wav2vec2_dir / "model.safetensors")
        _ensure_configs_from_base(hubert_dir, hubert_base_id)
        if enable_dual:
            _ensure_configs_from_base(wav2vec2_dir, wav2vec_base_id)

        repo_meta = "drive_weights"
        revision_meta = "n/a"
    else:
        if not repo_id:
            raise RuntimeError("HF_MODEL_REPO_ID is required unless USE_DRIVE_ZIP or USE_DRIVE_WEIGHTS is enabled.")
        local_snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(models_root),
            local_dir_use_symlinks=False,
        )
        snapshot_path = Path(local_snapshot)
        revision_meta = revision

    root_prefix = snapshot_path / "hf_models"
    hubert_dir = root_prefix / "hubert_stage3_large_ft" / "final"
    wav2vec2_dir = root_prefix / "stage2_wav2vec2_ctc" / "final"
    accent_head_path = root_prefix / "accent_head" / "accent_head_best.pt"
    accent_calib_path = root_prefix / "accent_head" / "calibration.json"

    _ensure_accent_ref_cache(base_dir, models_root)

    hubert_weights = hubert_dir / "model.safetensors"
    wav2vec_weights = wav2vec2_dir / "model.safetensors"

    if not hubert_weights.exists():
        raise RuntimeError(
            "Missing HuBERT weights at hf_models/hubert_stage3_large_ft/final/model.safetensors"
        )

    model_registry = {
        "active_model": "hubert_stage3_large_ft_v1",
        "final_model": "hubert_stage3_large_ft_v1",
        "dual_model": {
            "enabled": bool(enable_dual and wav2vec_weights.exists()),
            "content_model_key": "wav2vec2_stage2_best",
            "content_model_path": str(wav2vec2_dir),
        },
        "models": {
            "hubert_stage3_large_ft_v1": {
                "path": str(hubert_dir),
                "family": "HuBERT",
                "task": "CTC ASR for pronunciation scoring backend",
                "status": "deployed_hf_space",
            }
        },
    }

    scoring_calib = {
        "accent_ml": {
            "enabled": accent_head_path.exists(),
            "head_path": str(accent_head_path),
            "calibration_path": str(accent_calib_path),
        },
        "accent": {"z_baseline": 86.0, "z_slope": 6.5, "min_score": 35.0, "max_score": 98.0},
        "overall": {"w_content": 0.45, "w_accent": 0.3, "w_phoneme": 0.25},
        "feedback": {"weak_phone_threshold": 72.0},
    }

    (base_dir / "model_registry.json").write_text(json.dumps(model_registry, indent=2), encoding="utf-8")
    (base_dir / "scoring_calib.json").write_text(json.dumps(scoring_calib, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "bootstrapped": True,
                "source": repo_meta,
                "revision": revision_meta,
                "dual_enabled": model_registry["dual_model"]["enabled"],
                "accent_ml_enabled": scoring_calib["accent_ml"]["enabled"],
                "use_drive_zip": use_drive_zip,
                "use_drive_weights": use_drive_weights,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
