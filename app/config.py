import json
import os
from pathlib import Path
from typing import Dict, Tuple


BASE_DIR = Path(os.getenv("ACCENT_BASE_DIR", Path(__file__).resolve().parents[1]))
REGISTRY_PATH = BASE_DIR / "model_registry.json"
FALLBACK_MODEL_DIR = BASE_DIR / "runs" / "hubert_stage3_large_ft" / "final"
DATA_ROOT = BASE_DIR / "data" / "librispeech"
ACCENT_REF_CACHE = BASE_DIR / "runs" / "accent_reference_stats.pt"
SCORING_CALIB_PATH = BASE_DIR / "scoring_calib.json"


def load_scoring_calib() -> Dict:
    defaults = {
        "accent": {
            "z_baseline": 90.0,
            "z_slope": 8.0,
            "min_score": 20.0,
            "max_score": 99.0,
        },
        "overall": {"w_content": 0.2, "w_accent": 0.6, "w_phoneme": 0.2},
        "feedback": {"weak_phone_threshold": 85.0},
        "accent_ml": {
            "enabled": True,
            "head_path": str(BASE_DIR / "runs" / "accent_head" / "accent_head_best.pt"),
            "calibration_path": str(BASE_DIR / "runs" / "accent_head" / "calibration.json"),
        },
    }
    if not SCORING_CALIB_PATH.exists():
        return defaults
    try:
        data = json.loads(SCORING_CALIB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults
    acc = {**defaults["accent"], **(data.get("accent") or {})}
    ov = {**defaults["overall"], **(data.get("overall") or {})}
    fb = {**defaults["feedback"], **(data.get("feedback") or {})}
    aml = {**defaults["accent_ml"], **(data.get("accent_ml") or {})}
    return {"accent": acc, "overall": ov, "feedback": fb, "accent_ml": aml}


def resolve_calib_path(path_str: str) -> Path:
    """Paths in scoring_calib may be absolute or relative to BASE_DIR."""
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)


def load_model_registry() -> Dict:
    if not REGISTRY_PATH.exists():
        return {"active_model": "", "models": {}}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def resolve_active_model() -> Tuple[str, Path, Dict]:
    registry = load_model_registry()
    active_key = registry.get("active_model", "")
    models = registry.get("models", {})
    model_info = models.get(active_key, {})

    model_dir_str = model_info.get("path")
    if not model_dir_str:
        return "fallback", FALLBACK_MODEL_DIR, {}
    return active_key, Path(model_dir_str), model_info


def resolve_content_asr_model(registry: Dict) -> Tuple[bool, str, Path | None]:
    """Wav2Vec2 (or other CTC) for transcript / WER / alignment when dual_model is enabled."""
    dm = registry.get("dual_model") or {}
    if not dm.get("enabled", False):
        return False, "", None
    key = str(dm.get("content_model_key", "content_asr"))
    path_str = dm.get("content_model_path") or ""
    if not path_str:
        return False, "", None
    p = Path(path_str)
    if not p.exists():
        return False, "", None
    return True, key, p
