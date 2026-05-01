from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from app.schemas import ScoreResponse, compact_payload
from app.service import PronunciationService


app = FastAPI(title="Accent Scoring API", version="0.1.0")
svc = PronunciationService()
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
legacy_static = Path(__file__).parent / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
    # Backward-compat for older links while React build is active.
    app.mount("/static", StaticFiles(directory=str(frontend_dist / "assets")), name="static")
else:
    app.mount("/static", StaticFiles(directory=str(legacy_static)), name="static")


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


def _save_upload_to_temp(upload: UploadFile, data: bytes) -> Path:
    suffix = Path(upload.filename or "input.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def home():
    if frontend_dist.exists():
        return FileResponse(frontend_dist / "index.html")
    return FileResponse(legacy_static / "index.html")


@app.get("/model-info")
def model_info():
    return {
        "model_key": svc.active_model_key,
        "model_path": str(svc.model_dir),
        "dual_mode": getattr(svc, "dual_enabled", False),
        "content_asr_model_key": getattr(svc, "content_asr_key", ""),
        "metadata": svc.model_info,
        "device": svc.device,
        "accent_ml_active": getattr(svc, "accent_ml_active", False),
    }


@app.post("/score", response_model=ScoreResponse)
async def score(prompt_text: str = Form(...), audio_file: UploadFile = File(...)):
    data = await audio_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    tmp_path = _save_upload_to_temp(audio_file, data)

    try:
        result = svc.score_file(tmp_path, prompt_text)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/score-compact")
async def score_compact(prompt_text: str = Form(...), audio_file: UploadFile = File(...)):
    data = await audio_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    tmp_path = _save_upload_to_temp(audio_file, data)

    try:
        result = svc.score_file(tmp_path, prompt_text)
        return compact_payload(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
