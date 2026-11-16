"""Producer's Copilot HTTP API and web UI."""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .knowledge import KnowledgeBase
from .references import GENRE_TARGETS, ReferenceStore, compare, valid_workspace
from .tips import ClaudeTipWriter, generate_tips
from .tools import AudioLoadError, analyze_audio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("copilot")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "60"))
ALLOWED_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a"}
LLM_RATE_LIMIT_PER_HOUR = int(os.environ.get("LLM_RATE_LIMIT_PER_HOUR", "20"))


def _default_data_dir() -> Path:
    # Hugging Face Spaces mounts persistent storage at /data when enabled
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data")
    return Path(__file__).resolve().parents[1] / "data"


class RateLimiter:
    """Sliding-window limit per key (client IP), in memory."""

    def __init__(self, limit: int, window_s: float = 3600):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def create_app(*, store: ReferenceStore | None = None, kb: KnowledgeBase | None = None,
               writer: ClaudeTipWriter | None | bool = True) -> FastAPI:
    """`writer=True` builds a Claude writer when ANTHROPIC_API_KEY is set."""
    api = FastAPI(title="Producer's Copilot API", version="1.0.0",
                  description="DSP analysis, reference comparison and RAG production tips for your mixes.")
    api.state.store = store or ReferenceStore(_default_data_dir() / "references.json")
    api.state.kb = kb or KnowledgeBase()
    if writer is True:
        writer = ClaudeTipWriter() if os.environ.get("ANTHROPIC_API_KEY") else None
    api.state.writer = writer or None
    api.state.llm_limiter = RateLimiter(LLM_RATE_LIMIT_PER_HOUR)
    log.info("Tip engine: %s", "claude" if api.state.writer else "rules")

    # --- helpers -----------------------------------------------------------

    def analyze_upload(upload: UploadFile) -> dict:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{suffix or '?'}'. "
                                     f"Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            size = 0
            while chunk := upload.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    tmp.close()
                    os.unlink(tmp.name)
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                tmp.write(chunk)
            path = tmp.name
        try:
            return analyze_audio(path)
        except AudioLoadError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            os.unlink(path)

    def require_workspace(workspace: str | None) -> str:
        if not valid_workspace(workspace):
            raise HTTPException(400, "Missing or invalid X-Workspace header")
        return workspace

    def resolve_reference(workspace: str | None, reference_id: str | None,
                          target_id: str | None) -> tuple[dict | None, str | None]:
        if reference_id:
            ref = api.state.store.get(require_workspace(workspace), reference_id)
            if not ref:
                raise HTTPException(404, "Reference not found")
            return ref["metrics"], ref["name"]
        if target_id:
            target = GENRE_TARGETS.get(target_id)
            if not target:
                raise HTTPException(404, "Unknown genre target")
            return target["metrics"], f"the {target['name']} profile"
        return None, None

    # --- routes ------------------------------------------------------------

    @api.get("/health")
    def health():
        return {"status": "ok", "tip_engine": "claude" if api.state.writer else "rules",
                "knowledge_chunks": len(api.state.kb.chunks)}

    @api.post("/analyze")
    def analyze(file: UploadFile = File(...)):
        """Analyze an uploaded audio file and return DSP metrics."""
        return analyze_upload(file)

    @api.get("/targets")
    def list_targets():
        """Built-in genre profiles usable as references."""
        return {"targets": [{"id": k, "name": v["name"], "description": v["description"],
                             "metrics": v["metrics"]} for k, v in GENRE_TARGETS.items()]}

    @api.post("/references")
    def add_reference(file: UploadFile = File(...), name: str = Form(""),
                      x_workspace: str | None = Header(None)):
        """Analyze a reference track and store its metrics (not the audio)."""
        workspace = require_workspace(x_workspace)
        metrics = analyze_upload(file)
        label = name.strip() or Path(file.filename or "reference").stem
        return api.state.store.add(workspace, label, metrics)

    @api.get("/references")
    def list_references(x_workspace: str | None = Header(None)):
        return {"references": api.state.store.list(require_workspace(x_workspace))}

    @api.delete("/references/{ref_id}")
    def delete_reference(ref_id: str, x_workspace: str | None = Header(None)):
        if not api.state.store.delete(require_workspace(x_workspace), ref_id):
            raise HTTPException(404, "Reference not found")
        return {"deleted": ref_id}

    @api.post("/tips")
    def production_tips(request: Request, file: UploadFile = File(...),
                        reference_id: str | None = Form(None), target_id: str | None = Form(None),
                        use_llm: bool = Form(True), x_workspace: str | None = Header(None)):
        """Analyze a track, compare it with a reference or genre profile, and
        return production tips grounded in the knowledge base."""
        reference, reference_name = resolve_reference(x_workspace, reference_id or None, target_id or None)
        metrics = analyze_upload(file)
        comparison = compare(metrics, reference) if reference else []

        writer = api.state.writer if use_llm else None
        rate_limited = False
        if writer is not None:
            client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                request.client.host if request.client else "unknown")
            if not api.state.llm_limiter.allow(client_ip):
                writer, rate_limited = None, True

        result = generate_tips(metrics, api.state.kb, reference=reference, reference_name=reference_name,
                               comparison=comparison, writer=writer)
        if rate_limited:
            result["llm_error"] = "Hourly limit for AI-written tips reached; showing rule-based tips."
        return {"metrics": metrics, "reference_name": reference_name, "comparison": comparison, **result}

    # --- web UI ------------------------------------------------------------

    api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @api.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return api


app = create_app()
