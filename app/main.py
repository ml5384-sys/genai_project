"""JobFit FastAPI service.

Endpoints
  GET  /health                 service + index status
  POST /match                  one resume vs one job description (JSON)
  POST /rank                   rank the job database for a resume (JSON)
  POST /rank/upload            same, resume uploaded as PDF/DOCX/TXT (multipart)
  POST /extract/jd             structured JD extraction
  POST /extract/resume         structured resume extraction
  GET  /jobs/{job_id}          one posting (metadata, extracted profile, description)
  GET  /jobs/{job_id}/match    score a resume (query/body) against a stored posting
  GET  /                       web demo
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engine import JobFitEngine
from .resume_io import UnsupportedFile, file_to_text
from .schemas import ExtractRequest, JDProfile, MatchRequest, MatchResult, RankRequest, RankResponse, ResumeProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = JobFitEngine()
    yield
    state.clear()


app = FastAPI(
    title="JobFit API",
    version="1.0.0",
    description="Rank entry-level job postings by how well a resume covers their stated requirements. "
                "Scores are transparent and repeatable; they measure requirement coverage, not hiring probability.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def engine() -> JobFitEngine:
    return state["engine"]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "index": e.manifest, "extractor": e.extractor.name,
            "llm_enabled": settings.llm_enabled, "llm_model": settings.llm_model if settings.llm_enabled else None,
            "embed_model": e.emb.model_name}


@app.post("/match", response_model=dict)
def match(req: MatchRequest):
    """Score one resume against one job description, with per-requirement evidence."""
    m, jd, res = engine().match(req.resume_text, req.job_description, req.job_title, explain_it=req.explain)
    return {"match": m.model_dump(), "job_profile": jd.model_dump(), "resume_profile": res.model_dump()}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    """Rank every posting in the database for this resume (optionally filtered)."""
    return engine().rank(req)


@app.post("/rank/upload", response_model=RankResponse)
async def rank_upload(file: UploadFile = File(...), top_k: int = Form(10), explain_top: int = Form(3),
                      levels: Optional[str] = Form(None), functions: Optional[str] = Form(None),
                      remote_only: bool = Form(False), location_contains: Optional[str] = Form(None)):
    data = await file.read()
    if len(data) > 5_000_000:
        raise HTTPException(413, "File too large (max 5 MB)")
    try:
        text = file_to_text(file.filename or "", data)
    except UnsupportedFile as e:
        raise HTTPException(415, str(e))
    if len(text.strip()) < 20:
        raise HTTPException(422, "Could not read any text from the file (scanned PDF?)")
    req = RankRequest(resume_text=text, top_k=top_k, explain_top=explain_top,
                      levels=[s.strip() for s in levels.split(",")] if levels else None,
                      functions=[s.strip() for s in functions.split(",")] if functions else None,
                      remote_only=remote_only, location_contains=location_contains or None)
    return engine().rank(req)


@app.post("/extract/jd", response_model=JDProfile)
def extract_jd(req: ExtractRequest):
    return engine().extractor.extract_jd(req.text, req.title)


@app.post("/extract/resume", response_model=ResumeProfile)
def extract_resume(req: ExtractRequest):
    return engine().extractor.extract_resume(req.text)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    e = engine()
    i = e.by_id.get(job_id)
    if i is None:
        raise HTTPException(404, "Unknown job_id")
    return {"meta": e.meta[i], "profile": e.profiles[i].model_dump(), "description": e.description(job_id)}


@app.post("/jobs/{job_id}/match", response_model=MatchResult)
def match_stored(job_id: str, resume_text: str = Body(..., embed=True, min_length=20)):
    e = engine()
    i = e.by_id.get(job_id)
    if i is None:
        raise HTTPException(404, "Unknown job_id")
    return e.score_job(i, e.resume_context(resume_text), with_explanation=True)
