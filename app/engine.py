"""JobFit engine: loads the pre-built job index and serves /match and /rank."""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np

from .config import settings
from .embed import get_embedder
from .explain import explain
from .extract import RuleExtractor, get_extractor
from .schemas import JDProfile, MatchResult, RankRequest, RankResponse, ResumeProfile
from .scoring import BatchScorer, ResumeContext, score_pair
from .skills import embed_text

log = logging.getLogger(__name__)


def _rule_extract(args) -> JDProfile:
    return RuleExtractor().extract_jd(*args)


def build_index(jobs: list[dict], out_dir: Path, extractor=None, embedder=None, log_every: int = 500,
                workers: int = 1) -> dict:
    """Extract every posting once, embed unique requirement texts + titles, and save."""
    extractor = extractor or RuleExtractor()
    emb = embedder or get_embedder()
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles: list[JDProfile] = []
    t0 = time.time()
    if workers > 1 and isinstance(extractor, RuleExtractor):
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as pool:
            profiles = list(pool.map(_rule_extract, [(j["description"], j["title"], j["job_id"]) for j in jobs],
                                     chunksize=64))
    else:
        for i, j in enumerate(jobs):
            profiles.append(extractor.extract_jd(j["description"], j["title"], j["job_id"]))
            if log_every and (i + 1) % log_every == 0:
                log.info("extracted %d/%d (%.0fs)", i + 1, len(jobs), time.time() - t0)
    log.info("extraction done in %.0fs", time.time() - t0)
    req_texts = sorted({embed_text(r.text, r.skill) for p in profiles for r in p.requirements})
    req_emb = emb.encode(req_texts) if req_texts else np.zeros((0, emb.dim), np.float32)
    title_emb = emb.encode([j["title"] for j in jobs])
    with gzip.open(out_dir / "profiles.jsonl.gz", "wt", encoding="utf-8") as f:
        for j, p in zip(jobs, profiles):
            meta = {k: j.get(k) for k in ("job_id", "title", "company", "location", "level", "work_type", "remote",
                                          "salary", "url", "functions")}
            f.write(json.dumps({"meta": meta, "profile": p.model_dump()}) + "\n")
    (out_dir / "req_texts.json").write_text(json.dumps(req_texts))
    np.save(out_dir / "req_emb.npy", req_emb.astype(np.float16))
    np.save(out_dir / "title_emb.npy", title_emb.astype(np.float16))
    manifest = {"n_jobs": len(jobs), "n_unique_requirements": len(req_texts), "embed_model": emb.model_name,
                "extractor": extractor.name, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class JobFitEngine:
    def __init__(self, index_dir: Path | None = None, data_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.index_dir)
        self.data_dir = Path(data_dir or settings.data_dir)
        self.emb = get_embedder()
        self.extractor = get_extractor()
        self.rule = RuleExtractor()
        self._descriptions: dict[str, str] | None = None
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        d = self.index_dir
        self.manifest = json.loads((d / "manifest.json").read_text())
        self.meta, self.profiles = [], []
        with gzip.open(d / "profiles.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.meta.append(rec["meta"])
                self.profiles.append(JDProfile(**rec["profile"]))
        self.req_texts = json.loads((d / "req_texts.json").read_text())
        self.req_emb = np.load(d / "req_emb.npy").astype(np.float32)
        self.title_emb = np.load(d / "title_emb.npy").astype(np.float32)
        self.req_row = {t: i for i, t in enumerate(self.req_texts)}
        self.by_id = {m["job_id"]: i for i, m in enumerate(self.meta)}
        self.batch = BatchScorer(self.profiles, self.req_texts, self.req_emb, self.title_emb)
        self._levels = np.array([m.get("level") or "" for m in self.meta])
        self._funcs = [set(m.get("functions") or []) for m in self.meta]
        log.info("loaded index: %s", self.manifest)

    def description(self, job_id: str) -> str | None:
        if self._descriptions is None:
            self._descriptions = {}
            p = self.data_dir / "jobs.jsonl.gz"
            if p.exists():
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        j = json.loads(line)
                        self._descriptions[j["job_id"]] = j["description"]
        return self._descriptions.get(job_id)

    # --------------------------------------------------------------- operations
    def resume_context(self, resume_text: str, profile: ResumeProfile | None = None) -> ResumeContext:
        profile = profile or self.extractor.extract_resume(resume_text)
        return ResumeContext.build(profile, resume_text, self.emb)

    def _req_emb_for(self, jd: JDProfile) -> np.ndarray:
        keys = [embed_text(r.text, r.skill) for r in jd.requirements]
        rows = [self.req_row.get(k) for k in keys]
        if all(r is not None for r in rows):
            return self.req_emb[rows] if rows else np.zeros((0, self.emb.dim), np.float32)
        return self.emb.encode(keys)

    def score_job(self, idx: int, ctx: ResumeContext, with_explanation: bool = False) -> MatchResult:
        jd = self.profiles[idx]
        m = score_pair(jd, ctx, self._req_emb_for(jd), self.title_emb[idx], self.meta[idx])
        if with_explanation:
            m.explanation, m.explanation_source = explain(m, jd.min_years_experience, ctx.profile.years_experience)
        return m

    def match(self, resume_text: str, jd_text: str, title: str = "", explain_it: bool = True) -> tuple[MatchResult, JDProfile, ResumeProfile]:
        jd = self.extractor.extract_jd(jd_text, title)
        ctx = self.resume_context(resume_text)
        title_emb = self.emb.encode([title or jd_text[:200]])[0]
        m = score_pair(jd, ctx, self._req_emb_for(jd), title_emb, {"title": title})
        if explain_it:
            m.explanation, m.explanation_source = explain(m, jd.min_years_experience, ctx.profile.years_experience)
        return m, jd, ctx.profile

    def filter_mask(self, req: RankRequest) -> np.ndarray:
        mask = np.ones(len(self.meta), dtype=bool)
        if req.levels:
            mask &= np.isin(self._levels, req.levels)
        if req.functions:
            want = set(req.functions)
            mask &= np.array([bool(f & want) for f in self._funcs])
        if req.remote_only:
            mask &= np.array([bool(m.get("remote")) for m in self.meta])
        if req.location_contains:
            q = req.location_contains.lower()
            mask &= np.array([q in (m.get("location") or "").lower() for m in self.meta])
        return mask

    def rank(self, req: RankRequest, ctx: ResumeContext | None = None) -> RankResponse:
        t0 = time.perf_counter()
        ctx = ctx or self.resume_context(req.resume_text)
        overall, _ = self.batch.score_all(ctx)
        mask = self.filter_mask(req)
        cand = np.flatnonzero(mask)
        order, seen = [], set()
        for i in cand[np.lexsort((-self.title_emb[cand] @ ctx.head_emb, -overall[cand]))]:
            key = (self.meta[i]["title"].lower(), (self.meta[i].get("company") or "").lower())
            if key in seen:  # LinkedIn has many re-posts of the same role
                continue
            seen.add(key)
            order.append(int(i))
            if len(order) >= req.top_k:
                break
        results = [self.score_job(int(i), ctx, with_explanation=r < req.explain_top) for r, i in enumerate(order)]
        return RankResponse(resume=ctx.profile, total_postings_scored=int(mask.sum()),
                            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1), results=results)

    def scores_for(self, resume_text: str, profile: ResumeProfile | None = None) -> np.ndarray:
        """Overall score for every posting (used by evaluation)."""
        return self.batch.score_all(self.resume_context(resume_text, profile))[0]
