"""Transparent, deterministic scoring.

For a resume R and a posting J with requirements r_1..r_n:

  strength(r_i) = 1                         if r_i's canonical skill appears in R   (exact)
                = s_max * ramp(max_e cos(r_i, e))   otherwise, e over resume evidence units (semantic;
                                            s_max = 0.85, and 0 for certifications/licenses)
  ramp(c)       = clip((c - lo) / (hi - lo), 0, 1)

  w_i           = importance_weight[r_i] * (soft_skill_factor if r_i is a soft skill else 1)
  coverage(S)   = (sum_{i in S} w_i * strength_i + k * prior) / (sum_{i in S} w_i + k)      (smoothed)

  sub-scores    required = coverage(required reqs), preferred = coverage(preferred reqs),
                experience = min(1, years_R / years_J), education = 0.6*level_fit + 0.4*field_fit,
                role = ramp_role(cos(job title, resume headline / best evidence))
  overall       = 100 * sum_c W_c * sub_c / sum_c W_c   over the sub-scores that are defined

All constants live in app/config.py.  The same formula is implemented twice: per pair
(with evidence, for display) and vectorised over the whole job database (for ranking);
tests/test_scoring.py checks they agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings
from .embed import Embedder
from .schemas import EDU_LEVELS, JDProfile, MatchResult, RequirementMatch, ResumeProfile, SubScores
from .skills import SKILLS, embed_text, find_skills, group_of

SKILL_IDS = {s: i for i, s in enumerate(SKILLS)}
EDU_FAMILIES = [
    {"accounting", "finance", "economics", "business", "mathematics / statistics"},
    {"computer science", "information technology", "engineering", "data science", "mathematics / statistics"},
]


def ramp(x, lo: float, hi: float):
    return np.clip((np.asarray(x, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def req_weight(importance: str, skill: str | None) -> float:
    w = settings.importance_weight[importance]
    if skill and group_of(skill) == "soft":
        w *= settings.soft_skill_factor
    return w


def coverage(weights: np.ndarray, strengths: np.ndarray) -> float | None:
    if len(weights) == 0:
        return None
    k, p = settings.smoothing_k, settings.smoothing_prior
    return float((np.dot(weights, strengths) + k * p) / (weights.sum() + k))


def experience_fit(jd_years: float | None, res_years: float | None) -> float | None:
    if jd_years is None:
        return None
    if res_years is None:
        return 0.5
    return float(min(1.0, res_years / jd_years))


def education_fit(jd_level, jd_fields, res_level, res_fields) -> float | None:
    if jd_level is None and not jd_fields:
        return None
    if jd_level is None or jd_level == "none":
        lv = 1.0
    elif res_level is None:
        lv = 0.5
    else:
        gap = EDU_LEVELS[jd_level] - EDU_LEVELS[res_level]
        lv = 1.0 if gap <= 0 else (0.5 if gap == 1 else 0.0)
    if not jd_fields:
        fd = 1.0
    elif not res_fields:
        fd = 0.5
    elif set(jd_fields) & set(res_fields):
        fd = 1.0
    elif any(set(jd_fields) & fam and set(res_fields) & fam for fam in EDU_FAMILIES):
        fd = 0.7
    else:
        fd = 0.3
    return 0.6 * lv + 0.4 * fd


def combine(sub: dict[str, float | None]) -> float:
    W = settings.weights
    num = sum(W[c] * v for c, v in sub.items() if v is not None)
    den = sum(W[c] for c, v in sub.items() if v is not None)
    return round(100.0 * num / den, 1) if den else 0.0


def status_of(strength: float) -> str:
    if strength >= settings.matched_threshold:
        return "matched"
    if strength >= settings.partial_threshold:
        return "partial"
    return "missing"


# ------------------------------------------------------------- resume context
@dataclass
class ResumeContext:
    profile: ResumeProfile
    units: list[str]               # evidence units used for semantic matching
    unit_emb: np.ndarray           # (n_units, d)
    head_emb: np.ndarray           # (d,)  resume headline / summary embedding
    skill_mask: np.ndarray         # (n_taxonomy,) bool
    skill_evidence: dict[str, str]  # canonical skill -> first resume unit mentioning it

    @classmethod
    def build(cls, profile: ResumeProfile, resume_text: str, emb: Embedder) -> "ResumeContext":
        units = list(dict.fromkeys(profile.evidence + profile.skills))
        if not units:
            units = [resume_text[:300]]
        unit_emb = emb.encode(units)
        head = " ".join(resume_text.split())[:600]
        head_emb = emb.encode([head])[0]
        mask = np.zeros(len(SKILL_IDS), dtype=bool)
        for s in profile.skills:
            if s in SKILL_IDS:
                mask[SKILL_IDS[s]] = True
        skill_ev: dict[str, str] = {}
        for u in profile.evidence:
            for s in find_skills(u):
                skill_ev.setdefault(s, u)
        return cls(profile, units, unit_emb, head_emb, mask, skill_ev)

    def display(self, j: int) -> str:
        """Evidence shown to the user: skill-name units are replaced by the resume line that mentions them."""
        u = self.units[j]
        return self.skill_evidence.get(u, u)

    def role_score(self, title_emb: np.ndarray) -> np.ndarray:
        """title_emb: (m, d) -> (m,) role alignment in [0,1]."""
        head = title_emb @ self.head_emb
        sims = title_emb @ self.unit_emb.T
        k = min(3, sims.shape[1])
        top = -np.sort(-sims, axis=1)[:, :k].mean(axis=1)
        return ramp(0.5 * head + 0.5 * top, settings.role_lo, settings.role_hi)


# --------------------------------------------------------------- single pair
def score_pair(jd: JDProfile, ctx: ResumeContext, req_emb: np.ndarray, title_emb: np.ndarray,
               meta: dict | None = None) -> MatchResult:
    """Detailed score for one posting. req_emb: (n_reqs, d) aligned with jd.requirements."""
    meta = meta or {}
    p = ctx.profile
    details: list[RequirementMatch] = []
    w_req, s_req, w_pref, s_pref = [], [], [], []
    sims = req_emb @ ctx.unit_emb.T if len(jd.requirements) else np.zeros((0, len(ctx.units)))
    for i, r in enumerate(jd.requirements):
        best_j = int(np.argmax(sims[i])) if sims.shape[1] else 0
        best_sim = float(sims[i, best_j]) if sims.shape[1] else 0.0
        sem = settings.semantic_max * float(ramp(best_sim, settings.sim_lo, settings.sim_hi))
        if r.category in settings.exact_only_categories:
            sem = 0.0
        if r.skill and r.skill in SKILL_IDS and ctx.skill_mask[SKILL_IDS[r.skill]]:
            strength, method, ev = 1.0, "exact", ctx.skill_evidence.get(r.skill, r.skill)
        elif sem > 0:
            strength, method, ev = sem, "semantic", ctx.display(best_j)
        else:
            strength, method, ev = 0.0, "none", None
        w = req_weight(r.importance, r.skill)
        (w_req if r.importance == "required" else w_pref).append(w)
        (s_req if r.importance == "required" else s_pref).append(strength)
        details.append(RequirementMatch(requirement=r.text, category=r.category, importance=r.importance,
                                        status=status_of(strength), strength=round(strength, 3), method=method,
                                        evidence=ev, similarity=round(best_sim, 3)))
    req_cov = coverage(np.array(w_req), np.array(s_req))
    if not jd.requirements:  # nothing extracted -> no evidence of fit; fall back to the prior
        req_cov = settings.smoothing_prior
    sub = {
        "required": req_cov,
        "preferred": coverage(np.array(w_pref), np.array(s_pref)),
        "experience": experience_fit(jd.min_years_experience, p.years_experience),
        "education": education_fit(jd.education_level, jd.education_fields, p.education_level, p.education_fields),
        "role": float(ctx.role_score(title_emb[None, :])[0]),
    }
    details.sort(key=lambda d: ({"required": 0, "preferred": 1}[d.importance], -d.strength))
    return MatchResult(
        job_id=jd.job_id or meta.get("job_id", ""), title=jd.title or meta.get("title", ""),
        company=meta.get("company", ""), location=meta.get("location", ""), level=meta.get("level", ""),
        url=meta.get("url", ""), score=combine(sub),
        subscores=SubScores(**{k: (round(v, 3) if v is not None else None) for k, v in sub.items()}),
        matched=sum(d.status == "matched" for d in details), partial=sum(d.status == "partial" for d in details),
        missing=sum(d.status == "missing" for d in details), requirements=details)


# ------------------------------------------------------------ batch (ranking)
class BatchScorer:
    """Vectorised version of score_pair over a fixed job database (CSR layout)."""

    def __init__(self, profiles: list[JDProfile], req_texts: list[str], req_emb: np.ndarray, title_emb: np.ndarray):
        self.n = len(profiles)
        self.req_emb = req_emb.astype(np.float32)
        self.title_emb = title_emb.astype(np.float32)
        tid = {t: i for i, t in enumerate(req_texts)}
        u, sk, soft, isreq, job, cat = [], [], [], [], [], []
        for j, p in enumerate(profiles):
            for r in p.requirements:
                u.append(tid[embed_text(r.text, r.skill)])
                sk.append(SKILL_IDS.get(r.skill, -1) if r.skill else -1)
                soft.append(bool(r.skill) and group_of(r.skill) == "soft")
                isreq.append(r.importance == "required")
                job.append(j)
                cat.append(r.category)
        self.u, self.sk = np.array(u, dtype=np.int64), np.array(sk, dtype=np.int64)
        self.soft, self.isreq, self.job = np.array(soft, bool), np.array(isreq, bool), np.array(job, np.int64)
        self.cat = np.array(cat, dtype=object)
        self.n_reqs = np.array([len(p.requirements) for p in profiles])
        self.jd_years = np.array([np.nan if p.min_years_experience is None else p.min_years_experience for p in profiles])
        self.profiles = profiles

    def score_all(self, ctx: ResumeContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # weights / exact-only flags are derived from settings at call time (so ablations take effect)
        iw = settings.importance_weight
        w = np.where(self.isreq, iw["required"], iw["preferred"]) * np.where(self.soft, settings.soft_skill_factor, 1.0)
        sem_ok = ~np.isin(self.cat, list(settings.exact_only_categories))
        best_sim = (self.req_emb @ ctx.unit_emb.T).max(axis=1)          # (U,)
        sem = settings.semantic_max * ramp(best_sim, settings.sim_lo, settings.sim_hi)[self.u]
        sem = np.where(sem_ok, sem, 0.0)
        exact = np.where(self.sk >= 0, ctx.skill_mask[np.maximum(self.sk, 0)], False)
        strength = np.where(exact, 1.0, sem).astype(np.float32)
        k, prior = settings.smoothing_k, settings.smoothing_prior

        def cov(mask):
            num = np.bincount(self.job[mask], weights=(w * strength)[mask], minlength=self.n)
            den = np.bincount(self.job[mask], weights=w[mask], minlength=self.n)
            has = np.bincount(self.job[mask], minlength=self.n) > 0
            return np.where(has, (num + k * prior) / (den + k), np.nan)

        req, pref = cov(self.isreq), cov(~self.isreq)
        req = np.where(self.n_reqs == 0, prior, req)
        ry = ctx.profile.years_experience
        if ry is None:
            exp = np.where(np.isnan(self.jd_years), np.nan, 0.5)
        else:
            exp = np.where(np.isnan(self.jd_years), np.nan, np.minimum(1.0, ry / np.where(np.isnan(self.jd_years), 1, self.jd_years)))
        edu = np.array([np.nan if (v := education_fit(p.education_level, p.education_fields, ctx.profile.education_level,
                                                       ctx.profile.education_fields)) is None else v for p in self.profiles])
        role = ctx.role_score(self.title_emb)
        sub = {"required": req, "preferred": pref, "experience": exp, "education": edu, "role": role}
        W = settings.weights
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        for c, v in sub.items():
            ok = ~np.isnan(v)
            num += np.where(ok, W[c] * np.nan_to_num(v), 0)
            den += np.where(ok, W[c], 0)
        overall = np.round(100.0 * num / np.maximum(den, 1e-9), 1)
        return overall, sub
