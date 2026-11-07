"""Explanations built ONLY from the structured match output (never the raw texts).

* template: deterministic, free, always available
* llm:      short natural-language summary; the prompt contains just the match JSON,
            so the model can only talk about requirements we actually matched/missed.
"""
from __future__ import annotations

import json
import logging

from .config import settings
from .llm import LLMError, get_llm
from .schemas import MatchResult

log = logging.getLogger(__name__)

_LEVEL_NAMES = {"high_school": "a high-school diploma", "associate": "an associate degree", "bachelor": "a bachelor's degree",
                "master": "a master's degree", "doctorate": "a doctorate"}


def _band(score: float) -> str:
    return "Strong fit" if score >= 70 else "Good fit" if score >= 55 else "Partial fit" if score >= 40 else "Weak fit"


def _join(items: list[str], n: int = 4) -> str:
    items = items[:n]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def template_explanation(m: MatchResult, jd_years: float | None = None, res_years: float | None = None) -> str:
    req = [r for r in m.requirements if r.importance == "required"]
    pref = [r for r in m.requirements if r.importance == "preferred"]
    hit = [r for r in req if r.status == "matched"]
    part = [r for r in m.requirements if r.status == "partial"]
    miss = [r.requirement for r in req if r.status == "missing"]
    miss_pref = [r.requirement for r in pref if r.status == "missing"]
    out = [f"{_band(m.score)} ({int(m.score + 0.5)}/100)."]
    if req:
        out.append(f"Your resume covers {len(hit)} of {len(req)} required qualifications"
                   + (f", including {_join([r.requirement for r in hit], 4)}." if hit else "."))
    if part:
        p = part[0]
        out.append(f"Related but not exact evidence for {_join([r.requirement for r in part], 3)}"
                   + (f" (e.g. “{p.evidence[:90]}”)." if p.evidence else "."))
    if miss:
        out.append(f"No supporting evidence found for {_join(miss, 4)}.")
    elif miss_pref:
        out.append(f"Nice-to-haves not shown: {_join(miss_pref, 3)}.")
    if m.subscores.experience is not None and jd_years:
        yrs = f"~{res_years:.0f} years" if res_years is not None else "no clear dates"
        out.append(f"The posting asks for {jd_years:g}+ years of experience; your resume shows {yrs}.")
    return " ".join(out)


_PROMPT = """You write a 2-4 sentence explanation of how well a resume fits a job posting, for the job seeker.
Use ONLY the structured match result below. Mention the strongest matched required qualifications, any partial
matches, and the most important missing requirements. Do not invent skills or facts. Do not restate the numeric
sub-scores; you may mention the overall score once. Address the reader as "you".

Match result (JSON):
{payload}"""


def llm_explanation(m: MatchResult) -> str:
    payload = {
        "job_title": m.title, "overall_score": m.score,
        "requirements": [{"requirement": r.requirement, "importance": r.importance, "status": r.status,
                          "evidence": (r.evidence or "")[:120]} for r in m.requirements[:20]],
        "experience_fit": m.subscores.experience, "education_fit": m.subscores.education,
    }
    return get_llm().chat([{"role": "user", "content": _PROMPT.format(payload=json.dumps(payload, indent=1))}],
                          temperature=0.0, max_tokens=300).strip()


def explain(m: MatchResult, jd_years=None, res_years=None, mode: str | None = None) -> tuple[str, str]:
    mode = mode or settings.explainer
    if mode == "llm" or (mode == "auto" and settings.llm_enabled):
        try:
            return llm_explanation(m), "llm"
        except LLMError as e:
            log.warning("LLM explanation failed (%s); using template", e)
    return template_explanation(m, jd_years, res_years), "template"
