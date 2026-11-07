"""Structured extraction of job descriptions and resumes.

Two interchangeable back-ends produce the same pydantic schema:

* ``RuleExtractor``  - deterministic: skill taxonomy + section heuristics + regexes.
                       No API key needed; used offline, in tests, and as fallback.
* ``LLMExtractor``   - an LLM constrained to a fixed JSON schema at temperature 0,
                       with results cached (JD extraction runs once per posting).
                       LLM skill names are canonicalised onto the taxonomy so exact
                       matching still works.
"""
from __future__ import annotations

import logging

from .config import settings
from .llm import LLMError, get_llm
from .schemas import EDU_LEVELS, JDProfile, Requirement, ResumeProfile
from .skills import canonicalize, find_skills, group_of
from .text import education, jd_min_years, jd_sections, resume_years, segments

log = logging.getLogger(__name__)

_CERT_GROUP = "certification"
MAX_REQS = 25


def _category_for(skill: str) -> str:
    g = group_of(skill)
    if g == _CERT_GROUP:
        return "certification"
    if g in ("finance", "accounting", "business"):
        return "domain"
    if g == "tools":
        return "tool"
    return "skill"


class RuleExtractor:
    name = "rule"

    def extract_jd(self, text: str, title: str = "", job_id: str = "") -> JDProfile:
        sections = jd_sections(text)
        has_req_section = any(lbl == "requirement" for lbl, _ in sections)
        best: dict[str, str] = {}  # skill -> importance
        implicit: set[str] = set()  # preferred only because it appeared outside the requirement list
        rank = {"required": 2, "preferred": 1}
        for label, seg in sections:
            if label == "benefit":
                continue
            if label == "preferred":
                imp, impl = "preferred", False
            elif label == "requirement" or not has_req_section:
                imp, impl = "required", False
            else:  # skills only mentioned in duties/overview when an explicit requirement list exists
                imp, impl = "preferred", True
            for skill in find_skills(seg):
                if rank[imp] > rank.get(best.get(skill, ""), 0):
                    best[skill] = imp
                    implicit.discard(skill)
                    if impl:
                        implicit.add(skill)
        if best and "required" not in best.values():
            # the requirement list named no taxonomy skills: duties are the best evidence of what is needed
            for skill in implicit:
                best[skill] = "required"
        # skills named in the title are always required (e.g. "SQL Developer")
        for skill in find_skills(title):
            best[skill] = "required"
        reqs = [Requirement(text=s, category=_category_for(s), importance=imp, skill=s) for s, imp in best.items()]
        reqs.sort(key=lambda r: (r.importance != "required", group_of(r.skill or "") == "soft"))
        req_text = " ".join(seg for lbl, seg in sections if lbl in ("requirement", "preferred")) or text
        level, fields = education(req_text, mode="min")
        return JDProfile(job_id=job_id, title=title, requirements=reqs[:MAX_REQS],
                         min_years_experience=jd_min_years(req_text) or jd_min_years(text),
                         education_level=level, education_fields=fields, source="rule")

    def extract_resume(self, text: str) -> ResumeProfile:
        level, fields = education(text, mode="max")
        return ResumeProfile(skills=sorted(find_skills(text)), evidence=segments(text, min_len=15, min_words=3)[:250],
                             years_experience=resume_years(text), education_level=level,
                             education_fields=fields, source="rule")


# ------------------------------------------------------------------- LLM based
_JD_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "short phrase, max 12 words"},
                    "category": {"type": "string", "enum": ["skill", "tool", "certification", "domain", "other"]},
                    "importance": {"type": "string", "enum": ["required", "preferred"]},
                },
                "required": ["text", "category", "importance"],
            },
        },
        "min_years_experience": {"type": ["number", "null"]},
        "education_level": {"type": ["string", "null"], "enum": list(EDU_LEVELS) + [None]},
        "education_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requirements", "min_years_experience", "education_level", "education_fields"],
}

_JD_PROMPT = """You extract hiring requirements from a job posting into JSON.
Rules:
- List the concrete qualifications a candidate must or should have: skills, tools/software, certifications/licenses, domain knowledge. Max 20 items, each a short phrase (<= 12 words).
- importance = "required" unless the posting marks it preferred / nice-to-have / a plus / bonus.
- Do NOT list duties that are not qualifications, benefits, salary, or company descriptions.
- Do NOT list years of experience or degrees as requirements; put them in min_years_experience (smallest number of years required, null if none) and education_level / education_fields (minimum degree; null if none).
- Use only information present in the posting.

Job title: {title}
Job posting:
\"\"\"{text}\"\"\""""

_RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"},
                     "description": "verbatim or lightly trimmed resume statements showing skills/experience"},
        "years_experience": {"type": ["number", "null"]},
        "education_level": {"type": ["string", "null"], "enum": list(EDU_LEVELS) + [None]},
        "education_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["skills", "evidence", "years_experience", "education_level", "education_fields"],
}

_RESUME_PROMPT = """You extract a structured profile from a resume into JSON.
- skills: every skill, tool, software, certification or domain area the candidate demonstrates (short names).
- evidence: up to 40 short statements copied from the resume that show what the candidate has done (keep the original wording).
- years_experience: total professional experience in years (null if unknown).
- education_level: highest degree (none/high_school/associate/bachelor/master/doctorate) and education_fields: fields of study.
Use only information present in the resume.

Resume:
\"\"\"{text}\"\"\""""


class LLMExtractor:
    name = "llm"

    def __init__(self, fallback: RuleExtractor | None = None):
        self.llm = get_llm()
        self.fallback = fallback or RuleExtractor()

    def extract_jd(self, text: str, title: str = "", job_id: str = "") -> JDProfile:
        try:
            out = self.llm.json([{"role": "user", "content": _JD_PROMPT.format(title=title, text=text[:12000])}],
                                _JD_SCHEMA)
        except (LLMError, ValueError) as e:
            log.warning("LLM JD extraction failed (%s); using rule extractor", e)
            return self.fallback.extract_jd(text, title, job_id)
        reqs, seen = [], set()
        for r in out.get("requirements", [])[:MAX_REQS]:
            t = str(r.get("text", "")).strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            cat = r.get("category", "other")
            if cat not in ("skill", "tool", "certification", "domain", "other"):
                cat = "other"
            imp = "preferred" if r.get("importance") == "preferred" else "required"
            reqs.append(Requirement(text=t, category=cat, importance=imp, skill=canonicalize(t)))
        lvl = out.get("education_level")
        lvl = lvl if lvl in EDU_LEVELS else None
        yrs = out.get("min_years_experience")
        yrs = float(yrs) if isinstance(yrs, (int, float)) and 0 < yrs <= 15 else None
        return JDProfile(job_id=job_id, title=title, requirements=reqs, min_years_experience=yrs,
                         education_level=lvl, education_fields=[str(f).lower() for f in out.get("education_fields", [])][:5],
                         source="llm")

    def extract_resume(self, text: str) -> ResumeProfile:
        base = self.fallback.extract_resume(text)
        try:
            out = self.llm.json([{"role": "user", "content": _RESUME_PROMPT.format(text=text[:15000])}], _RESUME_SCHEMA)
        except (LLMError, ValueError) as e:
            log.warning("LLM resume extraction failed (%s); using rule extractor", e)
            return base
        canon = {c for s in out.get("skills", []) if (c := canonicalize(str(s)))}
        skills = sorted(set(base.skills) | canon)
        raw_skills = [str(s) for s in out.get("skills", [])]
        # evidence = LLM statements + skill names + rule segments (union keeps recall high)
        evidence = list(dict.fromkeys([str(e) for e in out.get("evidence", [])] + raw_skills + base.evidence))[:300]
        lvl = out.get("education_level")
        yrs = out.get("years_experience")
        return ResumeProfile(skills=skills, evidence=evidence,
                             years_experience=float(yrs) if isinstance(yrs, (int, float)) else base.years_experience,
                             education_level=lvl if lvl in EDU_LEVELS else base.education_level,
                             education_fields=[str(f).lower() for f in out.get("education_fields", [])] or base.education_fields,
                             source="llm")


def get_extractor(kind: str | None = None):
    kind = kind or settings.extractor
    if kind == "llm" or (kind == "auto" and settings.llm_enabled):
        return LLMExtractor()
    return RuleExtractor()
