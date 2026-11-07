"""Pydantic schemas: the fixed JSON contracts for extraction, matching and the API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Importance = Literal["required", "preferred"]
ReqCategory = Literal["skill", "tool", "certification", "domain", "experience", "education", "other"]
Status = Literal["matched", "partial", "missing"]

EDU_LEVELS = {"none": 0, "high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "doctorate": 5}
EduLevel = Literal["none", "high_school", "associate", "bachelor", "master", "doctorate"]


class Requirement(BaseModel):
    text: str = Field(..., description="Short requirement phrase, e.g. 'SQL' or 'Experience reconciling bank statements'")
    category: ReqCategory = "skill"
    importance: Importance = "required"
    skill: Optional[str] = Field(None, description="Canonical taxonomy skill, when the requirement maps to one")


class JDProfile(BaseModel):
    job_id: str = ""
    title: str = ""
    requirements: list[Requirement] = []
    min_years_experience: Optional[float] = None
    education_level: Optional[EduLevel] = None
    education_fields: list[str] = []
    source: Literal["rule", "llm"] = "rule"


class ResumeProfile(BaseModel):
    skills: list[str] = Field(default_factory=list, description="Canonical skills found in the resume")
    evidence: list[str] = Field(default_factory=list, description="Resume sentences/bullets used as matching evidence")
    years_experience: Optional[float] = None
    education_level: Optional[EduLevel] = None
    education_fields: list[str] = []
    source: Literal["rule", "llm"] = "rule"


class RequirementMatch(BaseModel):
    requirement: str
    category: ReqCategory
    importance: Importance
    status: Status
    strength: float = Field(..., ge=0, le=1)
    method: Literal["exact", "semantic", "none"]
    evidence: Optional[str] = None
    similarity: Optional[float] = None


class SubScores(BaseModel):
    required: Optional[float] = None
    preferred: Optional[float] = None
    experience: Optional[float] = None
    education: Optional[float] = None
    role: Optional[float] = None


class MatchResult(BaseModel):
    job_id: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    level: str = ""
    url: str = ""
    score: float = Field(..., description="Overall 0-100 requirement-coverage score (not a hiring probability)")
    subscores: SubScores
    matched: int
    partial: int
    missing: int
    requirements: list[RequirementMatch] = []
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None


# ----------------------------- API request bodies -----------------------------
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: str = Field(..., min_length=20)
    job_title: str = ""
    explain: bool = True


class RankRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    top_k: int = Field(10, ge=1, le=100)
    levels: Optional[list[str]] = Field(None, description="Filter: 'Entry level', 'Associate', 'Internship'")
    functions: Optional[list[str]] = Field(None, description="Filter: FIN, ACCT, IT, CNSL, ANLS")
    remote_only: bool = False
    location_contains: Optional[str] = None
    explain_top: int = Field(3, ge=0, le=10, description="Generate explanations for the first N results")


class RankResponse(BaseModel):
    resume: ResumeProfile
    total_postings_scored: int
    elapsed_ms: float
    results: list[MatchResult]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=20)
    title: str = ""
