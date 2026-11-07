"""Central configuration.  Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(_env("JOBFIT_DATA_DIR", str(ROOT / "data" / "processed")))
    index_dir: Path = Path(_env("JOBFIT_INDEX_DIR", str(ROOT / "data" / "processed" / "index")))
    model_dir: Path = Path(_env("JOBFIT_MODEL_DIR", str(ROOT / "models")))
    embed_model: str = _env("JOBFIT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # LLM (Gemini through its OpenAI-compatible endpoint by default; any OpenAI-compatible
    # server such as OpenAI, vLLM or Ollama works by changing the base URL / model).
    llm_base_url: str = _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    llm_model: str = _env("LLM_MODEL", "gemini-2.5-flash-lite")
    llm_api_key: str = _env("LLM_API_KEY", _env("GEMINI_API_KEY", ""))
    llm_timeout: float = float(_env("LLM_TIMEOUT", "60"))
    # "auto": use the LLM when a key is configured, else the rule-based path.
    extractor: str = _env("JOBFIT_EXTRACTOR", "auto")      # auto | llm | rule
    explainer: str = _env("JOBFIT_EXPLAINER", "auto")      # auto | llm | template
    cache_path: Path = Path(_env("JOBFIT_CACHE", str(ROOT / "data" / "cache" / "llm_cache.sqlite")))

    # ---- scoring formula (see docs / report section "Scoring") ----
    weights: dict = field(default_factory=lambda: {
        "required": 0.45, "preferred": 0.10, "experience": 0.15, "education": 0.10, "role": 0.20,
    })
    importance_weight: dict = field(default_factory=lambda: {"required": 1.0, "preferred": 0.5})
    soft_skill_factor: float = 0.5       # soft skills count half (everyone claims them)
    smoothing_k: float = 2.0             # pseudo-requirements pulling coverage toward the prior
    smoothing_prior: float = 0.3
    # semantic similarity -> match strength (linear ramp between lo and hi cosine);
    # calibrated per model with scripts/calibrate.py
    sim_lo: float = float(_env("JOBFIT_SIM_LO", "0.32"))
    sim_hi: float = float(_env("JOBFIT_SIM_HI", "0.52"))
    role_lo: float = float(_env("JOBFIT_ROLE_LO", "0.16"))
    role_hi: float = float(_env("JOBFIT_ROLE_HI", "0.50"))
    semantic_max: float = 0.85           # related (semantic) evidence is worth at most 85% of direct evidence
    exact_only_categories: tuple = ("certification",)  # licenses/certifications are binary: no semantic credit
    matched_threshold: float = 0.75
    partial_threshold: float = 0.35

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
