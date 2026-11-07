"""Extract requirements from every posting once and build the embedding index.

  python scripts/build_index.py                     # rule-based extractor (offline)
  python scripts/build_index.py --extractor llm     # Gemini structured extraction (cached)
"""
from __future__ import annotations

import argparse
import os
import gzip
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import build_index  # noqa: E402
from app.extract import LLMExtractor, RuleExtractor  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="data/processed/jobs.jsonl.gz")
    ap.add_argument("--out", default="data/processed/index")
    ap.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    jobs = [json.loads(line) for line in gzip.open(a.jobs, "rt", encoding="utf-8")]
    if a.limit:
        jobs = jobs[: a.limit]
    ex = LLMExtractor() if a.extractor == "llm" else RuleExtractor()
    print(json.dumps(build_index(jobs, Path(a.out), extractor=ex, workers=a.workers), indent=2))
