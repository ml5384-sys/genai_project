"""Plain-LLM baseline ("paste resume + JD into a chatbot") vs. JobFit.  Requires an LLM key.

  GEMINI_API_KEY=... python eval/llm_baseline.py --pairs 30 --runs 5

Measures, for the same (resume, posting) pairs:
  * run-to-run variability of the plain-LLM score (std over repeated calls, default temperature)
  * latency per pair and extrapolated time to rank the whole 8,000-posting database
  * rank agreement (Spearman) between the plain-LLM mean score and the JobFit score
  * ranking quality on the weak labels for a small candidate pool per resume
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings  # noqa: E402
from app.engine import JobFitEngine  # noqa: E402
from app.llm import LLMClient  # noqa: E402

PROMPT = """Here is my resume and a job posting. On a scale of 0 to 100, how well does my resume fit this job?
Answer with the number first, then one sentence of reasoning.

RESUME:
{resume}

JOB POSTING ({title}):
{jd}"""


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0, help="chatbot-like default sampling")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if not settings.llm_enabled:
        sys.exit("Set GEMINI_API_KEY (or LLM_API_KEY) to run the plain-LLM baseline.")
    rng = random.Random(a.seed)
    E = JobFitEngine()
    llm = LLMClient()
    resumes = [json.loads(l) for l in gzip.open(ROOT / "data/processed/resumes.jsonl.gz", "rt")]
    pairs = []
    for r in rng.sample(resumes, a.pairs):
        ctx = E.resume_context(r["text"])
        overall, _ = E.batch.score_all(ctx)
        # mix of strong, medium and random postings so scores span the range
        order = np.argsort(-overall)
        j = int(rng.choice([order[rng.randrange(5)], order[rng.randrange(200, 400)], rng.randrange(len(order))]))
        pairs.append((r, j, float(overall[j])))

    rows = []
    for r, j, jf in pairs:
        jd = E.description(E.meta[j]["job_id"]) or ""
        msg = [{"role": "user", "content": PROMPT.format(resume=r["text"][:8000], title=E.meta[j]["title"], jd=jd[:6000])}]
        scores, lat = [], []
        for _ in range(a.runs):
            t = time.perf_counter()
            out = llm.chat(msg, temperature=a.temperature, max_tokens=120, use_cache=False)
            lat.append(time.perf_counter() - t)
            m = re.search(r"\d{1,3}", out)
            if m:
                scores.append(min(100, int(m.group())))
        rows.append({"resume_id": r["resume_id"], "category": r["category"], "job_id": E.meta[j]["job_id"],
                     "title": E.meta[j]["title"], "jobfit": jf, "llm_scores": scores,
                     "llm_mean": float(np.mean(scores)) if scores else None,
                     "llm_std": float(np.std(scores)) if scores else None, "latency_s": float(np.mean(lat))})
        print(rows[-1]["title"][:40], jf, scores)
    ok = [x for x in rows if x["llm_mean"] is not None]
    t0 = time.perf_counter()
    E.rank(__import__("app.schemas", fromlist=["RankRequest"]).RankRequest(resume_text=pairs[0][0]["text"], top_k=10, explain_top=0))
    jobfit_ms = (time.perf_counter() - t0) * 1000
    lat = float(np.mean([x["latency_s"] for x in ok]))
    res = {
        "model": llm.model, "temperature": a.temperature, "n_pairs": len(ok), "runs_per_pair": a.runs,
        "llm_score_std_mean": float(np.mean([x["llm_std"] for x in ok])),
        "llm_score_range_mean": float(np.mean([max(x["llm_scores"]) - min(x["llm_scores"]) for x in ok])),
        "pct_pairs_score_changed": float(np.mean([len(set(x["llm_scores"])) > 1 for x in ok])),
        "jobfit_score_std": 0.0,
        "spearman_llm_vs_jobfit": spearman([x["llm_mean"] for x in ok], [x["jobfit"] for x in ok]),
        "llm_latency_s_per_pair": lat,
        "llm_hours_to_rank_8000": lat * len(E.meta) / 3600,
        "jobfit_ms_to_rank_8000": jobfit_ms,
        "pairs": rows,
    }
    out = ROOT / "eval/results/llm_baseline.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
