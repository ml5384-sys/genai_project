"""LLM vs rule-based JD extraction on a stratified sample, plus explanation faithfulness.

  GEMINI_API_KEY=... LLM_RPM=14 python eval/llm_extraction.py --n 300

Outputs eval/results/llm_extraction.json:
  * extraction statistics (requirements per JD, required share, years/degree found, agreement)
  * ranking quality (weak labels) within the sample using rule vs LLM JD profiles
  * faithfulness of LLM explanations: share whose skill mentions are all present in the match input
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings  # noqa: E402
from app.embed import get_embedder  # noqa: E402
from app.explain import llm_explanation, template_explanation  # noqa: E402
from app.extract import LLMExtractor, RuleExtractor  # noqa: E402
from app.scoring import BatchScorer, ResumeContext, score_pair  # noqa: E402
from app.skills import embed_text, find_skills  # noqa: E402
from eval.evaluate import RELEVANCE, metrics  # noqa: E402


def load(p):
    return [json.loads(l) for l in gzip.open(ROOT / p, "rt", encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-explain", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not settings.llm_enabled:
        sys.exit("Set GEMINI_API_KEY")
    rng = random.Random(11)
    jobs = load("data/processed/jobs.jsonl.gz")
    # stratify by primary function so every category is represented
    by_f = {}
    for j in jobs:
        by_f.setdefault(j["functions"][0], []).append(j)
    per = a.n // len(by_f)
    sample = [j for f, js in sorted(by_f.items()) for j in rng.sample(js, min(per, len(js)))]
    rule, llm = RuleExtractor(), LLMExtractor()
    t0 = time.time()
    rp = [rule.extract_jd(j["description"], j["title"], j["job_id"]) for j in sample]
    with ThreadPoolExecutor(a.workers) as pool:
        lp = list(pool.map(lambda j: llm.extract_jd(j["description"], j["title"], j["job_id"]), sample))
    print(f"extracted {len(sample)} JDs in {time.time()-t0:.0f}s")
    ok = [i for i, p in enumerate(lp) if p.source == "llm"]
    print("LLM succeeded:", len(ok))

    emb = get_embedder()

    def stats(ps):
        return {"reqs_per_jd": float(np.mean([len(p.requirements) for p in ps])),
                "required_share": float(np.mean([np.mean([r.importance == "required" for r in p.requirements]) if p.requirements else 0 for p in ps])),
                "with_canonical_skill_share": float(np.mean([np.mean([r.skill is not None for r in p.requirements]) if p.requirements else 0 for p in ps])),
                "no_requirements_pct": float(np.mean([not p.requirements for p in ps])),
                "years_found_pct": float(np.mean([p.min_years_experience is not None for p in ps])),
                "degree_found_pct": float(np.mean([p.education_level is not None for p in ps]))}

    # agreement: share of rule requirements semantically covered by some LLM requirement and vice versa
    def covered(a_reqs, b_reqs, thr=0.5):
        if not a_reqs or not b_reqs:
            return None
        A = emb.encode([embed_text(r.text, r.skill) for r in a_reqs])
        B = emb.encode([embed_text(r.text, r.skill) for r in b_reqs])
        return float(((A @ B.T).max(1) >= thr).mean())

    rule_in_llm = [c for i in ok if (c := covered(rp[i].requirements, lp[i].requirements)) is not None]
    llm_in_rule = [c for i in ok if (c := covered(lp[i].requirements, rp[i].requirements)) is not None]
    yrs_agree = [rp[i].min_years_experience == lp[i].min_years_experience for i in ok
                 if rp[i].min_years_experience is not None or lp[i].min_years_experience is not None]
    out = {"model": settings.llm_model, "n_sample": len(sample), "n_llm_ok": len(ok),
           "rule": stats([rp[i] for i in ok]), "llm": stats([lp[i] for i in ok]),
           "rule_reqs_covered_by_llm": float(np.mean(rule_in_llm)), "llm_reqs_covered_by_rule": float(np.mean(llm_in_rule)),
           "min_years_agreement": float(np.mean(yrs_agree)) if yrs_agree else None,
           "examples": [{"title": sample[i]["title"], "rule": [r.text for r in rp[i].requirements][:10],
                         "llm": [f"{r.text} ({r.importance})" for r in lp[i].requirements][:10]} for i in ok[:4]]}

    # ranking within the sample, rule vs LLM JD profiles (resumes: rule extraction for both)
    resumes = load("data/processed/resumes.jsonl.gz")
    sj = [sample[i] for i in ok]
    rel = np.array([[max([RELEVANCE[r["category"]].get(f, 0) for f in j["functions"]] + [0]) for j in sj] for r in resumes])
    title_emb = emb.encode([j["title"] for j in sj])
    ctxs = [ResumeContext.build(rule.extract_resume(r["text"]), r["text"], emb) for r in resumes]
    for name, profs in (("rule", [rp[i] for i in ok]), ("llm", [lp[i] for i in ok])):
        texts = sorted({embed_text(r.text, r.skill) for p in profs for r in p.requirements})
        bs = BatchScorer(profs, texts, emb.encode(texts), title_emb)
        S = np.stack([bs.score_all(c)[0] for c in ctxs]) + 1e-6 * np.stack([title_emb @ c.head_emb for c in ctxs])
        out[f"ranking_{name}"] = metrics(S, rel)

    # explanation faithfulness
    faith, ex = [], []
    for k in range(min(a.n_explain, len(ok))):
        r = resumes[rng.randrange(len(resumes))]
        i = ok[k]
        ctx = ctxs[resumes.index(r)]
        p = lp[i]
        m = score_pair(p, ctx, emb.encode([embed_text(q.text, q.skill) for q in p.requirements]), title_emb[ok.index(i)],
                       {"title": sample[i]["title"]})
        try:
            text = llm_explanation(m)
        except Exception as e:  # quota etc.
            print("explanation failed:", e)
            break
        allowed = set()
        for q in m.requirements:
            allowed |= set(find_skills(q.requirement)) | set(find_skills(q.evidence or ""))
        mentioned = set(find_skills(text))
        faith.append(mentioned <= allowed)
        ex.append({"title": m.title, "score": m.score, "llm": text, "template": template_explanation(m, p.min_years_experience, ctx.profile.years_experience),
                   "unsupported_skill_mentions": sorted(mentioned - allowed)})
    out["explanations"] = {"n": len(faith), "faithful_pct": float(np.mean(faith)) if faith else None, "examples": ex[:4]}
    (ROOT / "eval/results/llm_extraction.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("examples",)}, indent=2)[:4000])


if __name__ == "__main__":
    main()
