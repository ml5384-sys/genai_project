"""Offline evaluation of JobFit.  Writes eval/results/*.json and figures to docs/figures/.

  python eval/evaluate.py                # everything
  python eval/evaluate.py --only ranking --n-resumes 200

Experiments
  1. ranking     weak-label ranking quality over the 8,000-posting database vs. baselines + ablations
  2. paraphrase  keyword vs. semantic vs. combined requirement matching on hand-written paraphrases
  3. counterfactual  does the score move the right way when evidence is removed / added?
  4. stability   determinism + latency of /rank
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.embed import get_embedder  # noqa: E402
from app.engine import JobFitEngine  # noqa: E402
from app.schemas import RankRequest  # noqa: E402
from app.scoring import ramp  # noqa: E402
from app.skills import embed_text, find_skills  # noqa: E402

RES_DIR = ROOT / "eval" / "results"
FIG_DIR = ROOT / "docs" / "figures"

# resume category -> {function: graded relevance}
RELEVANCE = {
    "ACCOUNTANT": {"ACCT": 2, "FIN": 1},
    "FINANCE": {"FIN": 2, "ACCT": 1, "ANLS": 1},
    "BANKING": {"FIN": 2},
    "INFORMATION-TECHNOLOGY": {"IT": 2},
    "CONSULTANT": {"CNSL": 2, "ANLS": 1, "IT": 1},
}


@contextmanager
def patched(**kw):
    """Temporarily override frozen settings (for ablations)."""
    old = {k: getattr(config.settings, k) for k in kw}
    for k, v in kw.items():
        object.__setattr__(config.settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            object.__setattr__(config.settings, k, v)


def load_jsonl(p):
    return [json.loads(line) for line in gzip.open(p, "rt", encoding="utf-8")]


# ----------------------------------------------------------------- metrics
def rel_matrix(resumes, meta):
    R = np.zeros((len(resumes), len(meta)), dtype=np.int8)
    for i, r in enumerate(resumes):
        g = RELEVANCE[r["category"]]
        for j, m in enumerate(meta):
            R[i, j] = max([g.get(f, 0) for f in m["functions"]] + [0])
    return R


def metrics(scores: np.ndarray, rel: np.ndarray, k: int = 10, dedup_keys=None) -> dict:
    """scores, rel: (n_resumes, n_jobs)."""
    ndcg, p_at, p_strict, mrr = [], [], [], []
    disc = 1 / np.log2(np.arange(2, k + 2))
    for s, r in zip(scores, rel):
        order = np.argsort(-s, kind="stable")[:k]
        g = (2.0 ** r[order] - 1)
        ideal = np.sort(2.0 ** r - 1)[::-1][:k]
        ndcg.append(float((g * disc).sum() / max((ideal * disc).sum(), 1e-9)))
        p_at.append(float((r[order] > 0).mean()))
        p_strict.append(float((r[order] == 2).mean()))
        first = np.flatnonzero(r[np.argsort(-s, kind="stable")] == 2)
        mrr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
    ci = lambda a: 1.96 * np.std(a) / np.sqrt(len(a))  # noqa: E731
    return {f"nDCG@{k}": float(np.mean(ndcg)), f"nDCG@{k}_ci": float(ci(ndcg)),
            f"P@{k}": float(np.mean(p_at)), f"P@{k}_strict": float(np.mean(p_strict)),
            "MRR": float(np.mean(mrr)), "n": len(ndcg)}


# ------------------------------------------------------------- 1. ranking
def exp_ranking(E: JobFitEngine, resumes, n: int, rng) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    resumes = rng.sample(resumes, min(n, len(resumes)))
    jobs = load_jsonl(ROOT / "data/processed/jobs.jsonl.gz")
    assert [j["job_id"] for j in jobs] == [m["job_id"] for m in E.meta]
    rel = rel_matrix(resumes, E.meta)
    out = {"n_resumes": len(resumes), "n_jobs": len(jobs),
           "base_rate_relevant": float((rel > 0).mean()), "base_rate_strict": float((rel == 2).mean())}

    print("baselines ...")
    res = {}
    res["Random"] = metrics(np.random.default_rng(0).random(rel.shape), rel)
    tf = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True, min_df=2)
    J = tf.fit_transform([j["title"] + " " + j["description"] for j in jobs])
    Rv = tf.transform([r["text"] for r in resumes])
    res["TF-IDF (keyword)"] = metrics((Rv @ J.T).toarray(), rel)

    emb = E.emb
    jd_doc = emb.encode([(j["title"] + ". " + j["description"])[:1000] for j in jobs])
    rs_doc = emb.encode([" ".join(r["text"].split())[:1000] for r in resumes])
    res["Embedding doc-cosine"] = metrics(rs_doc @ jd_doc.T, rel)

    print("building resume contexts ...")
    ctxs = [E.resume_context(r["text"]) for r in resumes]
    # skill-overlap baseline (taxonomy Jaccard)
    job_sk = [set(r.skill for r in p.requirements if r.skill) for p in E.profiles]
    jac = np.array([[len(set(c.profile.skills) & s) / max(len(set(c.profile.skills) | s), 1) for s in job_sk] for c in ctxs])
    res["Skill Jaccard (exact)"] = metrics(jac, rel)

    W = dict(config.settings.weights)
    variants = {
        "JobFit (full)": {},
        "  - semantic matching (exact only)": {"sim_lo": 2.0, "sim_hi": 3.0},
        "  - role alignment": {"weights": {**W, "role": 0.0}},
        "  - required/preferred weighting": {"importance_weight": {"required": 1.0, "preferred": 1.0}},
        "  - smoothing": {"smoothing_k": 0.0},
        "  - experience & education": {"weights": {**W, "experience": 0.0, "education": 0.0}},
        "Coverage only (required+preferred)": {"weights": {"required": 0.8, "preferred": 0.2, "experience": 0.0,
                                                           "education": 0.0, "role": 0.0}},
    }
    for name, kw in variants.items():
        print("  ", name)
        with patched(**kw):
            S = np.stack([E.batch.score_all(c)[0] for c in ctxs])
        # tie-break exactly as /rank does (role/title similarity), via a tiny epsilon
        S = S + 1e-6 * np.stack([E.title_emb @ c.head_emb for c in ctxs])
        res[name] = metrics(S, rel)
        if name == "JobFit (full)":
            per_cat = {}
            for cat in RELEVANCE:
                idx = [i for i, r in enumerate(resumes) if r["category"] == cat]
                per_cat[cat] = metrics(S[idx], rel[idx])
            out["per_category_full"] = per_cat
            # hybrid: JobFit score + doc-cosine as a retrieval prior (reported as an extension)
            z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)  # noqa: E731
            res["Hybrid: JobFit + doc-cosine (z-avg)"] = metrics(z(S) + z(rs_doc @ jd_doc.T), rel)
    out["methods"] = res
    return out


# ----------------------------------------------------------- 2. paraphrase
def exp_paraphrase(models: list[str]) -> dict:
    cases = json.loads((ROOT / "eval/paraphrase_cases.json").read_text())
    out = {"n_cases": len(cases), "models": {}}
    kw_pos = [c["skill"] in find_skills(c["positive"]) for c in cases]
    kw_neg = [c["skill"] in find_skills(c["negative"]) for c in cases]
    out["keyword"] = {"recall": float(np.mean(kw_pos)), "false_positive_rate": float(np.mean(kw_neg))}
    for m in models:
        emb = get_embedder(m)
        cal_path = ROOT / ("eval/calibration.json" if "MiniLM" in m else "eval/calibration_bge.json")
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else {"sim_lo": 0.32, "sim_hi": 0.52}
        req = emb.encode([embed_text(c["skill"], c["skill"]) for c in cases])
        bare = emb.encode([c["skill"] for c in cases])
        pos = emb.encode([c["positive"] for c in cases])
        neg = emb.encode([c["negative"] for c in cases])
        row = {}
        for label, R in (("with descriptor", req), ("bare skill name", bare)):
            sp, sn = (R * pos).sum(1), (R * neg).sum(1)
            st_p = ramp(sp, cal["sim_lo"], cal["sim_hi"])
            st_n = ramp(sn, cal["sim_lo"], cal["sim_hi"])
            thr = config.settings.partial_threshold
            sem_pos, sem_neg = st_p >= thr, st_n >= thr
            row[label] = {
                "pairwise_accuracy": float((sp > sn).mean()),
                "semantic_recall": float(sem_pos.mean()), "semantic_fpr": float(sem_neg.mean()),
                "combined_recall": float(np.mean(np.array(kw_pos) | sem_pos)),
                "combined_fpr": float(np.mean(np.array(kw_neg) | sem_neg)),
                "recovered_beyond_keyword": int((~np.array(kw_pos) & sem_pos).sum()),
                "mean_cos_pos": float(sp.mean()), "mean_cos_neg": float(sn.mean()),
            }
        out["models"][m] = row
    return out


# ------------------------------------------------------- 3. counterfactual
def exp_counterfactual(E: JobFitEngine, resumes, n: int, rng) -> dict:
    """Remove lines evidencing a matched required skill -> score should drop;
    add a line evidencing a missing required skill -> score should rise."""
    drops, rises = [], []
    for r in rng.sample(resumes, min(n, len(resumes))):
        ctx = E.resume_context(r["text"])
        overall, _ = E.batch.score_all(ctx)
        j = int(np.argmax(overall))
        m = E.score_job(j, ctx)
        req_hit = [q for q in m.requirements if q.importance == "required" and q.method == "exact"]
        req_miss = [q for q in m.requirements if q.importance == "required" and q.status == "missing"]
        jd = E.profiles[j]
        if req_hit:
            sk = next(x.skill for x in jd.requirements if x.text == req_hit[0].requirement)
            lines = [ln for ln in r["text"].split("\n") if sk not in find_skills(ln)]
            m2 = E.score_job(j, E.resume_context("\n".join(lines)))
            drops.append(m2.score - m.score)
        if req_miss:
            add = r["text"] + f"\nSkills: {req_miss[0].requirement}"
            m3 = E.score_job(j, E.resume_context(add))
            rises.append(m3.score - m.score)
    d, u = np.array(drops), np.array(rises)
    return {"remove_matched_skill": {"n": len(d), "mean_delta": float(d.mean()), "pct_decreased": float((d < 0).mean()),
                                     "pct_increased": float((d > 0).mean())},
            "add_missing_skill": {"n": len(u), "mean_delta": float(u.mean()), "pct_increased": float((u > 0).mean()),
                                  "pct_decreased": float((u < 0).mean())}}


# ------------------------------------------------------------ 4. stability
def exp_stability(E: JobFitEngine, resumes, n: int, rng) -> dict:
    sample = rng.sample(resumes, min(n, len(resumes)))
    lat, identical = [], 0
    for r in sample:
        outs = []
        for _ in range(3):
            t = time.perf_counter()
            res = E.rank(RankRequest(resume_text=r["text"], top_k=10, explain_top=0))
            lat.append((time.perf_counter() - t) * 1000)
            outs.append([(x.job_id, x.score) for x in res.results])
        identical += all(o == outs[0] for o in outs)
    return {"n_resumes": len(sample), "runs_per_resume": 3, "identical_rankings_pct": identical / len(sample),
            "latency_ms_mean": float(np.mean(lat)), "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)), "postings_per_request": len(E.meta)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["ranking", "paraphrase", "counterfactual", "stability"])
    ap.add_argument("--n-resumes", type=int, default=586)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    resumes = load_jsonl(ROOT / "data/processed/resumes.jsonl.gz")
    E = JobFitEngine()
    summary = {"config": {k: (v if not isinstance(v, Path) else str(v)) for k, v in dataclasses.asdict(config.settings).items()
                          if k not in ("llm_api_key",)}, "index": E.manifest}
    if "ranking" in a.only:
        summary["ranking"] = exp_ranking(E, resumes, a.n_resumes, rng)
    if "paraphrase" in a.only:
        summary["paraphrase"] = exp_paraphrase(["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"])
    if "counterfactual" in a.only:
        summary["counterfactual"] = exp_counterfactual(E, resumes, 150, rng)
    if "stability" in a.only:
        summary["stability"] = exp_stability(E, resumes, 30, rng)
    for k, v in summary.items():
        if k != "config":
            (RES_DIR / f"{k}.json").write_text(json.dumps(v, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
