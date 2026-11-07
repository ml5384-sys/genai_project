"""Calibrate the cosine -> strength ramps from data (reported in the final report).

Semantic ramp (requirement vs. resume evidence):
  positives: (canonical skill name, resume sentence that mentions that skill)
  negatives: (canonical skill name, resume sentence that mentions no skill of the same group)
  lo = 95th percentile of negatives, hi = max(75th percentile of positives, lo + 0.2)
Role ramp (job title vs resume): 5th / 95th percentile of the raw role similarity.
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.embed import get_embedder  # noqa: E402
from app.skills import SKILLS, embed_text, find_skills, group_of  # noqa: E402
from app.text import segments  # noqa: E402


def main(model: str | None = None, out: str = "eval/calibration.json"):
    rng = random.Random(0)
    emb = get_embedder(model)
    R = [json.loads(line) for line in gzip.open("data/processed/resumes.jsonl.gz", "rt")]
    pos, neg = [], []
    for r in rng.sample(R, 200):
        segs = segments(r["text"])
        for s in rng.sample(segs, min(15, len(segs))):
            hits = find_skills(s)
            groups = {group_of(h) for h in hits}
            for h in hits:
                pos.append((h, s))
            cands = [k for k in SKILLS if group_of(k) not in groups and group_of(k) != "soft"]
            for k in rng.sample(cands, 2):
                neg.append((k, s))
    rng.shuffle(pos)
    pos, neg = pos[:3000], neg[:3000]

    def cos(pairs):
        a = emb.encode([embed_text(p[0], p[0]) for p in pairs])
        b = emb.encode([p[1] for p in pairs])
        return (a * b).sum(1)

    cp, cn = cos(pos), cos(neg)
    lo = float(np.percentile(cn, 95))
    hi = max(float(np.percentile(cp, 75)), lo + 0.2)  # avoid an overly steep ramp

    # role similarity distribution
    J = [json.loads(line) for line in gzip.open("data/processed/jobs.jsonl.gz", "rt")]
    titles = emb.encode([j["title"] for j in rng.sample(J, 1000)])
    raws = []
    for r in rng.sample(R, 100):
        segs = segments(r["text"])[:200]
        u = emb.encode(segs + [" ".join(r["text"].split())[:600]])
        head, units = u[-1], u[:-1]
        sims = titles @ units.T
        top = -np.sort(-sims, axis=1)[:, :3].mean(1)
        raws.append(0.5 * (titles @ head) + 0.5 * top)
    raws = np.concatenate(raws)
    res = {"model": emb.model_name, "n_pos": len(cp), "n_neg": len(cn),
           "pos_mean": float(cp.mean()), "neg_mean": float(cn.mean()),
           "pos_pcts": {p: float(np.percentile(cp, p)) for p in (10, 25, 50, 75)},
           "neg_pcts": {p: float(np.percentile(cn, p)) for p in (50, 90, 95, 99)},
           "auc": float((cp[:, None] > cn[None, :]).mean()),
           "sim_lo": round(lo, 3), "sim_hi": round(hi, 3),
           "role_lo": round(float(np.percentile(raws, 5)), 3), "role_hi": round(float(np.percentile(raws, 95)), 3)}
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [None]), *(sys.argv[2:3] or []))
