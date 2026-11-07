"""Build the processed job database and resume evaluation set from the raw Kaggle files.

Raw inputs (download with scripts/download_data.sh):
  data/raw/linkedin/postings.csv, data/raw/linkedin/jobs/job_skills.csv
  data/raw/resume/Resume/Resume.csv

Outputs:
  data/processed/jobs.jsonl.gz      ~8,000 entry-level postings in target functions
  data/processed/resumes.jsonl.gz   resumes in the 5 target categories
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

TARGET_FUNCS = {"FIN", "ACCT", "IT", "CNSL", "ANLS"}
LEVELS = ["Entry level", "Associate", "Internship"]
RESUME_CATS = ["FINANCE", "BANKING", "ACCOUNTANT", "INFORMATION-TECHNOLOGY", "CONSULTANT"]


def clean_text(t: str, keep_breaks: bool = False) -> str:
    t = str(t or "")
    t = t.replace(" ", " ").replace("​", " ")
    if keep_breaks:
        # Kaggle resumes encode bullet/section boundaries as runs of 2+ spaces
        t = re.sub(r" {2,}", "\n", t)
    # LinkedIn descriptions often lose newlines: "transactions.Oversee" -> "transactions. Oversee"
    t = re.sub(r"([a-z0-9\)])([.!?:;])([A-Z])", r"\1\2 \3", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-jobs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["job_id", "title", "company_name", "location", "formatted_experience_level",
            "description", "skills_desc", "formatted_work_type", "remote_allowed",
            "normalized_salary", "job_posting_url", "listed_time"]
    posts = pd.read_csv(raw / "linkedin/postings.csv", usecols=cols)
    tags = pd.read_csv(raw / "linkedin/jobs/job_skills.csv")
    posts = posts[posts.formatted_experience_level.isin(LEVELS)]
    print(f"entry-level/associate/internship postings: {len(posts):,}")

    tagset = tags[tags.job_id.isin(posts.job_id)].groupby("job_id").skill_abr.apply(lambda s: sorted(set(s)))
    posts = posts.join(tagset.rename("functions"), on="job_id")
    posts = posts[posts.functions.apply(lambda f: isinstance(f, list) and bool(set(f) & TARGET_FUNCS))]
    posts = posts[posts.description.fillna("").str.len() > 300]
    print(f"in target functions (FIN/ACCT/IT/CNSL/ANLS): {len(posts):,}")

    # Stratified sample: IT dominates (~68%), so keep every non-IT-only posting
    # and fill the remainder with IT-only postings.
    it_only = posts.functions.apply(lambda f: set(f) & TARGET_FUNCS == {"IT"})
    other = posts[~it_only]
    n_it = max(0, args.n_jobs - len(other))
    sample = pd.concat([other, posts[it_only].sample(n=min(n_it, it_only.sum()), random_state=args.seed)])
    sample = sample.sample(frac=1.0, random_state=args.seed).head(args.n_jobs)
    print(f"sampled: {len(sample):,}")

    n = 0
    with gzip.open(out / "jobs.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in sample.itertuples(index=False):
            desc = clean_text(r.description)
            if isinstance(r.skills_desc, str) and r.skills_desc.strip():
                desc += "\nSkills: " + clean_text(r.skills_desc)
            rec = {
                "job_id": str(r.job_id), "title": str(r.title), "company": str(r.company_name) if pd.notna(r.company_name) else "",
                "location": str(r.location) if pd.notna(r.location) else "", "level": r.formatted_experience_level,
                "work_type": str(r.formatted_work_type) if pd.notna(r.formatted_work_type) else "",
                "remote": bool(r.remote_allowed == 1.0), "salary": float(r.normalized_salary) if pd.notna(r.normalized_salary) else None,
                "url": str(r.job_posting_url) if pd.notna(r.job_posting_url) else "",
                "functions": [x for x in r.functions if x in TARGET_FUNCS], "all_functions": r.functions,
                "description": desc,
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n:,} jobs -> {out/'jobs.jsonl.gz'}")

    res = pd.read_csv(raw / "resume/Resume/Resume.csv")
    res = res[res.Category.isin(RESUME_CATS)]
    with gzip.open(out / "resumes.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in res.itertuples(index=False):
            f.write(json.dumps({"resume_id": str(r.ID), "category": r.Category, "text": clean_text(r.Resume_str, keep_breaks=True)}) + "\n")
    print(f"wrote {len(res):,} resumes -> {out/'resumes.jsonl.gz'}")
    print(res.Category.value_counts().to_string())


if __name__ == "__main__":
    main()
