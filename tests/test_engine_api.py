"""Integration tests: needs the built index (data/processed/index) and the embedding model."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/index/manifest.json").exists(), reason="index not built")

SAMPLE = (ROOT / "app/static/sample_resume.txt").read_text()
JD = ("Staff Accountant. Requirements: Bachelor's degree in Accounting. 1+ years of experience. Proficiency in Excel "
      "and NetSuite. Experience with account reconciliations, journal entries and month-end close. Knowledge of GAAP. "
      "Power BI experience is a plus.")


@pytest.fixture(scope="module")
def engine():
    from app.engine import JobFitEngine
    return JobFitEngine()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_batch_equals_pair(engine):
    """The vectorised ranking scorer and the detailed per-pair scorer implement the same formula."""
    resumes = [json.loads(l) for l in gzip.open(ROOT / "data/processed/resumes.jsonl.gz", "rt")][:5]
    rng = np.random.default_rng(0)
    for r in resumes + [{"text": SAMPLE}]:
        ctx = engine.resume_context(r["text"])
        overall, _ = engine.batch.score_all(ctx)
        for j in rng.choice(len(engine.meta), 25, replace=False):
            assert engine.score_job(int(j), ctx).score == pytest.approx(overall[j], abs=0.11)


def test_rank_deterministic(engine):
    from app.schemas import RankRequest
    a = engine.rank(RankRequest(resume_text=SAMPLE, top_k=10, explain_top=0))
    b = engine.rank(RankRequest(resume_text=SAMPLE, top_k=10, explain_top=0))
    assert [(x.job_id, x.score) for x in a.results] == [(x.job_id, x.score) for x in b.results]
    assert a.total_postings_scored == len(engine.meta)
    scores = [x.score for x in a.results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_partial_credit(engine):
    """'Power BI' should get partial credit from 'Tableau dashboards' (semantic), not exact credit."""
    m, jd, _ = engine.match(SAMPLE, JD, "Staff Accountant", explain_it=False)
    pbi = next(r for r in m.requirements if r.requirement == "Power BI")
    assert pbi.method == "semantic" and pbi.strength > 0
    excel = next(r for r in m.requirements if r.requirement == "Excel")
    assert excel.method == "exact" and excel.status == "matched"


def test_removing_evidence_lowers_score(engine):
    m1, _, _ = engine.match(SAMPLE, JD, "Staff Accountant", explain_it=False)
    weaker = "\n".join(l for l in SAMPLE.split("\n") if "GAAP" not in l and "reconciliation" not in l)
    m2, _, _ = engine.match(weaker, JD, "Staff Accountant", explain_it=False)
    assert m2.score < m1.score


def test_api_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["index"]["n_jobs"] > 0


def test_api_match(client):
    r = client.post("/match", json={"resume_text": SAMPLE, "job_description": JD, "job_title": "Staff Accountant"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["match"]["score"] <= 100
    assert body["match"]["explanation"]
    assert body["job_profile"]["min_years_experience"] == 1.0


def test_api_rank_and_filters(client):
    r = client.post("/rank", json={"resume_text": SAMPLE, "top_k": 5, "levels": ["Internship"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 5 and all(x["level"] == "Internship" for x in res)
    assert res[0]["explanation"]


def test_api_upload(client):
    r = client.post("/rank/upload", files={"file": ("resume.txt", SAMPLE.encode())}, data={"top_k": "3"})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    bad = client.post("/rank/upload", files={"file": ("resume.exe", b"MZ....")})
    assert bad.status_code == 415


def test_api_job_lookup(client, engine):
    jid = engine.meta[0]["job_id"]
    r = client.get(f"/jobs/{jid}")
    assert r.status_code == 200 and r.json()["description"]
    assert client.get("/jobs/does-not-exist").status_code == 404
    r = client.post(f"/jobs/{jid}/match", json={"resume_text": SAMPLE})
    assert r.status_code == 200 and "score" in r.json()


def test_api_validation(client):
    assert client.post("/rank", json={"resume_text": "short"}).status_code == 422
