"""Unit tests for extraction, parsing and the scoring formula (no index needed)."""
import numpy as np
import pytest

from app.extract import RuleExtractor
from app.schemas import JDProfile, Requirement, ResumeProfile
from app.scoring import coverage, education_fit, experience_fit, combine, ramp
from app.skills import canonicalize, embed_text, find_skills
from app.text import education, jd_min_years, resume_years, segments


def test_find_skills_basic():
    hits = find_skills("Proficient in Python, SQL and C++; built Tableau dashboards; prepared journal entries.")
    assert {"Python", "SQL", "C/C++", "Tableau", "Data visualization", "General ledger"} <= set(hits)


def test_find_skills_no_generic_false_positives():
    assert find_skills("In other words, the rest of the team was there.") == {}


def test_canonicalize():
    assert canonicalize("PostgreSQL") == "SQL"
    assert canonicalize("Microsoft Power BI") == "Power BI"
    assert canonicalize("underwater basket weaving") is None


def test_embed_text_adds_descriptor():
    assert embed_text("Power BI", "Power BI").startswith("Power BI (")
    assert embed_text("free text", None) == "free text"


def test_segments_split_glued_bullets():
    segs = segments("Several years of experience in credit- MBA/CFA a plus- Comprehensive knowledge of tax.")
    assert segs == ["Several years of experience in credit", "MBA/CFA a plus", "Comprehensive knowledge of tax."]


@pytest.mark.parametrize("text,expected", [
    ("Requires 3+ years of accounting experience", 3.0),
    ("Minimum of two years experience in banking; 5 years experience preferred", 2.0),
    ("We have been in business for 30 years.", None),
])
def test_jd_min_years(text, expected):
    assert jd_min_years(text) == expected


def test_resume_years_merges_overlaps():
    t = "Analyst 01/2018 to 12/2019\nIntern 06/2019 to 08/2019\nAssociate January 2020 to January 2022"
    assert resume_years(t) == pytest.approx(3.9, abs=0.1)  # 1.9 + 2.0, overlap not double counted


def test_education():
    assert education("Bachelor's degree in Accounting or Finance required; MBA preferred", mode="min")[0] == "bachelor"
    lvl, fields = education("M.S. in Computer Science, B.S. in Mathematics", mode="max")
    assert lvl == "master" and "computer science" in fields


def test_rule_extractor_required_vs_preferred():
    jd = ("Staff Accountant\nRequirements:\n- Bachelor's degree in Accounting\n- 2+ years of experience\n"
          "- Strong Excel skills and knowledge of GAAP\nPreferred Qualifications:\n- CPA\n- NetSuite experience")
    p = RuleExtractor().extract_jd(jd, "Staff Accountant")
    imp = {r.skill: r.importance for r in p.requirements}
    assert imp["Excel"] == "required" and imp["GAAP"] == "required"
    assert imp["CPA"] == "preferred" and imp["ERP systems"] == "preferred"
    assert p.min_years_experience == 2.0 and p.education_level == "bachelor"
    assert "accounting" in p.education_fields


def test_rule_extractor_resume():
    p = RuleExtractor().extract_resume("Jane Doe\nB.S. in Finance\nAnalyst, ACME 01/2020 to 01/2023\n"
                                       "- Built DCF models in Excel and wrote SQL queries")
    assert {"Excel", "SQL", "Financial modeling"} <= set(p.skills)
    assert p.years_experience == pytest.approx(3.0, abs=0.1)
    assert p.education_level == "bachelor"


# ----------------------------------------------------------------- formula
def test_ramp():
    assert list(np.round(ramp([0.0, 0.42, 1.0], 0.32, 0.52), 2)) == [0.0, 0.5, 1.0]


def test_coverage_smoothing():
    # one fully-matched requirement is pulled toward the prior (0.3) by k=2 pseudo-requirements
    assert coverage(np.array([1.0]), np.array([1.0])) == pytest.approx((1 + 0.6) / 3)
    assert coverage(np.array([]), np.array([])) is None
    many = coverage(np.ones(20), np.ones(20))
    assert many > 0.9


def test_experience_and_education_fit():
    assert experience_fit(None, 3) is None
    assert experience_fit(4, 2) == 0.5
    assert experience_fit(2, 10) == 1.0
    assert education_fit(None, [], "bachelor", []) is None
    assert education_fit("bachelor", ["accounting"], "bachelor", ["accounting"]) == 1.0
    assert education_fit("master", [], "bachelor", []) == pytest.approx(0.6 * 0.5 + 0.4)
    assert education_fit("bachelor", ["accounting"], "bachelor", ["finance"]) == pytest.approx(0.6 + 0.4 * 0.7)


def test_combine_renormalises_missing_components():
    full = combine({"required": 1.0, "preferred": 1.0, "experience": 1.0, "education": 1.0, "role": 1.0})
    partial = combine({"required": 1.0, "preferred": None, "experience": None, "education": None, "role": 1.0})
    assert full == partial == 100.0
