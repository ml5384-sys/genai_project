"""Deterministic text utilities: segmentation, experience years, education parsing."""
from __future__ import annotations

import re
from datetime import date

from .schemas import EDU_LEVELS

# ----------------------------------------------------------------- segmentation
_BULLETS = re.compile(r"^[\s\-\*•●▪–·•●▪◦o]+\s*")


def segments(text: str, min_len: int = 12, max_len: int = 300, min_words: int = 1) -> list[str]:
    """Split a resume/JD into sentence- or bullet-sized evidence units."""
    out: list[str] = []
    for line in re.split(r"[\n\r]+", text):
        line = _BULLETS.sub("", line).strip()
        if not line:
            continue
        # sentences, inline bullets, and bullets glued with "- " ("... a plus- Comprehensive knowledge")
        for sent in re.split(r"(?<=[.!?;])\s+(?=[A-Z(])|\s+[•●▪]\s+|(?<=[a-z0-9).])\s?-\s+(?=[A-Z])", line):
            sent = sent.strip(" -\t")
            if len(sent) < min_len:
                continue
            while len(sent) > max_len:  # very long run-on lines: cut on commas
                cut = sent.rfind(",", 0, max_len)
                cut = cut if cut > 40 else max_len
                out.append(sent[:cut].strip())
                sent = sent[cut + 1:].strip()
            if len(sent) >= min_len and len(sent.split()) >= min_words:
                out.append(sent)
    # de-duplicate (Kaggle resumes repeat whole blocks), keep order
    seen, uniq = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


# ----------------------------------------------------------- JD section context
REQ_HEAD = re.compile(
    r"\b(requirements?|qualifications?|what you(?:'ll| will)? (?:need|bring)|you (?:have|bring)|must[- ]haves?|"
    r"minimum qualifications|basic qualifications|required skills|who you are|skills(?: and| &)? (?:experience|abilities)|"
    r"knowledge,? skills|what we(?:'re| are) looking for|ideal candidate|competencies)\b", re.I)
PREF_HEAD = re.compile(r"\b(preferred|nice[- ]to[- ]have|bonus|desired|desirable|a plus|is a plus|advantageous|ideally|"
                       r"not required|highly desired|preference)\b", re.I)
DUTY_HEAD = re.compile(r"\b(responsibilities|what you(?:'ll| will) do|duties|the role|day[- ]to[- ]day|job summary|"
                       r"about the (?:role|job|position)|position summary|overview)\b", re.I)
BENEFIT_HEAD = re.compile(r"\b(benefits|perks|we offer|compensation|pay range|salary range|equal opportunity|eeo|"
                          r"about us|about the company|who we are)\b", re.I)


def jd_sections(text: str) -> list[tuple[str, str]]:
    """Label each JD segment with its section: requirement / preferred / duty / benefit / other.

    Headings are detected heuristically; a segment that itself contains a preference
    cue ("... is a plus") is labelled preferred regardless of section.
    """
    labelled: list[tuple[str, str]] = []
    mode = "other"
    for seg in segments(text, min_len=2, max_len=400):
        head = seg[:60]
        if len(seg) < 70 or seg.endswith(":"):
            if PREF_HEAD.search(head):
                mode = "preferred"
            elif REQ_HEAD.search(head):
                mode = "requirement"
            elif BENEFIT_HEAD.search(head):
                mode = "benefit"
            elif DUTY_HEAD.search(head):
                mode = "duty"
        else:
            # inline heading, e.g. "Qualifications: 3+ years of ..."
            m = re.match(r"^([A-Za-z' ,&/]{3,60}):\s", seg)
            if m:
                h = m.group(1)
                if PREF_HEAD.search(h):
                    mode = "preferred"
                elif REQ_HEAD.search(h):
                    mode = "requirement"
                elif BENEFIT_HEAD.search(h):
                    mode = "benefit"
                elif DUTY_HEAD.search(h):
                    mode = "duty"
        label = mode
        if label in ("requirement", "duty", "other") and PREF_HEAD.search(seg):
            label = "preferred"
        labelled.append((label, seg))
    return labelled


# ------------------------------------------------------------- experience years
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_YEARS_REQ = re.compile(
    r"(?P<n>\d{1,2}(?:\.\d)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*\+?\s*"
    r"(?:(?:-|to|–)\s*\d{1,2}\s*)?\+?\s*(?:years?|yrs?)\b(?P<tail>[^.;\n]{0,60})", re.I)


def jd_min_years(text: str) -> float | None:
    vals = []
    for m in _YEARS_REQ.finditer(text):
        tail = m.group("tail").lower()
        ctx = text[max(0, m.start() - 40): m.start()].lower()
        if "experience" not in tail and "experience" not in ctx:
            continue
        if any(w in tail for w in ("old", "age", "warranty", "in business", "company", "history")):
            continue
        n = m.group("n").lower()
        v = float(_NUM_WORDS.get(n, n))
        if 0 < v <= 15:
            vals.append(v)
    return min(vals) if vals else None


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE = r"(?:(?P<{p}m>[A-Za-z]{{3,9}})\.?\s+|(?P<{p}n>\d{{1,2}})\s*/\s*)?(?P<{p}y>(?:19|20)\d{{2}})"
_RANGE = re.compile(_DATE.format(p="a") + r"\s*(?:to|-|–|—|until)\s*(?:" + _DATE.format(p="b") +
                    r"|(?P<cur>current|present|now|today))", re.I)


def _ym(month: str | None, num: str | None, year: str) -> float:
    m = 1
    if month and month[:3].lower() in _MONTHS:
        m = _MONTHS[month[:3].lower()]
    elif num and 1 <= int(num) <= 12:
        m = int(num)
    return int(year) + (m - 1) / 12


def resume_years(text: str, today: date | None = None) -> float | None:
    """Total years of experience: union of employment date ranges (overlaps merged).

    Falls back to explicit statements like "5 years of experience".
    """
    today = today or date(2024, 6, 1)  # dataset snapshot; keeps results reproducible
    now = today.year + (today.month - 1) / 12
    flat = re.sub(r"\s+", " ", text)
    spans = []
    for m in _RANGE.finditer(flat):
        start = _ym(m.group("am"), m.group("an"), m.group("ay"))
        end = now if m.group("cur") else _ym(m.group("bm"), m.group("bn"), m.group("by"))
        if 1970 <= start <= end <= now + 0.1 and end - start < 45:
            spans.append((start, end))
    total = 0.0
    if spans:
        spans.sort()
        cs, ce = spans[0]
        for s, e in spans[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                total += ce - cs
                cs, ce = s, e
        total += ce - cs
    stated = [float(_NUM_WORDS.get(m.group("n").lower(), m.group("n")))
              for m in _YEARS_REQ.finditer(flat) if "experience" in m.group("tail").lower()]
    stated = [v for v in stated if 0 < v <= 45]
    best = max([total] + stated) if (spans or stated) else None
    return round(best, 1) if best is not None else None


# -------------------------------------------------------------------- education
_EDU_PATTERNS = [
    ("doctorate", r"\b(ph\.?\s?d|doctorate|doctoral)(?!\w)"),
    ("master", r"\b(master'?s?|m\.?b\.?a|mba|m\.s\.|msc|m\.sc|ms in|ma in|master of)(?!\w)"),
    ("bachelor", r"\b(bachelor'?s?|b\.?s\.?c?|b\.a\.|ba in|bs in|bba|b\.b\.a|undergraduate degree|4[- ]year degree|"
                 r"four[- ]year degree|college degree|university degree|bachelor of)(?!\w)"),
    ("associate", r"\b(associate'?s? degree|associate of|a\.a\.s?|2[- ]year degree|two[- ]year degree)(?!\w)"),
    ("high_school", r"\b(high school|ged|h\.s\. diploma)(?!\w)"),
]
EDU_FIELDS = {
    "accounting": r"accounting|accountancy",
    "finance": r"\bfinance\b|financial",
    "economics": r"economics|econ\b",
    "business": r"business|management|\bmba\b|commerce",
    "computer science": r"computer science|\bcs\b|software engineering|computer engineering",
    "information technology": r"information technology|information systems|\bmis\b|\bcis\b|computer information",
    "mathematics / statistics": r"mathematics|\bmath\b|statistics|actuarial|quantitative",
    "engineering": r"engineering",
    "data science": r"data science|analytics",
}


def education(text: str, mode: str = "max") -> tuple[str | None, list[str]]:
    """Return (level, fields). mode='max' for resumes, 'min' for JD minimum requirement."""
    found = []
    for level, pat in _EDU_PATTERNS:
        if re.search(pat, text, re.I):
            found.append(level)
    level = None
    if found:
        key = (lambda lv: EDU_LEVELS[lv])
        level = max(found, key=key) if mode == "max" else min(found, key=key)
        if mode == "min" and level == "high_school" and len(found) > 1:
            # "high school diploma required, bachelor's preferred" -> keep high school
            pass
    fields = []
    # only look for degree fields near degree words to avoid picking up every "business"
    for m in re.finditer(r"(degree|bachelor|master|b\.?s\.?|b\.?a\.?|mba|major|diploma)[^.\n]{0,80}", text, re.I):
        window = text[max(0, m.start() - 60): m.end()]
        for f, pat in EDU_FIELDS.items():
            if re.search(pat, window, re.I) and f not in fields:
                fields.append(f)
    return level, fields
