"""Skill taxonomy used by the rule-based extractor and for exact-match evidence.

Each canonical skill maps to a group and a list of surface forms (aliases).
Aliases are matched case-insensitively on word boundaries.  The taxonomy is
deliberately focused on the five target domains of the project (Finance,
Banking, Accounting, IT, Consulting/Analyst) plus transferable skills.
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical -> (group, [aliases])
SKILLS: dict[str, tuple[str, list[str]]] = {
    # ---------------- programming / data ----------------
    "Python": ("programming", ["python", "pandas", "numpy"]),
    "R": ("programming", [r"\bR\b(?= programming| language|,| and | studio|studio)", "rstudio"]),
    "SQL": ("programming", ["sql", "t-sql", "pl/sql", "mysql", "postgresql", "postgres", "sql server", "tsql"]),
    "Java": ("programming", [r"java(?!script)"]),
    "JavaScript": ("programming", ["javascript", "typescript", "node.js", "nodejs", "react", "angular", "vue.js"]),
    "C/C++": ("programming", [r"c\+\+", r"\bC\b(?= programming| language)"]),
    "C#/.NET": ("programming", ["c#", r"\.net", "asp.net", "vb.net"]),
    "HTML/CSS": ("programming", ["html", "css"]),
    "Shell scripting": ("programming", ["bash", "shell scripting", "powershell", "unix shell"]),
    "VBA": ("programming", ["vba", "excel macros", "macros"]),
    "SAS": ("analytics", [r"\bsas\b"]),
    "SPSS": ("analytics", ["spss"]),
    "Stata": ("analytics", ["stata"]),
    "MATLAB": ("analytics", ["matlab"]),
    "Machine learning": ("analytics", ["machine learning", "scikit-learn", "sklearn", "predictive model", "deep learning", "tensorflow", "pytorch"]),
    "Statistics": ("analytics", ["statistics", "statistical analysis", "regression", "hypothesis testing", "statistical modeling"]),
    "Data analysis": ("analytics", ["data analysis", "data analytics", "analyze data", "analyzing data", "data-driven", "analytical skills"]),
    "Data visualization": ("analytics", ["data visualization", "dashboards", "dashboard", "visualizations", "data viz"]),
    "Tableau": ("analytics", ["tableau"]),
    "Power BI": ("analytics", ["power bi", "powerbi"]),
    "Looker": ("analytics", ["looker"]),
    "Excel": ("tools", ["excel", "spreadsheets", "spreadsheet", "pivot tables", "vlookup", "xlookup"]),
    "Advanced Excel": ("tools", ["advanced excel", "pivot tables", "vlookup", "index match", "excel modeling"]),
    "Microsoft Office": ("tools", ["microsoft office", "ms office", "office suite", "microsoft 365", "office 365", "ms word", "microsoft word", "powerpoint", "outlook"]),
    "PowerPoint": ("tools", ["powerpoint", "slide decks", "presentations"]),
    "Google Workspace": ("tools", ["google sheets", "google workspace", "g suite", "gsuite"]),
    "ETL": ("data_eng", ["etl", "data pipeline", "data pipelines", "informatica", "ssis", "data integration"]),
    "Big data": ("data_eng", ["hadoop", "spark", "pyspark", "hive", "big data", "databricks", "kafka"]),
    "Data warehousing": ("data_eng", ["data warehouse", "data warehousing", "snowflake", "redshift", "bigquery"]),
    "Databases": ("data_eng", ["database", "databases", "oracle database", "mongodb", "nosql", "dbms", "database administration"]),
    # ---------------- IT / infrastructure ----------------
    "Cloud (AWS/Azure/GCP)": ("it", ["aws", "amazon web services", "azure", "gcp", "google cloud", "cloud computing", "cloud services"]),
    "Networking": ("it", ["networking", "tcp/ip", "lan", "wan", "routers", "switches", "dns", "dhcp", "vpn", "cisco", "ccna", "network administration", "firewalls"]),
    "Cybersecurity": ("it", ["cybersecurity", "cyber security", "information security", "security+", "siem", "vulnerability", "penetration testing", "incident response", "network security"]),
    "Linux/Unix": ("it", ["linux", "unix", "red hat", "ubuntu"]),
    "Windows administration": ("it", ["windows server", "active directory", "group policy", "windows 10", "windows 11", "microsoft windows", "sccm", "intune"]),
    "Technical support": ("it", ["technical support", "help desk", "helpdesk", "service desk", "desktop support", "troubleshooting", "troubleshoot", "it support", "end user support"]),
    "IT service management": ("it", ["itil", "servicenow", "ticketing system", "jira service", "itsm", "remedy"]),
    "Virtualization": ("it", ["vmware", "virtualization", "hyper-v", "citrix", "virtual machines"]),
    "DevOps": ("it", ["devops", "ci/cd", "jenkins", "docker", "kubernetes", "terraform", "ansible", "github actions"]),
    "Version control": ("it", ["git", "github", "gitlab", "version control", "bitbucket"]),
    "Software development": ("it", ["software development", "software engineering", "object-oriented", "sdlc", "full stack", "web development", "application development", "coding"]),
    "Software testing / QA": ("it", ["quality assurance", "software testing", "test cases", "selenium", "qa testing", "unit testing", "test automation"]),
    "APIs": ("it", ["api", "apis", "rest api", "restful", "web services", "microservices"]),
    "Hardware": ("it", ["hardware", "printers", "peripherals", "laptops", "desktops", "hardware installation"]),
    "ERP systems": ("tools", ["erp", "sap", "oracle financials", "netsuite", "workday", "peoplesoft", "dynamics 365", "microsoft dynamics", "jd edwards"]),
    "CRM systems": ("tools", ["crm", "salesforce", "hubspot", "dynamics crm"]),
    "Agile/Scrum": ("business", ["agile", "scrum", "kanban", "sprint", "jira"]),
    # ---------------- accounting ----------------
    "GAAP": ("accounting", ["gaap", "generally accepted accounting principles", "us gaap"]),
    "IFRS": ("accounting", ["ifrs"]),
    "General ledger": ("accounting", ["general ledger", "journal entries", "journal entry", "ledger", "chart of accounts"]),
    "Account reconciliation": ("accounting", ["reconciliation", "reconciliations", "reconcile", "reconciling", "bank rec"]),
    "Accounts payable": ("accounting", ["accounts payable", r"\bA/P\b", r"\bAP\b(?= process| specialist| clerk)", "vendor payments", "invoice processing", "invoices"]),
    "Accounts receivable": ("accounting", ["accounts receivable", r"\bA/R\b", "collections", "billing", "invoicing", "cash application"]),
    "Month-end close": ("accounting", ["month-end", "month end", "year-end", "closing process", "financial close", "close process", "quarter-end"]),
    "Financial statements": ("accounting", ["financial statements", "balance sheet", "income statement", "cash flow statement", "p&l", "profit and loss", "financial reporting"]),
    "Bookkeeping": ("accounting", ["bookkeeping", "bookkeeper"]),
    "Payroll": ("accounting", ["payroll", "adp", "paychex"]),
    "Tax": ("accounting", ["tax", "taxation", "tax returns", "tax preparation", "sales tax", "1099", "tax compliance"]),
    "Audit": ("accounting", ["audit", "auditing", "internal audit", "external audit", "audits", "sox", "sarbanes-oxley", "internal controls"]),
    "Cost accounting": ("accounting", ["cost accounting", "inventory accounting", "standard costing", "cost analysis"]),
    "Fixed assets": ("accounting", ["fixed assets", "depreciation", "amortization"]),
    "Accruals": ("accounting", ["accruals", "accrual", "prepaid", "deferred revenue", "revenue recognition"]),
    "Accounting software": ("tools", ["quickbooks", "sage", "xero", "peachtree", "great plains", "blackline", "accounting software", "yardi"]),
    # ---------------- finance / banking ----------------
    "Financial analysis": ("finance", ["financial analysis", "financial analyst", "variance analysis", "ratio analysis", "analyze financial"]),
    "Financial modeling": ("finance", ["financial modeling", "financial model", "financial models", "dcf", "discounted cash flow", "three-statement"]),
    "Valuation": ("finance", ["valuation", "valuations", "comparable companies", "precedent transactions"]),
    "Budgeting & forecasting": ("finance", ["budgeting", "budget", "budgets", "forecasting", "forecast", "forecasts", "fp&a", "financial planning"]),
    "Corporate finance": ("finance", ["corporate finance", "capital budgeting", "treasury", "cash management", "working capital", "capital structure"]),
    "Investment analysis": ("finance", ["investment", "investments", "portfolio", "equity research", "asset management", "securities", "fixed income", "equities", "wealth management"]),
    "Risk management": ("finance", ["risk management", "risk assessment", "credit risk", "market risk", "operational risk", "risk analysis", "risk mitigation"]),
    "Credit analysis": ("finance", ["credit analysis", "underwriting", "underwrite", "credit review", "loan analysis", "credit memos"]),
    "Banking operations": ("finance", ["banking", "bank operations", "branch operations", "teller", "cash handling", "deposits", "wire transfers", "commercial banking", "retail banking"]),
    "Lending": ("finance", ["lending", "loan", "loans", "mortgage", "mortgages", "loan origination", "loan processing"]),
    "Compliance / regulatory": ("finance", ["compliance", "regulatory", "aml", "anti-money laundering", "kyc", "bsa", "finra", "regulations"]),
    "Bloomberg": ("tools", ["bloomberg"]),
    "Capital markets": ("finance", ["capital markets", "trading", "derivatives", "investment banking", "m&a", "mergers and acquisitions"]),
    "Insurance": ("finance", ["insurance", "claims", "actuarial"]),
    # ---------------- consulting / business ----------------
    "Project management": ("business", ["project management", "project manager", "managing projects", "project planning", "pmp", "project coordination", "project lifecycle"]),
    "Business analysis": ("business", ["business analysis", "business analyst", "business requirements", "requirements gathering", "process mapping", "use cases", "user stories"]),
    "Process improvement": ("business", ["process improvement", "lean", "six sigma", "continuous improvement", "streamline", "process optimization", "efficiency improvements"]),
    "Strategy": ("business", ["strategy", "strategic planning", "business strategy", "strategic initiatives", "market research", "competitive analysis"]),
    "Change management": ("business", ["change management", "organizational change"]),
    "Stakeholder management": ("business", ["stakeholder", "stakeholders", "stakeholder management", "cross-functional", "cross functional"]),
    "Client relationship management": ("business", ["client relations", "client relationships", "client-facing", "client facing", "customer relationships", "account management", "relationship management"]),
    "Customer service": ("business", ["customer service", "customer support", "customer satisfaction", "customer experience", "client service"]),
    "Sales": ("business", ["sales", "selling", "business development", "lead generation", "quota"]),
    "Operations management": ("business", ["operations management", "business operations", "logistics", "supply chain", "inventory management", "procurement", "purchasing"]),
    "Reporting": ("business", ["reporting", "reports", "kpi", "kpis", "metrics", "ad hoc reports"]),
    "Documentation": ("business", ["documentation", "technical writing", "standard operating procedures", "sops"]),
    "Vendor management": ("business", ["vendor management", "vendors", "contract negotiation", "contracts"]),
    "Training & mentoring": ("business", ["training", "mentoring", "coaching", "onboarding", "train new"]),
    "Research": ("business", ["research", "researching"]),
    # ---------------- soft skills ----------------
    "Communication": ("soft", ["communication skills", "communication", "written and verbal", "verbal and written", "communicate"]),
    "Teamwork": ("soft", ["teamwork", "team player", "collaborate", "collaboration", "collaborative", "team environment"]),
    "Leadership": ("soft", ["leadership", "lead a team", "led a team", "supervised", "supervise", "supervisory", "team lead", "managed a team"]),
    "Problem solving": ("soft", ["problem solving", "problem-solving", "critical thinking", "solve problems", "analytical thinking"]),
    "Attention to detail": ("soft", ["attention to detail", "detail-oriented", "detail oriented", "accuracy", "meticulous"]),
    "Time management": ("soft", ["time management", "multitask", "multi-task", "multitasking", "prioritize", "deadlines", "deadline-driven", "organizational skills", "organized"]),
    "Presentation skills": ("soft", ["presentation skills", "presenting", "public speaking", "present findings"]),
    "Negotiation": ("soft", ["negotiation", "negotiate", "negotiating"]),
    "Adaptability": ("soft", ["adaptability", "adaptable", "fast-paced", "fast paced", "flexible"]),
    "Interpersonal skills": ("soft", ["interpersonal", "relationship building", "people skills"]),
    "Bilingual": ("soft", ["bilingual", "spanish", "mandarin", "french", "multilingual"]),
    # ---------------- certifications ----------------
    "CPA": ("certification", ["cpa", "certified public accountant"]),
    "CFA": ("certification", ["cfa", "chartered financial analyst"]),
    "CMA": ("certification", [r"\bcma\b", "certified management accountant"]),
    "FINRA licenses (SIE/Series 7/63)": ("certification", ["series 7", "series 63", "series 65", "series 66", r"\bsie\b", "finra license"]),
    "CompTIA (A+/Network+/Security+)": ("certification", ["comptia", "a+ certification", "network+", "security+", r"\bA\+"]),
    "PMP / CAPM": ("certification", ["pmp", "capm", "project management professional"]),
    "Cloud certification": ("certification", ["aws certified", "azure certification", "az-900", "solutions architect"]),
}

GROUP_LABELS = {
    "programming": "Programming", "analytics": "Analytics", "tools": "Tools & Software",
    "data_eng": "Data Engineering", "it": "IT & Infrastructure", "accounting": "Accounting",
    "finance": "Finance & Banking", "business": "Business & Consulting", "soft": "Soft Skills",
    "certification": "Certifications",
}

def _alias_to_regex(alias: str) -> str:
    if any(c in alias for c in "\\(?"):
        return alias  # already a regex
    esc = re.escape(alias.strip())
    left = r"(?<![A-Za-z0-9])"
    right = r"(?![A-Za-z0-9])"
    return left + esc + right


@lru_cache(maxsize=1)
def compiled() -> list[tuple[str, str, re.Pattern]]:
    out = []
    for canon, (group, aliases) in SKILLS.items():
        # case-sensitive patterns for single-letter languages
        parts = []
        for a in aliases:
            parts.append(_alias_to_regex(a))
        flags = re.IGNORECASE
        if canon in {"R", "C/C++"}:
            # the special R/C patterns must stay case sensitive; others in group ignore case
            cs = [p for a, p in zip(aliases, parts) if a.startswith(r"\b")]
            ci = [p for a, p in zip(aliases, parts) if not a.startswith(r"\b")]
            if cs:
                out.append((canon, group, re.compile("|".join(cs))))
            if ci:
                out.append((canon, group, re.compile("|".join(ci), flags)))
            continue
        out.append((canon, group, re.compile("|".join(parts), flags)))
    return out


def find_skills(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return {canonical_skill: [(start, end), ...]} for all skills found in text."""
    hits: dict[str, list[tuple[int, int]]] = {}
    for canon, _group, pat in compiled():
        for m in pat.finditer(text):
            hits.setdefault(canon, []).append((m.start(), m.end()))
    return hits


def group_of(skill: str) -> str:
    return SKILLS.get(skill, ("other", []))[0]


def canonicalize(name: str) -> str | None:
    """Map a free-text skill name (e.g. from an LLM) onto the taxonomy, if possible."""
    name = name.strip()
    if not name:
        return None
    low = name.lower()
    for canon in SKILLS:
        if canon.lower() == low:
            return canon
    hits = find_skills(name)
    if len(hits) == 1:
        return next(iter(hits))
    if hits:
        # prefer the skill whose match covers the largest share of the string
        best = max(hits.items(), key=lambda kv: max(e - s for s, e in kv[1]))
        return best[0]
    return None


# Short descriptions used ONLY for embedding a requirement (a bare tool name such as
# "Power BI" carries little meaning for a sentence encoder; "Power BI (business
# intelligence dashboards ...)" does).  Skills not listed here use their aliases.
DESCRIPTIONS: dict[str, str] = {
    "Python": "programming and scripting for data analysis and automation",
    "R": "statistical programming language",
    "SQL": "querying relational databases",
    "VBA": "Excel macros and automation",
    "SAS": "statistical analysis software",
    "SPSS": "statistical analysis software",
    "Stata": "statistical / econometric software",
    "Tableau": "business intelligence dashboards and data visualization",
    "Power BI": "business intelligence dashboards and data visualization",
    "Looker": "business intelligence dashboards and data visualization",
    "Excel": "spreadsheets, pivot tables, formulas",
    "Advanced Excel": "pivot tables, VLOOKUP, spreadsheet modeling",
    "ERP systems": "enterprise resource planning and accounting systems such as SAP, Oracle, NetSuite",
    "CRM systems": "customer relationship management software such as Salesforce",
    "Accounting software": "bookkeeping and accounting systems such as QuickBooks",
    "Bloomberg": "financial market data terminal",
    "GAAP": "US accounting standards and principles",
    "IFRS": "international accounting standards",
    "General ledger": "journal entries and maintaining the ledger",
    "Month-end close": "closing the books at month end and year end",
    "Cloud (AWS/Azure/GCP)": "cloud computing platforms and services",
    "DevOps": "CI/CD pipelines, containers, deployment automation",
    "Agile/Scrum": "agile software delivery, sprints",
    "CPA": "certified public accountant license",
    "CFA": "chartered financial analyst investment credential",
    "CMA": "certified management accountant credential",
    "FINRA licenses (SIE/Series 7/63)": "securities licenses",
    "CompTIA (A+/Network+/Security+)": "IT support and networking certifications",
    "PMP / CAPM": "project management certification",
    "Budgeting & forecasting": "budgets, forecasts, FP&A, variance to plan",
    "Financial modeling": "building financial models, DCF, three-statement models in Excel",
    "Capital markets": "trading, securities, investment banking deals",
    "Technical support": "help desk, troubleshooting user hardware and software issues",
}


def describe(skill: str) -> str:
    if skill in DESCRIPTIONS:
        return DESCRIPTIONS[skill]
    aliases = [a for a in SKILLS.get(skill, ("", []))[1] if "\\" not in a and "(" not in a]
    return ", ".join(aliases[:4])


def embed_text(text: str, skill: str | None) -> str:
    """The string actually embedded for a requirement."""
    if not skill:
        return text
    d = describe(skill)
    return f"{text} ({d})" if d and d.lower() != text.lower() else text
