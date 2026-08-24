"""Runtime config. No secrets in this file."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.getenv("LEG_OUT_DIR", str(PROJECT_ROOT / "out")))
GRAPH_DIR = OUT_DIR / "graph"
DEMO_DIR = OUT_DIR / "demo"
FIXTURES_DIR = PROJECT_ROOT / "fixtures"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "evidence_graph")
USE_FIRESTORE = os.getenv("LEG_USE_FIRESTORE", "false").strip().lower() in {"1", "true", "yes"}

# Public API endpoints
CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DAILYMED_SPLS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL_XML = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENTARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_MOLECULE_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
CHEMBL_MECHANISM = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
CHEMBL_TARGET = "https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"

USER_AGENT = os.getenv(
    "LEG_USER_AGENT",
    "living-evidence-graph/0.1 (contest scaffold; public data only; not a medical product)",
)
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "").strip()
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "contest@example.com").strip()

HTTP_TIMEOUT = float(os.getenv("LEG_HTTP_TIMEOUT", "20"))

# Demo vertical: cancer immunotherapy (public evidence only).
# Brand / ingredient strings for API queries — not invented IDs.
DEMO_DRUG_BRAND = "Keytruda"
DEMO_DRUG_INGREDIENT = "pembrolizumab"
DEMO_CONDITION = "non-small cell lung cancer"
DEMO_GOAL = "pembrolizumab / Keytruda NSCLC solid tumor evidence graph"


def gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def has_gemini_key() -> bool:
    return bool(gemini_api_key())
