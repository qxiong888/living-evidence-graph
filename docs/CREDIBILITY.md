# Credibility / trust formula

Implemented in `living_evidence_graph/credibility.py`. Apply to **every edge**.

## Core graph motif (LLM retrieval spine)

First-class triangle for drug–target–disease retrieval (Open Targets + ChEMBL):

| Edge type | Meaning (not causation) |
|-----------|-------------------------|
| `drug_targets_gene` | Drug → gene/protein target (mechanism) |
| `gene_associated_with_disease` | Gene → disease association |
| `drug_indicated_for_disease` | Drug → disease indication / label context |

ClinicalTrials.gov, PubMed, openFDA, DailyMed, and Europe PMC are **corroboration layers** around that spine (`treats_indication`, `studied_in`, `supports`, `reports_ae`, `warns_ae`, retraction flags, etc.).

## Formula

\[
trust = \mathrm{clip}_{[0,1]}\big(
0.35\cdot source\_tier
+ 0.30\cdot corroboration
+ 0.20\cdot recency
+ 0.15\cdot consistency
- retraction\_penalty
\big)
\]

### Components

| Signal | Definition |
|--------|------------|
| **source_tier** | Max tier among edge `sources[]` (see table). Unknown → 0.3. |
| **corroboration** | `min(1, distinct_source_families / 3)` using `SOURCE_FAMILY` collapse. |
| **recency** | `exp(-age_days / 365)`; missing age → **0.5**. |
| **consistency** | **1.0** if no `contradicts` relationship touches the same endpoints; else **0.4**. |
| **retraction_penalty** | **0.5** if retracted/erratum flagged (e.g. Europe PMC); else **0**. |

### source_tier table (exact)

| Source tag | Family | Tier | Notes |
|------------|--------|------|-------|
| `dailymed_label` | dailymed | **0.95** | FDA SPL labeled indication / warning text via NLM DailyMed |
| `clinicaltrials_registry` | clinicaltrials | **0.9** | ClinicalTrials.gov registry |
| `pubmed_peer_reviewed` | pubmed | **0.8** | PubMed / NCBI E-utilities (metadata/IDs) |
| `europepmc` | europepmc | **0.75** | Europe PMC status / literature signals (feeds retraction_penalty) |
| `opentargets_kb` | opentargets | **0.7** | Open Targets structured KB (CC0) |
| `chembl` | chembl | **0.65** | ChEMBL molecule/mechanism (CC BY-SA 3.0) |
| `openfda_faers` | openfda | **0.55** | FAERS **reports only** — not rates |
| `preprint` | preprint | **0.4** | Preprint (if ever used) |

**Seven active public source families in the demo:** clinicaltrials, pubmed, openfda, dailymed, europepmc, opentargets, chembl.

openFDA edges must remain labeled as **voluntary reports**, not incidence rates, regardless of score. DailyMed `warns_ae` is label text, not rates. No causation claims on any edge.

## Worked example (one edge)

**Edge:** `treats_indication`  
Keytruda/pembrolizumab → non-small cell lung cancer  
**sources:** `clinicaltrials_registry`, `pubmed_peer_reviewed`  
**age_days:** 120 · **contradict:** false · **retracted:** false

| Term | Value |
|------|-------|
| source_tier | max(0.9, 0.8) = **0.9** |
| corroboration | min(1, 2/3) = **0.6667** |
| recency | exp(-120/365) ≈ **0.7195** |
| consistency | **1.0** |
| retraction_penalty | **0** |

\[
raw = 0.35(0.9) + 0.30(0.6667) + 0.20(0.7195) + 0.15(1.0)
= 0.315 + 0.2000 + 0.1439 + 0.15
\approx 0.8089
\]

\[
trust\_score = clip(0,1,0.8089) \approx \mathbf{0.8089}
\]

If the same edge were later marked retracted (Europe PMC signal), subtract 0.5 → ≈ 0.3089.
