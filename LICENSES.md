# Licenses & attribution (upstream data)

This repository’s **code** is the contest submission. Upstream **data** remains under each provider’s terms. This project does **not** claim endorsement by FDA, NLM, NIH, NCBI, Open Targets, EMBL-EBI, ChEMBL, or Europe PMC.

## Upstream sources

| Source | Access | License / terms (summary) | How we use it |
|--------|--------|---------------------------|---------------|
| **Open Targets Platform** | Public GraphQL API | **CC0 1.0** | Drug–target–disease associations on the graph spine |
| **ChEMBL** | Public REST API | **CC BY-SA 3.0** — cite Mendez et al., *Nucleic Acids Res.* 2019;47(D1):D930–D940. doi:10.1093/nar/gky1075. Record `chembl_db_version` when returned. | Molecule / mechanism → `drug_targets_gene`. **ChEMBL-derived graph subsets are ShareAlike.** |
| **openFDA (FAERS)** | Public REST API | FDA openFDA terms | Adverse-event **report** counts only — not rates, not causation |
| **ClinicalTrials.gov** | Public API v2 | Courtesy of the **U.S. National Library of Medicine** | Trial registry rows → `studied_in` |
| **DailyMed (FDA SPL)** | Public NLM REST | Courtesy of the **U.S. National Library of Medicine** / FDA SPL | Label indication / warning text → `drug_indicated_for_disease`, `warns_ae` |
| **PubMed (NCBI E-utilities)** | Public API | Courtesy of NLM; subject to **NCBI policies/disclaimer** | PMID metadata only — retrieval links, not abstract dumps for training |
| **Europe PMC** | Public REST | **Per-article licenses** for full text | Retraction/erratum/correction **status metadata** only — no non-OA full-text ingest |

## ShareAlike note (ChEMBL)

Edges and node properties clearly derived from ChEMBL mechanism/molecule payloads should be treated as **CC BY-SA 3.0** ShareAlike when redistributed as a derived dataset. Attribute ChEMBL, cite the paper, and include the database version when known.

## LLM / training posture

- Prefer **retrieval over the living graph** (IDs, edge types, trust scores, evidence URLs).
- **Do not** dump PubMed abstracts or non-OA Europe PMC full text into training corpora from this pipeline.
- Prefer **public APIs** (as implemented under `living_evidence_graph/tools/`) — not HTML scraping — as the primary acquisition path.

## Disclaimers (always on)

- No PHI.
- No causation claims; no comparative efficacy claims.
- openFDA / FAERS = voluntary **reports ≠ rates**.
- Not a medical product; not clinical advice.
- Not endorsed by FDA, NLM, NIH, NCBI, Open Targets, or ChEMBL.
