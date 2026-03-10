# CarePlan Notes Summary

**LLM-Based Symptom-Severity Extraction from AML Clinical Notes via Targeted Summarization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the code for extracting symptom severity from clinical notes of patients with acute myeloid leukemia (AML) using a **summarization-first** approach. Rather than applying large language models (LLMs) directly to lengthy inpatient notes, we first generate **symptom-targeted summaries** that isolate clinically actionable content—symptom presence, negation, severity, interventions, and functional impact—then prompt GPT-4.1 to produce standardized severity ratings aligned with the **Nursing Outcomes Classification (NOC)** 1–5 Likert scale.

The pipeline supports five NOC outcomes:
- **Pain Level**
- **Anxiety Level**
- **Nausea & Vomiting Severity**
- **Tissue Integrity: Skin & Mucous Membranes**
- **Infection Severity**



## Symptom-Targeted Prompt for Summarization

Unlike generic summarization, our prompts use **structured extraction** with symptom-specific sections. For example, the Pain summarization prompt extracts:

| Section | Content |
|---|---|
| **PAIN REPORT** | Direct patient reports, presence or negation |
| **PAIN CHARACTERISTICS** | Location, quality, onset, aggravating/alleviating factors |
| **SEVERITY/INTENSITY** | Pain scale scores, NOC-aligned 1–5 rating |
| **PHARMACOLOGICAL TREATMENT** | Medications, doses, routes, PRN vs scheduled |
| **FUNCTIONAL IMPACT** | Effect on mobility, sleep, appetite, ADLs |
| **BEHAVIORAL INDICATORS** | Observable signs (grimacing, guarding, restlessness) |
| **PATIENT RESPONSE** | Response to interventions, re-assessment scores |

Each outcome (Pain, Anxiety, Infection, Nausea, Tissue) has its own tailored prompt with clinically relevant sections and indicators. If no symptom data is found, the model returns a one-line general clinical summary as fallback.



## Evaluation Metrics

### Downstream NOC Rating

| Metric | Description |
|---|---|
| Within ±1 Accuracy | Proportion of LLM ratings within 1 point of nurse-documented NOC score |
| Cohen's Kappa | Inter-rater agreement between LLM and nurse ratings |
| Confusion Matrix | Per-class agreement patterns |
| Per-class F1 | Precision, recall, F1 for each rating level (1–5) |

### Summary Quality

| Metric | Description |
|---|---|
| BERTScore F1 | Semantic similarity between source notes and summaries |
| ROUGE-1 Precision | Unigram overlap (summary words grounded in source) |
| Compression Ratio | Summary length / source length |
| Human Evaluation | Expert ratings on completeness, correctness, temporal alignment (1–5 scale) |

## Key Results

| NOC Outcome | Within ±1 Rating |
|---|---|
| Tissue Integrity: Skin & Mucous Membranes | 92.3% |
| Anxiety Level | 86.7% |
| Pain Level | 84.8% |
| Nausea & Vomiting Severity | 78.6% |
| Infection Severity | 64.8% |

- **Summarization**: BERTScore F1 = 0.184, compression ratio = 0.055 (1,771 → 74 words, >95% reduction)
- **Human evaluation**: Expert ratings 4.0–5.0 across completeness, correctness, and temporal alignment with 80–100% exact agreement

## Data

This study analyzed **2,147 admissions from 790 AML patients** (2006–2021). Clinical notes and NOC scores were extracted from the institutional EHR. Due to patient privacy, raw clinical data cannot be shared. All code, prompts, and evaluation scripts are provided for reproducibility on similar datasets.




## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, please open an issue or contact the corresponding author.
