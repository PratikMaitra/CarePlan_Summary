# CarePlan_Summary

**LLM-Based Symptom-Severity Extraction from AML Clinical Notes via Targeted Summarization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the code and evaluation pipeline for extracting symptom severity from clinical notes of patients with acute myeloid leukemia (AML) using a **summarization-first** approach. Rather than applying large language models (LLMs) directly to lengthy inpatient notes, we first generate **symptom-targeted summaries** that isolate clinically actionable content—symptom presence, negation, severity, interventions, and functional impact—then prompt GPT-4.1 to produce standardized severity ratings aligned with the **Nursing Outcomes Classification (NOC)** 1–5 Likert scale.

The pipeline supports five NOC outcomes:
- **Pain Level**
- **Anxiety Level**
- **Nausea & Vomiting Severity**
- **Tissue Integrity: Skin & Mucous Membranes**
- **Infection Severity**

## Pipeline Architecture

```
┌─────────────────────┐
│  Raw Clinical Notes  │
│  (EHR / CSV)         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  1. Date-Matched     │  Match notes to NOC observation dates
│     Note Merging     │  (EditDate from CarePlan_goals.csv)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Symptom-Targeted │  GPT-4.1 extracts ONLY symptom-relevant
│     Summarization    │  information using structured prompts
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. NOC Prompt       │  Embed summary into NOC rating prompt
│     Generation       │  with scale definitions & indicators
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. LLM Rating       │  GPT-4.1 returns <rating>X</rating>
│     (GPT-4.1)        │  for each encounter
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Evaluation        │  Compare LLM ratings to nurse-documented
│     & Visualization   │  NOC scores
└─────────────────────┘
```

## Repository Structure

```
CarePlan_Summary/
│
├── README.md
│
├── Step_0_Preprocessing/
│   ├── filter_notes_by_author.py          # Remove non-clinical author types
│   ├── merge_filtered_notes_to_careplan_datematched.py  # Date-matched note merging
│   ├── analyze_mrns.py                    # MRN overlap analysis
│   └── verify_merged_files.py             # Verify merge outputs
│
├── Step_1_Summarization/
│   ├── summarize_notes_pain.py            # Pain-specific summarization (GPT-4.1)
│   ├── summarize_notes_anxiety.py         # Anxiety-specific summarization
│   ├── summarize_notes_infection.py       # Infection-specific summarization
│   ├── summarize_notes_nausea.py          # Nausea-specific summarization
│   ├── summarize_notes_tissue.py          # Tissue integrity-specific summarization
│   └── summarize_notes_azure.py           # Generic summarization (Llama 3.1 8B)
│
├── Step_2_Prompt_Generation/
│   └── generate_noc_prompts.py            # Generate NOC rating prompts (all 5 outcomes)
│
├── Step_3_LLM_Rating/
│   └── call_azure_noc.py                  # Send prompts to GPT-4.1, collect ratings
│
├── Step_4_Evaluation/
│   ├── evaluate_noc_performance.py        # Accuracy, F1, Kappa, confusion matrix
│   ├── compute_summary_metrics.py         # BERTScore, ROUGE-1, compression ratio
│   ├── describe_summarized.py             # Descriptive statistics for summarized files
│   ├── plot_gold_vs_predicted.py          # Publication-quality visualizations
│   └── sample_20_notes.py                 # Random sampling for manual review
│
└── Utilities/
    ├── analyze_notes_file.py              # Descriptive stats for notes files
    ├── collect_merged_notes.py            # Collect merged files across folders
    └── consolidate_careplan_notes.py      # Consolidate notes per encounter
```

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy requests tqdm scikit-learn openpyxl

# For summary evaluation metrics
pip install rouge-score bert-score

# For visualization
pip install matplotlib
```

### Azure OpenAI Configuration

The pipeline uses **Azure OpenAI GPT-4.1** for both summarization and NOC rating. Update the following variables in each script:

```python
AZURE_API_KEY = "your-api-key"
AZURE_ENDPOINT = "https://your-endpoint.cognitiveservices.azure.com"
DEPLOYMENT_NAME = "gpt-4.1"
API_VERSION = "2024-05-01-preview"
```

### Running the Pipeline

The pipeline is designed to run from within each CarePlan subfolder (e.g., `CarePlan_Painlevel/`). Scripts auto-detect the NOC outcome from the folder name.

```bash
# Step 0: Preprocess and merge notes (run from CarePlanGoals root)
python filter_notes_by_author.py
python merge_filtered_notes_to_careplan_datematched.py

# Step 1-4: Run from each CarePlan subfolder
cd CarePlan_Painlevel/

# Step 1: Summarize clinical notes
python summarize_notes_pain.py

# Step 2: Generate NOC rating prompts
python generate_noc_prompts.py

# Step 3: Get LLM ratings
python call_azure_noc.py

# Step 4: Evaluate
python evaluate_noc_performance.py
python compute_summary_metrics.py
python plot_gold_vs_predicted.py
python describe_summarized.py
```

## Symptom-Targeted Summarization

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

## Date-Matched Note Merging

A key preprocessing step matches clinical notes to **specific NOC observation dates** rather than pulling all notes from an entire encounter:

- **Previous approach**: Merged all notes from any encounter with a NOC score → excessively long Notes column for lengthy hospitalizations
- **Current approach**: Uses `EditDate` from `CarePlan_goals.csv` to merge only notes written on the date the NOC score was documented → concise, clinically relevant notes

Configurable `DATE_WINDOW_DAYS` parameter (default: 0) allows ±N day flexibility.

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

## Citation

If you use this code or pipeline in your research, please cite:

```
@inproceedings{maitra2025careplan,
  title={LLM-Based Symptom-Severity Extraction from AML Clinical Notes via Targeted Summarization},
  author={Maitra, Pratik and [co-authors]},
  booktitle={AMIA Annual Symposium Proceedings},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, please open an issue or contact the corresponding author.
