# MythBench

**MythBench: Measuring Parametric Misconceptions Across Small Language Models**

Samuel Stephen · R. Vignesh  
Karunya Institute of Technology and Sciences, Coimbatore, India

---

## Overview

MythBench is a multiple-choice benchmark for evaluating **parametric misconceptions** in small language models (sub-7B). Parametric misconceptions are culturally entrenched false beliefs encoded in model weights that reproduce consistently across independently trained models.

The benchmark contains:
- **48 misconception items** — empirically validated across scientific, historical, and geographic categories
- **48 matched factual controls** — domain-matched questions without entrenched wrong distractors

All items use a four-option MCQ format with shuffled option order.

---

## Key Findings

| Metric | Value |
|--------|-------|
| CMO (misconception items) | 0.617 |
| CMO (control items) | 0.333 |
| CMO gap | **+0.284** |
| SEA (misconception items) | 0.853 |
| PMD = 1.0 items | 29 / 48 (60.4%) |

Models evaluated: Gemma-2B, Qwen-1.8B, SmolLM-1.7B, TinyLlama-1.1B  
Clean models (no position bias): Gemma-2B, Qwen-1.8B

---

## Repository Contents

```
MythBench/
├── MythBench_v10.json              # Full benchmark (48 misconceptions + 48 controls)
├── mythbench_inference_FINAL.py    # Inference script (Kaggle T4 x2)
├── results/
│   ├── mythbench_results_Gemma2B_v10.csv
│   ├── mythbench_results_Qwen18B_v10.csv
│   ├── mythbench_results_SmolLM_v10.csv
│   └── mythbench_results_TinyLlama_v10.csv
└── README.md
```

---

## Benchmark Format

Each misconception item:

```json
{
  "id": "M-006",
  "misconception_label": "great_wall_visible_from_space",
  "category": "geographic_attribution",
  "question": "Is the Great Wall of China visible from space with the naked eye?",
  "options": {
    "A": "Only visible from low Earth orbit under perfect conditions",
    "B": "Yes, it is clearly visible from orbit",
    "C": "No, it is too narrow (~9m wide) to see without optical aids",
    "D": "Visible only from the Moon on clear days"
  },
  "correct": "C",
  "misconception": "B",
  "source": "Confirmed by Chinese astronaut Yang Liwei and multiple NASA astronauts."
}
```

Each control item is matched to a misconception item on subject domain but contains no culturally entrenched distractor.

---

## Running Inference

```python
# Set MODEL_NAME in Cell 2 before running
# Options: "Gemma2B", "Qwen18B", "SmolLM", "TinyLlama"
MODEL_NAME = "Gemma2B"
```

Run `mythbench_inference_FINAL.py` on Kaggle (T4 x2 GPU).  
Requires: `HF_TOKEN` in Kaggle Secrets.

Output: `mythbench_results_{MODEL_NAME}_v10.csv`

---

## Metrics

**Cross-Model Overlap (CMO)**  
Fraction of questions (where at least one model is wrong) on which two models make the identical wrong answer.

**Shared Error Agreement (SEA)**  
Fraction of cases where both models are wrong and select the same answer.

**Parametric Misconception Density (PMD)**  
Fraction of models that select the misconception option for a given item.  
PMD = 1.0 indicates all clean models independently selected the same false belief.

---

## Category Distribution

| Category | Items | PMD = 1.0 | Avg PMD |
|----------|-------|-----------|---------|
| Scientific | 23 | 12 | 0.761 |
| Historical Narrative | 19 | 16 | 0.842 |
| Geographic Attribution | 6 | 4 | 0.750 |
| **Total** | **48** | **29** | **0.786** |

---

## Citation

```bibtex
@article{stephen2025mythbench,
  title={MythBench: Measuring Parametric Misconceptions Across Small Language Models},
  author={Stephen, Samuel and Vignesh, R.},
  journal={KI - K{\"u}nstliche Intelligenz},
  year={2025},
  note={Under review}
}
```

---

## License

MIT License. Dataset released for research use.

---

## Contact

Samuel Stephen — samuels24@karunya.edu.in  
ORCID: [0009-0002-9446-000X](https://orcid.org/0009-0002-9446-000X)
