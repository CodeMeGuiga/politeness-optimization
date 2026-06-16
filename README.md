# Teaching Politeness vs. Asking for It: A Comparison of DPO Fine-Tuning and System Prompting for Pragmatic Language Alignment

A comparative study of three approaches to making a language model respond more politely: a no-intervention baseline, system-prompt engineering, and Direct Preference Optimisation (DPO) fine-tuning.

**Model:** Llama 3.2 3B (Instruct)  
**Dataset:** 50 politeness scenarios across 5 emotionally sensitive scenario types  
**Evaluation:** Automated politeness classifier, semantic similarity to human-written references, and LLM-as-judge rubric scoring

---

## Research Question

Does DPO fine-tuning produce more consistently polite responses than system prompting, and does more elaborate system-prompt phrasing yield better results than a simple one?

---

## Scenario Types

The dataset covers five categories of high-stakes communication where tone matters:

| Type | Description |
|------|-------------|
| `critical_feedback` | Giving honest feedback on work the person is proud of |
| `declining_request` | Refusing inappropriate or unethical requests |
| `delivering_bad_news` | Communicating realistic but discouraging assessments |
| `correcting_error` | Correcting factual myths the user believes |
| `advising_under_stress` | Responding to someone in emotional distress |

Each scenario has a `chosen` (polite, empathetic) and `rejected` (blunt) reference response used for DPO training and embedding evaluation.

---

## Approaches

### Baseline
Llama 3.2 3B via Ollama with no system prompt. Responses are capped at 100 tokens, temperature 0.0.

### Prompted (A / B / C)
The same model receives a system prompt before each user message. Three phrasings test whether prompt wording affects consistency:

- **Phrasing A:** `"Respond politely."`
- **Phrasing B:** `"Respond in a polite and considerate manner, acknowledging the user's situation before giving advice."`
- **Phrasing C:** `"Be warm, thoughtful, and sensitive in your response. Acknowledge what the person is going through, soften any suggestions, and match your tone to the context."`

### DPO Fine-tuning
Llama 3.2 3B loaded in 4-bit (NF4) quantisation with LoRA adapters (r=4, targeting `q_proj`/`v_proj`) trained using TRL's `DPOTrainer` on 40 preference pairs. Training runs for 3 epochs with beta=0.1. The adapter is saved to `dpo_model/`.

---

## Results

### Politeness Classifier (Intel/polite-guard)

Percentage of responses classified as **polite** or **somewhat polite** out of 10 test prompts:

| Condition | % Polite/Somewhat Polite |
|-----------|--------------------------|
| Baseline | 40% |
| Prompted A | 30% |
| Prompted B | **80%** |
| Prompted C | **80%** |
| DPO | 50% |

### Semantic Similarity to Human References (all-MiniLM-L6-v2)

Average cosine similarity against `chosen` reference responses:

| Condition | Avg Cosine Similarity |
|-----------|-----------------------|
| Baseline | 0.592 |
| Prompted A | 0.602 |
| Prompted B | 0.593 |
| Prompted C | 0.541 |
| DPO | 0.540 |

### LLM-as-Judge (Claude Sonnet 4.6, May 2026)

Three rubric dimensions scored 1–5 (1 = very poor, 5 = excellent):

| Condition | ACK (Acknowledgement) | HEDGE (Hedging) | TONE |
|-----------|----------------------|-----------------|------|
| Baseline | 1.5 | 2.6 | 2.7 |
| Prompted A | 2.2 | 2.9 | 3.3 |
| Prompted B | **3.0** | **3.1** | 3.7 |
| Prompted C | 2.8 | 2.8 | 3.5 |
| DPO | 1.6 | 2.0 | 2.0 |

### Key Findings

- **Prompted B** performed most consistently across all three metrics — the more detailed prompt (acknowledge before advising) outperformed both the terse Phrasing A and the overly warm Phrasing C.
- **Prompted C** showed a notable safety failure on a declining-request prompt (wrote a thesis introduction it should have refused) and suppressed factual corrections entirely in favour of warmth.
- **DPO underperformed** relative to prompting on all metrics. The fine-tuned model hallucinated conversations, invented fictional facts, and failed safety checks on declining-request scenarios — likely due to the very small training set (40 pairs) and lightweight LoRA config (r=4) being insufficient for reliable alignment.
- **Embedding similarity** was broadly flat across conditions (~0.54–0.60), suggesting all approaches produce semantically similar content to references but differ mainly in tone and safety behaviour, which the classifier and judge capture better.

---

## Setup

### Dependencies

```
torch
transformers
peft
trl
datasets
bitsandbytes
ollama
sentence-transformers
```

### Prerequisites

- **Ollama** installed and running locally with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`)
- **CUDA GPU** required for DPO training (`dpo.py`)
- Base model `unsloth/Llama-3.2-3B-Instruct` downloaded from Hugging Face on first run

---

## Running the Pipeline

```bash
python json_export.py          # Generate dataset → politeness_dataset_correct.json
python baseline.py             # → baseline_responses.json
python prompted.py             # → prompted_responses.json
python dpo.py                  # → dpo_responses.json + dpo_model/  (GPU, ~5–10 min)
python evaluate_bert.py        # → bert_scores.json
python evaluate_embeddings.py  # → embedding_scores.json
```

Results are consolidated in `scores_comparison.csv`.

---

## File Overview

| File | Purpose |
|------|---------|
| `json_export.py` | Generates the 50-item dataset with chosen/rejected pairs |
| `baseline.py` | Runs test prompts through unmodified Llama via Ollama |
| `prompted.py` | Tests three system prompt phrasings (A, B, C) |
| `dpo.py` | Fine-tunes with DPO then runs inference |
| `evaluate_bert.py` | Politeness classification using Intel/polite-guard |
| `evaluate_embeddings.py` | Cosine similarity vs. reference responses |
| `politeness_dataset_correct.json` | Full 50-item dataset (5 types × 10 examples) |
| `training_f.json` | 40 preference pairs used for DPO training |
| `test_f.json` | 10-item test set (2 per scenario type) |
| `bert_scores.json` | Classifier output for all conditions |
| `embedding_scores.json` | Cosine similarity scores for all conditions |
| `judge_scores.json` | LLM judge rubric scores (ACK, HEDGE, TONE) with reasoning |
| `scores_comparison.csv` | Consolidated CSV of all BERT metrics |
| `dpo_model/` | Saved LoRA adapter and tokenizer |
