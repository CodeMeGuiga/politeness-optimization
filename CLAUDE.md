# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP research project comparing three approaches to **politeness optimization** for Llama 3.2 3B responses:
- **Baseline**: Unmodified model responses via Ollama
- **Prompted**: Three system-prompt phrasings (A, B, C) to test robustness
- **DPO**: Fine-tuned with Direct Preference Optimization using LoRA

Evaluated across 5 scenario types (critical feedback, declining requests, delivering bad news, correcting errors, advising under stress).

## Running the Pipeline

### Prerequisites
- Ollama running locally with `llama3.2:3b` model (for baseline and prompted)
- CUDA GPU with sufficient VRAM (for DPO training)
- Python packages: `torch`, `transformers`, `peft`, `trl`, `datasets`, `ollama`, `sentence-transformers`, `bitsandbytes`

### Execution Order

```bash
python json_export.py          # Generate dataset (politeness_dataset_correct.json)
python baseline.py             # → baseline_responses.json
python prompted.py             # → prompted_responses.json
python dpo.py                  # → dpo_responses.json + dpo_model/ (GPU required, ~5-10 min)
python evaluate_bert.py        # → bert_scores.json (uses Intel/polite-guard)
python evaluate_embeddings.py  # → embedding_scores.json (uses all-MiniLM-L6-v2)
```

## Architecture

```
politeness_dataset_correct.json (50 Q&A pairs: prompt, chosen, rejected)
    ├── training_f.json  (40 pairs for DPO training)
    └── test_f.json      (10 pairs for evaluation, 2 per scenario type)

Three response generation approaches (all use test_f.json):
    baseline.py   → Ollama API, no system prompt
    prompted.py   → Ollama API, 3 system prompt variants
    dpo.py        → HF Transformers + PEFT LoRA fine-tune, then inference

Two evaluation metrics:
    evaluate_bert.py        → politeness classification (polite/somewhat_polite/neutral/impolite)
    evaluate_embeddings.py  → cosine similarity vs. "chosen" reference responses
```

## Key Design Decisions

- **4-bit quantization** (`bitsandbytes`) in `dpo.py` to reduce VRAM during training
- **LoRA config**: r=4, lora_alpha=8, targeting `q_proj` and `v_proj` only
- **DPO hyperparameters**: beta=0.1, 3 epochs, batch_size=1, grad_accum=8
- **Deterministic generation**: temperature=0.0 for Ollama scripts; `do_sample=False` for DPO inference
- Trained LoRA adapter saved to `dpo_model/`; base model loaded fresh for inference

## Data Format

`politeness_dataset_correct.json` entries:
```json
{"id": 1, "type": "critical_feedback", "prompt": "...", "chosen": "...", "rejected": "..."}
```
`chosen` = polite reference response; `rejected` = less polite response used in DPO preference training.
