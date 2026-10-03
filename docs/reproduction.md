# Reproduction Guide

## 1. Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

## 2. Dataset

Download VFND from the source documented in the academic report.

Prepare either:

```text
data/processed/fnd_dataset.csv
```

with:

```text
text,label
```

or use `scripts/build_dataset.py` on the relevant JSON folders.

## 3. Source-aligned run

```bash
python train.py --data data/processed/fnd_dataset.csv --epochs 10 --batch-size 4 --learning-rate 2e-5 --entity-mode first_tokens
```

## 4. Outputs

A run writes:

```text
results/runs/default/
├── best.pt
├── history.csv
└── final_metrics.json
```

## 5. Why exact metrics may differ

The modular code intentionally fixes the original notebook's graph mini-batching index problem. Therefore an exact numerical match to the notebook output should not be assumed.

If exact historical evidence is needed, use:

```text
notebooks/original/17_SourceCode.ipynb
results/report/
```
