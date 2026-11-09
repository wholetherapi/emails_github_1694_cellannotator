# CellAnnotator

Official code for the paper **CellAnnotator: Weakly Supervised Single-Cell Type Annotation via Prototype Alignment** (RECOMB 2025).

CellAnnotator builds cell-type prototypes from marker-gene priors and aligns
unlabeled single-cell embeddings to those prototypes with a contrastive
objective. It supports cross-tissue and cross-platform annotation without
dataset-specific fine-tuning.

## Layout

- `src/prototypes.py` — marker-prior prototype construction
- `src/aligner.py` — contrastive prototype-alignment model
- `src/evaluate.py` — annotation metrics and confidence calibration
- `train.py` — training entry point
- `configs/base.yaml` — reference hyper-parameters

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
```
