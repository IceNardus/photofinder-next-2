# Phase 21 Identity Benchmark Report

## Final Verdict

- **Dataset**: DATASET_READY
- **Face**: FAIL
- **Body**: READY
- **Fusion**: EXPERIMENTAL

## Recommended Settings

- Alpha: 0.00
- Robust Alpha Range: [0.00, 1.00]
- Threshold: 0.65
- FAR: 0.0000
- FRR: 1.0000
- EER: 0.5000
- False Merge Rate: 0.0000
- False Split Rate: 0.0000
- LOO Pass Rate: 0.00%

## Face Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 0.0000 |
| EER | 0.5000 |
| FAR@0.1% | 0.0000 |
| FAR@0.5% | 0.0000 |
| FAR@1% | 0.0000 |
| FAR@5% | 0.0000 |
| FRR | 1.0000 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 0.00% |
| Top-3 | 0.00% |
| Top-5 | 0.00% |
| Top-10 | 0.00% |

### Margin Analysis

| Metric | Value |
|--------|-------|
| Positive Best | 0.0000 |
| Positive Mean | -0.0000 |
| Positive Min | 0.0000 |
| Negative Max | 0.0000 |
| Median | 0.0000 |
| P10 | 0.0000 |
| P25 | 0.0000 |

### Prototype LOO

| Type | Pass Rate |
|------|-----------|
| Single | 0.00% |
| Mean | 0.00% |
| Median | 0.00% |
| Geomean | 0.00% |

## Body Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 1.0000 |
| EER | 0.0000 |
| FAR@0.1% | 0.0000 |
| FAR@0.5% | 0.0000 |
| FAR@1% | 0.0000 |
| FAR@5% | 0.0000 |
| FRR | 0.0000 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 100.00% |
| Top-3 | 100.00% |
| Top-5 | 100.00% |
| Top-10 | 100.00% |

### Prototype LOO

| Type | Pass Rate |
|------|-----------|
| Single | 100.00% |
| Mean | 100.00% |
| Median | 100.00% |
| Geomean | 100.00% |

## Fusion Alpha Sweep

| Alpha | EER | FAR | FRR | AUC | Margin Median |
|-------|-----|-----|-----|-----|---------------|
| 0.00 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.05 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.10 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.15 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.20 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.25 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.30 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.35 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.40 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.45 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.55 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.60 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.65 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.70 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.75 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.80 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.85 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.90 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 0.95 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 1.00 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## False Merge Analysis

| Metric | Value |
|--------|-------|
| Face False Merges | 0 |
| Body False Merges | 0 |
| Fusion False Merges | 0 |
| False Split Rate | 0.0000 |
| Cluster Purity | 1.0000 |
| Cluster Recall | 1.0000 |
| Pairwise Precision | 1.0000 |
| Pairwise Recall | 1.0000 |
| Pairwise F1 | 1.0000 |

## Missing Channel Analysis

| Scenario | Correct |
|----------|---------|
| Face+Body | 0 |
| Face Only | 0 |
| Body Only | 299 |
| Face Missing | 0 |
| Body Missing | 0 |
| Both Missing | 0 |
| Total | 300 |

## Hard Negatives

| Type | Count |
|------|-------|
| Face Hard Negative | 0 |
| Body Hard Negative | 0 |
| Dual Hard Negative | 0 |

