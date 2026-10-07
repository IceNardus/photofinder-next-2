# Phase 22 Identity Real-World Benchmark Report

## Final Verdict

- **Dataset**: CHECK_DATASET
- **Face**: EXPERIMENTAL
- **Body**: EXPERIMENTAL
- **Fusion**: EXPERIMENTAL

## Production Gate Criteria

| Metric | Face | Body | Fusion | Gate |
|--------|------|------|--------|------|
| Top-1 | 91.31% | 0.00% | - | >= 95% |
| Top-3 | 94.71% | 0.00% | - | >= 98% |
| Top-5 | 96.10% | 0.00% | - | >= 99% |
| AUC | 0.9391 | 0.0000 | - | >= 0.98 |
| EER | 12.06% | 50.00% | - | <= 2% |
| FAR@0.1% | 0.0000 | 0.0000 | - | <= 0.1% |
| TAR@FAR=0.1% | 0.0000 | 0.0000 | - | - |
| Median Margin | 0.2991 | - | - | - |
| P10 Margin | -0.0033 | - | - | - |
| LOO Pass Rate | 89.96% | 0.00% | - | >= 95% |
| False Merge Rate | 0.0667 | 0.0000 | 0.0000 | <= 0.1% |
| Chain Contamination | NONE | - | - | false |

## Face Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 0.9391 |
| EER | 0.1206 |
| FAR@0.1% | 0.0000 |
| FAR@0.5% | 0.0000 |
| FAR@1% | 0.0000 |
| FAR@5% | 0.0000 |
| FRR | 0.9986 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 91.31% |
| Top-3 | 94.71% |
| Top-5 | 96.10% |
| Top-10 | 98.24% |

### Margin Analysis

| Metric | Value |
|--------|-------|
| Positive Best | 0.2991 |
| Positive Mean | 0.2690 |
| Positive Min | -0.5627 |
| Negative Max | 0.7163 |
| Median | 0.2991 |
| P10 | -0.0033 |
| P25 | 0.1593 |
| P50 | 0.2991 |
| P75 | 0.4100 |
| P90 | 0.4888 |

### Prototype LOO

| Metric | Value |
|--------|-------|
| Pass Rate | 89.96% |

## Body Benchmark

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

### Prototype LOO

| Metric | Value |
|--------|-------|
| Pass Rate | 0.00% |

## False Merge Analysis

| Metric | Value |
|--------|-------|
| Face-only False Merges | 29 |
| Body-only False Merges | 0 |
| Fusion False Merges | 0 |
| Total Pairs | 435 |
| Face False Merge Rate | 0.066667 |
| Body False Merge Rate | 0.000000 |
| Fusion False Merge Rate | 0.000000 |

## Chain Contamination Testing

| Steps | Detected | Margin A-B | Margin B-C | Margin C-D | Margin A-D |
|-------|----------|------------|------------|------------|------------|
| 1 | NO | 0.2152 | -0.0218 | 0.0235 | 0.1404 |
| 2 | NO | 0.2152 | -0.0218 | 0.0235 | 0.1404 |
| 5 | NO | 0.2152 | -0.0218 | 0.0235 | 0.1404 |
| 10 | NO | 0.2152 | -0.0218 | 0.0235 | 0.1404 |
| 20 | NO | 0.2152 | -0.0218 | 0.0235 | 0.1404 |

## Anti-Chaining Verification

| Metric | Value |
|--------|-------|
| Ambiguous Pairs | 854 |
| Total Pairs | 966 |
| Ambiguous Rate | 0.8841 |

