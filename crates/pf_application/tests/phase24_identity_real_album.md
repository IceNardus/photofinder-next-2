# Phase 24 Real Album Identity Benchmark Report

## Final Verdict

- **Dataset**: READY
- **Face**: EXPERIMENTAL
- **Body**: EXPERIMENTAL
- **Fusion**: EXPERIMENTAL

## Data Quality Summary

| Category | Count |
|----------|-------|
| Face-Usable | 966 |
| Body-Usable | 967 |
| Dual-Channel | 966 |
| Face-Only | 0 |
| Body-Only | 1 |
| Neither | 0 |

## Production Gate Criteria

| Metric | Face | Body | Fusion | Gate |
|--------|------|------|--------|------|
| Top-1 | 100.00% | 100.00% | - | >= 95% |
| Top-3 | 100.00% | 100.00% | - | >= 98% |
| Top-5 | 100.00% | 100.00% | - | >= 99% |
| AUC | 0.9374 | 0.7608 | - | >= 0.98 |
| EER | 12.23% | 30.38% | - | <= 2% |
| FAR@0.1% | 0.0001 | 0.0003 | - | <= 0.1% |
| LOO Pass Rate | 1.86% | 28.75% | - | >= 95% |
| False Merge Rate | 0.000000 | 0.000000 | 0.000000 | <= 0.1% |

## Face Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 0.9374 |
| EER | 0.1223 |
| FAR@0.1% | 0.0001 |
| FAR@0.5% | 0.0016 |
| FAR@1% | 0.0001 |
| FRR | 0.9986 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 100.00% |
| Top-3 | 100.00% |
| Top-5 | 100.00% |
| Top-10 | 100.00% |

### Margin Analysis

| Metric | Value |
|--------|-------|
| Positive Best | 0.6623 |
| Positive Mean | 0.2552 |
| Positive Min | -0.4103 |
| Negative Max | -340282346638528859811704183484516925440.0000 |
| Median | 0.2838 |
| P10 | -0.0223 |
| P25 | 0.1459 |
| P50 | 0.2838 |
| P75 | 0.4052 |
| P90 | 0.4810 |

### LOO Prototype

| Metric | Value |
|--------|-------|
| Pass Rate | 1.86% |

## Body Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 0.7608 |
| EER | 0.3038 |
| FAR@0.1% | 0.0003 |
| FAR@0.5% | 0.0845 |
| FAR@1% | 0.0000 |
| FRR | 0.9903 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 100.00% |
| Top-3 | 100.00% |
| Top-5 | 100.00% |
| Top-10 | 100.00% |

### LOO Prototype

| Metric | Value |
|--------|-------|
| Pass Rate | 28.75% |

## False Merge Analysis

| Metric | Value |
|--------|-------|
| Face-only False Merges | 0 |
| Body-only False Merges | 0 |
| Fusion False Merges | 0 |
| Total Pairs | 482 |
| Face False Merge Rate | 0.000000 |
| Body False Merge Rate | 0.000000 |
| Fusion False Merge Rate | 0.000000 |

## Chain Contamination Testing

| Steps | Detected | Margin A-B | Margin B-C | Margin C-D | Margin A-D |
|-------|----------|------------|------------|------------|------------|
| 1 | YES | 0.2839 | 0.4186 | 0.4430 | 0.4445 |
| 2 | YES | 0.2839 | 0.4186 | 0.4430 | 0.4445 |
| 5 | YES | 0.2839 | 0.4186 | 0.4430 | 0.4445 |
| 10 | YES | 0.2839 | 0.4186 | 0.4430 | 0.4445 |
| 20 | YES | 0.2839 | 0.4186 | 0.4430 | 0.4445 |

## Anti-Chaining Verification

| Metric | Value |
|--------|-------|
| Ambiguous Pairs | 50 |
| Total Pairs | 422059 |
| Ambiguous Rate | 0.0001 |
