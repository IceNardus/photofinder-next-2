# Phase 23 Identity Full-Body Benchmark Report

## Final Verdict

- **Dataset**: DATASET_READY
- **Face**: EXPERIMENTAL
- **Body**: PRODUCTION_READY
- **Fusion**: EXPERIMENTAL

## Production Gate Criteria

| Metric | Face | Body | Fusion | Gate |
|--------|------|------|--------|------|
| Top-1 | 75.00% | 100.00% | - | >= 95% |
| Top-3 | 75.00% | 100.00% | - | >= 98% |
| Top-5 | 75.00% | 100.00% | - | >= 99% |
| AUC | 0.4020 | 1.0000 | - | >= 0.98 |
| EER | 48.63% | 0.01% | - | <= 2% |
| FAR@0.1% | 0.0061 | 0.0000 | - | <= 0.1% |
| TAR@FAR=0.1% | 0.0000 | 0.0000 | - | - |
| Median Margin | -0.0408 | - | - | - |
| P10 Margin | -0.1185 | - | - | - |
| LOO Pass Rate | 16.00% | 100.00% | - | >= 95% |
| False Merge Rate | 0.5287 | 0.0000 | 0.0138 | <= 0.1% |
| Chain Contamination | NONE | - | - | false |

## Face Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 0.4020 |
| EER | 0.4863 |
| FAR@0.1% | 0.0061 |
| FAR@0.5% | 0.0790 |
| FAR@1% | 0.2462 |
| FAR@5% | 0.5714 |
| FRR | 0.4490 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 75.00% |
| Top-3 | 75.00% |
| Top-5 | 75.00% |
| Top-10 | 100.00% |

### Margin Analysis

| Metric | Value |
|--------|-------|
| Positive Best | -0.0408 |
| Positive Mean | -0.0253 |
| Positive Min | -0.1772 |
| Negative Max | 0.3081 |
| Median | -0.0408 |
| P10 | -0.1185 |
| P25 | -0.0567 |
| P50 | -0.0408 |
| P75 | -0.0121 |
| P90 | 0.0129 |

### Prototype LOO

| Metric | Value |
|--------|-------|
| Pass Rate | 16.00% |

## Body Benchmark

### Verification

| Metric | Value |
|--------|-------|
| AUC | 1.0000 |
| EER | 0.0001 |
| FAR@0.1% | 0.0000 |
| FAR@0.5% | 0.0000 |
| FAR@1% | 0.0000 |
| FAR@5% | 0.0000 |
| FRR | 0.0163 |

### Identification

| Metric | Value |
|--------|-------|
| Top-1 | 100.00% |
| Top-3 | 100.00% |
| Top-5 | 100.00% |
| Top-10 | 100.00% |

### Prototype LOO

| Metric | Value |
|--------|-------|
| Pass Rate | 100.00% |

## False Merge Analysis

| Metric | Value |
|--------|-------|
| Face-only False Merges | 230 |
| Body-only False Merges | 0 |
| Fusion False Merges | 6 |
| Total Pairs | 435 |
| Face False Merge Rate | 0.528736 |
| Body False Merge Rate | 0.000000 |
| Fusion False Merge Rate | 0.013793 |

## Chain Contamination Testing

| Steps | Detected | Margin A-B | Margin B-C | Margin C-D | Margin A-D |
|-------|----------|------------|------------|------------|------------|
| 1 | NO | 0.8406 | 0.9693 | 0.9847 | 0.8906 |
| 2 | NO | 0.8406 | 0.9693 | 0.9847 | 0.8906 |
| 5 | NO | 0.8406 | 0.9693 | 0.9847 | 0.8906 |
| 10 | NO | 0.8406 | 0.9693 | 0.9847 | 0.8906 |
| 20 | NO | 0.8406 | 0.9693 | 0.9847 | 0.8906 |

## Anti-Chaining Verification

| Metric | Value |
|--------|-------|
| Ambiguous Pairs | 22 |
| Total Pairs | 25 |
| Ambiguous Rate | 0.8800 |

