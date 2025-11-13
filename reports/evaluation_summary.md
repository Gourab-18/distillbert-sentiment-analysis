# Model Evaluation Summary Report

**Date:** 2025-11-13T06:52:50.703670

**Dataset:** IMDb Movie Reviews (Test Set)

**Test Samples:** 25,000

---

## Performance Comparison

| Metric | Baseline | DistilBERT | Improvement |
|--------|----------|------------|-------------|
| **Accuracy** | 0.8778 | 0.9265 | +0.0487 |
| **Precision** | 0.8778 | 0.9266 | +0.0487 |
| **Recall** | 0.8778 | 0.9265 | +0.0487 |
| **F1 Score** | 0.8778 | 0.9265 | +0.0487 |
| **ROC AUC** | 0.9499 | 0.9791 | +0.0292 |

---

## Error Analysis

- **Baseline Errors:** 3054 (12.22%)
- **DistilBERT Errors:** 1837 (7.35%)
- **Error Reduction:** 1217 samples

- Errors fixed by DistilBERT: 2057
- New errors in DistilBERT: 840
- Errors in both models: 997

---

## Visualizations

- `confusion_matrix_comparison.png` - Side-by-side confusion matrices
- `performance_comparison.png` - Overall performance metrics comparison
- `per_class_metrics.png` - Per-class precision, recall, and F1 scores

---

## Conclusion

✅ **Test accuracy target (≥92%) ACHIEVED: 92.65%**

DistilBERT shows clear performance improvement over the baseline:
- Accuracy improved by 4.87 percentage points
- Reduced errors by 1217 samples
- Achieved 0.9791 ROC AUC score
