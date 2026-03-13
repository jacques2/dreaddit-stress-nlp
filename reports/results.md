# Results Summary

This document summarizes the final results of the project, from exploratory analysis to model comparison and explainability.

## Exploratory Analysis

Main observations from EDA:

- The dataset is nearly balanced between stress and non-stress posts.
- Text length alone has limited discriminative power, with substantial overlap between classes.
- Stress posts show more long outliers, often corresponding to personal narratives.
- Affective variables such as pleasantness and sentiment are more informative than simple surface statistics.

These findings motivate the comparison between:

- an interpretable feature-based baseline
- a contextual Transformer model

### Label Distribution

![Label Distribution](../results/eda/label_distribution.png)

### Text Length by Label

![Text Length By Label](../results/eda/text_length_by_label.png)

### Pleasantness by Label

![Pleasantness By Label Distribution](../results/eda/pleasantness_by_label.png)

## Baseline Model Performance

The baseline is a Logistic Regression classifier trained on a compact set of affective, psycholinguistic, and readability features.

### Hold-out Test Performance

- Accuracy: **0.748**
- Macro F1-score: **0.748**
- Majority baseline accuracy: **0.516**

The baseline clearly outperforms the naive majority classifier and shows balanced behavior across the two classes.

### Classification Report

```text
              precision    recall  f1-score   support

           0      0.732     0.754     0.743        69
           1      0.764     0.743     0.753        74

    accuracy                          0.748       143
   macro avg      0.748     0.748     0.748       143
weighted avg      0.749     0.748     0.748       143
```

The confusion pattern is fairly symmetric, which suggests that the baseline is not strongly biased toward one class.

### Cross-Validation Performance

- 5-fold stratified CV macro F1 mean: **0.755**
- Standard deviation: **0.025**

This confirms that the baseline is stable and not overly dependent on a single favorable split.

### Ranking-Based Evaluation

- ROC-AUC: **0.814**
- Average Precision (PR-AUC): **0.769**

The baseline is therefore competitive not only as a classifier, but also as a probabilistic scorer.

### Threshold Optimization

- Best threshold: **0.54**
- Best macro F1 after threshold tuning: **0.762**

This modest improvement over the default threshold suggests that Logistic Regression produces reasonably calibrated probabilities.

### Baseline Diagnostic Plots

![Baseline Confusion Matrix](../results/baseline/confusion_matrix.png)

![Baseline ROC Curve](../results/baseline/roc_curve.png)

![Baseline Precision-Recall Curve](../results/baseline/pr_curve.png)

## Transformer Model Performance (DistilBERT)

The Transformer model is a fine-tuned DistilBERT classifier evaluated on the same held-out test split (`n_test = 143`).

### Hold-out Test Performance

- Accuracy: **0.748**
- Macro F1-score: **0.745**
- ROC-AUC: **0.819**
- Average Precision (PR-AUC / AP): **0.835**

### Classification Report

```text
              precision    recall  f1-score   support

           0      0.780     0.667     0.719        69
           1      0.726     0.824     0.772        74

    accuracy                          0.748       143
   macro avg      0.753     0.745     0.745       143
weighted avg      0.752     0.748     0.746       143
```

Compared with the baseline, the Transformer is more sensitive to the stress class, but this comes with lower recall on non-stress and more false positives overall.

Approximate confusion profile:

- TN = 46
- FP = 23
- FN = 13
- TP = 61

### Transformer Diagnostic Plots

![Transformer Confusion Matrix](../results/transformer/confusion_matrix.png)

![Transformer ROC Curve](../results/transformer/roc_curve.png)

![Transformer PR Curve](../results/transformer/pr_curve.png)

## Baseline vs Transformer

| Metric | Baseline (LogReg) | Transformer (DistilBERT) |
|---|---:|---:|
| Accuracy | 0.748 | 0.748 |
| Macro F1 | 0.748 | 0.745 |
| ROC-AUC | 0.814 | 0.819 |
| AP (PR-AUC) | 0.769 | 0.835 |

### Interpretation

- Threshold-based classification performance does not improve with the Transformer.
- Accuracy is identical for the two models.
- Macro F1 is slightly lower for DistilBERT than for the baseline.
- The tuned baseline macro F1 (`0.762`) remains clearly above the final Transformer macro F1 (`0.745`).
- The Transformer performs better on ranking-based metrics, especially AP, which indicates stronger probabilistic separation.

This means the fair conclusion is not that DistilBERT is universally better. The actual result is more nuanced:

- Logistic Regression remains a very strong, simple, and interpretable classifier.
- DistilBERT adds value mainly as a scorer, thanks to better contextual ranking quality.

Given the small test set, these differences should be interpreted cautiously.

## Explainability

The project includes explainability for both the baseline and the Transformer.

### Baseline Explainability

For Logistic Regression, interpretation is direct through coefficient signs and magnitudes.

Main positive contributors to stress:

- `lex_liwc_i`
- `lex_liwc_WC`
- `lex_liwc_negemo`
- `lex_liwc_anx`
- `lex_dal_avg_imagery`

Main negative contributors to stress:

- `lex_liwc_Tone`
- `lex_dal_avg_pleasantness`
- `sentiment`
- `lex_liwc_social`
- readability features

These patterns are consistent with the EDA: stress-related posts are often more self-focused, more negative, and more narratively detailed.

![Baseline Coefficients](../results/explainability/baseline_coefficients.png)

### Transformer Explainability with SHAP

SHAP provides local token-level explanations for Transformer predictions.

In the analyzed stress example, the model prediction is pushed upward by tokens such as:

- `panic`
- `attacks`
- `experience`
- `my`
- `help`

These are psychologically plausible cues and support the idea that the model is relying on meaningful contextual evidence rather than arbitrary artifacts.

![SHAP Stress Explanation](../results/explainability/SHAP_values_stress.png)

![SHAP Non-Stress Explanation](../results/explainability/SHAP_values_non-stress.png)

Global SHAP analysis also highlights stress-related tokens such as `panic`, `attacks`, and `anxiety` as consistently important across examples.

![Global SHAP Token Importance](../results/explainability/global_token.png)

The waterfall explanation confirms that the final stress score is driven by the cumulative effect of multiple meaningful tokens, not by a single isolated keyword.

![SHAP Waterfall Example](../results/explainability/waterfall_example1.png)

### Transformer Explainability with Integrated Gradients

Integrated Gradients provides a gradient-based perspective on token importance.

The main high-attribution tokens are consistent with SHAP:

- `attacks`
- `anxiety`
- `severe`
- `panic`
- self-referential tokens such as `I` and `my`

This agreement between SHAP and Integrated Gradients is useful because the two methods rely on different principles:

- SHAP is perturbation-based
- Integrated Gradients is gradient-based

Their qualitative consistency suggests that the Transformer is using linguistically meaningful stress-related signals.

At the same time, these explanations should still be interpreted as approximate tools for inspection, not as exact proofs of model reasoning.

## Final Interpretation

The final project outcome is methodologically meaningful even though the Transformer does not beat the baseline on final hard-label classification.

The key findings are:

- The feature-based baseline is strong, stable, and highly interpretable.
- DistilBERT does not improve accuracy and slightly underperforms on macro F1.
- DistilBERT does improve ROC-AUC and especially AP, so it provides better ranking quality.
- Explainability confirms that both models rely on plausible stress-related signals.

Overall, the project shows that handcrafted affective and psycholinguistic features already capture a large portion of the signal in Dreaddit. Contextual modeling still adds useful information, but its benefit appears more clearly in probabilistic ranking than in final threshold-based classification performance.

## Limitations

- The dataset is relatively small.
- Metric differences between models are therefore sensitive to small changes in predictions.
- Transformer explainability methods are informative but approximate.
- The task concerns stress-related language patterns, not clinical diagnosis.
