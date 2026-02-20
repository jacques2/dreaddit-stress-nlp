# Results Summary

This section summarizes the main findings obtained so far.

## Exploratory Analysis
- The dataset is nearly balanced between stress and non-stress posts.
![Label Distribution](./label_distribution.png)
- Text length alone shows limited discriminative power, with substantial overlap between classes.
![Text Length By Label](./text_length_by_label.png)
- Stress-related posts exhibit a higher number of long outliers, often corresponding to extended personal narratives.
- Affective features such as sentiment and lexical pleasantness show clearer differences between classes.
![Pleasantness By Label Distribution](./pleasantness_by_label.png)
These findings motivate the use of interpretable affective baselines and contextual NLP models.

## Baseline Model Performance

An interpretable logistic regression model was trained using affective, psycholinguistic, and readability features.

### Hold-out Test Performance
- Accuracy: **0.748**
- Macro F1-score: **0.748**
- The model substantially outperforms the majority baseline (0.516).
- Precision and recall are balanced across both classes.

### Cross-Validation Performance
- 5-fold stratified CV macro F1 (mean): **0.755**
- Standard deviation: **0.025**

Cross-validation confirms that performance is stable and not dependent on a specific train/test split.

### Ranking-Based Evaluation
- ROC-AUC: **0.814**
- Average Precision (PR-AUC): **0.769**

The ROC-AUC indicates strong ranking capability, while the Precision–Recall curve confirms that positive (stress) predictions remain reasonably reliable across thresholds.

### Threshold Optimization
- Optimal threshold: **0.54**
- Best macro F1: **0.762**

The modest improvement over the default threshold suggests that logistic regression produces reasonably calibrated probability estimates.

---

## Feature Interpretation

The most influential signals for stress prediction include:

- First-person singular pronouns (`lex_liwc_i`)
- Negative emotional language (`lex_liwc_negemo`, `lex_liwc_anx`)
- Text length (`lex_liwc_WC`)
- Imagery-related lexical features

Features associated with positive emotional tone, social references, and higher readability reduce the likelihood of stress predictions.

Because features were standardized, coefficient magnitudes are directly comparable.

---

## Error Analysis

High-confidence misclassifications mainly correspond to emotionally expressive but non-stress posts. These cases suggest that feature-based models may conflate personal narrative style with psychological stress.

This limitation motivates the adoption of semantic models (e.g., Transformer-based architectures) capable of capturing contextual meaning beyond surface-level affective cues.