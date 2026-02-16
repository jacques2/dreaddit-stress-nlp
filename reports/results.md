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

## Baseline Model
- An interpretable logistic regression baseline was trained using affective, psycholinguistic, and readability features.
- The most influential signals for stress prediction include first-person singular pronouns, negative emotional language, and text length.
- Features associated with positive emotional tone, social references, and higher readability reduce the likelihood of stress predictions.

## Error Analysis
- High-confidence errors mainly correspond to emotionally expressive but non-stress posts.
- These cases suggest that feature-based models tend to conflate personal narrative style with psychological stress.
- This limitation motivates the adoption of semantic models capable of capturing contextual meaning beyond surface-level affective cues.
