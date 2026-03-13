# Stress Detection in Social Media (NLP)

This project studies stress detection from Reddit posts using NLP methods with two complementary goals:

- compare an interpretable feature-based baseline with a contextual Transformer model
- analyze not only predictive performance, but also interpretability, limitations, and methodological trade-offs

The task is binary classification: given a post, predict whether it is labeled as `stress` or `non-stress`.

This project is not intended for clinical diagnosis. It is an academic study of linguistic patterns and model behavior on a sensitive NLP task.

## Dataset

- **Dataset**: Dreaddit
- **Source**: public and anonymized Reddit posts labeled for stress-related content
- **Size used in the project**: 715 posts
- **Class balance**: nearly balanced, with stress posts slightly above 50%

The near-balanced label distribution allows comparison with standard classification metrics without heavy imbalance correction.

## Notebook Workflow

The project is primarily notebook-driven. The recommended order is:

1. `notebooks/01_eda.ipynb.ipynb`
   Exploratory analysis of class balance, text length, and affective signals.
2. `notebooks/02_train_baseline.ipynb`
   Interpretable baseline with handcrafted features and Logistic Regression.
3. `notebooks/03_train_transformer.ipynb`
   DistilBERT fine-tuning, evaluation, and final comparison with the baseline.
4. `notebooks/04_explainability.ipynb`
   Explainability for both baseline and Transformer models.

## Methods

### Baseline

The baseline model is a Logistic Regression classifier trained on a compact set of handcrafted features:

- readability and style features
- LIWC-style psycholinguistic features
- DAL affective features
- sentiment-related signals

The modeling pipeline includes:

- median imputation
- feature scaling
- Logistic Regression

This baseline is intentionally simple and interpretable, so feature effects can be inspected directly through model coefficients.

### Transformer

The contextual model is a fine-tuned **DistilBERT** classifier.

Main choices:

- pretrained model: `distilbert-base-uncased`
- maximum sequence length: 128
- training epochs: 3
- Hugging Face `Trainer` workflow
- same train/test split used by the baseline for a fair comparison

### Evaluation

The project uses both threshold-based and ranking-based metrics:

- Accuracy
- Precision / Recall / F1
- Macro F1
- ROC-AUC
- Average Precision (PR-AUC / AP)

For the baseline, threshold optimization is also included:

- best threshold: `0.54`
- best macro F1 after threshold tuning: `0.762`

## Final Results

Final comparison on the shared held-out test split:

| Metric | Baseline (LogReg) | Transformer (DistilBERT) |
|---|---:|---:|
| Accuracy | 0.748 | 0.748 |
| Macro F1 | 0.748 | 0.745 |
| ROC-AUC | 0.814 | 0.819 |
| AP (PR-AUC) | 0.769 | 0.835 |

Main takeaway:

- The Transformer does **not** improve final threshold-based classification performance over the baseline.
- The baseline matches Transformer accuracy and slightly exceeds it on macro F1.
- The tuned baseline remains stronger on macro F1 (`0.762`) than the final Transformer run (`0.745`).
- The Transformer performs better on ranking-based metrics, especially AP, suggesting better probabilistic discrimination.

This means the baseline remains extremely competitive, while DistilBERT adds value mainly as a scorer rather than as a clearly better final classifier on this dataset.

## Explainability

The project includes explainability for both model families.

### Baseline Explainability

For Logistic Regression, interpretation is direct through feature coefficients.

Main signals associated with stress:

- first-person singular pronouns
- negative emotional language
- anxiety-related vocabulary
- text length and narrative detail

Signals associated with non-stress:

- more positive emotional tone
- greater pleasantness
- social references
- higher readability

### Transformer Explainability

For DistilBERT, the project applies:

- SHAP for local and global token-level explanations
- Integrated Gradients for gradient-based token attribution

The explainability analysis shows that the model often focuses on semantically meaningful tokens such as:

- `panic`
- `attacks`
- `anxiety`
- `severe`
- self-referential tokens like `I` and `my`

SHAP and Integrated Gradients are qualitatively consistent, which supports the interpretation that the Transformer relies on plausible stress-related language cues rather than arbitrary artifacts.

## Repository Structure

`dreaddit-stress-nlp/`

- `data/`
  Dataset instructions and local data placement.
- `notebooks/`
  Main experimental workflow for EDA, training, evaluation, and explainability.
- `results/`
  Generated plots and exported artifacts.
- `reports/`
  Written summary of final results.
- `src/`
  Helper code used by parts of the workflow.

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare the dataset:

- Option A: place `data/raw/dreaddit.csv` locally
- Option B: rely on notebook download fallback via `kagglehub` when available

3. Open the notebooks and run them in order:

- `notebooks/01_eda.ipynb.ipynb`
- `notebooks/02_train_baseline.ipynb`
- `notebooks/03_train_transformer.ipynb`
- `notebooks/04_explainability.ipynb`

4. Optional Hugging Face login for the Transformer notebook:

```bash
export HF_TOKEN=your_token_here
```

Generated figures are saved under `results/`, and the written summary is available in `reports/results.md`.

## Limitations

- The dataset is relatively small, so differences between models should be interpreted cautiously.
- The Transformer does not deliver a clear improvement on final hard-label classification metrics.
- Explainability methods for Transformers are useful but approximate.
- The task concerns stress-related language patterns, not clinical diagnosis.

## Ethical Considerations

This project uses sensitive user-generated text, even though the dataset is public and anonymized.

Results should be interpreted carefully:

- they are not suitable for clinical decision-making
- they should not be treated as psychological diagnosis
- they are intended only for academic analysis of NLP methods in a sensitive domain

## Status

This repository contains the final version of an academic project for the course *AI in Industry*.

For the complete written discussion of findings, see `reports/results.md`.
