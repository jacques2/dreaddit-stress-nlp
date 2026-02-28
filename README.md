# Stress Detection in Social Media (NLP)

This project explores the use of Natural Language Processing (NLP) techniques
to classify social media posts as indicative of psychological stress or not.

The goal is **not** to perform any form of clinical diagnosis, but to analyze
statistical and linguistic patterns in publicly available and anonymized datasets,
and to study the behavior, limitations, and interpretability of NLP models
in sensitive domains.

## Dataset
- **Dreaddit**: a dataset of Reddit posts labeled for stress-related content.
- The dataset is publicly available and anonymized.
- See `data/README.md` for instructions on how to obtain and place the data.

## Methods
The project includes:
- Interpretable baseline with handcrafted linguistic/affective features + Logistic Regression
- Fine-tuned Transformer model (DistilBERT)
- Evaluation with classification and ranking metrics
- Simple feature-based explainability for baseline coefficients

## Evaluation
Models are evaluated using:
- Precision, Recall, F1-score
- Analysis of class imbalance
- Qualitative error analysis

## Explainability
To better understand model decisions, we apply:
- Feature-based explanations for baseline models
- Explainable AI (XAI) techniques for Transformer models (e.g., SHAP, Integrated Gradients)

## Repository Structure
dreaddit-stress-nlp/
├── data/ # dataset instructions
├── notebooks/ # experiments and analysis
├── src/ # reusable training, evaluation and explainability code
└── reports/ # results, tables and qualitative analysis


## Ethical Considerations
This project deals with sensitive user-generated content.
All analyses are conducted on anonymized, publicly available data.
Results should be interpreted with caution and are not intended
for clinical or real-world decision-making.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data (see `data/README.md`):
- Option A (recommended): place `data/raw/dreaddit.csv` locally
- Option B: if the file is missing, scripts/notebooks can download via `kagglehub`

3. Run baseline training (reproducible split + validation-based threshold):
   ```bash
   python src/train.py --seed 42
   ```
   Outputs are saved in `reports/results/`.

4. Optionally run/inspect notebooks in `notebooks/`.

## Status

This repository is under active development as part of an academic project.
