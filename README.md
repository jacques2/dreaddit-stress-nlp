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
- Traditional baselines (e.g., TF-IDF + Logistic Regression)
- Fine-tuned Transformer models (e.g., BERT / RoBERTa)
- Evaluation using appropriate classification metrics
- Explainability techniques to interpret model predictions

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

2. Download and place the dataset as described in data/README.md

3. Run experiments using notebooks or Python scripts in src/

## Status

This repository is under active development as part of an academic project.