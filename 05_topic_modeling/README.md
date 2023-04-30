# Topic Modeling

This module contains a topic modeling analysis of the text extracted from the PDF attachments to the AI Act proposal public consultation. For the analysis, Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) are used, with a slight focus on the latter.

The module is structured as follows:
- `topic_models.py`: Setup for LDA and NMF topic models using sklearn and hyperparameter tuning tools.
- `evaluation.py`: Tools to evaluate topic models, like the computation of topic coherence or visualizations of the topic-term and topic-document matrices.
- `images/`: Folder containing saved plots.
- `hyperparameter_tuning/`: (Intermediate) Results of hyperparameter tuning procedures.
- `topic_modeling.ipynb`: Notebook containing the complete analysis.
- `top2vec/`: Our experiments with Top2vec.
- `duplicates.ipynb`: Our experiments to identify duplicate submissions and collaborations.

## Setup and usage

- (Optionally) Create a new virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- Install the necessary libraries
    ```bash
    pip install -r requirements.txt
    ```
- Execute the `topic_modeling.ipynb` notebook using a method of your choice!