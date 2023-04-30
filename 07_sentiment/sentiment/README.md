# Converting the attachments from PDF to text files

This module contains the functionality needed to convert the PDF attachments into text files. This allows for simple and fast downstream processing. The converted dataset can be easily accessed using a provided dataloader that can also detect the language of a text and tokenize it.

Everything is structured as follows:
- `embedding_clustering.ipynb`: Run some experiments with embedding results in 2D.
- `patterns.ipynb`: Validation of regex ATE.
- `absa.ipynb`: Transformers model
- `pyabsa_sentiment.py`: PyABSA model

## Setup and usage

### Transformers model (and additional notebooks)

- (Optionally) Create a new virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- Install the necessary libraries
    ```bash
    pip install -r requirements.txt
    ```
- Execute the `absa.ipynb` notebook using the method of your choice!


### PyABSA model

- (Optionally) Create a new virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- Install the necessary libraries
    ```bash
    pip install -r pyabsa_requirements.txt
    ```
- Run the ABSA
    ```bash
    python pyabsa_sentiment.py
    ```