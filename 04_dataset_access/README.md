# Converting the attachments from PDF to text files

This module contains the functionality needed to convert the PDF attachments into text files. This allows for simple and fast downstream processing. The converted dataset can be easily accessed using a provided dataloader that can also detect the language of a text and tokenize it.

Everything is structured as follows:
- `notebook.ipynb` contains an example on how the functionality of this module can be used and highlights the importance of multiprocessing.
- `text_extraction.py`, `footnote_resolution.py`, and `clean_extract.py` provide functionality for converting a dataset of PDF files into one of text files.
- `dataloader.py` provides a simple way of reading in the converted dataset, as well as tokenizing it.

## Setup and usage

- (Optionally) Create a new virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- Install the necessary libraries
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
- Include in your programs - see docstrings for details
    ```python
    import dataloader
    import pandas as pd

    df = pd.read_csv(
        "../24212003_requirements_for_artificial_intelligence/patched_feedbacks.csv"
    )
    df = dataloader.Dataloader().from_folder(
        "../24212003_requirements_for_artificial_intelligence/attachments/", df
    )
    ```