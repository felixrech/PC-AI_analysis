# Finding the best PDF text extractor

This module contains multiple evaluations to identify the best PDF text extraction library for our dataset. The following libraries were evaluated:

| Library    | Github                                                                    | PyPI                                           |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| PyMuPDF    | [pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF)                     | [here](https://pypi.org/project/PyMuPDF/)      |
| Tika       | [chrismattmann/tika-python](https://github.com/chrismattmann/tika-python) | [here](https://pypi.org/project/tika/)         |
| PyPDF2     | [py-pdf/PyPDF2](https://github.com/py-pdf/PyPDF2)                         | [here](https://pypi.org/project/PyPDF2/)       |
| pdfminer   | [pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six)         | [here](https://pypi.org/project/pdfminer.six/) |
| pdfplumber | [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber)                 | [here](https://pypi.org/project/pdfplumber/)   |
| borb       | [jorisschellekens/borb](https://github.com/jorisschellekens/borb)         | [here](https://pypi.org/project/borb/)         |

The module is structure as follows:

- `results/`: Results of the experiments in CSV format (mostly human unreadable).
- `footnote_resolution_v1.py`: Outdated version of footnote reference resolution for the sake of reproducibility, see `../04_dataset_access` for a current version.
- `pdf_extractors.ipynb`: Contains all experiments and their results. Automatically installs the necessary libraries.
