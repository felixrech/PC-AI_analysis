## Using NLP to Study the Public Consultation on the AI Act Proposal

This repository contains the code for the Master's thesis "Computer, how should we regulate artificial intelligence? Using NLP to Study the Public Consultation on the AI Act Proposal". Feel free to use it to reproduce our research or adapt it to your topic!

Note that the 'Have your Say' scraper was also developed for this thesis. You can find it on [PyPI](https://pypi.org/project/hys-scraper/) and [GitHub](https://github.com/FelixRech/hys_scraper).

## Structure

This repository is structured as follows:

- `01_pdf_extractor_comparison/`: contains the code for the evaluation of PDF extraction libraries (see Section 4.2.1 for an explanation)
- `02_eval_tool/`: contains a tool internally used to help with the evaluation but might also be used later on in the research (no corresponding section in the thesis)
- `03_dataset_statistics_biases/`: computes statistics about the dataset and its biases - powers Section 2.3 of the thesis
- `04_dataset_access`: the code used to convert the dataset into text files and easily access, tokenize, and lemmatize it (Sections 4.2.2, 4.2.3, 4.3.1 of the thesis)
- `05_topic_modeling`: the code powering Sections 4.3 and 5.1 of the thesis
- `06_explorer/explorer`: the code for our interactive visualization dashboard, see Section 4.5 for details
- `07_sentiment/sentiment`: the code to process the dataset using ABSA, see Sections 4.4 and 5.2

## Getting started

All parts expect the dataset to be already downloaded. You can either use [hys_scraper](https://github.com/FelixRech/hys_scraper/) to scrape it from the 'Have your Say' platform yourself (and fix some [issues](https://github.com/felixrech/PC-AI#Patches) yourself, too) or use our version (with said patches already applied):

```bash
git clone git@github.com:felixrech/PC-AI.git -b main --single-branch --depth=1 24212003_requirements_for_artificial_intelligence
git clone git@github.com:felixrech/PC-AI.git -b white_paper --single-branch --depth=1 7639546_requirements_for_artificial_intelligence
```

Each part then has its own setup instructions in its README.
