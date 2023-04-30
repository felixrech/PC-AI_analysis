# Explorer

This module contains a interactive visualization dashboard.

Everything is structured as follows:
- `app.py`: Main server executable.
- `exports.ipynb`: Computes some statistics and figures for export.
- `summarization_test.ipynb`: Some experiments concerning the use of summarization.

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
- Start the server
    ```bash
    python3 app.py
    ```
- Enjoy: [http://localhost:8050/](http://localhost:8050/)