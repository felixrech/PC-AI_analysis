# Dataset statistics & biases

This module is for computing some basic statistics for the dataset, as well as evaluating the biases present in the dataset.

This module is structured as follows:
- `utils.py` and `plots.py`: Containing some general and visualization code to make the notebook less cluttered.
- `datasets/`: Folder containing two additional datasets: GDP and population numbers (from the World Bank Group, a UN subsidiary).
- `figures/`: Folder that contains the saved plots.
- `stats.ipynb`: Contains the computation of statistics and all the plots. Note that the results are not commented on, please refer to the thesis for that.

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
- Execute the notebook (e.g. VS Code, Browser, etc. - might require additional packages depending on the method chosen)