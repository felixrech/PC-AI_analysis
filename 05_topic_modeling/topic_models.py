"""This module contains utilities to train and hyperparameter-tune topic modeling
pipelines."""


import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import chain
from typing import Union, Any, Iterable

from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator

# Enable dataloader import
for path in ["../04_dataset_access/", "../../04_dataset_access/"]:
    module_path = os.path.abspath(os.path.join(path))
    if module_path not in sys.path:
        sys.path.append(module_path)

import dataloader


tokenizer = dataloader.Tokenizer(n_jobs=1)


def tokenize(x: Iterable[str] | str) -> Iterable[str]:
    """Tokenize a string or leave a already tokenized string (list of strings) as is.

    Parameters
    ----------
    x : Iterable[str] | str
        String or list (iterable) of strings.

    Returns
    -------
    Iterable[str]
        Tokenized string.
    """
    if type(x) is not str:  # The input should already be tokenized ...
        return x  # which can be recognized as it being of list type

    return tokenizer.transform([x]).iloc[0]  # Stop sklearn warning


# Common options
common_vectorizer_options = {
    "min_df": 10,  # Discard some of the less common words
    "stop_words": "english",  # Discard some of the too common words
}
external_tokenizer_options = {
    "lowercase": False,
    "tokenizer": tokenize,
    "token_pattern": None,  # Suppress token_pattern not used warnings
}

# NMF
tfidf_vectorizer = TfidfVectorizer(
    **common_vectorizer_options, **external_tokenizer_options
)
nmf = NMF(
    n_components=10,  # Default to 10 topics (changed later)
    random_state=5,  # Reproducible results
    max_iter=int(1e4),  # Make sure the model converges
    init="nndsvda",  # Stop sklearn warnings
    solver="mu",  # Use the MU solver from the original paper
)
nmf_pipeline = Pipeline(
    [
        ("vectorizer", tfidf_vectorizer),
        ("topic_model", nmf),
    ]
)

# LDA
count_vectorizer = CountVectorizer(
    **common_vectorizer_options, **external_tokenizer_options
)
lda = LatentDirichletAllocation(
    n_components=10,  # Default to 10 topics (changed later)
    n_jobs=-1,  # Enable multiprocessing
    random_state=42,  # Reproducible results
)
lda_pipeline = Pipeline(
    [
        ("vectorizer", count_vectorizer),
        ("topic_model", lda),
    ]
)


class Tokenizer(TransformerMixin, BaseEstimator):
    """Sklearn-compatible tokenizer. Uses the dataloader tokenizer under the hood."""

    def __init__(self, n_jobs=-1) -> None:
        """Initialize the tokenizer.

        Parameters
        ----------
        n_jobs : int, optional
            Number of processes to use for multiprocessing, by default -1, i.e. use as
            many as there are CPU cores.
        """
        self.n_jobs = n_jobs

    def fit(self, X, **fit_params):
        """Does nothing. For compatibility with BaseEstimator."""
        return self

    def transform(self, X: Union[list, pd.Series], **fit_params) -> pd.Series:
        """Tokenize an input corpus.

        Parameters
        ----------
        X : Union[list, pd.Series]
            Corpus to tokenize. Should be either a list of strings or a Pandas series
            of type string.

        Returns
        -------
        pd.Series
            Pandas series of type list[str].
        """
        tokenizer = dataloader.Tokenizer(n_jobs=self.n_jobs)
        return tokenizer.transform(X)


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into training and testing datasets using a 90-10 split and
    stratification using the user_type column.

    Parameters
    ----------
    df : pd.DataFrame
        Arbitrary dataframe. Has to have a user_type column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and testing dataframes.
    """
    train_idx, test_idx = model_selection.train_test_split(
        df.index, test_size=0.1, stratify=df["user_type"], shuffle=True, random_state=42
    )
    return df.loc[train_idx], df.loc[test_idx]


def hyperparameter_tune(
    df: pd.DataFrame,
    pipeline: Pipeline,
    name: str,
    hyperparameter_grid: Union[dict, list] = {},
    overwrite: bool = False,
    use_cv=False,
    scorer=None,
) -> pd.DataFrame:
    """Conduct hyperparameter tuning for the given dataset and pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains a "tokenized" column and, if use_cv is True a
        "user_type" column.
    pipeline : Pipeline
        Sklearn pipeline, should contain a "topic_model" component (and whatever other
        components are referenced in the input hyperparameter grid).
    name : str
        Name of the hyperparameter tuning procedure. Will be used to save intermediate
        results (as f"./hyperparameter_tuning/{name}.csv").
    hyperparameter_grid : Union[dict, list], optional
        Hyperparameter tuning grid. When using a dict, use the sklearn parameter grid
        input format, i.e. {'param_a': [val1, val2], 'param_b': [x1]}. You can also
        use a list which should contain hyperparameter combinations to try out. The
        example above would be [{'param_a': val1, 'param_b': x1}, {'param_a': val2,
        'param_b': x1}]. By default {}, i.e. use a default grid.
    overwrite : bool, optional
        Whether to overwrite and ignore previous results if intermediate results file
        exists, by default False.
    use_cv : bool, optional
        Whether to use (5-fold) cross validation, by default False.
    scorer : optional
        Scorer which takes a pipeline and a dataset and, by default
        evaluation.compute_coherence_from_pipeline.

    Returns
    -------
    pd.DataFrame
        Dataframe of the hyperparameter tuning procedure. Contains a column for each
        hyperparameter tuned and an additional "score" column.
    """
    if scorer is None:
        import evaluation

        scorer = evaluation.compute_coherence_from_pipeline

    # Read from existing tuning file, if it exists
    filename = f"./hyperparameter_tuning/{name}.csv"
    if f"{name}.csv" in os.listdir("./hyperparameter_tuning/") and not overwrite:
        print(
            f"Existing hyperparameter tuning for pipeline {name} found, "
            + "continuing there. You can force the use your (or the default) "
            + "hyperparameter by setting overwrite=True."
        )
        hyperparameters = pd.read_csv(
            filename,
            dtype={
                "topic_model__doc_topic_prior": "float",
                "topic_model__n_components": "int",
                "topic_model__topic_word_prior": "float",
                "score": "float",
            },
        )
    else:
        # Use default grid if hyperparameter grid is not specified
        if len(hyperparameter_grid) == 0:
            hyperparameter_grid = _get_default_grid(pipeline)
        # Turn grid into a dataframe
        hyperparameters = pd.DataFrame(
            list(model_selection.ParameterGrid(hyperparameter_grid))
            if type(hyperparameter_grid) is dict
            else hyperparameter_grid
        )
        hyperparameters["score"] = None  # Add score column
    hyperparameters = hyperparameters.replace({np.nan: None})  # None instead of NaN

    # Iterate over hyperparameter combinations (with progress bar)
    for i, row in tqdm(hyperparameters.iterrows(), total=len(hyperparameters)):
        # If the row has a score entry, then we're already finished
        if not row["score"] is None:
            continue

        # Extract and set the current hyperparameter combination
        params = hyperparameters.loc[i, hyperparameters.columns != "score"]  # type: ignore
        pipeline.set_params(**params.to_dict())

        # Fit the model and evaluate it
        if use_cv:
            skf = model_selection.StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42
            )
            score = model_selection.cross_val_score(
                pipeline,
                df["tokenized"],
                scoring=scorer,
                cv=skf.split(df["tokenized"], df["user_type"]),
            ).mean()
        else:
            pipeline.fit(df["tokenized"])
            score = scorer(pipeline, df["tokenized"])
        hyperparameters.loc[i, "score"] = score  # type: ignore

        # Save the results
        hyperparameters.to_csv(filename, index=False)

    return hyperparameters


def _get_default_grid(pipeline: Pipeline) -> dict[str, Any]:
    """Returns a default hyperparameter tuning grid for input pipeline.

    For all topic models, compare 3, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30, 35, 40, 45,
    and 50 topics.

    For LDA, add tuning over alpha and eta.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "topic_model" component.

    Returns
    -------
    dict[str, Any]
        Sklearn parameter grid input style grid.
    """
    # We default to just the number of components (= topics)
    default_components = {
        "topic_model__n_components": list(chain(range(3, 18, 2), range(20, 51, 5)))
    }

    # For the LDA, add the hyperparamters alpha and eta
    if isinstance(
        pipeline.named_steps["topic_model"],
        LatentDirichletAllocation,
    ):
        return {
            "topic_model__doc_topic_prior": [None, 0.05, 0.1, 0.5, 1, 5, 10],
            "topic_model__topic_word_prior": [None, 0.05, 0.1, 0.5, 1, 5, 10],
        } | default_components

    # Warn if we're defaulting to just topic models
    else:
        warnings.warn(
            f"No default hyperparameters available for "
            + str(type(pipeline.named_steps["topic_model"]))
            + ", falling back to number of topics only!"
        )
        return default_components


def trained_pipeline_from_hyperparameters(
    pipeline: Pipeline, hyperparameters: pd.DataFrame, df: pd.DataFrame, n_topics: int
) -> tuple[Pipeline, np.ndarray]:
    """Fits a pipeline using the results from hyperparameter tuning.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "topic_model" component.
    hyperparameters : pd.DataFrame
        Hyperparameter tuning results, in the format that hyperparameter_tune outputs.
    df : pd.DataFrame
        Dataframe that contains a "tokenized" column.
    n_topics : int
        Number of topics to extract the top scoring hyperparameter combination for.

    Returns
    -------
    tuple[Pipeline, np.ndarray]
        Fitted pipeline using the optimal hyperparameters and topic-document-matrix.
    """
    # Select the correct row and extract hyperparameters as dictionary
    limited = hyperparameters.query("topic_model__n_components == @n_topics")
    limited = limited.sort_values("score", ascending=False)
    params = limited.iloc[
        0, ~hyperparameters.columns.isin(("score", "topic_model"))
    ].to_dict()

    # Ensure number of topics is an int
    params["topic_model__n_components"] = int(params["topic_model__n_components"])

    # Set the hyperparameter and fit the pipeline
    pipeline.set_params(**params)  # type: ignore
    H = pipeline.fit_transform(df["tokenized"])
    return pipeline, H
