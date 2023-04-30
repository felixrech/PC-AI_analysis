"""Module for wrangling a dataset into the correct format for aspect-based sentiment
analysis and actually performing such."""

import re
import numpy as np
import pandas as pd
import itertools as it
import more_itertools as mit
from functools import partial

import torch
import transformers
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer as Tokenizer,
    AutoModelForSequenceClassification as Model,
)

from utils import patterns


DEFAULT_CHUNK_SIZE = 3


def add_aspect_and_string(df: pd.DataFrame, max_context=500) -> pd.DataFrame:
    """Add columns to prepare for aspect-based sentiment analysis to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with "sentence_index" and "text" columns.
    max_context : int, optional
        Maximum content window size before and after aspect, by default 500

    Returns
    -------
    pd.DataFrame
        Dataframe with "sentence_before", "aspect", "sentence_after", "aspect_type", and
        "aspect_subtype" columns.
    """
    df = (
        df.assign(extract=df["text"].str.findall(patterns.PATTERN))
        .explode("extract")
        .assign(
            # Extract aspect and before and after strings
            sentence_before=lambda df: df["extract"]
            .map(lambda x: x[0] if type(x) is tuple else None)
            .fillna(""),
            aspect=lambda df: df["extract"].map(
                lambda x: x[1] if type(x) is tuple else None
            ),
            sentence_after=lambda df: df["extract"]
            .map(lambda x: x[2] if type(x) is tuple else None)
            .fillna(""),
            # Create string to input into the ABSA model
            sentiment_string=lambda df: (
                "[CLS]"
                + df["sentence_before"].map(
                    partial(_context_before, max_length=max_context)
                )
                + df["aspect"]
                + df["sentence_after"].map(
                    partial(_context_after, max_length=max_context)
                )
                + "[SEP]"
                + df["aspect"]
                + "[SEP]"
            ),
            # Classify aspect, e.g. "Article 5" to "article" (type) and "5" (subtype)
            aspect_type=lambda df: df["aspect"]
            .apply(patterns.classify_aspect)["aspect_type"]
            .map(lambda x: str(x) if x is not None else x),
            aspect_subtype=lambda df: df["aspect"]
            .apply(patterns.classify_aspect)["aspect_subtype"]
            .map(lambda x: str(x) if x is not None else x),
        )
        .drop(columns=["extract"])
        # Drop repeated aspect in same sentence (e.g. "the Artificial [..] Act (AI Act)")
        .drop_duplicates(subset=["sentence_index", "aspect_type", "aspect_subtype"])
    )
    return df


def _context_before(s: str, max_length: int) -> str:
    """Cut the context before an aspect to a maximum length, using whitespace wherever
    possible.

    Parameters
    ----------
    s : str
        Arbitrary string.
    max_length : int
        Maximum length the string may have after cutting.

    Returns
    -------
    str
        Cut string.
    """
    splitted = re.split(r"\s+", s)
    if len(closest := splitted[0]) >= max_length:
        return closest[: max_length - 1] + " "

    tmp, i = "", 0
    while i < len(splitted) and len(new := tmp + " " + splitted[i]) <= max_length:
        tmp = new
        i += 1
    return tmp[1:] + " "


def _context_after(s: str, max_length: int) -> str:
    """Cut the context after an aspect to a maximum length, using whitespace wherever
    possible.

    Parameters
    ----------
    s : str
        Arbitrary string.
    max_length : int
        Maximum length the string may have after cutting.

    Returns
    -------
    str
        Cut string.
    """
    splitted = re.split(r"\s+", s)
    if len(closest := splitted[-1]) >= max_length:
        return closest[-max_length:] + " "

    tmp, i = "", -1
    while i >= -len(splitted) and len(new := splitted[i] + " " + tmp) <= max_length:
        tmp = new
        i += -1
    return " " + tmp[:-1]


def _absa_single(
    text: list[str],
    tokenizer: transformers.DebertaV2TokenizerFast,
    model: transformers.DebertaV2ForSequenceClassification,
) -> np.ndarray:
    """Compute aspect-based sentiment analysis.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    tokenizer : transformers.DebertaV2TokenizerFast
        Tokenizer matching the model
    model : transformers.DebertaV2ForSequenceClassification
        ABSA model matching the tokenizer.

    Returns
    -------
    np.ndarray
        [n, 3]-dimensional arrays with "negative", "neutral", and "positive" columns.
    """
    # Tokenize input and copy to GPU
    tokens = tokenizer(text, padding=True, return_tensors="pt")
    tokens = tokens.to("cuda")

    # Predict and softmax to get final answer
    predictions = model(**tokens)
    results_tensor = softmax(predictions.logits, dim=1)
    results = results_tensor.to("cpu").detach().numpy()

    # Release the GPU memory the computations take up
    del tokens, predictions, results_tensor
    torch.cuda.empty_cache()

    return results


def absa(text: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
    """Compute aspect-based sentiment analysis and return it as a numpy array.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    chunk_size : int, optional
        Chunk size (to adapt to available GPU memory), by default DEFAULT_CHUNK_SIZE.

    Returns
    -------
    np.ndarray
        [n, 3]-dimensional arrays with "negative", "neutral", and "positive" columns.
    """
    # Initialize the model
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer: transformers.DebertaV2TokenizerFast = Tokenizer.from_pretrained(
        model_name
    )  # type: ignore
    model = Model.from_pretrained(model_name).to("cuda")

    # Compute the result in small chunks to fit into GPU memory
    results = np.vstack(
        list(
            it.chain(
                *map(
                    partial(_absa_single, tokenizer=tokenizer, model=model),
                    mit.chunked(text, chunk_size),
                )
            )
        )
    )

    # Release the GPU memory the model takes up
    del model
    torch.cuda.empty_cache()
    return results


def absa_df(texts: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    """Compute aspect-based sentiment analysis and return it as a dataframe.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    chunk_size : int, optional
        Chunk size (to adapt to available GPU memory), by default DEFAULT_CHUNK_SIZE.

    Returns
    -------
    pd.DataFrame
        Dataframe with "negative", "neutral", and "positive" columns.
    """
    results = absa(texts, chunk_size=chunk_size)

    return pd.DataFrame(results, columns=["negative", "neutral", "positive"])


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add aspect-based sentiment analysis columns to any rows of a dataframe with
    non-null "sentiment_string"

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with "sentiment_string" and "sentence_index" columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with added "negative", "neutral", "positive", and "sentiment" columns.
    """
    mentions = df.query("sentiment_string.notnull()").copy().reset_index(drop=True)
    others = df.query("sentiment_string.isnull()").copy()

    results = absa_df(mentions["sentiment_string"].to_list())

    results_df = pd.concat((mentions, results), axis=1)
    results_df["sentiment"] = pd.Series(
        np.argmax(results_df[["negative", "neutral", "positive"]].values, axis=1)
    ).map({0: "negative", 1: "neutral", 2: "positive"})

    return (
        pd.concat((results_df, others), ignore_index=True, axis=0)
        .sort_values("sentence_index")
        .reset_index(drop=True)
    )
