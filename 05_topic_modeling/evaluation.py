"""This module contains utilities to interpret, evaluate, and analyze topic modeling
results."""

import os
import fitz
import scipy
import shutil
import gensim
import warnings
import itertools
import numpy as np
import pandas as pd

import multiprocessing
from zipfile import ZipFile
from typing import Iterable
from functools import partial
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def get_topics_from_pipeline(pipeline: Pipeline, n_terms: int = 10) -> pd.DataFrame:
    """Gets a dataframe of the top terms for each of a pipeline's topics.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "vectorizer" and a "topic_model" component.
    n_terms : int, optional
        Number of terms to return for each topic, by default 10.

    Returns
    -------
    pd.DataFrame
        Dataframe of the top terms for each of a pipeline's topics. Will be of shape
        (n_terms, n_topics).
    """
    return get_topics(
        pipeline.named_steps["topic_model"].components_,
        pipeline.named_steps["vectorizer"].get_feature_names_out(),
        n_terms=n_terms,
    )


def get_topic_from_pipeline(pipeline, topic: int, n_terms: int = -1) -> pd.Series:
    """Compute a series of the top terms in a topic and their term-topic-factors.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "vectorizer" and a "topic_model" component.
    topic : int
        Number of the topic to extract, zero-indexed.
    n_terms : int, optional
        Number of terms to return for the topic, by default -1, representing all terms.

    Returns
    -------
    pd.Series
        Series containing the top terms of the given pipeline's topics. Will have the
        terms as the index and the term-topic-factors as the values.
    """
    return get_topic(
        pipeline.named_steps["topic_model"].components_,
        pipeline.named_steps["vectorizer"].get_feature_names_out(),
        topic=topic,
        n_terms=n_terms,
    )


def get_topics(
    W: np.ndarray,
    terms: list[str],
    n_terms: int = 10,
) -> pd.DataFrame:
    """Computes a dataframe of the top terms for each of topics from a
    term-topic-matrix.

    Parameters
    ----------
    W : np.ndarray
        Term-topic matrix (sklearn's topic_model.components_).
    terms : list[str]
        The terms used in W, i.e. its column names (sklearn's
        vectorizer.get_feature_names_out()).
    n_terms : int, optional
        Number of terms to return for each topic, by default 10.

    Returns
    -------
    pd.DataFrame
        Dataframe of the top terms for each of a pipeline's topics. Will be of shape
        (n_terms, n_topics).
    """
    topics = []
    for topic in range(W.shape[0]):
        topics.append(get_topic(W, terms, topic, n_terms).index.tolist())

    return pd.DataFrame(
        {f"topic_{str(i).zfill(2)}": topics[i] for i in range(len(topics))}
    )


def get_topic(
    W: np.ndarray,
    terms: list[str],
    topic: int,
    n_terms: int = 10,
) -> pd.Series:
    """Compute a series of the top terms in a topic and their term-topic-factors based
    on the term-topic-matrix.

    Parameters
    ----------
    W : np.ndarray
        Term-topic matrix (sklearn's topic_model.components_).
    terms : list[str]
        The terms used in W, i.e. its column names (sklearn's
        vectorizer.get_feature_names_out()).
    topic : int
        Number of the topic to extract, zero-indexed.
    n_terms : int, optional
        Number of terms to return for the topic, by default -1, representing all terms.

    Returns
    -------
    pd.Series
        Series containing the top terms of the given pipeline's topics. Will have the
        terms as the index and the term-topic-factors as the values.
    """
    s = pd.Series(W[topic], index=terms)
    if n_terms > 0:
        return s.nlargest(n_terms)
    return s.sort_index().sort_values(ascending=False, kind="stable")


def compute_perplexity_from_pipeline(pipeline: Pipeline, X: pd.Series) -> float:
    """Computes perplexity of pipeline for given input.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "vectorizer" and a "topic_model" component.
    X : pd.Series
        Input series with each element being a list of strings.

    Returns
    -------
    float
        The perplexity.
    """
    X = pipeline.named_steps["vectorizer"].transform(X)
    return pipeline.named_steps["topic_model"].perplexity(X)


def compute_coherence_from_pipeline(
    pipeline: Pipeline,
    X,
    coherence_measure: str = "c_v",
    n_terms: int = 20,
) -> float:
    """Compute topic coherence for given pipeline and input.

    Parameters
    ----------

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "vectorizer" and a "topic_model" component.
    X : pd.Series
        Input series with each element being a list of strings.
    coherence_measure : str, optional
        Coherence measure to use, one of 'c_v', 'u_mass', 'c_uci', or 'c_npmi'. By
        default "c_v".
    n_terms : int, optional
        Number of terms to extract from each topic to evaluate the topic coherence,
        by default 20.

    Returns
    -------
    float
        Topic coherence.
    """
    return compute_coherence(
        pipeline.named_steps["topic_model"].components_,
        pipeline.named_steps["vectorizer"].get_feature_names_out(),
        X.to_list(),
        coherence_measure,
        n_terms=n_terms,
    )


def compute_coherence(
    W: np.ndarray,
    terms: list[str],
    texts: Iterable[list[str]],
    coherence_measure: str = "c_v",
    n_terms: int = 20,
) -> float:
    """Compute topic coherence for given term-topic matrix W.

    Parameters
    ----------
    W : np.ndarray
        Term-topic matrix of dimension N*k (number of terms by number of topics).
    terms : list[str]
        List of the terms used for the term-topic matrix, i.e. the column names of W.
    texts : Iterable[list[str]]
        List of tokenized documents to use for the coherence computation.
    coherence_measure : str, optional
        Coherence measure to use, one of 'c_v', 'u_mass', 'c_uci', or 'c_npmi'. By
        default "c_v".
    n_terms : int, optional
        Number of terms to extract from each topic to evaluate the topic coherence,
        by default 20.

    Returns
    -------
    float
        Topic coherence.
    """
    topics = get_topics(W=W, terms=terms, n_terms=n_terms)
    topics = [topics[col].to_list() for col in topics.columns]

    dictionary = gensim.corpora.dictionary.Dictionary(texts)

    cm = gensim.models.CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_measure,
    )
    return cm.get_coherence()


def plot_coherence_against_topics(
    hyperparameters: pd.DataFrame,
) -> tuple[plt.figure, plt.axis]:
    """Plot topic coherence against the number of topics used.

    Parameters
    ----------
    hyperparameters : pd.DataFrame
        Hyperparameter tuning results, in the format that hyperparameter_tune outputs.

    Returns
    -------
    tuple[plt.figure, plt.axis]
        Figure and axis of the resulting plot.
    """
    grouped = hyperparameters.groupby(
        ["topic_model", "topic_model__n_components"], group_keys=False
    )
    best = grouped.apply(lambda df: df[df["score"] == df["score"].max()]).reset_index(
        drop=True
    )

    palette = {
        "nmf": [0.21568627, 0.52941176, 0.75424837, 1.0],
        "nmf_line": [0.34646674, 0.63240292, 0.81067282, 1.0],
        "nmf_sentence": [0.67189542, 0.81437908, 0.90065359, 1.0],
        "nmf_doc": [0.51058824, 0.73230296, 0.85883891, 1.0],
        "lda": [0.51058824, 0.73230296, 0.85883891, 1.0],
    }

    fig, ax = plt.subplots(figsize=(7.5, 5))
    with warnings.catch_warnings():  # Suppress the annoying warning about single marker
        warnings.simplefilter("ignore")  # but two classes...
        sns.lineplot(
            x="topic_model__n_components",
            y="score",
            data=best,
            hue="topic_model",
            ax=ax,
            color=[0.215, 0.529, 0.754, 1.0],
            style="topic_model",
            marker="o",
            dashes=False if len(best["topic_model"].unique()) < 3 else True,
            palette=palette,
        )

    # Add highlight at each model's optimum
    for topic_model in best["topic_model"].unique():
        limited = best.query("topic_model == @topic_model")
        max_ = limited.iloc[limited["score"].argmax()]
        ax.scatter(
            x=max_["topic_model__n_components"],
            y=max_["score"],
            color=[1.000, 0.000, 0.000, 0.75],
            zorder=10,
        )
        ax.annotate(
            xy=(max_["topic_model__n_components"], max_["score"]),
            xytext=(0, 5),
            textcoords="offset points",
            text=(max_["topic_model__n_components"], round(max_["score"], 2)),
            ha="center",
            va="bottom",
            color="#444444",
        )

    # Label the axes and set up legend
    ax.set_xlabel("Number of topics", color="#444444")
    ax.set_ylabel("Topic coherence ($C_V$)", color="#444444")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="Topic model",
        handles=handles,
        labels=[l.upper() for l in labels],
        loc="center right"
        # if "nmf_doc" not in best["topic_model"].unique()
        # else "upper right",
    )

    # Apply common theme to plot
    _theme_plot(ax)
    return fig, ax


def merge_pdfs(files: list[str], name: str, delete: bool = True) -> None:
    """Merge multiple PDF files into a single file.

    Parameters
    ----------
    files : list[str]
        Filenames of PDF files to merge, order will be kept.
    name : str
        Filename of the output file.
    delete : bool, optional
        Whether to delete the input files, by default True.
    """
    doc = fitz.open(files[0])
    for file in files[1:]:
        doc.insert_pdf(fitz.open(file))
    doc.save("/tmp/topic_modeling_merge_result.pdf")
    if delete:
        [os.remove(file) for file in files]
    shutil.move("/tmp/topic_modeling_merge_result.pdf", name)


def plot_topic_terms(
    pipeline: Pipeline, topic: int, n_terms: int = 10
) -> tuple[plt.figure, plt.axis]:
    """Plot the top terms in a given topic.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline, should contain a "vectorizer" and a "topic_model" component.
    topic : int
        Number of the topic to extract, zero-indexed.
    n_terms : int, optional
        Number of terms to return for the topic, by default 10.

    Returns
    -------
    tuple[plt.figure, plt.axis]
        Figure and axis of the resulting plot.
    """
    # Prepare the data
    topic = get_topic_from_pipeline(pipeline, topic, -1)
    topic = topic.to_frame("factor").reset_index().rename(columns={"index": "term"})
    topic["relative"] = topic["factor"] / topic["factor"].sum()
    topic["importance"] = pd.cut(
        topic["relative"],
        [0, 0.01, 0.03, 0.05, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    # Plot the plot
    fig, ax = plt.subplots(figsize=(7.5, 5 + ((n_terms - 10) / 3)))
    palette = {
        "low": [0.79935409, 0.8740792, 0.94488274, 1.0],
        "medium": [0.67189542, 0.81437908, 0.90065359, 1.0],
        "high": [0.34646674, 0.63240292, 0.81067282, 1.0],
        "very_high": [0.10557478, 0.41262591, 0.68596694, 1.0],
    }
    sns.barplot(
        x="relative",
        y="term",
        data=topic.iloc[:n_terms],
        hue="importance",
        dodge=False,
        palette=palette,
        ax=ax,
    )

    # Disable legend, as well as x and y axis labels
    ax.get_legend().set_visible(False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # Format x axis as percentages
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Some more theming
    _theme_plot(ax)
    plt.yticks(color="#111111")
    return fig, ax


def _theme_plot(ax: plt.axis):
    """Apply a common theme to a plot.

    Parameters
    ----------
    ax : plt.axis
        Axis of the plot.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="x", which="both", colors="#999999")
    ax.tick_params(axis="y", colors="#999999")
    [l.set_color("#444444") for l in ax.xaxis.get_ticklabels()]
    [l.set_color("#444444") for l in ax.yaxis.get_ticklabels()]


def _is_outlier(df: pd.DataFrame, method="Tukey", ease=1.0) -> pd.Series:
    """Helper for the plot_topics_to_user_types method: Computes which dataframe rows
    are outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing "variable" and "value" columns.
    method : str, optional
        Outlier detection method to use, either "mean" (mean + 2*std) or "Tukey"
        (median + 1.5iqr), by default "Tukey".
    ease : float, optional
        Control the amount of outliers there are by scaling the second addend, by
        default 1.0.

    Returns
    -------
    pd.Series
        Boolean mask.
    """
    if method == "Tukey":
        limits = df.groupby("variable")["value"].median()
        limits += ease * 1.5 * df.groupby("variable")["value"].agg(scipy.stats.iqr)

        df = pd.merge(
            df, limits.to_frame("limit"), left_on="variable", right_index=True
        )
        return df["value"] > df["limit"]
    elif method == "mean":
        limits = df.groupby("variable")["value"].mean()
        limits += ease * 2 * df.groupby("variable")["value"].std()

        df = pd.merge(
            df, limits.to_frame("limit"), left_on="variable", right_index=True
        )
        return df["value"] > df["limit"]


def plot_topics_to_user_types(
    df: pd.DataFrame,
    H: np.ndarray,
    topic: str = "",
    topic_names: list[str] = [],
    normalize=True,
) -> sns.FacetGrid:
    """Plot the average topic-document frequencies based a given topic-document-matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a "user_type" column and of shape (n_documents, X).
    H : np.ndarray
        Topic-document-matrix of shape (n_documents, n_topics).
    topic : str, optional
        If specified, limit plot to a single topic based on its name, by default "".
    topic_names : list[str], optional
        (Manually) assigned names of the topics, by default [], i.e. just numbering
        them.
    normalize : bool, optional
        Whether to normalize row-wise (i.e. make topic-document factors sum up to one
        for each document), by default True.

    Returns
    -------
    sns.FacetGrid
        Final plot.
    """
    df = _H_aggregate_statistics(df, H, topic_names, normalize)

    # Limit to single topic if specified
    if topic != "":
        df = df.query(f"variable == @topic").copy()

    # Add some additional columns for plotting
    df["user_type_name"] = df["user_type"].map(_user_type_map)
    df["outlier"] = _is_outlier(df, ease=2 / 3)

    light = [0.51058824, 0.73230296, 0.85883891, 1.0]
    dark = [0.21568627, 0.52941176, 0.75424837, 1.0]
    palette = {True: dark, False: light}

    # Create barplot out of dataframe
    g = sns.catplot(
        x="user_type_name",
        y="value",
        col="variable",
        col_wrap=5,
        data=df,
        kind="bar",
        sharex=False,
        sharey=False,
        order=map(lambda x: _user_type_map[x], _user_type_order),
        hue="value",
        palette="Blues",
        dodge=False,
        legend=False,  # type: ignore
        height=4.5,  # type: ignore
        aspect=1.5 if len(df["variable"].unique()) == 1 else 1.1,  # type: ignore
    )

    # Properly label the subplots
    g.set_titles(template="Topic {col_name}")

    for ax in g.axes.flat:
        # Rotate user types
        ax.set_xticklabels(
            map(lambda x: _user_type_map[x], _user_type_order),
            rotation=25,
            horizontalalignment="right",
        )

        # Highlight outlier user types
        outliers = df.query(f"variable == '{ax.get_title()[6:]}'").set_index(
            "user_type"
        )
        colors = map(
            lambda x: {True: "#111111", False: "#555555"}[x],
            outliers["outlier"].loc[_user_type_order],
        )
        [l.set_color(c) for l, c in zip(ax.xaxis.get_ticklabels(), colors)]

        if topic != "":
            ax.set_title("")

        # Other theming: remove x label and legend; set y label
        ax.set_xlabel(None)
        ax.set_ylabel("Topic frequency")
        ax.legend().set_visible(False)

    # Avoid plots sticking too close together
    g.fig.subplots_adjust(hspace=0.5)

    return g


def plot_user_types_to_topics(df, H, topic_names, normalize=True, top_n=10):
    """Plot which user types have the highest average topic-document frequencies based
    on a given topic-document-matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a "user_type" column and of shape (n_documents, X).
    H : np.ndarray
        Topic-document-matrix of shape (n_documents, n_topics).
    topic_names : list[str]
        (Manually) assigned names of the topics.
    normalize : bool, optional
        Whether to normalize row-wise (i.e. make topic-document factors sum up to one
        for each document), by default True.
    top_n : int, optional
        Limit each topic to the top_n most frequent terms, by default 10.

    Returns
    -------
    sns.FacetGrid
        Final plot.
    """
    df = _H_aggregate_statistics(df, H, topic_names, normalize)
    df = df.sort_values(["user_type", "value"], ascending=[True, False])

    if top_n > 0:
        grouped = df.groupby("user_type")
        df = grouped.apply(lambda df: df.iloc[:top_n]).reset_index(drop=True)
    df = df.set_index("user_type").loc[_user_type_order].reset_index()

    # Add some additional columns for plotting
    df["user_type_name"] = df["user_type"].map(_user_type_map)

    g = sns.catplot(
        x="value",
        y="variable",
        col="user_type_name",
        col_wrap=3,
        data=df,
        kind="bar",
        sharex=False,
        sharey=False,
        palette="Blues",
        hue="value",
        dodge=False,
        aspect=1.2,
    )

    # Format x axis as percentages
    import matplotlib.ticker as mtick

    for ax in g.axes.flat:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        _theme_plot(ax)
        plt.setp(ax.get_yticklabels(), color="#111111")

    # Properly label the subplots
    g.set_titles(template="{col_name}", color="#111111")

    g.fig.subplots_adjust(hspace=0.2)
    return g


def H_to_dataframe(
    H: np.ndarray, topic_names: list[str], normalize: bool
) -> pd.DataFrame:
    """Turn the topic-document-matrix H into a dataframe.

    Parameters
    ----------
    H : np.ndarray
        Topic-document-matrix of shape (n_documents, n_topics).
    topic_names : list[str]
        (Manually) assigned names of the topics.
    normalize : bool
        Whether to normalize row-wise (i.e. make topic-document factors sum up to one
        for each document).

    Returns
    -------
    pd.DataFrame
        Dataframe with topics as columns and documents as rows.
    """
    H = pd.DataFrame(H)

    if len(topic_names) > 0:
        topic_names = list(topic_names)
        topic_names.extend(
            f"Nr. {i}" for i in range(len(topic_names) + 1, H.shape[1] + 1)
        )
        H.columns = topic_names

    # Normalize H by row (= document)
    if normalize:
        H = H.div(H.sum(axis=1), axis=0)

    return H


def _H_aggregate_statistics(
    df: pd.DataFrame, H: np.ndarray, topic_names: list[str], normalize: bool
) -> pd.DataFrame:
    """Compute the mean topic-document frequencies for each user type.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a "user_type" column and of shape (n_documents, X).
    H : np.ndarray
        Topic-document-matrix of shape (n_documents, n_topics).
    topic_names : list[str]
        (Manually) assigned names of the topics.
    normalize : bool
        Whether to normalize row-wise (i.e. make topic-document factors sum up to one
        for each document).

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "user_type", "variable" containing topic names, and
        "value" containing mean topic-document factors.
    """
    H = H_to_dataframe(H, topic_names, normalize)

    # Compute mean term frequency for each user type and topic combination
    df = pd.concat((df.reset_index(drop=True), H.reset_index(drop=True)), axis=1)
    # Replace to fix some weird Pandas error (AttributeError; df has no attribute name)
    # df = df.groupby("user_type").agg({topic: "mean" for topic in H.columns})
    df = df.groupby("user_type")[topic_names].agg("mean")
    df = df.reset_index().melt("user_type")
    return df


def examples_for(
    df: pd.DataFrame, H: np.ndarray, topic_names: list[str], topic: str, user_type: str
) -> pd.DataFrame:
    """Compute a dataframe of examples for a topic, sorted by descending topic-document
    factors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a "user_type" column and of shape (n_documents, X).
    H : np.ndarray
        Topic-document-matrix of shape (n_documents, n_topics).
    topic_names : list[str]
        (Manually) assigned names of the topics.
    topic : str
        Limit dataframe to a specific topic.
    user_type : str
        Limit dataframe to a specific user_type.

    Returns
    -------
    pd.DataFrame
        Dataframe with the columns of the input df, as well as a "variable" column
        containing topic names and a "value" column containing mean topic-document
        factors.
    """
    df = pd.concat(
        (
            df.reset_index(drop=True),
            H_to_dataframe(H, topic_names, True).reset_index(drop=True),
        ),
        axis=1,
    )

    return df.query("user_type == @user_type").sort_values(topic, ascending=False)


def _tsne_embedding(H_norm: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    """Compute a TSNE embedding for a normalized topic-document-matrix.

    Parameters
    ----------
    H_norm : pd.DataFrame
        (Document-)Normalized topic-document-matrix H.
    topics : list[str]
        Names of the topics. Have to be columns in the H_norm data frame.

    Returns
    -------
    pd.DataFrame
        Dataframe with with each row representing the TSNE embedding of one document's
        topic distribution, i.e. the two columns named 'PC1' and 'PC2' being the two
        principal components.
    """
    tsne = TSNE(n_components=2, n_jobs=-1)
    X = H_norm[topics].values.astype(np.float32)  # Conversion might stop weird bugs

    embedding = tsne.fit_transform(X)
    return pd.DataFrame(embedding, columns=["PC1", "PC2"])


def _similarity_ratio_row(
    i: int, texts: pd.Series, method: str, cutoff: float
) -> list[float]:
    """Compute one row of the similarity matrix. Has to be mirrored along the diagonal.

    Parameters
    ----------
    i : int
        Row index, i.e. in [0, len(texts)].
    texts : pd.Series[str]
        Pandas series of the strings to compute similarity ratios for.
    method : str
        Similarity measure to use. Currently supported: 'levenshtein'
    cutoff : float
        Cutoff, can speed up computations.

    Returns
    -------
    list[float]
        Row of similarity ratios as list.
    """
    import Levenshtein

    if cutoff > 1 or cutoff < 0:
        raise ValueError("Cutoff has to be in [0, 1]")

    s2 = texts.iloc[i]
    if method == "levenshtein":
        func = partial(Levenshtein.ratio, s2=s2, score_cutoff=cutoff)
    else:
        raise ValueError(
            f"Method '{method}' is unknown! Currently implemented: 'levenshtein'"
        )

    return [0] * i + texts.iloc[i:].map(func).tolist()


def _similarity_ratio_full(
    texts: pd.Series,
    method: str = "levenshtein",
    cutoff: float = 0.1,
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute a similarity (ratio) matrix.

    Parameters
    ----------
    texts : pd.Series[str]
        Pandas series of the strings to compute similarity ratios for.
    method : str, optional
        Similarity measure to use. Currently supported: 'levenshtein'.
        Defaults to "levenshtein".
    cutoff : float, optional
        Cutoff, can speed up computations, by default 0.1.
    n_jobs : int, optional
        Number of (multiprocessing) processes to use for computation, by default -1
        representing the number of CPU cores available.

    Returns
    -------
    np.ndarray
        (len(texts), len(texts)) shaped array of similarity ratios.
    """
    n_jobs = multiprocessing.cpu_count() if n_jobs < 1 else n_jobs
    pool = multiprocessing.Pool(n_jobs)
    func = partial(_similarity_ratio_row, texts=texts, method=method, cutoff=cutoff)
    results = np.array(pool.map(func, range(len(texts))))

    results = results + np.triu(results, k=1).T  # Mirror along the diagonal
    return results


def _lda_vis_html(texts: pd.Series, pipeline: Pipeline) -> None:
    """Compute the LDAvis visualization for given dataset-pipeline-combo and save it to
    "ldavis.html".

    Parameters
    ----------
    texts : pd.Series
        Texts used for training the pipeline. Can use whatever dtype the pipeline's
        vectorizer can transform.
    pipeline : Pipeline
        Topic modeling pipeline consisting of a "vectorizer" and a "topic model".
    """
    with warnings.catch_warnings():  # Suppress the many LDAvis warnings
        warnings.simplefilter("ignore", category=DeprecationWarning)
        import pyLDAvis
        import pyLDAvis.sklearn

    # Monkey patch to enable compatibility with sklearn 1.2 and upwards
    vectorizer = pipeline.named_steps["vectorizer"]
    if not hasattr(vectorizer, "get_feature_names"):
        vectorizer.get_feature_names = vectorizer.get_feature_names_out

    # Prepare the LDAvis inputs
    vectorized = vectorizer.transform(texts)
    topic_model = pipeline.named_steps["topic_model"]

    # Save the LDAvis visualization to an html file
    with warnings.catch_warnings():  # Suppress the many LDAvis warnings
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lda_vis = pyLDAvis.sklearn.prepare(topic_model, vectorized, vectorizer)
        pyLDAvis.save_html(lda_vis, "ldavis.html")


def export_for_indepth_analysis(
    name: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    H_array: np.ndarray,
    topic_names: list[str],
) -> None:
    """Save topic modeling results and some additional information to a file for later
    analysis.

    Exported file used the .indepth extension but is just a zip file containing:
    - term-topic-matrix W   (W.arrow)
    - topic-document-matrix H   (H.arrow)
    - dataset used to train the topic model   (texts.arrow)
    - TSNE embeddings of H   (tsne.arrow)
    - SBERT-like document embeddings   (embeddings.npy)
    - Three similarity matrices based on H_norm, TSNE embeddings, and
      Levenshtein-distance (similarities_tm.npy, similarities_embedding.npy, and
      similarities_levenshtein.npy)
    - LDAvis visualization (ldavis.html)

    Unsurprisingly, the resulting file can easily get quite large, depending on the
    training dataset and number of topics.

    Parameters
    ----------
    name : str
        Filename (without extension). File will be saved as in_depth/data/<name>.indepth
    df : pd.DataFrame
        Training dataset. Needs to have "id", "tokenized", and "text" columns.
    pipeline : Pipeline
        Topic modeling pipeline consisting of a "vectorizer" and a "topic model".
    H_array : np.ndarray
        Normalized topic-document-matrix H.
    topic_names : list[str]
        Names of the topics (manually assigned labels).
    """
    import pickle
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Create and save the term-topic-matrix W
    W = pd.concat(
        (
            get_topic_from_pipeline(pipeline, i, n_terms=-1)
            for i in range(len(topic_names))
        ),
        axis=1,
    )
    W.columns = pd.Index(topic_names)
    W = W.reset_index(names=["lemma"])
    W.to_feather("W.arrow")

    # Create and save the topic-document-matrix H
    H = pd.concat(
        (
            df["id"].astype(int).reset_index(drop=True),
            H_to_dataframe(H_array, topic_names, False),
        ),
        axis=1,
    )
    H.to_feather("H.arrow")

    # Normalize the H into topic distributions for each document
    H_norm = H.copy()
    topics = [c for c in H.columns if c != "id"]
    for topic in topics:
        H_norm[topic] = H[topic] / H[topics].sum(axis="columns")

    # Save the dataset used to train the topic model
    texts = df.reset_index(drop=True)
    texts.to_feather("texts.arrow")

    # Compute and save TSNE embeddings of the topic-document-matrix
    H_tsne = _tsne_embedding(H_norm, topics)
    H_tsne.to_feather("tsne.arrow")

    # Compute SBERT-like document embeddings and save them
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid strange warnings
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(texts["text"].to_list())
    np.save("embeddings.npy", embeddings)

    # Compute three similarity matrices and save them
    H_norm_arr = H_norm.drop(columns="id").values
    similarities_tm = cos_sim(H_norm_arr, H_norm_arr)
    similarities_embedding = cos_sim(embeddings, embeddings)
    similarities_levenshtein = _similarity_ratio_full(
        texts["text"], method="levenshtein"
    )
    np.save("similarities_tm.npy", similarities_tm.astype(np.float16))
    np.save("similarities_embedding.npy", similarities_embedding.astype(np.float16))
    np.save("similarities_levenshtein.npy", similarities_levenshtein.astype(np.float16))

    # Compute and save the LDAvis visualization
    _lda_vis_html(df["tokenized"], pipeline)

    with open("pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Write all files into a combined zip file and then delete the component files
    files = (
        ["W.arrow", "H.arrow", "texts.arrow", "tsne.arrow", "embeddings.npy"]
        + ["similarities_tm.npy", "similarities_embedding.npy"]
        + ["similarities_levenshtein.npy", "ldavis.html", "pipeline.pkl"]
    )
    with ZipFile(f"{name}.indepth", "w") as zip:
        for file in files:
            zip.write(file)
            os.remove(file)


def duplicates(texts: pd.DataFrame, cutoff: float = 0.5) -> pd.DataFrame:
    """Identify duplicates in given dataframe based on a Levenshtein-distance based
    similarity measure.

    Parameters
    ----------
    texts : pd.DataFrame
        Dataframe containing a "text" column
    cutoff : float, optional
        How similar string have to be to be considered duplicates, by default 0.5.

    Returns
    -------
    pd.DataFrame
        Dataframe containing "i", "j", and "similarity", as well as the respective
        rows of the texts dataframe, suffixed with "_i" and "_j".
    """
    # Compute similarity based on Levenshtein distance
    sims = _similarity_ratio_full(texts["text"], method="levenshtein")

    # Identify duplicates and transform them into a data frame
    duplicates = []
    for i, j in itertools.product(range(sims.shape[0]), range(sims.shape[0])):
        if (
            j > i  # Capture each (candidate) duplication only once
            and sims[i, j] > cutoff  # Similarity above threshold
            and texts.iloc[i]["id"] != texts.iloc[j]["id"]  # Different users
            and min(len(texts.iloc[i]["text"]), len(texts.iloc[j]["text"]))
            > 500  # Non-trivial
        ):
            duplicates.append(dict(i=i, j=j, similarity=sims[i, j]))
    duplicates_df = pd.DataFrame(duplicates)

    # Merge duplicates with the data frame they originate from
    df = pd.merge(duplicates_df, texts, left_on="i", right_index=True)
    df = pd.merge(df, texts, left_on="j", right_index=True, suffixes=("_i", "_j"))
    return df


_user_type_map = {
    "company": "Company",
    "business_association": "Business Association",
    "academic_research_institution": "Academia",
    "ngo": "NGO",
    "trade_union": "Trade Union",
    "public_authority": "Public Authority",
    "consumer_organisation": "Consumer Organisation",
    "eu_citizen": "EU Citizen",
    "standardizing_body": "Standardizing Body",
}
_user_type_order = [
    "company",
    "business_association",
    "ngo",
    "trade_union",
    "public_authority",
    "consumer_organisation",
    "academic_research_institution",
    "eu_citizen",
    "standardizing_body",
]


_dataset_duplicates_page_false_positives = [
    (536, 1821),
    (450, 536),
    (450, 1821),
    (1821, 2465),
    (536, 2465),
    (982, 1821),
    (382, 1821),
    (536, 982),
    (450, 2465),
    (450, 1907),
    (982, 2465),
    (382, 536),
    (536, 1907),
    (1555, 1793),
    (382, 450),
    (450, 982),
    (1821, 1907),
    (536, 1365),
    (382, 982),
    (450, 1365),
    (1365, 1821),
    (536, 2061),
    (450, 2061),
    (450, 1005),
    (382, 2465),
    (1365, 1907),
    (1005, 1907),
    (536, 1005),
    (1365, 2465),
    (1907, 2465),
    (2061, 2465),
    (1907, 2423),
    (1821, 2061),
    (1907, 2061),
    (1005, 1821),
    (982, 1907),
    (1365, 2061),
    (536, 1471),
    (1471, 1907),
    (450, 1471),
    (982, 1005),
    (982, 2061),
    (382, 1005),
    (1005, 1365),
    (382, 2061),
    (982, 1365),
    (382, 1907),
    (1793, 1907),
    (382, 1365),
    (300, 1939),
    (304, 497),
    (366, 1284),
    (980, 1248),
    (996, 2057),
    (905, 2057),
    (370, 2058),
    (366, 460),
    (334, 1290),
    (980, 1951),
    (298, 496),
    (562, 1289),
    (1645, 2164),
    (369, 831),
    (475, 2251),
    (1288, 1790),
    (314, 2057),
    (335, 1290),
    (1248, 1951),
]

_dataset_duplicates_page_true_positives = [
    (2665649, 2665234),
    (2663486, 2662780),
    (2665205, 2663486),
    (2665205, 2662780),
    (2665563, 2665479),
    (2665574, 2665497),
    (2665227, 2663395),
    (2663356, 2663263),
    (2665628, 2665293),
    (2663486, 2661971),
    (2662780, 2661971),
    (2665205, 2661971),
    (2665515, 2665168),
    (2665168, 2663339),
    (2665515, 2663339),
]
