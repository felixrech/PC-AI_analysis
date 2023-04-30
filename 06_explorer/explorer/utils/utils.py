"""Module containing general utility functions."""


import os
import sys
import json
import pickle
import urllib
import numpy as np
import pandas as pd
from typing import IO
import itertools as it
from pathlib import Path
from sklearn import metrics
from zipfile import ZipFile
from sklearn.pipeline import Pipeline

from utils.caching import memoize


DEFAULT_TM = "nmf_page"


def load_tm(tm: str = DEFAULT_TM) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a topic model (topic-document matrix H, term-topic matrix W, and training
    data) from a .indepth file.

    - The H dataframe has one row for each document and "id" and the topic names as
      columns.
    - The W dataframe has one row for each lemma and "lemma" and the topic names as
      columns.

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Topic-document matrix H, term-topic matrix W, and training data dataframes.
    """
    H: pd.DataFrame = _load_pandas(tm, "H")
    W: pd.DataFrame = _load_pandas(tm, "W")
    texts: pd.DataFrame = _load_pandas(tm, "texts")

    # Add internal index and the PDF page text is found on
    texts = texts.reset_index(names="internal_index")
    texts["page"] = texts.groupby("id").cumcount()
    return H, W, texts


def load_tsne(tm: str = DEFAULT_TM) -> pd.DataFrame:
    """Load a TSNE embedding of the topic-document matrix H.

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    pd.DataFrame
        TSNE embedding with "PC1" and "PC2" dataframe.
    """
    return _load_pandas(tm, "tsne")


def load_sbert_embeddings(tm: str = DEFAULT_TM) -> np.ndarray:
    """Load SBERT embeddings of each document.

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    np.ndarray
        (n_documents, 768)-dimensional array of SBERT embeddings.
    """
    return _load_numpy(tm, "embeddings")


def load_similarities(tm: str = DEFAULT_TM) -> dict[str, np.ndarray]:
    """Load the three types of similarity matrices.

    - Cosine similarity based on unit-normed topic distribution vectors of each document
      from the topic-document matrix H.
    - Cosine similarity of the SBERT embeddings of each document.
    - Levenshtein-distance based similarity between documents, defined as:
        sim(s_1, s_2) = 1 - lev(s_1, s_2) / (|s_1| + |s_2|)

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys "similarities_tm", "similarities_embedding", and
        "similarities_levenshtein" and values being the numpy arrays of the similarity
        matrices explained above.
    """
    return dict(
        similarities_tm=_load_numpy(tm, "similarities_tm"),
        similarities_embedding=_load_numpy(tm, "similarities_embedding"),
        similarities_levenshtein=_load_numpy(tm, "similarities_levenshtein"),
    )


class UnzippedFile:
    """Context manager providing access to a file within a zip archive."""

    def __init__(self, tm: str, file: str) -> None:
        """Initialize a new context manager.

        Parameters
        ----------
        tm : str
            File name of the .indepth file without the file extension and relative to
            the data folder.
        file : str
            Filename of file to extract from archive.
        """
        self.tm, self.file = tm, file
        self.file_path = Path(__file__).parents[1] / f"data/{self.tm}.indepth"

    def __enter__(self) -> IO[bytes]:
        """Return the extracted file's IO.

        Returns
        -------
        IO[bytes]
            Extracted file's IO.
        """
        self.zip = ZipFile(self.file_path)
        self.file_io = self.zip.open(self.file)
        return self.file_io

    def __exit__(self, *_) -> None:
        """Close the file and zip archive descriptors."""
        self.file_io.close()
        self.zip.close()


def _load_pandas(tm: str, file: str) -> pd.DataFrame:
    """Load a dataframe saved to a zip archive (.indepth file).

    Parameters
    ----------
    tm : str
        File name of the .indepth file without the file extension and relative to the
        data folder.
    file : str
        Filename of file to extract from archive (excluding the file extension .arrow).

    Returns
    -------
    pd.DataFrame
        Extracted dataframe.
    """
    with UnzippedFile(tm, f"{file}.arrow") as f:
        return pd.read_feather(f)


def _load_numpy(tm: str, file: str) -> np.ndarray:
    """Load a numpy array saved to a zip archive (.indepth file).

    Parameters
    ----------
    tm : str
        File name of the .indepth file without the file extension and relative to the
        data folder.
    file : str
        Filename of file to extract from archive (excluding the file extension .npy).

    Returns
    -------
    pd.DataFrame
        Extracted dataframe.
    """
    with UnzippedFile(tm, f"{file}.npy") as f:
        return np.load(f)  # type: ignore


def load_ldavis(tm: str = DEFAULT_TM) -> str:
    """Load an LDAvis analysis saved to a zip archive (.indepth file).

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    str
        Extracted LDAvis html.
    """
    with UnzippedFile(tm, "ldavis.html") as f:
        return f.read().decode("utf-8")


def load_pipeline(tm: str = DEFAULT_TM) -> Pipeline:
    """Load the vectorizer-topic model pipeline from a .indepth file.

    Parameters
    ----------
    tm : str, optional
        File name of the .indepth file without the file extension and relative to the
        data folder, by default DEFAULT_TM.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline object.
    """
    # Allow for import of modules from 05_topic_modeling folder
    module_path = os.path.abspath(os.path.join("../../05_topic_modeling/"))
    if module_path not in sys.path:
        sys.path.append(module_path)
    import topic_models  # type: ignore[import] # noqa[F401]

    # Loading the pipeline requires the topic_models module to be loaded
    # from utils import topic_models as topic_models  # noqa[F401]

    with UnzippedFile(tm, "pipeline.pkl") as f:
        return pickle.load(f)


def get_news() -> list[dict[str, str]]:
    """Read the list of news from assets/notifications.json.

    Returns
    -------
    list[dict[str, str]]
        List of dicts (with keys "id" and "msg").
    """
    with open("assets/notifications.json", "r") as f:
        news: list[dict[str, str]] = json.loads(f.read())
    return news


def normalize_H(H: pd.DataFrame) -> pd.DataFrame:
    """Normalize the topic-document matrix H into proper topic distributions for each
    document.

    Parameters
    ----------
    H : pd.DataFrame
        Topic-document matrix H.

    Returns
    -------
    pd.DataFrame
        (Row) normalized topic-document matrix H.
    """
    H_norm = H.copy()
    topics = get_topics(H)

    for topic in topics:
        H_norm[topic] = H[topic] / H[topics].sum(axis="columns")
    return H_norm


def normalize_W(W: pd.DataFrame) -> pd.DataFrame:
    """Normalize the term-topic matrix W into proper token distributions for each topic.

    Parameters
    ----------
    W : pd.DataFrame
        Term-topic matrix W

    Returns
    -------
    pd.DataFrame
        (Column) normalized term-topic matrix W.
    """
    W_norm = W.copy()
    topics = get_topics(W)

    W_norm[topics] = W[topics] / W[topics].sum(axis="index").T
    return W_norm


def get_topics(H: pd.DataFrame) -> list[str]:
    """Extract the list of topic labels from a (normalized or unnormalized)
    topic-document matrix H or term-topic matrix W.

    Simply returns the column names as a list after discarding the "id" and "lemma"
    columns, respectively.

    Parameters
    ----------
    H : pd.DataFrame
        Topic-document matrix H (or term-topic matrix W)

    Returns
    -------
    list[str]
        List of topic labels.
    """
    return [str(topic) for topic in H.columns if topic != "id" and topic != "lemma"]


def document_dominant_topic(H: pd.DataFrame) -> pd.DataFrame:
    """Computes each document's dominant topic based on the (normalized or unnormalized)
    topic-document matrix H.

    Parameters
    ----------
    H : pd.DataFrame
        Topic-document matrix H.

    Returns
    -------
    pd.DataFrame
        Dataframe with "internal_id", "dominant_topic", and "value" columns. The
        internal_id column matches the document's position in the input dataframe and
        the share the topic share of the dominant topic (not automatically normalized if
        the input is unnormalized).
    """
    dominant_topic: pd.DataFrame = (
        H.reset_index(names="internal_id")  # type: ignore
        .melt(
            id_vars="internal_id", value_vars=get_topics(H), var_name="dominant_topic"
        )
        .groupby("internal_id")
        .apply(lambda df: df[df["value"] == df["value"].max()].iloc[0])  # type: ignore
    )
    return dominant_topic.reset_index(drop=True)


def current_is_dominant_topic(
    current: pd.Series, H: pd.DataFrame, topic_selected: str
) -> bool:
    """Compute whether the currently selected document has topic_selected as its
    dominant topic, i.e. has the highest topic share.

    Parameters
    ----------
    current : pd.Series
        Information about currently selected document with "internal_index" and
        topic_selected labels.
    H : pd.DataFrame
        Normalized or unnormalized topic-document matrix H.
    topic_selected : str
        Currently selected topic, i.e. a column name of H.

    Returns
    -------
    bool
        Whether topic_selected is the dominant topic of current.
    """
    topics = get_topics(H)

    return bool(
        current[topic_selected] >= H.iloc[current["internal_index"]][topics].max()  # type: ignore # noqa: E501
    )


def tm_similarity_matrix(
    W_norm_main: pd.DataFrame,
    W_norm_other: pd.DataFrame,
    normalize: bool = False,
) -> tuple[np.ndarray, tuple[list[str], list[str]]]:
    """Compute a matrix of similarities between topics of two topic models based on the
    Hellinger distance between columns of the normalized term-topic matrices.

    Parameters
    ----------
    W_norm_main : pd.DataFrame
        Normalized term-topic matrix W of the currently active topic model.
    W_norm_other : pd.DataFrame
        Normalized term-topic matrix W of the topic model to compare to.
    normalize : bool, optional
        Whether to normalize similarities using min-max scaling, by default False.
        (Even without this option, similarities will lie in [0, 1]. But e.g. no two
        topics might actually have similarity 1.)

    Returns
    -------
    tuple[np.ndarray, tuple[list[str], list[str]]]
        Numpy array of similarities and tuple containing list of associated topics. The
        first (active TM) labels the rows, the second (compare TM) labels the columns.
    """
    W_main = W_norm_main.drop(columns="lemma").values.T
    W_other = W_norm_other.drop(columns="lemma").values.T
    topics_main, topics_other = get_topics(W_norm_main), get_topics(W_norm_other)

    diff = np.zeros((W_main.shape[0], W_other.shape[0]))
    for topic in np.ndindex(diff.shape):  # type: ignore
        diff[topic] = hellinger(W_main[topic[0]], W_other[topic[1]])

    if normalize:
        similarity = 1 - (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    else:
        similarity = 1 - diff

    return similarity, (topics_main, topics_other)


def tm_similarity_matrix_df(
    W_norm_main: pd.DataFrame,
    W_norm_other: pd.DataFrame,
    normalize: bool = False,
) -> pd.DataFrame:
    """Compute a matrix of similarities between topics of two topic models based on the
    Hellinger distance between columns of the normalized term-topic matrices.

    Parameters
    ----------
    W_norm_main : pd.DataFrame
        Normalized term-topic matrix W of the currently active topic model.
    W_norm_other : pd.DataFrame
        Normalized term-topic matrix W of the topic model to compare to.
    normalize : bool, optional
        Whether to normalize similarities using min-max scaling, by default False.
        (Even without this option, similarities will lie in [0, 1]. But e.g. no two
        topics might actually have similarity 1.)

    Returns
    -------
    pd.DataFrame
        Dataframe of similarities with rows representing the topics of the active topic
        model and the columns representing the topics of the compare topic model.
    """
    similarity, (topics_main, topics_other) = tm_similarity_matrix(
        W_norm_main, W_norm_other, normalize
    )
    return pd.DataFrame(similarity, index=topics_main, columns=topics_other)


def tm_comparison_terms(
    W_norm_main: pd.DataFrame,
    W_norm_other: pd.DataFrame,
    n_terms: int = 20,
    normalize: str = "n_terms",
) -> pd.DataFrame:
    """Compute a matrix of similarities between topics of two topic models based on the
    number of terms that are present in both topics.

    Parameters
    ----------
    W_norm_main : pd.DataFrame
        Normalized term-topic matrix W of the currently active topic model.
    W_norm_other : pd.DataFrame
        Normalized term-topic matrix W of the topic model to compare to.
    n_terms : int, optional
        The n_terms terms with highest topic-specific probability will be used.
    normalize : str, optional
        How to normalize the similarities to [0, 1]. The default, "n_terms", divides by
        maximum number of overlapping terms, while "total_terms" divides by the number
        of terms in either topic.

    Returns
    -------
    pd.DataFrame
        Dataframe of similarities with rows representing the topics of the active topic
        model and the columns representing the topics of the compare topic model.
    """
    topics_main, topics_other = get_topics(W_norm_main), get_topics(W_norm_other)

    df_main, df_other = (
        (
            df.melt(id_vars="lemma", var_name="topic", value_name="share")
            .groupby("topic")
            .apply(lambda df: df.sort_values("share", ascending=False).head(n_terms))  # type: ignore # noqa: E501
            .reset_index(drop=True)
        )
        for df in [W_norm_main, W_norm_other]
    )

    # Find overlap between topic terms
    df = pd.merge(df_main, df_other, how="outer", on=["lemma"], indicator=True)
    df = (
        df.groupby(["topic_x", "topic_y"])
        .apply(lambda df: len(df.query("_merge == 'both'")))
        .reset_index()
        .rename(columns={0: "n_common"})
    )

    # Add default 0 overlap for topics without overlap
    defaults = pd.DataFrame(
        it.product(topics_main, topics_other, [0]), columns=df.columns
    )
    df = pd.concat((df, defaults)).drop_duplicates(["topic_x", "topic_y"])

    # Normalize number of terms into [0,1]-similarity
    if normalize == "total_terms":
        df["total"] = df.apply(
            lambda row: len(
                set(df_main[df_main["topic"] == row["topic_x"]]["lemma"])
                | set(df_other[df_other["topic"] == row["topic_y"]]["lemma"])
            ),
            axis="columns",
        )
    elif normalize == "n_terms":
        df["total"] = n_terms
    else:
        raise ValueError(
            f"Normalization method '{normalize}' unknown! "
            "Please use 'total_terms' or 'n_terms'."
        )

    df["similarity"] = df["n_common"] / df["total"]

    return df


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Hellinger distance between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector.

    Returns
    -------
    float
        Hellinger distance.
    """
    return 1 / np.sqrt(2) * np.sqrt(np.sum(np.square(np.sqrt(a) - np.sqrt(b))))  # type: ignore # noqa: E501


def most_similar_docs(
    similarities_dict: dict[str, np.ndarray],
    current_index: int,
    method: str = "levenshtein",
    top_n: int = 30,
) -> pd.DataFrame:
    """Computes the top_n most similar documents according to chosen similarity measure.

    Parameters
    ----------
    similarities_dict : dict[str, np.ndarray]
        Similarity measures as returned by load_similarities().
    current_index : int
        internal_index of the currently selected document.
    method : str, optional
        Similarity measure to use, by default "levenshtein".
    top_n : int, optional
        How many documents to return, by default 30.

    Returns
    -------
    pd.DataFrame
        Most similar documents as dataframe with "internal_index" and "similarity"
        columns.
    """
    similarities = similarities_dict[f"similarities_{method}"]
    indices = np.argsort(similarities[current_index, :])[::-1]
    used_similarities = [similarities[current_index, i] for i in indices]

    df = pd.DataFrame(
        zip(indices, used_similarities), columns=["internal_index", "similarity"]
    )
    df = (
        df.query("internal_index != @current_index")
        .sort_values(["similarity", "internal_index"], ascending=[False, True])
        .iloc[:top_n]
    )
    return df


def cutoff_text(text: str, max_length: int) -> str:
    """Shorten a string to given maximum length. If longer cuts off at max_length and
    adds "…" to denote the cut.

    Parameters
    ----------
    text : str
        Arbitrary string.
    max_length : int
        Maximum length.

    Returns
    -------
    str
        String of maximum length max_length.
    """
    return text if len(text) < max_length else text[: max_length - 1] + "…"


def hys_url(id: str | int) -> str:
    """Compute the 'Have your Say' URL from a given feedback id.

    Parameters
    ----------
    id : str | int
        Feedback id, e.g. "2256824".

    Returns
    -------
    str
        Feedback URL.
    """
    url = (
        "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/"
        "12527-Artificial-intelligence-ethical-and-legal-requirements/F{}_en"
    )
    return url.format(id)


def local_attachment_exists(id: str | int) -> bool:
    """Checks whether a feedback's attachment is available locally (in the
    assets/attachments/ folder).

    Parameters
    ----------
    id : str | int
        Feedback id, e.g. "2256824".

    Returns
    -------
    bool
        Whether the feedback's attachment is available locally.
    """
    attachments_loc = "assets/attachments/"
    c = os.path.exists(attachments_loc) and f"{id}.pdf" in os.listdir(attachments_loc)
    return c


def local_attachment_href(
    id: str | int, page: int = 0, reminder: str | None = None
) -> str:
    """Compute the href to display the local attachment for a feedback.

    Parameters
    ----------
    id : str | int
        Feedback id, e.g. "2256824".
    page : int, optional
        Page of attachment to display, by default 0, i.e. no page in particular.
    reminder : str | None, optional
        Short len(reminder) <= 3 string to display at the lower right corner, by default
        None, i.e. do not display anything at all.

    Returns
    -------
    str
        Relative href.
    """
    href = f"/assets/attachments/{id}.pdf"
    href += f"#page={page}" if page > 0 else ""

    search = urllib.parse.urlencode(  # type: ignore
        dict(href=href) | (dict() if reminder is None else dict(reminder=reminder))
    )
    return f"/pdf/?{search}"


def user_type_to_name(user_type: str) -> str:
    """Transform a user type from its snake case form to full string.

    Parameters
    ----------
    user_type : str
        User type in snake case, e.g. "eu_citizen".

    Returns
    -------
    str
        User type as a full string, e.g. "EU Citizen".
    """
    user_type_map = {
        "company": "Company",
        "business_association": "Business Association",
        "academic_research_institution": "Academia",
        "ngo": "NGO",
        "trade_union": "Trade Union",
        "public_authority": "Public Authority",
        "consumer_organisation": "Consumer Organization",
        "eu_citizen": "EU Citizen",
        "standardizing_body": "Standardizing Body",
    }

    return user_type_map[user_type]


def user_type_to_abbreviation(user_type: str) -> str:
    """Transform a user type from its snake case form to a short abbreviation.

    Parameters
    ----------
    user_type : str
        User type in snake case, e.g. "eu_citizen".

    Returns
    -------
    str
        User type abbreviation, e.g. "Cit".
    """
    user_type_map = {
        "company": "C",
        "business_association": "B",
        "academic_research_institution": "A",
        "ngo": "N",
        "trade_union": "TU",
        "public_authority": "PA",
        "consumer_organisation": "CO",
        "eu_citizen": "Cit",
        "standardizing_body": "SB",
    }

    return user_type_map[user_type]


def user_type_to_interest(user_type: str) -> str:
    """Transform a user type from its snake case form to its interest group.

    Parameters
    ----------
    user_type : str
        User type in snake case, e.g. "business_association".

    Returns
    -------
    str
        User type interest group, e.g. "corporate".
    """
    user_type_map = {
        "company": "corporate",
        "business_association": "corporate",
        "ngo": "public",
        "trade_union": "public",
        "consumer_organisation": "public",
        "eu_citizen": "public",
        "public_authority": "public",
        "academic_research_institution": "other",
        "standardizing_body": "other",
    }

    return user_type_map[user_type]


def interest_to_user_types(interest: str) -> list[str]:
    """Reverse map an interest to the list of its user types.

    Parameters
    ----------
    interest : str
        Interest, i.e. "corporate", "public", or "other".

    Returns
    -------
    list[str]
        List of user types.
    """
    interest_map = {
        "corporate": ["company", "business_association"],
        "public": "ngo;trade_union;consumer_organisation;public_authority;"
        "eu_citizen".split(";"),
        "other": ["academic_research_institution", "standardizing_body"],
    }

    return interest_map[interest]


def user_types_present(user_types: pd.Series) -> pd.DataFrame:
    """Compute a dataframe of user types that are in input, ordered like
    utils.user_type_order.

    Parameters
    ----------
    user_types : pd.Series
        Series of user types like "eu_citizen".

    Returns
    -------
    pd.DataFrame
        Dataframe with "user_type" and "user_type_name" columns, ordered like
        utils.user_type_order.
    """
    user_types_df = (
        user_types.to_frame("user_type")
        .drop_duplicates()
        .copy()
        .assign(user_type_name=user_types.map(user_type_to_name))
    )
    user_type_names = pd.Series(user_type_order).to_frame("user_type_name")
    return pd.merge(user_type_names, user_types_df)


@memoize
def rand_index(main: str, other: str) -> float:
    """Compute rand index between the clusterings induced by document-dominant topic
    assignments.

    Parameters
    ----------
    main : str
        File name of the active topic model's .indepth file without the file extension
        and relative to the data folder.
    other : str
        File name of the compared topic model's .indepth file without the file extension
        and relative to the data folder.

    Returns
    -------
    float
        Rand index R.
    """
    H_main, _, _ = load_tm(main)
    H_other, _, _ = load_tm(other)

    H_main = normalize_H(H_main)
    H_other = normalize_H(H_other)

    df = pd.merge(
        document_dominant_topic(H_main),
        document_dominant_topic(H_other),
        on="internal_id",
        suffixes=("_main", "_other"),
    )

    return float(
        metrics.rand_score(df["dominant_topic_main"], df["dominant_topic_other"])
    )


# fmt: off
user_type_order = [
    "Company", "Business Association", "NGO", "Trade Union", "Public Authority",
    "Consumer Organization", "Academia", "EU Citizen", "Standardizing Body"
]  # fmt: on
