"""Module with utility functions for dealing with aspects and sentiment."""

import re
import warnings
import textstat
import numpy as np
import pandas as pd
from typing import Any
import itertools as it
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from utils import utils
from utils.caching import memoize


def load_sentiment(
    aspect: str = "all",
    filter_: bool = True,
    aspect_type: str = "",
    aspect_subtype: str = "",
    sort: str = "",
    filter_user_types: list[str] = [],
) -> pd.DataFrame:
    """Load the sentiment from the dataset. Has to be saved as data/sentiments.arrow.

    Parameters
    ----------
    aspect : str, optional
        Whether to filter to "article" aspect_type, to any of the "others", or not
        filter and use "all" mentions, instead.
    filter_ : bool, optional
        Whether to remove any rows without predicted sentiment, by default True.
    aspect_type : str, optional
        Which aspect type (aspect_type column) to filter to, by default "", i.e. no
        filtering.
    aspect_subtype : str, optional
        Which aspect subtype (aspect_subtype column) to filter to, by default "", i.e.
        no filtering.
    sort : str, optional
        How to sort the dataset. By default "", i.e. sort by "sentence_index" column.
        Other options are "positive" - by ascending positive sentiment probability
        ("positive" column), "negative" - by ascending negative sentiment probability
        ("negative" column), or "extreme" by ascending maximum of positive and negative
        sentiment probability.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.

    Returns
    -------
    pd.DataFrame
        Sentiment dataset as dataframe.
    """
    sentiments = pd.read_feather("data/sentiments.arrow")

    # Remove non-sentiment rows (based on "filter_" argument)
    if filter_:
        sentiments = sentiments.query("sentiment.notnull()")

    # General aspect filtering (based on "aspect" argument)
    if aspect == "article":
        sentiments = sentiments.query("aspect_type == 'article'")
    elif aspect == "others":
        sentiments = sentiments.query("aspect_type != 'article'")
    elif aspect == "all":
        pass
    else:
        raise ValueError(
            f'Value "{aspect}" of aspect argument unknown! '
            'Use "article", "others", or "all".'
        )

    # Specific aspect filtering (based on "aspect_type" and "aspect_subtype" arguments)
    if aspect_type != "":
        sentiments = sentiments.query("aspect_type == @aspect_type")
    if aspect_subtype != "":
        sentiments = sentiments.query("aspect_subtype == @aspect_subtype")

    # Sorting (based on "sort" argument)
    if sort == "":
        sentiments = sentiments.sort_values("sentence_index", ascending=True)
    elif sort == "positive":
        sentiments = sentiments.sort_values("positive", ascending=False)
    elif sort == "negative":
        sentiments = sentiments.sort_values("negative", ascending=False)
    elif sort == "extreme":
        sentiments = (
            sentiments.assign(
                extreme=lambda df: df[["positive", "negative"]].max(axis="columns")
            )
            .sort_values("extreme", ascending=False)
            .drop(columns="extreme")
        )
    elif sort == "aspect":
        sentiments = sort_by_aspect(sentiments)
    else:
        raise ValueError(
            f"Sort value '{sort}' unknown! See the docstring for possible values."
        )

    # Filter by user type (based on "filter_user_types" argument)
    if len(filter_user_types) > 0:
        sentiments = (
            sentiments.assign(
                user_type_name=sentiments["user_type"].map(utils.user_type_to_name)
            )
            .query("user_type_name.isin(@filter_user_types)")
            .reset_index(drop=True)
        )

    return sentiments.reset_index(drop=True)


@memoize
def search_sentiment(
    search_term: str, regex: bool, filter_user_types: tuple = tuple()
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex : bool
        Whether the search term uses regular expressions.
    filter_user_types : tuple[str], optional
        Which user types ("user_type" column) to filter to, by default (), i.e. no
        filtering.

    Returns
    -------
    pd.DataFrame
        Filtered sentiment dataset as a dataframe. Removes sentiment related columns but
        keeps e.g. "aspect_type" and "aspect_subtype". Adds "result" column containing
        exact search term match.
    """
    # Load complete dataset
    all_sentiments = load_sentiment(
        filter_=False, filter_user_types=list(filter_user_types)
    )

    # Remove some unnecessary columns from dataset
    drop_columns = (
        "sentence_before;aspect;sentence_after;sentiment_string;negative;neutral;"
        "positive;sentiment"
    ).split(";")
    df = all_sentiments.drop(columns=drop_columns)

    # Exact match search is faster - filter w/o using regex first
    if not regex:
        df = df.query(f"text.str.lower().str.contains('{search_term}')")
        for r, s in [("\\", "\\\\")] + [(s, "\\" + s) for s in "^$.|"]:
            search_term = search_term.replace(r, s)

    # Use regex to find matches and sentence before and after
    return (
        df.assign(
            extract=df["text"].str.findall(f"(.*)({search_term})(.*)"),
            flags=re.IGNORECASE,
        )
        .query("extract.str.len() > 0")
        .explode("extract")
        .assign(
            sentence_before=lambda df: df["extract"]
            .map(lambda x: x[0] if type(x) is tuple else None)
            .fillna(""),
            result=lambda df: df["extract"].map(
                lambda x: x[1] if type(x) is tuple else None
            ),
            sentence_after=lambda df: df["extract"]
            .map(lambda x: x[2] if type(x) is tuple else None)
            .fillna(""),
        )
        .drop(columns=["extract"])
    )


def sort_by_aspect(sentiments: pd.DataFrame) -> pd.DataFrame:
    """Sorts a dataframe by aspect_type and aspect_subtypes columns in a way that makes
    sense to humans (e.g. "i" < "2" < "10").

    Parameters
    ----------
    sentiments : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.

    Returns
    -------
    pd.DataFrame
        Sorted sentiments dataset.
    """
    type_map = {
        None: -1,
        "ai_act": 1,
        "eaib": 2,
        "title": 3,
        "annex": 4,
        "article": 5,
    }
    subtype_map = {None: -1, "": -1, "i": 1, "ii": 2, "iii": 3, "iv": 4}

    return (
        sentiments.copy()
        .assign(
            aspect_type_numerical=sentiments["aspect_type"].map(type_map),
            aspect_subtype_numerical=sentiments["aspect_subtype"].map(
                lambda x: subtype_map[x] if x in subtype_map.keys() else int(x)
            ),
        )
        .sort_values(["aspect_type_numerical", "aspect_subtype_numerical"])
        .drop(columns=["aspect_type_numerical", "aspect_subtype_numerical"])
    )


def aggregate_mentions_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Computes various summary statistics after aggregating the sentiments dataset by
    aspect and user type interest:

    - The "user_type_interest" column contains the user type interest group, e.g.
        "public".
    - The "details" column containing a string with the number of mentions for each user
        type of each row's interest.
    - The "n_mentions" column contains the number of mentions of the row's aspect by the
        row's user type interest group.
    - The "total_mentions" column contains the total number of mentions of each row's
        aspect by any interest.
    - The "mentions_share" column is the quotient of "n_mentions" and "total_mentions".

    Parameters
    ----------
    df : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.

    Returns
    -------
    pd.DataFrame
        Dataframe with "aspect_type", "aspect_subtype", "user_type_interest", "details",
        "n_mentions", "total_mentions", "mentions_share" columns.
    """
    aspect = ["aspect_type", "aspect_subtype"]

    # Aggregate mentions of each aspect by user type
    tmp = (
        df.groupby(aspect + ["user_type"])
        .agg("size")
        .reset_index()
        .rename(columns={0: "n"})
    )

    tmp["user_type_interest"] = tmp["user_type"].map(utils.user_type_to_interest)

    def add_details_string(df: pd.DataFrame) -> pd.DataFrame:
        """Add a string of details about the number of mentions by user types within
        each interest type.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with "user_type" and "n" columns. Should be applied to a dataframe
            with only a single interest type.

        Returns
        -------
        pd.DataFrame
            Dataframe with "details" and "n_mentions" columns, where the latter sums
            up the "n" column along all rows.
        """
        details_string = _user_types_details_string(
            example_user_type=df["user_type"].iloc[0], existing=df
        )
        return pd.DataFrame(
            [[details_string, df["n"].sum()]], columns=["details", "n_mentions"]
        )

    # Compute details string for aggregated results
    tmp = (
        tmp.groupby(aspect + ["user_type_interest"])[["user_type", "n"]]
        .apply(add_details_string)
        .reset_index()
        .drop(columns="level_3")
    )

    # Create default values for combinations not present in dataset
    defaults = pd.merge(
        tmp[["aspect_type", "aspect_subtype"]].drop_duplicates(),
        tmp["user_type_interest"].drop_duplicates(),
        how="cross",
    ).assign(
        details=lambda df: df["user_type_interest"].map(
            lambda x: _user_types_details_string(interest=x)
        ),
        n_mentions=0,
    )

    # Combine default values with remaining data
    output = (
        pd.concat((tmp, defaults))
        .drop_duplicates(subset=aspect + ["user_type_interest"], keep="first")
        .sort_values(aspect + ["user_type_interest"])
    )

    # Add relative share of each interest type, too
    output["total_mentions"] = output.groupby(aspect)["n_mentions"].transform("sum")
    output["mentions_share"] = output["n_mentions"] / output["total_mentions"]
    return sort_by_aspect(output)


def aggregate_sentiments(
    df: pd.DataFrame, probs: bool, separate_interest_types: bool = False
) -> pd.DataFrame:
    """For each aspect, averages positive, neutral, and negative sentiment probabilities
    and computes the support (number of observations) for each aspect.

    Parameters
    ----------
    df : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.
    probs : bool
        Whether to use average predicted probabilities (True) or fraction of predicted
        sentiment (False).
    separate_interest_types : bool, optional
        Whether to separate results by interest type, by default False. Adds the
        "interest_type" column to the output dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with "aspect_type", "aspect_subtype", "sentiment", "share", and
        "support" columns.
    """
    if separate_interest_types:
        keys = ["aspect_type", "aspect_subtype", "interest_type"]
    else:
        keys = ["aspect_type", "aspect_subtype"]

    if probs:
        return sort_by_aspect(
            df.groupby(keys)
            .agg(
                {
                    "positive": "mean",
                    "neutral": "mean",
                    "negative": "mean",
                    "sentiment_string": "size",
                }
            )
            .rename(columns={"sentiment_string": "support"})
            .query("support >= 30")
            .reset_index()
            .melt(
                id_vars=keys + ["support"],
                var_name="sentiment",
                value_name="share",
            )
        )
    else:
        return sort_by_aspect(
            pd.concat(
                (
                    (
                        df.groupby(keys)["sentiment"]
                        .value_counts()
                        .to_frame("share")
                        .reset_index()
                    ),
                    (
                        pd.merge(
                            df[keys],
                            pd.DataFrame(
                                dict(sentiment=["positive", "neutral", "negative"])
                            ),
                            how="cross",
                        ).assign(share=0)
                    ),
                )
            )
            .drop_duplicates(subset=keys + ["sentiment"])
            .assign(
                support=lambda df: df.groupby(keys)["share"].transform("sum"),
                share=lambda df: df["share"] / df["support"],
            )
        )


def aspect_mentions_share() -> pd.DataFrame:
    """Compute the share of sentences containing mentions of articles or annexes among
    all sentences.

    Returns
    -------
    pd.DataFrame
        Dataframe containing "user_type", "user_type_name", "interest_type",
        "sentences", "mentions", "share", and "interest_share" share.
    """
    # Count total number of sentences
    df = (
        load_sentiment(filter_=False)
        .drop_duplicates(["sentence_index"])
        .groupby("user_type")["sentence_index"]
        .count()
        .to_frame("sentences")
    )

    # Count number of mentions and compute statistics
    df = (
        load_sentiment()
        .drop_duplicates(["sentence_index"])
        .query("aspect_type == 'article' or aspect_type == 'annex'")
        .groupby("user_type")["sentence_index"]
        .count()
        .to_frame("mentions")
        .merge(df, on="user_type")
        .reset_index()
        .assign(
            share=lambda df: df.eval("mentions / sentences"),
            user_type_name=lambda df: df["user_type"].map(utils.user_type_to_name),
            interest_type=lambda df: df["user_type"].map(utils.user_type_to_interest),
        )
    )

    # Compute share aggregated by interest type
    df = df.merge(
        (
            df.groupby("interest_type")["mentions"].sum()
            / df.groupby("interest_type")["sentences"].sum()
        )
        .to_frame("interest_share")
        .reset_index()
    )

    # Sort in usual order
    df = pd.merge(pd.Series(utils.user_type_order).to_frame("user_type_name"), df)
    return df


def aspect_sentiment_user_difference(user_type_name: str) -> pd.DataFrame:
    """Compute summary statistics by aggregating over user types (and aspects).

    Parameters
    ----------
    user_type_name : str
        Name of a user type, e.g. "Business Association".

    Returns
    -------
    pd.DataFrame
        Dataframe with "aspect_type", "aspect_subtype", "sentiment", "n_full",
        "total_full", "share_full", "n_user", "total_user", "share_user", "share_diff",
        "all_aspects_total_user", "all_aspects_total_full", "expected_total",
        "expected_rel_diff", and "aspect" columns.
    """
    full = load_sentiment()
    user = load_sentiment(filter_user_types=[user_type_name])

    def helper(df: pd.DataFrame) -> pd.DataFrame:
        """Small helper function that computes some summary statistics by aggregating
        each aspect's mentions.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with "sentiment", "aspect_type", and "aspect_subtype" columns.

        Returns
        -------
        pd.DataFrame
            Dataframe with "n", "total", and "share" columns added.
        """
        return (
            # Transform sentiment column into categorical
            df.assign(
                sentiment=lambda df: pd.Categorical(
                    df["sentiment"],
                    categories=["negative", "neutral", "positive"],
                    ordered=True,
                )
            )
            # Count sentiments for each aspect
            .groupby(aspect := ["aspect_type", "aspect_subtype"])["sentiment"]
            .value_counts()
            .to_frame("n")
            .reset_index()
            # Add e.g. share column
            .assign(
                total=lambda df: df.groupby(aspect)["n"].transform("sum"),
                share=lambda df: df["n"] / df["total"],
            )
            # Remove neutral sentiment
            .query("sentiment != 'neutral'")
        )

    df = (
        pd.merge(
            helper(full),
            helper(user),
            how="left",
            on=(aspect := ["aspect_type", "aspect_subtype"]) + ["sentiment"],
            suffixes=("_full", "_user"),
        )
        .fillna(0)
        .assign()
        .query("total_full >= 30 and total_user >= 15")
        .assign(
            share_diff=lambda df: df.eval("share_user - share_full"),
            sentiment=lambda df: df["sentiment"].astype(str),
            all_aspects_total_user=lambda df: df.drop_duplicates(subset=aspect)[
                "total_user"
            ].sum(),
            all_aspects_total_full=lambda df: df.drop_duplicates(subset=aspect)[
                "total_full"
            ].sum(),
            expected_total=lambda df: df.eval(
                "(total_full / all_aspects_total_full) * all_aspects_total_user"
            ),
            expected_rel_diff=lambda df: df.eval(
                "(total_user - expected_total) / expected_total"
            ),
            aspect=lambda df: df[aspect].apply(aspect_nice_format_series, axis=1),
        )
    )
    return sort_by_aspect(df).reset_index(drop=True)


def _user_types_details_string(
    example_user_type: str = "",
    interest: str = "",
    existing: pd.DataFrame | None = None,
) -> str:
    """Assemble a string with the number of mentions by each user type within a interest
    type.

    While both "example_user_type" and "interest" are optional, one of the two has to be
    specified.

    Parameters
    ----------
    example_user_type : str, optional
        Specification of interest type by a user type that belongs to it, for example
        "company" for interest type "corporate". By default unspecified ("").
    interest : str, optional
        Direct specification of interest type, for example "corporate". By default
        unspecified ("").
    existing : pd.DataFrame | None, optional
        Dataframe with "user_type" and "n" columns. By default None, i.e. 0 for each
        user type in given interest type.

    Returns
    -------
    str
        String with number of mentions for each user type. For example for the
        "corporate" interest type:
            Company: 18,   Business Association: 10
    """
    if example_user_type == "" and interest == "":
        raise ValueError("Either example_user_type or interest has to be specified!")

    # If interest is unspecified, infer it from the user type
    if interest == "":
        interest = utils.user_type_to_interest(example_user_type)

    # If existing is unspecified, initialize it as an empty dataset
    if existing is None:
        existing = pd.DataFrame(columns=["user_type", "n"])

    # Add default values to dataframe
    user_types = utils.interest_to_user_types(interest)
    defaults = pd.DataFrame(zip(user_types, it.repeat(0)), columns=existing.columns)  # type: ignore[arg-type] # noqa: E501
    all = (
        pd.concat((existing, defaults))
        .drop_duplicates(subset=["user_type"], keep="first")
        .set_index("user_type")
        .loc[user_types]
    )

    # Compute the output
    details_string = ",   ".join(
        [
            f"{utils.user_type_to_name(str(user_type))}: {row['n']}"
            for user_type, row in all.iterrows()
        ]
    )
    return details_string


def get_aspects() -> pd.DataFrame:
    """Extract the aspects their number of mentions from a sentiments dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with "aspect_type", "aspect_subtype", and "n_mentions" columns.
    """
    return (
        load_sentiment(filter_=True, sort="aspect")[
            ["aspect_type", "aspect_subtype", "sentence_index"]
        ]
        .groupby(["aspect_type", "aspect_subtype"], sort=False)
        .agg({"sentence_index": "size"})
        .rename(columns={"sentence_index": "n_mentions"})
        .reset_index()
    )


@memoize
def word_cloud(
    sentiment: str, aspect_type: str, aspect_subtype: str, spoof: bool = True
) -> str:
    """Draw word cloud for mentions of given sentiment and aspect.

    Word size(s):
    - Load sentiments dataset and limit to selected aspect
    - If spoof is true, replace sentiment with majority between positive and negative
      predicted probabilities if either is at least 33%
    - Tf-idf vectorize texts with positive or negative sentiment
    - Train a logistic regression classifier based on tf-idf scores
    - Draw word cloud with class-specific logistic regression coefficients as
      frequencies

    Parameters
    ----------
    sentiment : str
        Which sentiment to visualize, either "positive" or "negative".
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".
    spoof : bool, optional
        (Conditionally) replace neutral sentiments as specified above, by default True.

    Returns
    -------
    str
        Word cloud as SVG string.
    """
    # Load dataset
    sentiments = load_sentiment(aspect_type=aspect_type, aspect_subtype=aspect_subtype)
    if spoof:
        sentiments["sentiment"] = np.where(
            sentiments[["positive", "negative"]].max(axis="columns") > 1 / 3,
            np.where(
                sentiments["positive"] > sentiments["negative"], "positive", "negative"
            ),
            sentiments["sentiment"],
        )
    non_neutral = sentiments.query("sentiment == 'positive' or sentiment == 'negative'")

    # Compute tf-idf embedding
    filter_ = dict(max_df=0.9, min_df=2) if len(non_neutral) > 2 else dict()
    tfidf = TfidfVectorizer(
        **filter_, stop_words=list(_get_stop_words()), ngram_range=(1, 2)
    )
    try:
        X = tfidf.fit_transform(non_neutral["text"])
        tokens = tfidf.get_feature_names_out()
    except ValueError:
        # Fallback in case of errors
        X, tokens = np.array([[1]] * len(non_neutral)), ["error"]

    # Default coloring: randomly sample red (negative) or green (positive) colors
    color: dict[str, Any] = dict(
        colormap="Reds" if sentiment == "negative" else "Greens"
    )

    # Mentions of both sentiments
    if non_neutral["sentiment"].nunique() == 2:
        clf = LogisticRegression().fit(X, non_neutral["sentiment"])

        pred = {
            clf.classes_[i]: {
                tokens[j]: np.exp(clf.coef_.reshape(-1) * (-1 if i == 0 else 1))[j]
                for j in range(len(tokens))
            }
            for i in range(len(clf.classes_))
        }[sentiment]

    # Only mentions with sentiment X but we're plotting Y -> just say that
    elif sentiment not in non_neutral["sentiment"].unique():
        pred = {f"no {sentiment} mentions": 1}
        color = dict(color_func=lambda *args, **kwargs: "white")

    # Only mentions with sentiment X and we're plotting X -> fallback to tfidf
    else:
        tfidf_scores = (
            tfidf.transform(["\n".join(non_neutral["text"])]).toarray().reshape(-1)  # type: ignore[pylance] # noqa[E501]
        )
        pred = {token: tfidf_score for token, tfidf_score in zip(tokens, tfidf_scores)}

    word_cloud = (
        WordCloud(min_word_length=2, background_color=None, mode="RGBA", **color)  # type: ignore[pylance] # noqa[E501]
        .generate_from_frequencies(pred)
        .to_svg(embed_font=True, optimize_embedded_font=False)
    )
    return f"data:image/svg+xml,{word_cloud}"


@memoize
def _get_stop_words() -> set[str]:
    """Compute a list of stop words based on sklearn's built-in and any unigram or
    bigram appearing in at least 3% of the sentiment dataset's texts.

    Returns
    -------
    set[str]
        Set of stop word n-grams.
    """
    sentiments = load_sentiment(filter_=False)
    vec = CountVectorizer(stop_words="english", min_df=0.03, ngram_range=(1, 2))  # type: ignore[pylance] # noqa[E501]
    vec.fit(sentiments["text"])
    return set(ENGLISH_STOP_WORDS) | set(vec.get_feature_names_out())


def context(
    df: pd.DataFrame, row: pd.Series, direction: str, window_size: int = 2
) -> pd.DataFrame:
    """Compute the dataframe of sentences (rows) before or after the a specific mention.

    Parameters
    ----------
    df : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.
    row : pd.Series
        Row of the sentiment dataset. Needs "sentence_index" label.
    direction : str
        Direction of context, either "before" or "after".
    window_size : int, optional
        How many rows on each size to return, by default 2.

    Returns
    -------
    pd.DataFrame
        Context as dataframe.
    """
    comp = "<" if direction == "before" else ">"
    id, source, sentence_index = row["id"], row["source"], row["sentence_index"]  # noqa
    a = (
        df.query("id == @id and source == @source")
        .query(f"sentence_index {comp} @sentence_index")
        .sort_values("sentence_index")
    )

    if direction == "before":
        return a.iloc[-window_size:]
    elif direction == "after":
        return a.iloc[:window_size]
    else:
        raise ValueError(
            f'Unknown "direction" value "{direction}". Use "before" or "after" instead.'
        )


def context_string(
    df: pd.DataFrame, row: pd.Series, direction: str, window_size: int = 2
) -> str:
    """Compute the string before or after the a specific mention.

    Parameters
    ----------
    df : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.
    row : pd.Series
        Row of the sentiment dataset. Needs "sentence_index" label.
    direction : str
        Direction of context, either "before" or "after".
    window_size : int, optional
        How many rows on each size to return, by default 2.

    Returns
    -------
    str
        Context as string.
    """
    context_ = context(df, row, direction, window_size)
    return "" if len(context_) == 0 else str((context_["text"] + " ").sum())


def aspect_nice_format(aspect_type: str, aspect_subtype: str) -> str:
    """Turn aspect_type and aspect_subtype into a nice string representation, by e.g.
    converting snake cased aspect_type into proper "Article", "AI Act", and so on.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    str
        Formatted aspect.
    """
    maps = {"ai_act": "AI Act", "eaib": "EAIB"}
    if aspect_type in maps:
        return maps[aspect_type]

    aspect = aspect_type.replace("_", " ").capitalize()
    if aspect_type in ["annex", "title"]:
        aspect_subtype = _arabic_to_roman(aspect_subtype)
    aspect_subtype = aspect_subtype.upper()

    return f"{aspect} {aspect_subtype}"


def _arabic_to_roman(arabic: str) -> str:
    """Convert an Arabic number between 1 and 20 to its Roman equivalent.

    Parameters
    ----------
    arabic : str
        String containing an Arabic number between 1 and 20, e.g. "4".

    Returns
    -------
    str
        Roman equivalent, e.g. "iv".
    """
    ROMAN_NUMBERS = (
        "i;ii;iii;iv;v;vi;vii;viii;ix;x;xi;xii;xiii;xiv;xv;xvi;xvii;xviii;xix;xx"
    ).split(";")
    return ROMAN_NUMBERS[int(arabic) - 1]


def aspect_nice_format_series(aspect: pd.Series) -> str:
    """Apply the aspect_nice_format function to a series.

    Parameters
    ----------
    aspect : pd.Series
        Series with "aspect_type" and "aspect_subtype" labels.

    Returns
    -------
    str
        Nice aspect text.
    """
    return aspect_nice_format(aspect["aspect_type"], aspect["aspect_subtype"])


def sentiment_topic_modeling(
    aspect_type: str | None = None,
    aspect_subtype: str | None = None,
    search_term: str | None = None,
    regex: bool = False,
    filter_user_types: list[str] = [],
) -> pd.DataFrame:
    """Compute the topic modeling transformation of selected aspect's mentions.

    Parameters
    ----------
    aspect_type : str, optional
        General part of the aspect, e.g. "aspect" - by default None (unspecified).
        Either aspect_type and aspect_subtype or search_term needs to be specified. If
        both are specified, search_term is used.
    aspect_subtype : str, optional
        Specific part of the aspect, e.g. "5" - by default None (unspecified).
    search_term : str, optional
        Search term to use, e.g. "face" - by default None (unspecified). Either
        aspect_type and aspect_subtype or search_term needs to be specified. If both
        are specified, search_term is used.
    regex : bool, optional
        Whether the search term uses regular expressions, by default False.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.

    Returns
    -------
    pd.DataFrame
        Sentiment dataset with additional (normalized) topic share columns for each
        topic and a "dominant_topic" column.
    """
    # Throw error if neither aspect nor search term are specified
    if (aspect_type is None or aspect_subtype is None) and search_term is None:
        raise ValueError(
            "Either aspect_type and aspect_subtype, or search_term arguments have to "
            "be specified!"
        )

    # Setup TM pipeline
    pipeline = utils.load_pipeline()

    # Get search results or aspect mentions
    if search_term is not None:
        texts = search_sentiment(
            search_term=search_term, regex=regex, filter_user_types=filter_user_types
        )
    elif aspect_type is not None and aspect_subtype is not None:
        texts = load_sentiment(
            aspect_type=aspect_type,
            aspect_subtype=aspect_subtype,
            filter_user_types=tuple(filter_user_types),  # type: ignore
        )
    else:
        raise ValueError(
            "Either aspect_type and aspect_subtype, or search_term arguments have to "
            "be specified!"
        )
    topics = utils.get_topics(utils.load_tm()[0])

    # Fallback if search results/aspect mentions are empty
    if len(texts) == 0:
        return pd.DataFrame(
            columns=texts.columns.to_list() + topics + ["dominant_topic"]
        )

    # Process data using TM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tm = pipeline.transform(texts["tokenized"])
    H_norm = pd.DataFrame(
        tm / np.sum(tm, axis=1).reshape(-1, 1), columns=topics
    ).assign(
        dominant_topic=lambda df: list(
            map(lambda n: topics[n], np.argmax(df[topics].values, axis=1))  # type: ignore[no-any-return] # noqa[E501]
        )
    )
    return pd.concat((texts.reset_index(drop=True), H_norm), axis=1)


def get_user_type_names() -> list[str]:
    """Get a list of user type names present in sentiment dataset, sorted by the number
    of their mentions (across all aspects).

    Returns
    -------
    list[str]
        List of user type names, e.g. ["Company", "NGO", ...].
    """
    return list(
        map(
            utils.user_type_to_name,
            load_sentiment()
            .groupby("user_type")["sentence_index"]
            .size()
            .sort_values(ascending=False)
            .index.values,
        )
    )


def get_readability() -> pd.DataFrame:
    """Compute Flesch reading ease and Dale-Chall readability score for sentiment
    dataset, separated by interest type.

    Returns
    -------
    pd.DataFrame
        Dataframe with "interest_type", "score", and "median" columns.
    """
    # Flesch: Higher score = less difficult text (scale: 0-100, <30="very confusing")
    # Dale-Chall: base on US grade (8-8.99=11th or 12th grade student)
    return (
        load_sentiment(filter_=False)
        # Merge all sentences into a single text
        .assign(
            full_text=lambda df: df.groupby("id")["text"].transform(
                lambda x: " ".join(x)
            ),
            interest_type=lambda df: df["user_type"].map(utils.user_type_to_interest),
        )
        .drop_duplicates(subset=["id"])
        # Compute readability scores
        .assign(
            flesch=lambda df: df["full_text"].map(textstat.flesch_reading_ease),
            dale_chall=lambda df: df["full_text"].map(
                textstat.dale_chall_readability_score
            ),
        )
        # Compute aggregate statistics
        .groupby("interest_type")
        .agg(dict(flesch=["median", "std"], dale_chall=["median", "std"]))
        # Remove multilevel columns
        .rename_axis(columns=["score", None])  # type: ignore[list-item]
        .stack(0)
        .reset_index()
        .explode("interest_type")
    )
