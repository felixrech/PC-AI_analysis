"""Module that contains utilities for dealing with topics."""
import pandas as pd

from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from utils.caching import memoize
from utils import utils, topics as topic_utils


def docs(texts: pd.DataFrame, H: pd.DataFrame, topic_selected: str) -> pd.DataFrame:
    """Add the topic share of the selected topic to the training data dataframe.

    Parameters
    ----------
    texts : pd.DataFrame
        Training data of the topic model as saved using
        evaluation.export_for_indepth_analysis.
    H : pd.DataFrame
        (Normalized or unnormalized) topic-document matrix H.
    topic_selected : str
        Currently selected topic, i.e. a column name of H.

    Returns
    -------
    pd.DataFrame
        Columns of texts plus "internal_index" and topic_selected columns.
    """
    topic_docs = utils.normalize_H(H)[["id", topic_selected]]
    topic_docs = pd.merge(topic_docs.reset_index(names="internal_index"), texts)
    return topic_docs


def most_relevant_doc(
    texts: pd.DataFrame, H_norm: pd.DataFrame, topic_selected: str
) -> int:
    """Compute the index of the most relevant document in the selected topic.

    Parameters
    ----------
    texts : pd.DataFrame
        Training data of the topic model as saved using
        evaluation.export_for_indepth_analysis.
    H_norm : pd.DataFrame
        (Row-) Normalized topic-document matrix H.
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    int
        Index of the most relevant document.
    """
    df = docs(texts, H_norm, topic_selected).sort_values(
        topic_selected, ascending=False
    )
    return int(df.iloc[0]["internal_index"])


@memoize
def info_table(topic_selected: str) -> DashComponent:
    """A information table about the currently selected topic containing for how many
    documents it is relevant and for how many it is the dominant topic.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    DashComponent
        Information table.
    """
    H_norm = utils.normalize_H(utils.load_tm()[0])

    # For how many documents is this topic dominant/relevant
    dominant_topic = utils.document_dominant_topic(H_norm)["dominant_topic"]
    dominant_percentage = round(100 * (dominant_topic == topic_selected).mean(), 1)
    relevant_percentage = round(100 * (H_norm[topic_selected] > 0.1).mean(), 1)

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            "Dominant for",
                            title="Topic-document score > than for all other topics",
                            style=dict(width="50%"),
                        ),
                        html.Th("Relevant for", title="Topic-document score > 10%"),
                    ]
                )
            ),
            html.Tbody(
                html.Tr(
                    [
                        html.Td(f"{dominant_percentage}% of docs"),
                        html.Td(f"{relevant_percentage}% of docs"),
                    ]
                )
            ),
        ]
    )


@memoize
def user_types_doc_numbers(topic_selected: str) -> DashComponent:
    """Return a table of buttons to the five documents with highest topic shares for
    each user type.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    DashComponent
        Table of buttons.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)
    topic_docs = topic_utils.docs(texts, H_norm, topic_selected)

    # Find the top five documents for each user type
    topic_docs = (
        topic_docs.reset_index(drop=True)
        .sort_values(topic_selected, ascending=False)
        .reset_index(names="topic_index")
    )
    user_type_docs = topic_docs.groupby("user_type").head(5)
    user_type_docs_dict = {
        utils.user_type_to_name(user): zip(
            user_type_docs.query("user_type == @user")["internal_index"].to_list(),
            user_type_docs.query("user_type == @user").index.to_list(),
        )
        for user in user_type_docs["user_type"].unique()
    }

    # Transform numbers into buttons
    user_type_buttons = {
        k: [
            dbc.Button(
                children=i + 1,
                id=dict(type="document_selected_index_button_", index=n),
                n_clicks=-1,
                style={"min-width": "3rem"},
            )
            for n, i in user_type_docs_dict[k]
        ]
        for k in utils.user_type_order
    }

    # Add fake button to catch callbacks when adding the buttons to the page
    fake_button = dbc.Button(
        id=dict(type="document_selected_index_button_", index=-1),
        style=dict(display="none"),
    )

    # Return buttons in table layout
    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("User type", style=dict(width="15%")),
                        html.Th("Top 5 documents", style=dict(width="15%")),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(k if i != 0 else [fake_button, k]),
                            html.Td(dbc.ButtonGroup(user_type_buttons[k], size="sm")),
                        ]
                    )
                    for i, k in enumerate(utils.user_type_order)
                ]
            ),
        ]
    )
    return table


# @memoize
def popularity(
    H_norm: pd.DataFrame, texts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes how commonly a topic is dominant and relevant, separated by user_type.

    Output dataframe columns (besides "user_type" and "topic"):
    - "n_docs": How often is topic dominant/relevant for given user type?
    - "user_type_name": Full user type (from utils.user_type_to_name).
    - "topic_total": How often is topic dominant/relevant (across all user types)?
    - "user_type_topic_share": n_docs / topic_total.
    - "topic_share": n_docs / (sum of n_docs for given user type across all topics).

    Parameters
    ----------
    H_norm : pd.DataFrame
        (Row-) Normalized topic-document matrix H.
    texts : pd.DataFrame
        Training data of the topic model as saved
        using evaluation.export_for_indepth_analysis.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Popularity of topic as dominant and relevant as dataframes, each with the
        following columns: "user_type", "topic", "n_docs", "user_type_name",
        "topic_total", "user_type_topic_share", "topic_share"
    """
    topics = utils.get_topics(H_norm)

    df_dominant = (
        pd.concat((utils.document_dominant_topic(H_norm), texts), axis=1)
        .groupby(["user_type", "dominant_topic"])
        .agg({"internal_id": "count"})
        .reset_index()
        .rename(columns={"internal_id": "n_docs", "dominant_topic": "topic"})
    )

    df_relevant = (
        pd.concat((H_norm, texts), axis=1)[topics + ["user_type"]]
        .reset_index(names="internal_index")
        .melt(id_vars=["internal_index", "user_type"], var_name="topic")
        .query("value > 0.1")
        .groupby(["user_type", "topic"])
        .agg({"internal_index": "count"})
        .rename(columns={"internal_index": "n_docs"})
        .reset_index()
    )

    for df in [df_dominant, df_relevant]:
        df["user_type_name"] = df["user_type"].map(utils.user_type_to_name)
        df["topic_total"] = df.groupby("topic")["n_docs"].transform("sum")
        df["user_type_topic_share"] = df["n_docs"] / df["topic_total"]
        df["topic_share"] = (
            df["topic_total"]
            / df.drop_duplicates(subset=["topic"])["topic_total"].sum()
        )
    return df_dominant, df_relevant


def top_terms(W_norm: pd.DataFrame, topic_selected: str) -> pd.DataFrame:
    """Compute the 20 lemmas with higher topic-specific probability.

    Parameters
    ----------
    W_norm : pd.DataFrame
        (Column-) Normalized term-topic matrix W.
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    pd.DataFrame
        Dataframe with "lemma" and topic_selected columns and 20 rows.
    """
    topic_terms = (
        W_norm[["lemma", topic_selected]]
        .sort_values(topic_selected, ascending=False)
        .iloc[:20]
    )
    return topic_terms
