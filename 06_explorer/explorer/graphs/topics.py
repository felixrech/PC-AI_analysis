"""Module with some topics-specific graphs."""

import pandas as pd
import itertools as it
import plotly.express as px
import plotly.graph_objects as go

from graphs import graphs
from utils.caching import memoize
from utils import utils, topics as topic_utils


@memoize
def top_terms_barplot(topic_selected: str) -> go.Figure:
    """Make a barplot of a topics top terms.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    _, W, _ = utils.load_tm()
    W_norm = utils.normalize_W(W)
    topic_top_terms = topic_utils.top_terms(W_norm, topic_selected)

    fig = px.bar(
        topic_top_terms,
        y="lemma",
        x=topic_selected,
        labels={"lemma": "", topic_selected: ""},
    )

    order = topic_top_terms.sort_values(topic_selected, ascending=True)["lemma"]
    layout = {
        "yaxis_ticksuffix": "  ",
        "yaxis_categoryorder": "array",
        "yaxis_categoryarray": order.to_list(),
    }
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    graphs.format_axis_percent(fig, axis="x")
    return fig


@memoize
def user_types_avg_share_barplot(topic_selected: str) -> go.Figure:
    """Plots each user type's average topic share as a barplot.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)
    topic_docs = topic_utils.docs(texts, H_norm, topic_selected)

    # Compute the average share for each user type and reorder results
    user_types = topic_docs.groupby("user_type")[topic_selected].mean().to_frame()
    user_types.index = user_types.index.map(utils.user_type_to_name)
    user_types = user_types.loc[utils.user_type_order]

    fig = px.bar(
        user_types,
        x=user_types.index,
        y=topic_selected,
        labels={"user_type": "", topic_selected: ""},
    )

    layout = {"autosize": False, "height": 300}
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    graphs.format_axis_percent(fig, axis="y")
    return fig


@memoize
def user_types_dominant_barplot(topic_selected: str) -> go.Figure:
    """Show how often the selected topic is the dominant topic for documents of each
    user type as a barplot.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)

    # Get the data
    dominant_topics = utils.document_dominant_topic(H_norm)
    dominant_topics = pd.merge(texts.reset_index(names="internal_id"), dominant_topics)
    dominant_topics = (
        dominant_topics.groupby(["user_type"])["dominant_topic"]
        .value_counts(normalize=True)
        .to_frame("share_dominant")
        .reset_index(names=["user_type", "topic"])
        .query("topic == @topic_selected")
        .set_index("user_type")
    )

    # If a topic is not dominant for any docs of an user type, that user type would be
    # missing - add them with zeros
    missing_types = list(set(texts["user_type"].unique()) - set(dominant_topics.index))
    missing = pd.DataFrame(
        {
            "topic": it.repeat(topic_selected, len(missing_types)),
            "share_dominant": it.repeat(0, len(missing_types)),
        },
        index=missing_types,
    )
    dominant_topics = pd.concat((dominant_topics, missing))

    # Plotting
    dominant_topics.index = dominant_topics.index.map(utils.user_type_to_name)
    dominant_topics = dominant_topics.loc[utils.user_type_order].reset_index(
        names=["user_type"]
    )
    fig = px.bar(
        dominant_topics,
        x="user_type",
        y="share_dominant",
        labels={"user_type": "", "share_dominant": ""},
    )
    layout = {"autosize": False, "height": 300}
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    graphs.format_axis_percent(fig, axis="y")
    return fig


@memoize
def user_types_read_n_docs_barplot(
    topic_selected: str, top_n: int, total: bool = False
) -> go.Figure:
    """Show how many documents of each user type are among the top_n documents with
    highest topic share.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    top_n : int
        Number of documents which to analyze.
    total : bool
        Show the total number of each bar as an annotation.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)
    topic_docs = topic_utils.docs(texts, H_norm, topic_selected)

    # Set up the data
    reading = (
        topic_docs.sort_values(topic_selected, ascending=False)  # type: ignore
        .iloc[:top_n][["id", "organization", "user_type"]]
        .value_counts(dropna=False)
        .to_frame("n_read")
        .reset_index()
        .fillna("<Private>")
    )

    # If an user type is not present among the top 20, it wouldn't be part of dataset
    missing_types = list(set(texts["user_type"].unique()) - set(reading["user_type"]))
    missing = pd.DataFrame(
        {
            "n_read": list(it.repeat(0, len(missing_types))),
            "user_type": missing_types,
            "organization": list(it.repeat(None, len(missing_types))),
            "id": list(it.repeat(None, len(missing_types))),
        },
    )
    reading = pd.concat((reading, missing))

    # Plotting
    reading = (
        reading.set_index(reading["user_type"].map(utils.user_type_to_name))
        .loc[utils.user_type_order, ["id", "organization", "n_read"]]
        .reset_index(names="user_type")
    )
    fig = px.bar(
        reading,
        x="user_type",
        y="n_read",
        text="organization",
        custom_data=["id", "organization"],
        labels={"user_type": "", "n_read": ""},
    )
    fig.update_traces(
        hovertemplate=(
            "Feedback: %{customdata[0]}<br>"
            "Organization: %{customdata[1]}<br>"
            "#docs: %{y}"
        ),
        textposition="inside",
    )

    if total:
        totals = reading.groupby("user_type")["n_read"].sum().reset_index()
        annotations = [
            dict(
                x=row["user_type"],
                y=row["n_read"] + totals["n_read"].max() * 0.06,
                text=row["n_read"],
                showarrow=False,
                font_size=10,
                bgcolor="white",
            )
            for _, row in totals.iterrows()
        ]
    layout = {
        "autosize": False,
        "height": 300,
        "annotations": [] if not total else annotations,
    }
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    return fig
