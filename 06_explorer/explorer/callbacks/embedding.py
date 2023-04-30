"""Module providing the /embedding/ callbacks."""

from typing import Any
import itertools as it
import plotly.graph_objects as go

import dash
from dash import ctx
from dash._callback import NoUpdate
from dash.dependencies import Input, Output, State
from dash.dependencies import Component as DashComponent

from graphs import graphs
from utils import utils, caching

H, _, texts = utils.load_tm()


def initialize_callbacks() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Dendrogram (complete)
    - TSNE embedding colored by user types (common)
    - TSNE embedding colored by dominant topic (common)
    """
    # Get some common user types and dominant topics selection
    user_types = texts["user_type"].unique()
    user_type_lists = list(
        it.chain(
            [
                [user_type for user_type in user_types if user_type != excluded]
                for excluded in user_types
            ],
            [[user_type] for user_type in user_types],
        )
    )
    topics = utils.get_topics(H)
    topic_lists = list(
        it.chain(
            [[topic for topic in topics if topic != excluded] for excluded in topics],
            [[topic] for topic in topics],
        )
    )

    # Dendrogram
    graphs.tm_dendrogram()

    # TSNE embedding colored by user type name
    for user_type_selected in caching.progress_bar(
        user_type_lists,
        "/embedding/",
        initial=0,
        total=len(user_type_lists) + len(topic_lists),
    ):
        graphs.tsne_embedding(
            legend="dominant_topic",
            user_type_selected=user_type_selected,
        )

    # TSNE embedding colored by user type name
    caching.delete_last_line()
    for topic_selected in caching.progress_bar(
        topic_lists,
        "/embedding/",
        initial=len(user_type_lists),
        total=len(user_type_lists) + len(topic_lists),
    ):
        graphs.tsne_embedding(
            legend="user_type_name",
            topic_selected=topic_selected,
        )


def _update_dendrogram(_: Any) -> go.Figure:
    """Callback to set/update the dendrogram figure.

    Returns
    -------
    go.Figure
        Updated topic dendrogram figure.
    """
    return graphs.tm_dendrogram()


def _update_embedding_topic(
    restyle_data: None | list[dict[str, list[str]] | list[int]], fig: go.Figure
) -> go.Figure:
    """Update the TSNE embedding colored by dominant topic.

    Parameters
    ----------
    restyle_data : None | list[dict[str, list[str]] | list[int]]
        Restyle data of the TSNE embedding colored by user type.
    fig : go.Figure
        TSNE embedding scatter plot colored by user type.

    Returns
    -------
    go.Figure
        Updated dominant topic colored scatter plot.
    """
    # Page load
    if not ctx.triggered_id:
        return graphs.tsne_embedding(legend="dominant_topic")

    user_types = [user_type["legendgroup"] for user_type in fig["data"]]  # type: ignore[silence_vscode] # noqa[E501]
    visibilities = {
        i: visibility != "legendonly"
        for visibility, i in zip(restyle_data[0]["visible"], restyle_data[1])  # type: ignore[index,call-overload] # noqa[E501]
    }
    user_type_selected = [
        topic
        for i, topic in enumerate(user_types)
        if i not in visibilities or visibilities[i]
    ]

    return graphs.tsne_embedding(
        legend="dominant_topic",
        user_type_selected=user_type_selected,
    )


def _update_embedding_user_type(restyle_data, fig: go.Figure) -> go.Figure:
    """Update the TSNE embedding colored by user type.

    Parameters
    ----------
    restyle_data : None | list[dict[str, list[str]] | list[int]]
        Restyle data of the TSNE embedding colored by dominant topic.
    fig : go.Figure
        TSNE embedding scatter plot colored by dominant topic.

    Returns
    -------
    go.Figure
        Updated user type colored scatter plot.
    """
    # Page load
    if not ctx.triggered_id:
        return graphs.tsne_embedding(legend="user_type_name")

    topics = [topic["legendgroup"] for topic in fig["data"]]  # type: ignore[silence_vscode] # noqa[E501]
    visibilities = {
        i: visibility != "legendonly"
        for visibility, i in zip(restyle_data[0]["visible"], restyle_data[1])
    }
    topic_selected = [
        topic
        for i, topic in enumerate(topics)
        if i not in visibilities or visibilities[i]
    ]

    return graphs.tsne_embedding(
        legend="user_type_name",
        topic_selected=topic_selected,
    )


def _toggle_double_click_alert(
    is_open: bool, current_children: DashComponent | list[DashComponent]
) -> DashComponent | list[DashComponent] | NoUpdate:
    """Set cookie to remember dismissed double click alert.

    Parameters
    ----------
    is_open : bool
        Whether the alert is currently displayed (True) or dismissed (False).
    current_children : DashComponent | list[DashComponent]
        Current alert div children components.

    Returns
    -------
    DashComponent | list[DashComponent] | NoUpdate
        Dummy output (leave everything as-is).
    """
    if not is_open:
        dash.ctx.response.set_cookie(
            "alert_note_double_click_dismissed", "true", max_age=int(1e10)
        )
        return current_children

    return dash.no_update


def _update_ldavis_iframe(_) -> str:
    """Update the LDAvis iframe's srcDoc with actual content.

    Why is this in a callback if it's static content? Because dash can not handle
    initializing the iframe with the srcDoc and shows the pages' literal html instead
    of rendering them.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback).

    Returns
    -------
    str
        LDAvis visualization's html code.
    """
    return (
        "<style>\n\thtml, body { overflow: hidden; background-color: #303030; }\n"
        "\ttext { fill: white; stroke: white; }\n</style>\n\n" + utils.load_ldavis()
    )


def setup_callbacks() -> None:
    """Set up the callbacks for the /embedding/ page."""
    dash.get_app().callback(
        Output("embedding_dendrogram", "figure"),
        Input("embedding_page_load", "children"),
    )(_update_dendrogram)

    dash.get_app().callback(
        Output("embedding_topic", "figure"),
        Input("embedding_user_type", "restyleData"),
        State("embedding_user_type", "figure"),
    )(_update_embedding_topic)

    dash.get_app().callback(
        Output("embedding_user_type", "figure"),
        Input("embedding_topic", "restyleData"),
        State("embedding_topic", "figure"),
    )(_update_embedding_user_type)

    dash.get_app().callback(
        Output("alert_note_double_click", "children"),
        Input("alert_note_double_click", "is_open"),
        State("alert_note_double_click", "children"),
    )(_toggle_double_click_alert)

    dash.get_app().callback(
        Output("ldavis_iframe", "srcDoc"),
        Input("embedding_page_load", "children"),
    )(_update_ldavis_iframe)
