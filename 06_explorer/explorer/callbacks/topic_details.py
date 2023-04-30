"""Module providing the callbacks for the /topic_details/ page."""

import urllib
import pandas as pd
from typing import Any
import plotly.graph_objects as go

import dash
from dash import ctx
from dash._callback import NoUpdate
from dash._utils import AttributeDict
from dash.dependencies import Output, Input, State, ALL
from dash.dependencies import Component as DashComponent

from utils.caching import memoize
from graphs import topics as topic_graphs
from utils import utils, current, topics, caching
from layouts import layouts, navbar, topic_details as layout_


# Load the data
top_n = 20
H, W, texts = utils.load_tm()
similarities = utils.load_similarities()
H_norm, W_norm = utils.normalize_H(H), utils.normalize_W(W)


def initialize_caching() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Topic top terms barplot (complete)
    - Topic info table (complete)
    - Topic user type modal graphs (complete)
    - Topic user type document numbers (complete)
    """
    for topic in caching.progress_bar(utils.get_topics(H), "/topic_details/"):
        topic_graphs.top_terms_barplot(topic)
        topics.info_table(topic)
        topic_graphs.user_types_avg_share_barplot(topic)
        topic_graphs.user_types_dominant_barplot(topic)
        topic_graphs.user_types_read_n_docs_barplot(topic, top_n)
        topic_graphs.user_types_read_n_docs_barplot(topic, 50)
        topics.user_types_doc_numbers(topic)


def setup_callbacks() -> None:
    """Setup the callbacks for element on the /topic_details/ page."""
    _setup_invisible_callbacks()
    _setup_doc_change_callbacks()
    _setup_topic_change_callbacks()
    _setup_modal_callbacks()


########################################################################################
# BEGIN HELPER FUNCTIONS                                                               #
########################################################################################


def _get_current(topic_selected: str, document_selected: int) -> pd.Series:
    """Extract the document_selected row from texts and add topic_selected's topic
    share.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic, i.e. a column name of H.
    document_selected : int
        Currently selected topic, i.e. an int from [0, len(texts)).

    Returns
    -------
    pd.Series
        Columns of texts plus "internal_index" and topic_selected as entries.
    """
    topic_docs = topics.docs(texts, H_norm, topic_selected)
    return topic_docs.query("internal_index == @document_selected").iloc[0]


def _parse_search(search: str) -> dict:
    """Parse URL search into a dictionary.

    Parameters
    ----------
    search : str
        Search string.

    Returns
    -------
    dict
        Parsed search.
    """
    search_dict = urllib.parse.parse_qs(search)  # type: ignore

    # Rename ?something arguments
    rename = [k for k in search_dict.keys() if k.startswith("?")]
    for key in rename:
        search_dict[key[1:]] = search_dict.pop(key)

    # Replace lists by values
    for key in list(search_dict.keys()):
        if len(search_dict[key]) == 1:
            search_dict[key] = search_dict[key][0]  # type: ignore
    return search_dict


########################################################################################
# BEGIN INVISIBLE CALLBACKS                                                            #
########################################################################################


@memoize
def _update_id_div(topic_selected: str, document_selected: str) -> str:
    """Update the id div based on selections.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    document_selected : str
        (Internal index of the) Currently selected topic.

    Returns
    -------
    str
        New id.
    """
    return str(_get_current(topic_selected, int(document_selected))["id"])


def _update_similar_docs_type(*_) -> str | NoUpdate:
    """Update the selected similar documents mode.

    Returns
    -------
    str | NoUpdate
        Similarity modus.
    """
    if ctx.triggered_id is None:
        return dash.no_update
    return str(ctx.triggered_id).replace("similar_docs_type_", "")


def _update_search(topic_selected: str, document_selected: str, _, search: str) -> str:
    """Update the URL's search.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    document_selected : str
        (Internal index of the) Currently selected topic.
    _ : Any
        Ignored (only for triggering callback).
    search : str
        Current search string.

    Returns
    -------
    str
        New search.
    """
    search_dict = _parse_search(search)

    # Update search based on user selection
    if (
        type(ctx.triggered_id) is AttributeDict
        and ctx.triggered_id["type"] == "topic_selected_"
    ):
        search_dict["topic_selected"] = ctx.triggered_id["index"]
    else:
        search_dict["topic_selected"] = topic_selected
    search_dict["document_selected"] = document_selected

    # URL-encode search and return
    return "?" + urllib.parse.urlencode(search_dict)  # type: ignore


def _update_topic_selected(_) -> str | NoUpdate:
    """Update the currently selected topic.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback).

    Returns
    -------
    str | NoUpdate
        Currently selected topic.
    """
    if type(ctx.triggered_id) is not AttributeDict:
        return dash.no_update

    return ctx.triggered_id["index"]


def _update_document_selected(
    _a,
    _b,
    _c,
    _d,
    modal_clicks: int | None,
    topic_selected: str,
    document_selected_str: str,
    modal_open: bool,
) -> tuple[int, bool]:
    """Update the currently selected document.

    Parameters
    ----------
    _a : Any
        Ignored (for triggering callback only).
    _b : Any
        Ignored (for triggering callback only).
    _c : Any
        Ignored (for triggering callback only).
    _d : Any
        Ignored (for triggering callback only).
    modal_clicks : int | None
        Number of clicks on the user type document numbers modal.
    topic_selected : str
        Currently selected topic.
    document_selected_str : str
        (Internal index of the) Currently selected topic.
    modal_open : bool
        Whether the user type document numbers modal is currently open.

    Returns
    -------
    tuple[int, bool]
        Tuple containing currently selected document and whether the user type document
        numbers modal is currently open.
    """
    del _a, _b, _c, _d  # "Use" arguments

    if ctx.triggered_id is None:
        # If triggered on page-load, simply return current state to avoid breaking stuff
        return int(document_selected_str), modal_open

    if (
        type(ctx.triggered_id) is AttributeDict
        and ctx.triggered_id["type"] == "document_selected_index_button_"
    ):
        # Catch the callback fired based on adding the buttons to the page (when
        # changing topics)
        if ctx.triggered_id["index"] < 0:
            # We should actually raise PreventUpdate here but that causes Plotly to
            # ignore the real callback, so just set the same value as the real callback
            return topics.most_relevant_doc(texts, H_norm, topic_selected), modal_open

        return ctx.triggered_id["index"], False

    if ctx.triggered_id == "topic_user_types_nums_button":
        if modal_clicks:
            return int(document_selected_str), not modal_open
        return int(document_selected_str), modal_open

    if ctx.triggered_id == "topic_selected":
        return topics.most_relevant_doc(texts, H_norm, topic_selected), modal_open

    document_selected = int(document_selected_str)
    df = topics.docs(texts, H_norm, topic_selected).sort_values(
        topic_selected, ascending=False
    )

    if document_selected not in df["internal_index"]:
        print(f"Currently selected document ({document_selected}) invalid!")
        return 0, modal_open  # Default to something

    pos = df["internal_index"].to_list().index(document_selected)  # type: ignore
    prev = 0 if pos <= 0 else pos - 1
    next = len(df) - 1 if pos >= len(df) - 1 else pos + 1
    prev, next = df["internal_index"].iloc[[prev, next]]  # type: ignore

    if ctx.triggered_id == "document_selected_previous":
        return prev, modal_open
    else:
        return next, modal_open


def _setup_invisible_callbacks() -> None:
    """Setup callbacks for the invisible components."""
    dash.get_app().callback(
        Output("id_div", "children"),
        [
            Input("topic_selected", "children"),
            Input("document_selected", "value"),
        ],
    )(_update_id_div)

    dash.get_app().callback(
        Output("similar_docs_type", "children"),
        [
            Input("similar_docs_type_tm", "n_clicks"),
            Input("similar_docs_type_levenshtein", "n_clicks"),
            Input("similar_docs_type_embedding", "n_clicks"),
        ],
    )(_update_similar_docs_type)

    dash.get_app().callback(
        Output("dummy_1", "children"),
        [
            Input("topic_selected", "children"),
            Input("document_selected", "value"),
            Input({"type": "topic_selected_", "index": ALL}, "n_clicks"),
        ],
        [State("url", "search")],
    )(_update_search)

    dash.get_app().callback(
        Output("topic_selected", "children"),
        [
            Input({"type": "topic_selected_", "index": ALL}, "n_clicks"),
        ],
    )(_update_topic_selected)

    dash.get_app().callback(
        [
            Output("document_selected", "value"),
            Output("topic_user_types_nums_modal", "is_open"),
        ],
        [
            Input("topic_selected", "children"),
            Input("document_selected_previous", "n_clicks"),
            Input("document_selected_next", "n_clicks"),
            Input(
                {"type": "document_selected_index_button_", "index": ALL}, "n_clicks"
            ),
            Input("topic_user_types_nums_button", "n_clicks"),
        ],
        [
            State("topic_selected", "children"),
            State("document_selected", "value"),
            State("topic_user_types_nums_modal", "is_open"),
        ],
    )(_update_document_selected)


########################################################################################
# BEGIN DOCUMENT CHANGE CALLBACKS                                                      #
########################################################################################


def _update_document_topic_index(topic_selected: str, document_selected_: str) -> int:
    """Update the index of the selected document in the selected topic.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    document_selected_ : str
        (Internal index of the) Currently selected topic.

    Returns
    -------
    int
        Index of the selected document in the selected topic.
    """
    document_selected = int(document_selected_)
    df = topics.docs(texts, H_norm, topic_selected).sort_values(
        topic_selected, ascending=False
    )
    return df["internal_index"].to_list().index(document_selected) + 1  # type: ignore


def _update_current_info_table(
    topic_selected: str, document_selected: str, doc_topic_index: int
) -> DashComponent:
    """Update the information about the currently selected document.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    document_selected : str
        (Internal index of the) Currently selected topic.
    doc_topic_index : int
        Index of the selected document in the selected topic.

    Returns
    -------
    DashComponent
        Table component.
    """
    current_ = _get_current(topic_selected, int(document_selected))
    return current.info_table(current_, topic_selected, doc_topic_index, H_norm)


def _update_current_text(topic_selected: str, document_selected: str) -> DashComponent:
    """Update the text of the currently selected document.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    document_selected : str
        (Internal index of the) Currently selected topic.

    Returns
    -------
    DashComponent
        Text component.
    """
    current_ = _get_current(topic_selected, int(document_selected))
    topic_top_terms = topics.top_terms(W_norm, topic_selected)
    return current.text(current_, topic_top_terms, topic_selected)


def _setup_doc_change_callbacks() -> None:
    """Setup callbacks triggering on document change."""
    dash.get_app().callback(
        Output("document_selected_topic_index", "children"),
        [Input("topic_selected", "children"), Input("document_selected", "value")],
    )(_update_document_topic_index)

    dash.get_app().callback(
        Output("current_info_table", "children"),
        [
            Input("topic_selected", "children"),
            Input("document_selected", "value"),
            Input("document_selected_topic_index", "children"),
        ],
    )(_update_current_info_table)

    dash.get_app().callback(
        Output("current_text", "children"),
        [
            Input("topic_selected", "children"),
            Input("document_selected", "value"),
        ],
    )(_update_current_text)


########################################################################################
# BEGIN TOPIC CHANGE CALLBACKS                                                         #
########################################################################################


def _update_navbar(topic_selected: str) -> DashComponent:
    """Update the navigation bar.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    DashComponent
        Navbar component.
    """
    return navbar.get_navbar(f"Topic details: {topic_selected}")


def _update_topic_info_table(topic_selected: str, _: Any) -> DashComponent:
    """Update the topic information table.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    DashComponent
        Table component.
    """
    return topics.info_table(topic_selected)


def _update_topic_top_terms_plot(topic_selected: str, _: Any) -> go.Figure:
    """Update the topic's top terms barplot.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    go.Figure
        Barplots figure.
    """
    return topic_graphs.top_terms_barplot(topic_selected)


def _setup_topic_change_callbacks() -> None:
    """Setup callbacks triggering on topic change."""
    dash.get_app().callback(
        Output("navbar", "children"),
        Input("topic_selected", "children"),
    )(_update_navbar)

    dash.get_app().callback(
        Output("topic_info_table", "children"),
        [
            Input("topic_selected", "children"),
            Input("topic_details_page_load", "children"),
        ],
    )(_update_topic_info_table)

    dash.get_app().callback(
        Output("topic_top_terms_plot", "figure"),
        [
            Input("topic_selected", "children"),
            Input("topic_details_page_load", "children"),
        ],
    )(_update_topic_top_terms_plot)


########################################################################################
# BEGIN MODAL CALLBACKS                                                                #
########################################################################################


def _update_topic_user_types_modal_header(topic_selected: str) -> str:
    """Update the topic user types popularity modal's header.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    str
        Modal header.
    """
    return f"How common is {topic_selected}?"


def _update_topic_plots(
    topic_selected: str, _: Any
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Update the four user types topic popularity plots.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure, go.Figure]
        Tuple of the four barplots.
    """
    return (
        topic_graphs.user_types_avg_share_barplot(topic_selected),
        topic_graphs.user_types_dominant_barplot(topic_selected),
        topic_graphs.user_types_read_n_docs_barplot(topic_selected, top_n),
        topic_graphs.user_types_read_n_docs_barplot(topic_selected, 50),
    )


def _update_topic_user_types_doc_numbers(topic_selected: str, _: Any) -> DashComponent:
    """Update the document number buttons modal.

    Parameters
    ----------
    topic_selected : str
        Currently selected topic.
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    DashComponent
        Document number buttons modal content.
    """
    return topics.user_types_doc_numbers(topic_selected)


def _update_similar_docs(document_selected: str, method: str) -> list[DashComponent]:
    """Update the similar documents modal.

    Parameters
    ----------
    document_selected : str
        (Internal index of the) Currently selected topic.
    method : str
        Similarity measure used.

    Returns
    -------
    list[DashComponent]
        Similar documents modal component.
    """
    return layout_.layout_similar_docs(document_selected, method)  # type: ignore[no-any-return] # noqa[E501]


def _setup_modal_callbacks() -> None:
    """Setup callbacks for modal components."""
    dash.get_app().callback(
        Output("topic_user_types_modal", "is_open"),
        Input("topic_user_types_button", "n_clicks"),
        State("topic_user_types_modal", "is_open"),
    )(layouts.toggle_modal)
    dash.get_app().callback(
        Output("similar_docs_modal", "is_open"),
        Input("similar_docs_button", "n_clicks"),
        State("similar_docs_modal", "is_open"),
    )(layouts.toggle_modal)

    dash.get_app().callback(
        Output("topic_user_types_modal_header", "children"),
        Input("topic_selected", "children"),
    )(_update_topic_user_types_modal_header)

    dash.get_app().callback(
        [
            Output("topic_user_types_plot", "figure"),
            Output("topic_user_types_dominant_plot", "figure"),
            Output("topic_user_types_read", "figure"),
            Output("topic_user_types_read_50", "figure"),
        ],
        [
            Input("topic_selected", "children"),
            Input("topic_details_page_load", "children"),
        ],
    )(_update_topic_plots)

    dash.get_app().callback(
        Output("topic_user_types_doc_numbers", "children"),
        [
            Input("topic_selected", "children"),
            Input("topic_details_page_load", "children"),
        ],
    )(_update_topic_user_types_doc_numbers)

    dash.get_app().callback(
        Output("similar_docs", "children"),
        [
            Input("document_selected", "value"),
            Input("similar_docs_type", "children"),
        ],
    )(_update_similar_docs)
