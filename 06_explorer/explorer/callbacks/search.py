"""Module containing the callbacks for the /search/ page."""

import dash
from typing import Any
from functools import partial
import plotly.graph_objects as go

from utils import caching
from layouts import layouts
import layouts.search as layout
from graphs import sentiment as sentiment_graphs


import diskcache
from dash._callback import NoUpdate
from dash import ctx, DiskcacheManager
from dash.dependencies import Component as DashComponent
from dash.dependencies import Input, Output, State, MATCH, ALL


cache = diskcache.Cache("/tmp/explorer_cache_search")
background_callback_manager = DiskcacheManager(cache)


def _update_topics(
    search_term: str,
    regex_badge_children: list[dict],
    filter_open: bool,
    filter_badge_children: list[list[dict]],
    dominant_children: list[dict],
) -> go.Figure | NoUpdate:
    """Update the topics barplot.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex_badge_children : list[dict]
        "Children" property of the 'Use regex' button badge.
    filter_open : bool
        "is_open" property of the topic modal.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.
    dominant_children : list[dict]
        "Children" property of the 'dominant topic' button badge.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    # Delay callback until option modals are closed
    if filter_open:
        return dash.no_update

    regex = layouts.button_badge_get_checked(regex_badge_children)
    filter_user_types = layouts.filter_user_types_extract(filter_badge_children)
    variant = (
        "dominant"
        if layouts.button_badge_get_checked(dominant_children)
        else "avg_share"
    )

    return sentiment_graphs.sentiment_topics(
        search_term=search_term,
        regex=regex,
        filter_user_types=filter_user_types,
        variant=variant,
    )


def _update_aspects(
    search_term: str,
    regex_badge_children: list[dict],
    filter_open: bool,
    filter_badge_children: list[list[dict]],
) -> go.Figure | NoUpdate:
    """Update the aspects barplot.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex_badge_children : list[dict]
        "Children" property of the 'Use regex' button badge.
    filter_open : bool
        Whether the filter options modal is open or not.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    # Delay callback until option modals are closed
    if filter_open:
        return dash.no_update

    regex = layouts.button_badge_get_checked(regex_badge_children)
    filter_user_types = layouts.filter_user_types_extract(filter_badge_children)

    return sentiment_graphs.search_aspects(
        search_term=search_term,
        regex=regex,
        filter_user_types=filter_user_types,
    )


def _update_user_types(
    search_term: str,
    regex_badge_children: list[dict],
    filter_open: bool,
    filter_badge_children: list[list[dict]],
    user_types_badge_children: list[dict],
) -> go.Figure | NoUpdate:
    """Update the user types barplot.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex_badge_children : list[dict]
        "Children" property of the 'Use regex' button badge.
    filter_open : bool
        Whether the filter options modal is open or not.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.
    user_types_badge_children : list[dict]
        "Children" property of the "Relative" button badge.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    # Delay callback until option modals are closed
    if filter_open:
        return dash.no_update

    regex = layouts.button_badge_get_checked(regex_badge_children)
    filter_user_types = layouts.filter_user_types_extract(filter_badge_children)
    relative = layouts.button_badge_get_checked(user_types_badge_children)

    return sentiment_graphs.search_user_types(
        search_term=search_term,
        regex=regex,
        filter_user_types=filter_user_types,
        relative=relative,
    )


def _update_details(
    search_term: str,
    filter_open: bool,
    n: str,
    regex_badge_children: list[dict],
    multiselect_badge_children: list[dict],
    filter_badge_children: list[list[dict]],
) -> list[DashComponent] | NoUpdate:
    """Callback to update the search details accordion.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    filter_open : bool
        Whether the filter options modal is open or not.
    n : str
        "Children" property of the dummy component that contains the number mentions to
        display.
    regex_badge_children : list[dict]
        "Children" property of the 'Use regex' button badge.
    multiselect_badge_children : list[dict]
        "Children" property of the multiselect button badge.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.

    Returns
    -------
    list[DashComponent] | NoUpdate
        _description_
    """
    # Delay callback until option modals are closed
    if filter_open:
        return dash.no_update

    regex = layouts.button_badge_get_checked(regex_badge_children)
    filter_user_types = layouts.filter_user_types_extract(filter_badge_children)

    return layout.details(
        search_term=search_term,
        regex=regex,
        filter_user_types=filter_user_types,
        n=int(n),
        always_open=layouts.button_badge_get_checked(multiselect_badge_children),
    )


def _update_details_n(
    _a: Any, _b: Any, _c: Any, n_clicks: int | None, n: str
) -> int | NoUpdate:
    """Callback to update the number of search results shown.

    Parameters
    ----------
    _a : Any
        Search term - ignored, for triggering callback only.
    _b : Any
        Whether the search term is a regular expression - ignored, for triggering
        callback only.
    _c : Any
        Sentiment filter button badge children - ignored, for triggering callback only.
    n_clicks : int | None
        Number of clicks on the "Load more" button.
    n : str
        Current number of search results shown.

    Returns
    -------
    int | NoUpdate
        Updated number of search results shown.
    """
    del _a, _b, _c  # No not accessed warning

    # Button click triggers increase
    if ctx.triggered_id == "search_details_load_more" and n_clicks is not None:
        return int(n) + 20

    # Reset to default value
    if int(n) != 20:
        return 20

    # No need to trigger callback for details by returning something
    return dash.no_update


def initialize_caching() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Topic bar plot (two examples)
    - Aspect bar plot (two examples)
    - User types bar plot (two examples)
    - Search result details (two examples)
    """
    examples = [("face", False), ("face|facial", True)]

    for t, r in caching.progress_bar(examples, "/search/"):
        layout.details(t, r)
        for variant in ["dominant", "avg_share"]:
            sentiment_graphs.sentiment_topics(search_term=t, regex=r, variant=variant)
        sentiment_graphs.search_aspects(search_term=t, regex=r)
        for relative in [True, False]:
            sentiment_graphs.search_user_types(
                search_term=t, regex=r, relative=relative
            )


def setup_callbacks() -> None:
    """Setup the callbacks for element on the /search/ page."""
    # Search bar
    dash.get_app().callback(
        Output("search_value_regex", "children"),
        Input("search_value_regex", "n_clicks"),
        State("search_value_regex", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("search_info_modal", "is_open"),
        Input("search_info_modal_button", "n_clicks"),
        State("search_info_modal", "is_open"),
    )(layouts.toggle_modal_multiple)

    # Topic modal
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("search_topic_modal", "is_open"),
        Input("search_topic_modal_button", "n_clicks"),
        State("search_topic_modal", "is_open"),
    )
    dash.get_app().callback(
        [
            Output("search_topic_dominant", "children"),
            Output("search_topic_share", "children"),
        ],
        [
            Input("search_topic_dominant", "n_clicks"),
            Input("search_topic_share", "n_clicks"),
        ],
        [
            State("search_topic_dominant", "children"),
            State("search_topic_share", "children"),
        ],
    )(
        partial(
            layouts.update_buttons_badges_mutex,
            names=["dominant", "share"],
        )
    )
    dash.get_app().callback(
        Output("search_topic_fig", "figure"),
        [
            Input("search_value", "value"),
            Input("search_value_regex", "children"),
            Input("search_filter_modal", "is_open"),
            Input({"type": "search_filter_badge", "index": ALL}, "children"),
            Input("search_topic_dominant", "children"),
        ],
    )(_update_topics)

    # Aspects modal
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("search_aspects_modal", "is_open"),
        Input("search_aspects_modal_button", "n_clicks"),
        State("search_aspects_modal", "is_open"),
    )
    dash.get_app().callback(
        Output("search_aspects_fig", "figure"),
        [
            Input("search_value", "value"),
            Input("search_value_regex", "children"),
            Input("search_filter_modal", "is_open"),
            Input({"type": "search_filter_badge", "index": ALL}, "children"),
        ],
    )(_update_aspects)

    # User types modal
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("search_user_types_modal", "is_open"),
        Input("search_user_types_modal_button", "n_clicks"),
        State("search_user_types_modal", "is_open"),
    )
    dash.get_app().callback(
        Output("search_user_types_relative", "children"),
        Input("search_user_types_relative", "n_clicks"),
        State("search_user_types_relative", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("search_user_types_fig", "figure"),
        [
            Input("search_value", "value"),
            Input("search_value_regex", "children"),
            Input("search_filter_modal", "is_open"),
            Input({"type": "search_filter_badge", "index": ALL}, "children"),
            Input("search_user_types_relative", "children"),
        ],
    )(_update_user_types)

    # Settings
    dash.get_app().callback(
        Output("search_filter_modal", "is_open"),
        Input("search_filter_modal_button", "n_clicks"),
        State("search_filter_modal", "is_open"),
    )(layouts.toggle_modal_multiple)
    dash.get_app().callback(
        Output("search_filter_modal_button", "children"),
        Input("search_filter_modal", "is_open"),
        State({"type": "search_filter_badge", "index": ALL}, "children"),
    )(layouts.filter_user_types_modal_callback)
    dash.get_app().callback(
        Output({"type": "search_filter_badge", "index": MATCH}, "children"),
        Input({"type": "search_filter_badge", "index": MATCH}, "n_clicks"),
        State({"type": "search_filter_badge", "index": MATCH}, "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("search_multiselect", "children"),
        Input("search_multiselect", "n_clicks"),
        State("search_multiselect", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("search_details_n", "children"),
        [
            Input("search_value", "value"),
            Input("search_value_regex", "children"),
            Input({"type": "search_filter_badge", "index": ALL}, "children"),
            Input("search_details_load_more", "n_clicks"),
        ],
        [
            State("search_details_n", "children"),
        ],
    )(_update_details_n)

    # Details accordion
    dash.get_app().callback(
        Output("search_details", "children"),
        [
            Input("search_value", "value"),
            Input("search_filter_modal", "is_open"),
            Input("search_details_n", "children"),
            Input("search_value_regex", "children"),
            Input("search_multiselect", "children"),
        ],
        [
            State({"type": "search_filter_badge", "index": ALL}, "children"),
        ],
        background=True,
        running=[(Output("search_details_loading", "hidden"), False, True)],
        manager=background_callback_manager,
    )(_update_details)
