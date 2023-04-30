"""Module containing the callbacks for the /aspects/ page."""

from typing import Any
import itertools as it
from functools import partial
import plotly.graph_objects as go

import dash
from dash.dependencies import Input, Output, State

from layouts import layouts
from utils import caching, sentiment
from graphs import aspects as graphs


def _update_mentions_area_plot(
    filter_badge_children: list[dict],
    normalize_badge_children: list[dict],
    _: Any,
    articles: bool,
) -> go.Figure:
    """Callback to update the number of mentions area plot.

    Parameters
    ----------
    filter_badge_children : list[dict]
        "Children" property of the filter badge button.
    normalize_badge_children : list[dict]
        "Children" property of the normalize badge button.
    _ : Any
        Ignored, only for triggering a callback on page load.
    articles : bool
        Whether to plot the number of mentions of articles or the other aspects.

    Returns
    -------
    go.Figure
        Updated area plot figure.
    """
    filter_ = layouts.button_badge_get_checked(filter_badge_children)
    normalize = layouts.button_badge_get_checked(normalize_badge_children)

    return graphs.n_mentions_area_plot(
        normalize=normalize, filter_=filter_, aspect="article" if articles else "others"
    )


def _update_sentiment_area_plot(
    _: Any, probs_badge_children: list[dict], articles: bool
) -> go.Figure:
    """Callback to update the sentiment area plot.

    Parameters
    ----------
    _ : Any
        Ignored, only for triggering a callback on page load.
    probs_badge_children : list[dict]
        Children property of the "Use probabilities" button badge.
    articles : bool
        Whether to plot the number of mentions of articles or the other aspects.

    Returns
    -------
    go.Figure
        Updated area plot figure.
    """
    probs = layouts.button_badge_get_checked(probs_badge_children)

    return graphs.sentiment_area_plot(
        aspect="article" if articles else "others", probs=probs
    )


def initialize_caching() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Number of mentions area plot (complete)
    - Sentiment area plot (complete)
    - User type differences plot (complete)
    - Interest type's sentiment (single, complete)
    - Interest type's share of mentions barplot (single, complete)
    - Interest type's readability barplot (single, complete)
    """
    for a, b, f, v, u in caching.progress_bar(
        it.product(
            *it.repeat([True, False], 2),
            ["article", "others"],
            ["sentiment", "n_mentions"],
            sentiment.get_user_type_names(),
        ),
        "/aspects/",
    ):
        graphs.n_mentions_area_plot(normalize=a, filter_=b, aspect=f)
        graphs.sentiment_area_plot(aspect=f, probs=a)
        graphs.aspect_user_difference_barplot(user_type_name=u, variant=v)
        graphs.sentiment_area_plot(
            aspect="all", area_plot=False, separate_interest_types=True
        )
        graphs.aspect_interest_share_barplot()
        graphs.interest_readability_barplot()


def setup_callbacks() -> None:
    """Setup the callbacks for element on the /aspects/ page."""

    for tab in ["articles", "others"]:
        # Button badges
        dash.get_app().callback(
            Output(f"{tab}_mentions_filter_button", "children"),
            Input(f"{tab}_mentions_filter_button", "n_clicks"),
            State(f"{tab}_mentions_filter_button", "children"),
        )(layouts.update_buttons_badge)
        dash.get_app().callback(
            Output(f"{tab}_mentions_normalize_button", "children"),
            Input(f"{tab}_mentions_normalize_button", "n_clicks"),
            State(f"{tab}_mentions_normalize_button", "children"),
        )(layouts.update_buttons_badge)
        dash.get_app().callback(
            Output(f"{tab}_sentiment_probs", "children"),
            Input(f"{tab}_sentiment_probs", "n_clicks"),
            State(f"{tab}_sentiment_probs", "children"),
        )(layouts.update_buttons_badge)

        # Plot callbacks
        dash.get_app().callback(
            Output(f"{tab}_mentions_area_plot", "figure"),
            [
                Input(f"{tab}_mentions_filter_button", "children"),
                Input(f"{tab}_mentions_normalize_button", "children"),
                Input("aspects_page_load_trigger", "children"),
            ],
        )(partial(_update_mentions_area_plot, articles=(tab == "articles")))
        dash.get_app().callback(
            Output(f"{tab}_sentiment_area_plot", "figure"),
            [
                Input("aspects_page_load_trigger", "children"),
                Input(f"{tab}_sentiment_probs", "children"),
            ],
        )(partial(_update_sentiment_area_plot, articles=(tab == "articles")))

        # By user type
        dash.get_app().callback(
            Output(f"aspects_user_diff_modal_{tab}", "is_open"),
            Input(f"aspects_user_diff_n_mentions_modal_button_{tab}", "n_clicks"),
            Input(f"aspects_user_diff_sentiment_modal_button_{tab}", "n_clicks"),
            State(f"aspects_user_diff_modal_{tab}", "is_open"),
        )(layouts.toggle_modal_multiple)
        for variant in ["sentiment", "n_mentions"]:
            dash.get_app().callback(
                Output(f"aspects_user_diff_{variant}_fig_{tab}", "figure"),
                Input(f"aspects_user_diff_user_{tab}", "value"),
            )(partial(graphs.aspect_user_difference_barplot, variant=variant))
        dash.get_app().callback(
            Output(f"aspects_interest_sentiment_fig_{tab}", "figure"),
            Input("aspects_page_load_trigger", "children"),
        )(
            lambda _: graphs.sentiment_area_plot(
                aspect="all", area_plot=False, separate_interest_types=True
            )
        )
        dash.get_app().callback(
            Output(f"aspects_user_share_fig_{tab}", "figure"),
            Input("aspects_page_load_trigger", "children"),
        )(lambda _: graphs.aspect_interest_share_barplot())
        dash.get_app().callback(
            Output(f"aspects_user_readability_fig_{tab}", "figure"),
            Input("aspects_page_load_trigger", "children"),
        )(lambda _: graphs.interest_readability_barplot())
