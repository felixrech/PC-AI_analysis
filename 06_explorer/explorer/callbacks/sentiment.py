"""Module containing the callbacks for the /sentiment/ page."""

import dash
from typing import Any
import itertools as it
from functools import partial
import plotly.graph_objects as go

from layouts import layouts
import layouts.sentiment as layout
from graphs import sentiment as sentiment_graphs
from utils import caching, sentiment as sentiment_utils

from dash import ctx
from dash._callback import NoUpdate
from dash.dependencies import Component as DashComponent
from dash.dependencies import Input, Output, State, MATCH, ALL


def _update_navbar(aspect_type: str, aspect_subtype: str) -> DashComponent:
    """Callback to update the navigation bar based on the selected aspect.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    DashComponent
        Updated navigation bar layout.
    """
    return layout.navbar(aspect_type, aspect_subtype)


def _update_selection_options(show_all_badge_children: dict, _: Any) -> list[dict]:
    """Update the aspect selection options based on whether to show all or just those
    with at least 30 mentions.

    Parameters
    ----------
    show_all_badge_children : dict
        Children of the show_all_icon element.
    _ : Any
        Page load trigger - ignored (for triggering callback only).

    Returns
    -------
    list[dict]
        Dropdown options.
    """
    show_all = _get_selection_show_all(show_all_badge_children)  # noqa [pd.df.query]

    aspects = (
        sentiment_utils.get_aspects()
        .query("@show_all or n_mentions >= 30")
        .assign(
            # Bring aspect and n_mentions into nicer string format
            aspect_nice=lambda df: df.apply(
                lambda row: sentiment_utils.aspect_nice_format(
                    row["aspect_type"], row["aspect_subtype"]
                ),
                axis=1,
            ),
            n_mentions=lambda df: "(" + df["n_mentions"].astype(str) + " mentions)",
        )
        .assign(
            # Shorten and justify aspect and n_mentions
            aspect_nice=lambda df: df["aspect_nice"].str.ljust(
                df["aspect_nice"].str.len().max()
            ),
            n_mentions=lambda df: df["n_mentions"].str.rjust(
                df["n_mentions"].str.len().max()
            ),
            # Create final labels
            label=lambda df: df["aspect_nice"] + " " * 5 + df["n_mentions"],
        )
    )

    return [
        dict(label=row["label"], value=f"{row['aspect_type']};{row['aspect_subtype']}")
        for _, row in aspects.iterrows()
    ]


def _update_selection_show_all(
    n_clicks: int | None, badge_children: dict
) -> DashComponent | NoUpdate:
    """Update the badge icon denoting whether the aspect selection dropdown options
    should be filtered those aspects with at least mentions.

    Parameters
    ----------
    n_clicks : int | None
        Number of clicks on the button that contains the badge.
    badge_children : dict
        Current children of the badge icon.

    Returns
    -------
    DashComponent
        Updated badge.
    """
    # Ignore initial page callback
    if not n_clicks:
        return dash.no_update

    # Cycle between checked and x mark icons
    if not _get_selection_show_all(badge_children):
        return layouts.SINGLE_CHECK
    else:
        return layouts.X_MARK_NO_CIRC_ICON


def _update_aspect(value: str, options: list[dict[str, str]]) -> tuple[str, str]:
    """Callback to update the currently selected aspect.

    Resets to no aspect ("", "") if selected value is not in the options.

    Parameters
    ----------
    value : str
        Value of the selected entry in the aspect dropdown.
    options : list[dict[str, str]]
        List of dropdown options.

    Returns
    -------
    tuple[str, str]
        Aspect type and subtype as strings.
    """
    if not value or value not in [option["value"] for option in options]:
        return "", ""

    return tuple(value.split(";"))  # type: ignore[return-value]


def _update_aspect_mentions_warning_open(aspect_type: str, aspect_subtype: str) -> bool:
    """Callback to decide whether to open the modal warning about few mentions.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    bool
        Whether the modal should be opened or not.
    """
    sentiments = sentiment_utils.load_sentiment(
        aspect_type=aspect_type, aspect_subtype=aspect_subtype
    )

    if len(sentiments) < 30:
        return True

    return False


def _update_overview_pie_chart(aspect_type: str, aspect_subtype: str) -> go.Figure:
    """Update the sentiment overview pie chart.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    go.Figure
        Updated pie chart.
    """
    return sentiment_graphs.sentiment_overview_pie_chart(
        str(aspect_type), str(aspect_subtype)
    )


def _update_overview_barplot(
    is_open: bool,
    aspect_type: str,
    aspect_subtype: str,
    normalize_badge_children: list[dict],
    share_badge_children: list[dict],
    pos_neg_badge_children: list[dict],
) -> go.Figure | NoUpdate:
    """Update the sentiment overview barplot.

    Parameters
    ----------
    is_open : bool
        Whether the barplot options modal is currently open or not.
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".
    normalize_badge_children : list[dict]
        "Children" property of the normalize bars button badge.
    share_badge_children : list[dict]
        "Children" property of the average share (over dominant sentiment) button badge.
    pos_neg_badge_children : list[dict]
        "Children" property of the limit to positive and negative sentiments button
        badge.

    Returns
    -------
    go.Figure | NoUpdate
        Updated barplot.
    """
    # Delay figure update until the options modal is closed
    if is_open:
        return dash.no_update

    return sentiment_graphs.sentiment_overview_barplot(
        str(aspect_type),
        str(aspect_subtype),
        normalized=layouts.button_badge_get_checked(normalize_badge_children),
        probs=layouts.button_badge_get_checked(share_badge_children),
        limit_pos_neg=layouts.button_badge_get_checked(pos_neg_badge_children),
    )


def _update_word_clouds(aspect_type: str, aspect_subtype: str) -> list[str]:
    """Update the word cloud figures.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    list[str]
        List of the two word cloud figures.
    """
    return [
        sentiment_utils.word_cloud(sentiment, aspect_type, aspect_subtype)
        for sentiment in ["positive", "negative"]
    ]


def _update_word_clouds_visibility(button_badge_children: list[dict]) -> dict[str, str]:
    """Update whether or not the word clouds are visible.

    Parameters
    ----------
    button_badge_children : list[dict]
        "Children" property of the word clouds button badge.

    Returns
    -------
    dict[str, str]
        "Style" property of the div holding the word clouds.
    """
    if layouts.button_badge_get_checked(button_badge_children):
        return dict()
    else:
        return dict(display="none")


def _update_topics(
    aspect_type: str, aspect_subtype: str, dominant_children: list[dict]
) -> go.Figure:
    """Update the topics barplot.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".
    dominant_children : list[dict]
        "Children" property of the 'dominant topic' button badge.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    variant = (
        "dominant"
        if layouts.button_badge_get_checked(dominant_children)
        else "avg_share"
    )
    return sentiment_graphs.sentiment_topics(
        aspect_type=aspect_type, aspect_subtype=aspect_subtype, variant=variant
    )


def _update_sort_button(
    is_open: bool,
    sort_pos_children: list[dict],
    sort_neg_children: list[dict],
    sort_ext_children: list[dict],
) -> str | NoUpdate:
    """Update the text of the button to open the sort options modal.

    Parameters
    ----------
    is_open : bool
        Whether the sort options modal is open or not.
    sort_pos_children : list[dict]
        "Children" property of the positive sort order button badge.
    sort_neg_children : list[dict]
        "Children" property of the negative sort order button badge.
    sort_ext_children : list[dict]
        "Children" property of the extreme sort order button badge.

    Returns
    -------
    str | NoUpdate
        Updated text, e.g. "Sort: extreme".
    """
    # Delay button update until the options modal is closed
    if is_open:
        return dash.no_update

    return f"Sort: {_get_sort(sort_pos_children, sort_neg_children, sort_ext_children)}"


def _update_details(
    aspect_type: str,
    aspect_subtype: str,
    sort_open: bool,
    filter_open: bool,
    n: str,
    multiselect_badge_children: list[dict],
    sort_pos_children: list[dict],
    sort_neg_children: list[dict],
    sort_ext_children: list[dict],
    filter_badge_children: list[list[dict]],
) -> list[DashComponent] | NoUpdate:
    """Callback to update the sentiment details accordion.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        General part of the aspect, e.g. "article".
    sort_open : bool
        Whether the sort options modal is open or not.
    filter_open : bool
        Whether the filter options modal is open or not.
    n : str
        "Children" property of the dummy component that contains the number mentions to
        display.
    multiselect_badge_children : list[dict]
        "Children" property of the multiselect button badge.
    sort_pos_children : list[dict]
        "Children" property of the positive sort order button badge.
    sort_neg_children : list[dict]
        "Children" property of the negative sort order button badge.
    sort_ext_children : list[dict]
        "Children" property of the extreme sort order button badge.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.

    Returns
    -------
    list[DashComponent] | NoUpdate
        Updated layout.
    """
    # Delay callback until option modals are closed
    if sort_open or filter_open:
        return dash.no_update

    # Extract sorting method and user types from button badges
    sort = _get_sort(sort_pos_children, sort_neg_children, sort_ext_children)
    filter_user_types = layouts.filter_user_types_extract(filter_badge_children)

    return layout.details(
        aspect_type=aspect_type,
        aspect_subtype=aspect_subtype,
        sort=sort,
        filter_user_types=filter_user_types,
        n=int(n),
        always_open=layouts.button_badge_get_checked(multiselect_badge_children),
    )


def _update_details_n(
    _a: Any, _b: Any, _c: Any, _d: Any, _e: Any, _f: Any, n_clicks: int | None, n: str
) -> int | NoUpdate:
    """Callback to update the number of mentions shown.

    Parameters
    ----------
    _a : Any
        Aspect type - ignored, for triggering callback only.
    _b : Any
        Aspect subtype - ignored, for triggering callback only.
    _c : Any
        Sentiment sort (positive) - ignored, for triggering callback only.
    _d : Any
        Sentiment sort (negative) - ignored, for triggering callback only.
    _e : Any
        Sentiment sort (extreme) - ignored, for triggering callback only.
    _f : Any
        Sentiment filter button badge children - ignored, for triggering callback only.
    n_clicks : int | None
        Number of clicks on the "Load more" button.
    n : str
        Current number of mentions shown.

    Returns
    -------
    int | NoUpdate
        Updated number of mentions shown.
    """
    del _a, _b, _c, _d, _e, _f  # No not accessed warning

    # Button click triggers increase
    if ctx.triggered_id == "sentiment_details_load_more" and n_clicks is not None:
        return int(n) + 20

    # Reset to default value
    if int(n) != 20:
        return 20

    # No need to trigger callback for details by returning something
    return dash.no_update


def initialize_caching() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Sentiment overview pie chart (complete)
    - Sentiment by user type bar plot (complete)
    - Sentiment details (common)
    - Word clouds (complete)
    """
    aspects_ = ["aspect_type", "aspect_subtype"]
    aspects = it.chain(
        sentiment_utils.get_aspects()[aspects_].itertuples(index=False),
        [("", "")],
    )
    for t, s in caching.progress_bar(aspects, "/sentiment/"):
        sentiment_graphs.sentiment_overview_pie_chart(t, s)
        for n, p, l in it.product(*it.repeat([True, False], 3)):
            sentiment_graphs.sentiment_overview_barplot(
                aspect_type=t, aspect_subtype=s, normalized=n, probs=p, limit_pos_neg=l
            )
        for r, o in it.product(["extreme", "positive", "negative"], [True, False]):
            layout.details(
                aspect_type=t,
                aspect_subtype=s,
                sort=r,
                filter_user_types=[],
                n=20,
                always_open=o,
            )
        for p in ["positive", "negative"]:
            sentiment_utils.word_cloud(sentiment=p, aspect_type=t, aspect_subtype=s)
        for v in ["dominant", "avg_share"]:
            sentiment_graphs.sentiment_topics(
                aspect_type=t, aspect_subtype=s, variant=v
            )


def setup_callbacks() -> None:
    """Setup the callbacks for element on the /sentiment/ page."""

    # Navbar
    dash.get_app().callback(
        Output("sentiment_navbar", "children"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
        ],
    )(_update_navbar)

    # Dummy values
    dash.get_app().callback(
        [
            Output("aspect_type", "children"),
            Output("aspect_subtype", "children"),
        ],
        [
            Input("aspect_dropdown", "value"),
            Input("aspect_dropdown", "options"),
        ],
    )(_update_aspect)
    dash.get_app().callback(
        Output("sentiment_details_n", "children"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
            State("sentiment_sort_positive", "children"),
            Input("sentiment_sort_negative", "children"),
            Input("sentiment_sort_extreme", "children"),
            Input({"type": "sentiment_filter_badge", "index": ALL}, "children"),
            Input("sentiment_details_load_more", "n_clicks"),
        ],
        [
            State("sentiment_details_n", "children"),
        ],
    )(_update_details_n)

    # Selection
    dash.get_app().callback(
        Output("aspect_dropdown", "options"),
        [
            Input("selection_show_all_icon", "children"),
            Input("sentiment_page_load_trigger", "children"),
        ],
    )(_update_selection_options)
    dash.get_app().callback(
        Output("selection_show_all_icon", "children"),
        Input("selection_show_all", "n_clicks"),
        State("selection_show_all_icon", "children"),
    )(_update_selection_show_all)

    # Overview: pie chart and barplot
    dash.get_app().callback(
        Output("sentiment_overview_barplot", "figure"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
        ],
    )(_update_overview_pie_chart)
    dash.get_app().callback(
        Output("aspect_user_types_sentiment", "figure"),
        [
            Input("sentiment_barplot_modal", "is_open"),
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
        ],
        [
            State("sentiment_barplot_normalize", "children"),
            State("sentiment_barplot_share", "children"),
            State("sentiment_barplot_pos_neg", "children"),
        ],
    )(_update_overview_barplot)

    # Word clouds
    dash.get_app().callback(
        [
            Output("sentiment_word_cloud_positive", "src"),
            Output("sentiment_word_cloud_negative", "src"),
        ],
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
        ],
    )(_update_word_clouds)
    dash.get_app().callback(
        Output("sentiment_word_clouds_button_badge", "children"),
        Input("sentiment_word_clouds_button_badge", "n_clicks"),
        State("sentiment_word_clouds_button_badge", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("sentiment_word_clouds_div", "style"),
        Input("sentiment_word_clouds_button_badge", "children"),
    )(_update_word_clouds_visibility)

    # Topics modal figure
    dash.get_app().callback(
        Output("sentiment_topic_fig", "figure"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
            Input("sentiment_topic_dominant", "children"),
        ],
    )(_update_topics)

    # Details accordion
    dash.get_app().callback(
        Output("sentiment_details", "children"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
            Input("sentiment_sort_modal", "is_open"),
            Input("sentiment_filter_modal", "is_open"),
            Input("sentiment_details_n", "children"),
            Input("sentiment_multiselect", "children"),
        ],
        [
            State("sentiment_sort_positive", "children"),
            State("sentiment_sort_negative", "children"),
            State("sentiment_sort_extreme", "children"),
            State({"type": "sentiment_filter_badge", "index": ALL}, "children"),
        ],
    )(_update_details)

    # Modal opening callbacks
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("sentiment_topic_modal", "is_open"),
        Input("sentiment_topic_modal_button", "n_clicks"),
        State("sentiment_topic_modal", "is_open"),
    )
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("sentiment_sort_modal", "is_open"),
        Input("sentiment_sort_modal_button", "n_clicks"),
        State("sentiment_sort_modal", "is_open"),
    )
    dash.get_app().clientside_callback(
        layouts.TOGGLE_MODAL,
        Output("sentiment_filter_modal", "is_open"),
        Input("sentiment_filter_modal_button", "n_clicks"),
        State("sentiment_filter_modal", "is_open"),
    )
    dash.get_app().callback(
        Output("sentiment_barplot_modal", "is_open"),
        Input("sentiment_barplot_button", "n_clicks"),
        State("sentiment_barplot_modal", "is_open"),
    )(layouts.toggle_modal)
    dash.get_app().callback(
        Output("sentiment_aspect_mentions_warning", "is_open"),
        [
            Input("aspect_type", "children"),
            Input("aspect_subtype", "children"),
        ],
    )(_update_aspect_mentions_warning_open)

    # Modal buttons callbacks
    dash.get_app().callback(
        Output("sentiment_sort_modal_button", "children"),
        Input("sentiment_sort_modal", "is_open"),
        [
            State("sentiment_sort_positive", "children"),
            State("sentiment_sort_negative", "children"),
            State("sentiment_sort_extreme", "children"),
        ],
    )(_update_sort_button)
    dash.get_app().callback(
        Output("sentiment_filter_modal_button", "children"),
        Input("sentiment_filter_modal", "is_open"),
        State({"type": "sentiment_filter_badge", "index": ALL}, "children"),
    )(layouts.filter_user_types_modal_callback)

    # Badge buttons callbacks
    dash.get_app().callback(
        [
            Output("sentiment_sort_positive", "children"),
            Output("sentiment_sort_negative", "children"),
            Output("sentiment_sort_extreme", "children"),
        ],
        [
            Input("sentiment_sort_positive", "n_clicks"),
            Input("sentiment_sort_negative", "n_clicks"),
            Input("sentiment_sort_extreme", "n_clicks"),
        ],
        [
            State("sentiment_sort_positive", "children"),
            State("sentiment_sort_negative", "children"),
            State("sentiment_sort_extreme", "children"),
        ],
    )(
        partial(
            layouts.update_buttons_badges_mutex,
            names=["positive", "negative", "extreme"],
        )
    )
    dash.get_app().callback(
        [
            Output("sentiment_topic_dominant", "children"),
            Output("sentiment_topic_share", "children"),
        ],
        [
            Input("sentiment_topic_dominant", "n_clicks"),
            Input("sentiment_topic_share", "n_clicks"),
        ],
        [
            State("sentiment_topic_dominant", "children"),
            State("sentiment_topic_share", "children"),
        ],
    )(
        partial(
            layouts.update_buttons_badges_mutex,
            names=["dominant", "share"],
        )
    )
    dash.get_app().callback(
        Output({"type": "sentiment_filter_badge", "index": MATCH}, "children"),
        Input({"type": "sentiment_filter_badge", "index": MATCH}, "n_clicks"),
        State({"type": "sentiment_filter_badge", "index": MATCH}, "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("sentiment_barplot_normalize", "children"),
        Input("sentiment_barplot_normalize", "n_clicks"),
        State("sentiment_barplot_normalize", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        [
            Output("sentiment_barplot_share", "children"),
            Output("sentiment_barplot_dominant", "children"),
        ],
        [
            Input("sentiment_barplot_share", "n_clicks"),
            Input("sentiment_barplot_dominant", "n_clicks"),
        ],
        [
            State("sentiment_barplot_share", "children"),
            State("sentiment_barplot_dominant", "children"),
        ],
    )(partial(layouts.update_buttons_badges_mutex, names=["share", "dominant"]))
    dash.get_app().callback(
        Output("sentiment_barplot_pos_neg", "children"),
        Input("sentiment_barplot_pos_neg", "n_clicks"),
        State("sentiment_barplot_pos_neg", "children"),
    )(layouts.update_buttons_badge)
    dash.get_app().callback(
        Output("sentiment_multiselect", "children"),
        Input("sentiment_multiselect", "n_clicks"),
        State("sentiment_multiselect", "children"),
    )(layouts.update_buttons_badge)


def _get_sort(
    sort_pos_children: list[dict],
    sort_neg_children: list[dict],
    sort_ext_children: list[dict],
) -> str:
    """Extract the selected sort option from the three sort button badges.

    Parameters
    ----------
    sort_pos_children : list[dict]
        "Children" property of the positive sort order button badge.
    sort_neg_children : list[dict]
        "Children" property of the negative sort order button badge.
    sort_ext_children : list[dict]
        "Children" property of the extreme sort order button badge.

    Returns
    -------
    str
        Sort option value: "positive", "negative", or "extreme"
    """
    if layouts.button_badge_get_checked(sort_pos_children):
        return "positive"
    elif layouts.button_badge_get_checked(sort_neg_children):
        return "negative"
    elif layouts.button_badge_get_checked(sort_ext_children):
        return "extreme"
    else:  # Default to something in case of bugs
        return "extreme"


def _get_selection_show_all(badge_children: dict) -> bool:
    """Based on the icon badge, compute whether to show all aspects or only those with
    more than 30 mentions.

    Parameters
    ----------
    badge_children : dict
        Children of the show_all_icon element.

    Returns
    -------
    bool
        Whether to show all or filter.
    """
    check_class: str = layouts.SINGLE_CHECK.className  # type: ignore
    return bool(badge_children["props"]["className"] == check_class)
