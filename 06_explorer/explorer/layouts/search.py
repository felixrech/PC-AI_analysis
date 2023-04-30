"""Module providing the layout for the search page."""

import pandas as pd
from typing import Callable
from functools import partial

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from utils import utils
from graphs import graphs
from layouts import layouts
from utils.caching import memoize
from layouts import navbar as navbar_
from utils import sentiment as sentiment_utils


def get_layout(sentiments: pd.DataFrame) -> Callable[..., list[DashComponent]]:
    """Creates the layout() function.

    Parameters
    ----------
    sentiments : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module.

    Returns
    -------
    Callable[[str, str], list[DashComponent]]
        layout() function. Can use aspect_type and aspect_subtype values in the search
        but ignores any other parameters.
    """

    def layout(**kwargs) -> list[DashComponent]:
        """Compute the layout for the sentiment page.

        Returns
        -------
        list[DashComponent]
            Page layout.
        """
        return [
            html.Div(children=navbar_.get_navbar("Search")),
            _searchbar_layout(),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(width=2),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Center("Searching..."),
                                    html.Br(),
                                    dbc.Spinner(size="lg"),
                                    html.Br(),
                                    html.Br(),
                                ],
                                id="search_details_loading",
                            ),
                            html.Div(id="search_details"),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                layouts.topics_modal(
                                    "search",
                                    bottom_margin=False,
                                    modal_additions=[_DISCLAIMER_FOOTER],
                                ),
                                className="mm-a",
                            ),
                            html.Div(_aspects_modal(), className="mm-a"),
                            html.Div(_user_types_modal(), className="mm-a"),
                            html.Div(
                                layouts.filter_user_types_modal(sentiments, "search"),
                                className="mm-a",
                            ),
                            html.Div(
                                layouts.multiselect_button("search"), className="mm-a"
                            ),
                        ],
                        width=2,
                    ),
                ]
            ),
            *_dummy_components(),
        ]

    return layout


def _searchbar_layout() -> DashComponent:
    """Layout for the search bar containing search term entry box, search guide modal,
    and a button badge to switch between exact match and regex.

    Returns
    -------
    DashComponent
        Searchbar layout.
    """
    info_button = layouts.make_changes(
        dbc.Button(layouts.INFO_ICON, id="search_info_modal_button"),
        style=(
            small_button_style := {
                "padding-left": "0.1cm",
                "padding-right": "0.1cm",
                "--bs-btn-padding-y": "0",
            }
        ),
        className="button_invisible",
    )
    normal_info = (
        "Looks for any sentences that contain the search term as an exact match. "
        "Ignores casing."
    )
    regex_info = [
        dcc.Markdown(
            "Search using regular expressions (regex). For example, `face|facial` will "
            "match any sentences that contain `face` or `facial`. "
            "See [here](https://regexone.com/) for an interactive tutorial into regex. "
            "Search ignores casing.\n\n"
            "Search pattern must not contain any capturing groups (i.e. replace any "
            "`(subpattern)` with `(?:subpattern)`)."
        ),
    ]
    info_modal = dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Search guide"),
            ),
            dbc.ModalBody(
                [
                    html.Div(html.B("Normal search:")),
                    html.Div(normal_info),
                    html.Br(),
                    html.Br(),
                    html.Div(html.B("Regex search:")),
                    html.Div(regex_info),
                ]
            ),
        ],
        id="search_info_modal",
        size="lg",
    )
    info = html.Span([info_button, info_modal])

    search_bar = dcc.Input(
        placeholder="Enter search term here",
        id="search_value",
        **dict(type="text", debounce=True, persistence=True),
    )

    buffer, small_buffer = [
        html.Span(style=dict(display="inline-block", width=f"{i}rem")) for i in [2, 0.5]
    ]

    use_regex = layouts.make_changes(
        layouts.button_badge(
            "Use regex", id="search_value_regex", checked=False, fake=False
        ),
        style=dict(width="8rem", height="1.7rem") | small_button_style,
    )

    return html.Div([info, small_buffer, search_bar, buffer, use_regex])


def _aspects_modal() -> list[DashComponent]:
    """Layout of the button and modal containing information about which aspects are
    common among the search results.

    Returns
    -------
    list[DashComponent]
        Layout components.
    """
    modal = html.Div(
        [
            dbc.Button(
                "Aspects of search",
                id="search_aspects_modal_button",
                style={"width": "80%", "max-width": "5cm"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle("Which aspects do these sentences contain?")
                    ),
                    dbc.ModalBody(
                        dcc.Graph(
                            figure=graphs.dummy_fig(),
                            id="search_aspects_fig",
                            **graphs.GRAPH_CONF,
                        ),
                    ),
                    _DISCLAIMER_FOOTER,
                ],
                id="search_aspects_modal",
                **dict(is_open=False, size="xl"),
            ),
        ]
    )
    return [modal]


def _user_types_modal() -> list[DashComponent]:
    """Layout of the button and modal containing information about which user types are
    common among the search results.

    Returns
    -------
    list[DashComponent]
        Layout components.
    """
    button_style = {"width": "80%", "max-width": "5cm", "margin-bottom": "0.5cm"}
    modal = html.Div(
        [
            dbc.Button(
                "User types",
                id="search_user_types_modal_button",
                style=button_style,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            "Which user types do these sentences originate from?"
                        )
                    ),
                    dbc.ModalBody(
                        [
                            html.Center(
                                layouts.make_changes(
                                    layouts.button_badge(
                                        text="Relative",
                                        id="search_user_types_relative",
                                        checked=False,
                                        fake=False,
                                    ),
                                    style=button_style,
                                    title=(
                                        "Whether to use absolute numbers (❌) or share "
                                        "of matches among all sentences by user type "
                                        "(✔️)."
                                    ),
                                ),
                            ),
                            html.Br(),
                            dcc.Graph(
                                figure=graphs.dummy_fig(),
                                id="search_user_types_fig",
                                **graphs.GRAPH_CONF_UNRESPONSIVE,
                            ),
                        ]
                    ),
                    _DISCLAIMER_FOOTER,
                ],
                id="search_user_types_modal",
                **dict(is_open=False, size="xl"),
            ),
        ]
    )
    return [modal]


@memoize
def details(
    search_term: str = "",
    regex: bool = False,
    filter_user_types: list[str] = [],
    n: int = 20,
    always_open: bool = False,
) -> list[DashComponent]:
    """Compute layout with details about search term matches.

    Parameters
    ----------
    search_term : str, optional
        Search term to use, e.g. "face" - by default "".
    regex : bool, optional
        Whether the search term uses regular expressions, by default False.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.
    n : int, optional
        Number of matches to include, by default 20.
    always_open : bool, optional
        Whether to allow for the opening of more than one accordion item at a time, by
        default False.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    all_sentiments = sentiment_utils.load_sentiment(
        filter_=False, filter_user_types=list(filter_user_types)
    )
    search_results = sentiment_utils.search_sentiment(
        search_term, regex, tuple(filter_user_types)
    )

    button_args = (
        dict(style={"margin-top": "0.5cm"})
        if len(search_results) > n
        else layouts.STYLE_INVISIBLE
    )

    if len(search_results) == 0:
        return [html.Div("No search results. Try a different search term!")]

    return [
        html.Div(html.I(f"{len(search_results)} matches found!", className="fs-sm")),
        *layouts.accordion(
            search_results.iloc[:n],
            title=_details_single_title,
            text=partial(
                layouts.accordion_windowed_text,
                sentiments=all_sentiments,
                window_size=3,
                main_column="result",
            ),
            always_open=always_open,
        ),
        dbc.Button(
            "Load more",
            id="search_details_load_more",
            className="mm-a",
            **button_args,
        ),
        html.Div(
            html.I(
                f"(Currently showing {min(n, len(search_results))} matches.)",
                className="fs-sm",
            )
        ),
    ]


def _details_single_title(sentence: pd.Series) -> str:
    """Compute the title of a single mention accordion item.

    Parameters
    ----------
    sentence : pd.Series
        Row of the sentiment dataset. Needs "organization", "user_type", and "result"
        labels.

    Returns
    -------
    str
        Title.
    """
    org = (
        sentence["organization"]
        if sentence["organization"] is not None
        else "<Anonymous>"
    )
    org = utils.cutoff_text(org, 40).ljust(41)

    user_type = f"{utils.user_type_to_name(sentence['user_type'])}".rjust(1)

    match = (
        sentence["result"]
        if len(sentence["result"]) <= 13
        else f"{sentence['result'][:12]}…"
    )
    match = f'"{match}"'.ljust(15)

    return f"Match: {match}   by {org} ({user_type})"


def _dummy_components() -> list[DashComponent]:
    """Compute the layout of invisible dummy components.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        html.Div(children="20", id="search_details_n", **layouts.STYLE_INVISIBLE),
        html.Div(id="search_page_load_trigger", **layouts.STYLE_INVISIBLE),
    ]


_DISCLAIMER_FOOTER = dbc.ModalFooter(
    [
        layouts.INFO_ICON,
        html.Div(
            "Statistics are computed for actual search results - including "
            "potential user type filters.",
            style={"font-size": "small"},
        ),
    ]
)