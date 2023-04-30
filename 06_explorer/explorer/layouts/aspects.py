"""Module containing the layouts for the /aspects/ page."""

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from utils import sentiment
from layouts import layouts, navbar
from graphs import graphs as general_graphs


def layout(**kwargs) -> list[DashComponent]:
    """Compute the complete page layout. Ignores any arguments.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    tabs = [
        [
            *_layout_mentions(tab),
            *_layout_sentiment(tab),
            *_layout_dummy(),
        ]
        for tab in ["articles", "others"]
    ]
    labels = ["Articles", "Other aspects"]

    return [
        navbar.get_navbar("Aspects"),
        html.Br(),
        dbc.Tabs([dbc.Tab(tab, label=label) for tab, label in zip(tabs, labels)]),
    ]


def _layout_mentions(tab: str) -> list[DashComponent]:
    """Layout for an area plot about the number of mentions for each aspect.

    Parameters
    ----------
    tab : str
        Tab to place layout in, either "articles" or "others".

    Returns
    -------
    list[DashComponent]
        Layout component.
    """
    filter_ = layouts.button_badge(
        "Filter", f"{tab}_mentions_filter_button", checked=True
    )
    normalize = layouts.button_badge(
        "Normalize", f"{tab}_mentions_normalize_button", checked=False
    )
    buttons = [
        layouts.make_changes(
            filter_,
            additive=True,
            style={"margin-bottom": "0.2cm"},
            title="Filter to aspect's with at least 30 mentions "
            "(matching the sentiments plot below).",
        ),
        html.Br(),
        *_sentiment_user_difference_modal(tab=tab, variant="n_mentions"),
        html.Br(),
        layouts.make_changes(
            normalize,
            additive=True,
            title="Normalize each aspect's number of mentions to probabilities.",
        ),
    ]

    graph = dcc.Graph(
        id=f"{tab}_mentions_area_plot",
        figure=general_graphs.dummy_fig(),
        **general_graphs.GRAPH_CONF,
    )

    body = dbc.Row([dbc.Col(buttons, width=2), dbc.Col(graph, width=10)])

    return [
        dbc.Card(
            [dbc.CardHeader("Number of mentions"), dbc.CardBody(body)],
            className="fw-m2",
        )
    ]


def _layout_sentiment(tab: str) -> list[DashComponent]:
    """Layout for an area plot about the sentiment among mentions of each aspect.

    Parameters
    ----------
    tab : str
        Tab to place layout in, either "articles" or "others".

    Returns
    -------
    list[DashComponent]
        Layout component.
    """
    probs_button = layouts.make_changes(
        layouts.button_badge("Use probabilities", f"{tab}_sentiment_probs", False),
        additive=True,
        style={"margin-bottom": "0.2cm"},
        title="Whether to use average predicted probabilities (✓) "
        "or fraction of predicted sentiment (x).",
    )

    details_button = dbc.Button(
        [
            html.Div(
                [
                    layouts.make_changes(
                        html.I(className="bi bi-box-arrow-up-right"),
                        className="float_left v-h",
                        style={"margin-top": "0.1rem"},
                        additive=True,
                    ),
                    html.Span(
                        "Details",
                        style={"position": "relative", "top": "0.1rem"},
                        title="Go to sentiments page for more details.",
                    ),
                    layouts.make_changes(
                        html.I(className="bi bi-box-arrow-up-right"),
                        className="float_right",
                        style={"margin-top": "0.1rem"},
                        additive=True,
                    ),
                ]
            ),
        ],
        style=dict(width="100%"),
        href="/sentiment/",
        external_link=True,
    )

    sentiment_graph = dcc.Graph(
        id=f"{tab}_sentiment_area_plot",
        figure=general_graphs.dummy_fig(),
        **general_graphs.GRAPH_CONF,
    )

    body = dbc.Row(
        [
            dbc.Col(
                [
                    probs_button,
                    html.Br(),
                    *_sentiment_user_difference_modal(tab),
                    html.Br(),
                    details_button,
                ],
                width=2,
            ),
            dbc.Col(sentiment_graph, width=10),
        ]
    )

    return [
        dbc.Card([dbc.CardHeader("Sentiment"), dbc.CardBody(body)], className="fw-m2")
    ]


def _sentiment_user_difference_modal(
    tab: str, variant: str = "sentiment"
) -> list[DashComponent]:
    """Compute the layout of the user type differences modal.

    Parameters
    ----------
    tab : str
        Tab to place layout in, either "articles" or "others".
    variant : str, optional
        Parent layout the resulting layout is to be embedded into either "n_mentions" or
        "sentiment", by default "sentiment".

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    user_type_button = dbc.Button(
        "By user type",
        id=f"aspects_user_diff_{variant}_modal_button_{tab}",
        style={"margin-bottom": "0.2cm", "width": "100%"},
    )
    if variant == "n_mentions":
        return [user_type_button]

    interest_types_share = [
        dbc.ModalTitle("Differences in share of mentions between interest types"),
        dbc.ModalBody(
            [
                dcc.Graph(
                    figure=general_graphs.dummy_fig(),
                    id=f"aspects_user_share_fig_{tab}",
                    **general_graphs.GRAPH_CONF,
                )
            ]
        ),
    ]

    interest_types_readability = [
        dbc.ModalTitle("Differences in readability of mentions between interest types"),
        dbc.ModalBody(
            [
                dcc.Graph(
                    figure=general_graphs.dummy_fig(),
                    id=f"aspects_user_readability_fig_{tab}",
                    **general_graphs.GRAPH_CONF,
                )
            ]
        ),
        dbc.ModalFooter(
            [
                layouts.INFO_ICON,
                html.Div(
                    [
                        html.Span(
                            "You may find more information about the Dale-Chall "
                            "readability score, for example, on "
                        ),
                        html.A(
                            "Wikipedia",
                            href="https://en.wikipedia.org/wiki/"
                            "Dale%E2%80%93Chall_readability_formula",
                        ),
                        html.Span(" or in the "),
                        html.A(
                            "original publication",
                            href="https://www.jstor.org/stable/1473669",
                        ),
                        html.Span("("),
                        html.A(
                            "s",
                            href="https://www.goodreads.com/book/show/"
                            "1418721.Readability_Revisted",
                        ),
                        html.Span(")."),
                    ],
                    style={"font-size": "small", "margin-bottom": "0"},
                ),
            ],
            className="mf-lp",
        ),
    ]

    user_types = [
        dbc.ModalTitle("Differences between user types"),
        dbc.ModalBody(
            [
                html.Center(
                    html.Div(
                        dcc.Dropdown(
                            sentiment.get_user_type_names(),
                            "Company",
                            id=f"aspects_user_diff_user_{tab}",
                        ),
                        className="mw-15",
                    )
                ),
                dcc.Graph(
                    figure=general_graphs.dummy_fig(),
                    id=f"aspects_user_diff_n_mentions_fig_{tab}",
                    **general_graphs.GRAPH_CONF,
                    style={"margin-top": "0.3cm", "margin-bottom": "0.3cm"},
                ),
                dcc.Graph(
                    figure=general_graphs.dummy_fig(),
                    id=f"aspects_user_diff_sentiment_fig_{tab}",
                    **general_graphs.GRAPH_CONF,
                ),
            ]
        ),
        dbc.ModalFooter(
            [
                layouts.INFO_ICON,
                html.Div(
                    "Only showing aspects with at least 30 mentions from all user "
                    "types and 15 mentions from selected user type!",
                    style={"font-size": "small"},
                ),
            ],
            className="mf-lp",
        ),
    ]

    interest_sentiment = [
        dbc.ModalTitle("Differences in sentiment between interest types"),
        dbc.ModalBody(
            [
                html.Div(
                    html.Div(
                        dcc.Graph(
                            figure=general_graphs.dummy_fig(),
                            id=f"aspects_interest_sentiment_fig_{tab}",
                            **general_graphs.GRAPH_CONF,
                        )
                    ),
                    style=dict(width="100%", display="inline-block"),
                )
            ],
            style=dict(width="100%"),
        ),
        dbc.ModalFooter(
            [
                layouts.INFO_ICON,
                html.Div(
                    "Aspect-interest combinations with less than thirty mentions are "
                    "marked with a ⚠️. Beware of sentiment percentages with such few "
                    "observations.",
                    style={"font-size": "small"},
                ),
            ],
            className="mf-lp",
        ),
    ]

    user_type_modal = dbc.Modal(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        html.Div(user_types, style=dict(margin="0.5cm")),
                        label="User types vs. overall average",
                        tab_id="tab_mentions_sentiment",
                    ),
                    dbc.Tab(
                        html.Div(
                            interest_sentiment, style=dict(margin="0.5cm", width="100%")
                        ),
                        label="Interests vs. sentiment",
                        tab_id="tab_mentions_sentiment_interest",
                    ),
                    dbc.Tab(
                        html.Div(interest_types_share, style=dict(margin="0.5cm")),
                        label="Interests vs. mentions",
                        tab_id="tab_mentions_share",
                    ),
                    dbc.Tab(
                        html.Div(
                            interest_types_readability, style=dict(margin="0.5cm")
                        ),
                        label="Interests vs. readability",
                        tab_id="tab_mentions_readability",
                    ),
                ],
                active_tab="tab_mentions_sentiment",
            ),
        ],
        id=f"aspects_user_diff_modal_{tab}",
        is_open=False,
        # size="xl",
        style={"--bs-modal-width": "70vw"},
    )

    return [user_type_button, user_type_modal]


def _layout_dummy() -> list[DashComponent]:
    """Layout containing invisible dummy elements.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [html.Div(id="aspects_page_load_trigger", **layouts.STYLE_INVISIBLE)]
