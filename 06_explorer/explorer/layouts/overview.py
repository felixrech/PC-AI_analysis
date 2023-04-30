"""Module containing the layouts for the /overview/ page."""

import os
import more_itertools as mit

import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent
from dash import html, dcc

from utils import utils
from graphs import graphs
from layouts import navbar, layouts


H, _, texts = utils.load_tm()
topics_ = utils.get_topics(H)
H_norm = utils.normalize_H(H)


def layout(**kwargs) -> list[DashComponent]:
    """Compute page layout. Any arguments are ignored.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        navbar.get_navbar("Topic overview"),
        html.Br(),
        *_layout_popularity(),
        html.Br(),
        *_layout_user_types(),
        html.Br(),
        *_layout_comparison(),
        *_layout_dummy(),
    ]


def _layout_popularity() -> list[DashComponent]:
    """Compute the layouts for topic popularity (i.e. number of documents for which
    each topic is relevant/dominant).

    Returns
    -------
    list[DashComponent]
        Layout components.
    """

    hint = layouts.cbl(
        dcc.Markdown(
            "Note that for the purposes of this analysis, we "
            "will (somewhat arbitrarily) define `relevant` as "
            "having a topic share of $\\geq 10\\%$.",
            mathjax=True,
        )
    )
    hint = layouts.hint_accordion(hint, "How was this figure created?")

    dominant = dbc.AccordionItem(
        html.Div(
            [
                dcc.Graph(
                    id="overview_popularity_dominant",
                    figure=graphs.dummy_fig(),
                    config=dict(displayModeBar=False),
                ),
                *layouts.plotly_hint(),
            ]
        ),
        title="How popular is ... as a dominant topic?",
        className="accordion_buttonize",
    )
    relevant = dbc.AccordionItem(
        [
            dcc.Graph(
                id="overview_popularity_relevant",
                figure=graphs.dummy_fig(),
                config=dict(displayModeBar=False),
            ),
            html.Br(),
            hint,
            *layouts.plotly_hint(),
        ],
        title="How popular is ... as a relevant topic?",
        className="accordion_buttonize",
    )

    return [
        dbc.Card(
            dbc.CardBody(
                dbc.Accordion(
                    [dominant, relevant],
                    start_collapsed=True,
                    always_open=True,
                ),
            )
        )
    ]


def _layout_user_types_content(variant: str = "share") -> list[DashComponent]:
    """Compute the layout of barplots of which topics each of the different user types
    talk about.

    Parameters
    ----------
    variant : str, optional
        How to measure topic popularity. "share" is the average (normalized) topic share
        and "dominant" the fraction of documents for which the topic is dominant. By
        default "share".

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    buttons = html.Center(
        [
            dbc.Button(
                "Average share",
                id="comparison_user_types_buttons_share",
                style=dict(margin="0.05cm", width="20%"),
                disabled=variant == "share",
            ),
            dbc.Button(
                "Dominant topic",
                id="comparison_user_types_buttons_dominant",
                style=dict(margin="0.05cm", width="20%"),
                disabled=variant == "dominant",
            ),
        ]
    )

    user_type_plots = [
        dbc.Col(
            dcc.Graph(
                id=dict(
                    type="overview_user_type_barplots",
                    index=i,
                ),
                figure=graphs.dummy_fig(),
                config=dict(displayModeBar=False),
                style=dict(width="calc(32vw - 1cm)"),
            )
        )
        for i in range(texts["user_type"].nunique())
    ]
    user_type_plots = [dbc.Row(cols) for cols in mit.chunked(user_type_plots, 3)]

    return [buttons, html.Br(), *user_type_plots]


def _layout_user_types() -> list[DashComponent]:
    """Accordion wrapper around _layout_user_types_content.

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    content = html.Div(
        children=_layout_user_types_content(), id="comparison_user_types_accordion"
    )

    hint = layouts.cbl(
        dcc.Markdown(
            "- Average share:\n"
            "   - The topic-document matrix $H$ is first normalized by document.\n"
            "   - Then, the mean over topics is taken, aggregated by user type.\n"
            "- Dominant:\n"
            "   - The topic-document matrix $H$ is first normalized by document.\n"
            "   - The dominant topic is selected for each document.\n"
            "   - Then, the results are aggregated by user type.",
            mathjax=True,
        )
    )
    hint = layouts.hint_accordion(hint, "How were these figures created?")

    return [
        dbc.Card(
            dbc.CardBody(
                dbc.Accordion(
                    dbc.AccordionItem(
                        [content, hint],
                        title="What do the different user types talk about?",
                        className="accordion_buttonize",
                    ),
                    start_collapsed=True,
                    always_open=True,
                )
            )
        )
    ]


def _layout_comparison() -> list[DashComponent]:
    """Compute the layout for topic model comparisons (Rand index barplot and topic
    model comparison heatmap).

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    topic_models = [f[:-8] for f in os.listdir("data") if f.endswith(".indepth")]

    tm_selector = dbc.Col(
        dbc.DropdownMenu(
            label="Select a topic model to compare to",
            children=[
                dbc.DropdownMenuItem(
                    topic_model,
                    id=dict(
                        type="other_tm_selected_",
                        index=topic_model,
                    ),
                )
                for topic_model in topic_models
            ],
            className="m-1",
            id="other_tm_selected",
        )
    )
    rand_plot = dbc.Col(
        dcc.Graph(
            id="rand_index_pie",
            figure=graphs.rand_index_bar(0),
            animate=True,
            animation_options={
                "frame": {
                    "duration": 1250,
                    "redraw": False,
                },
                "transition": {
                    "duration": 1250,
                    "ease": "cubic-in-out",
                },
            },
            config=dict(displayModeBar=False),
        )
    )
    heatmap = [
        html.Center(
            [
                dbc.Button(
                    "Hellinger distance",
                    id="tm_topics_heatmap_type_button_hellinger",
                    className="mm",
                    disabled=True,
                ),
                dbc.Button(
                    "Top terms overlap",
                    id="tm_topics_heatmap_type_button_top_terms",
                    className="mm",
                ),
            ]
        ),
        dcc.Graph(
            id="tm_topics_heatmap",
            figure=graphs.dummy_fig("Please select a topic model first!"),
            config=dict(displayModeBar=False),
        ),
        html.Br(),
        html.Div(
            id="tm_topics_heatmap_explanation",
            style=dict(width="75%"),
        ),
    ]

    accordion = dbc.Accordion(
        dbc.AccordionItem(
            html.Div(
                [
                    dbc.Row([tm_selector, rand_plot]),
                    html.Br(),
                    *heatmap,
                ],
                id="rand_index_container",
            ),
            title="How does this topic modeling compare to ...?",
            className="accordion_buttonize",
        ),
        start_collapsed=True,
        always_open=True,
    )
    return [dbc.Card(dbc.CardBody(accordion))]


def _layout_dummy() -> list[DashComponent]:
    """Compute layout of invisible dummy components.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [html.Div(id="overview_page_load", **layouts.STYLE_INVISIBLE)]
