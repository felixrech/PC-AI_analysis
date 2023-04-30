"""Module providing the layout for the sentiment page."""

import pandas as pd
import itertools as it
from typing import Callable
from functools import partial

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent
from graphs import graphs
from layouts import layouts

from utils import utils
from utils.caching import memoize
from layouts import navbar as navbar_
from utils import sentiment as sentiment_utils


def get_layout(sentiments: pd.DataFrame) -> Callable[[str, str], list[DashComponent]]:
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

    def layout(
        aspect_type: str = "", aspect_subtype: str = "", **kwargs
    ) -> list[DashComponent]:
        """Compute the layout for the sentiment page.

        Parameters
        ----------
        aspect_type : str, optional
            General part of the aspect, e.g. "article" - by default "".
        aspect_subtype : str, optional
            Specific part of the aspect, e.g. "5" - by default "".

        Returns
        -------
        list[DashComponent]
            Page layout.
        """
        size = dict(className="mm-a")

        top = [
            dbc.Col(_overview_pie_chart(), width=4),
            dbc.Col(_overview_barplot(), width=7, className="px-0", align="center"),
            _overview_barplot_modal(),
        ]
        details_options = [
            html.Div(
                layouts.button_badge(
                    "Word clouds",
                    "sentiment_word_clouds_button_badge",
                    **dict(checked=True, fake=False),
                ),
                style={"width": "80%", "max-width": "5cm"},
                **size,
            ),
            html.Div(layouts.topics_modal("sentiment"), **size),
            html.Div(_details_sort_modal(), **size),
            html.Div(
                layouts.filter_user_types_modal(sentiments, "sentiment"),
                **size,
            ),
            html.Div(layouts.multiselect_button("sentiment"), **size),
        ]
        details = [
            dbc.Col(width=2),
            dbc.Col(html.Div(id="sentiment_details"), width=8),
            dbc.Col(details_options, width=2),
        ]

        return [
            html.Div(
                children=navbar(aspect_type, aspect_subtype), id="sentiment_navbar"
            ),
            _aspect_selection(),
            html.Br(),
            dbc.Row(top, justify="center", align="center"),
            html.Br(),
            _layout_word_clouds(),
            html.Br(),
            dbc.Row(details),
            *_dummy_components(aspect_type, aspect_subtype),
        ]

    return layout


def navbar(aspect_type: str, aspect_subtype: str) -> DashComponent:
    """Compute the navigation bar layout.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    DashComponent
        Navigation bar layout.
    """
    if aspect_type == "" or aspect_subtype == "":
        return navbar_.get_navbar("Sentiment")

    return navbar_.get_navbar(
        f"Sentiment:  {sentiment_utils.aspect_nice_format(aspect_type, aspect_subtype)}"
    )


def _aspect_selection() -> DashComponent:
    """Compute the layout for selection an aspect.

    Returns
    -------
    DashComponent
        Layout.
    """
    dropdown = html.Div(
        dcc.Dropdown(
            placeholder="All aspects (Click here to select an aspect!)",
            id="aspect_dropdown",
        ),
        className="mw-15",
    )

    filter_selection = dbc.Button(
        [
            dbc.Badge(
                layouts.DANGER_ICON,
                style=dict(top="-5px"),
                className="nr-r",
            ),
            dbc.Badge(
                layouts.X_MARK_NO_CIRC_ICON,
                id="selection_show_all_icon",
                style=dict(top="-5px"),
                className="nr-l",
            ),
        ],
        id="selection_show_all",
        title="Display aspects with less than 30 mentions?",
        className="button_invisible",
    )

    # Create the warning modal when selecting aspect with few mentions
    modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Warning!")),
            dbc.ModalBody(
                "You selected an aspect with less than 30 mentions! "
                "This might make some of the results less reliable, "
                "as noise has large impacts when few observations are used."
            ),
        ],
        id="sentiment_aspect_mentions_warning",
    )

    return html.Div([dropdown, filter_selection, modal])


def _overview_pie_chart() -> DashComponent:
    """Compute the layout for a pie chart giving an overview of sentiments in the
    selected aspect.

    Returns
    -------
    DashComponent
        Layout.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(
                    "sentiment_overview_barplot",
                    figure=graphs.dummy_fig(),
                    **graphs.GRAPH_CONF_UNRESPONSIVE,
                ),
            ]
        )
    )


def _overview_barplot() -> DashComponent:
    """Compute the layout for a bar plot giving an overview of sentiments in the
    selected aspect for each user type.

    Returns
    -------
    DashComponent
        Layout.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(
                    "aspect_user_types_sentiment",
                    figure=graphs.dummy_fig(),
                    **graphs.GRAPH_CONF_UNRESPONSIVE,
                )
            ]
        )
    )


def _overview_barplot_modal() -> DashComponent:
    """Compute the layout for the barplot's options in a modal.

    Returns
    -------
    DashComponent
        Layout.
    """
    badges = [
        layouts.button_badge(
            "Normalize barplot", "sentiment_barplot_normalize", checked=False
        ),
        layouts.button_badge(
            "Dominant sentiment", "sentiment_barplot_dominant", checked=True
        ),
        layouts.button_badge(
            "Average predicted probs.", "sentiment_barplot_share", checked=False
        ),
        layouts.button_badge(
            "Limit to positive or negative sentiment",
            "sentiment_barplot_pos_neg",
            checked=False,
        ),
    ]
    badges = [layouts.make_changes(badge, className="mm-a") for badge in badges]

    hints = [
        "Normalizes each bar into [0,1] by dividing by the sum of its components. "
        "Useful for comparing sentiment between user types.",
        "Plot based on dominant sentiment, i.e. the sentiment class with highest "
        "predicted probability for each mention, or by averaging the predicted "
        "probabilities directly. The former yields results easier to interpret, while"
        "the latter might help when predicted sentiments are mostly neutral.",
        "Drop the neutral sentiment. (Removes the mentions with neutral sentiment "
        "when dominant sentiment is used and the neutral sentiment probabilities "
        "otherwise.)",
    ]
    hints = list(map(layouts.hint_row, hints))

    return html.Div(
        [
            dbc.Button(
                html.Div(),
                id="sentiment_barplot_button",
                className="triangle_button",
                title="Barplot options",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Barplot options")),
                    dbc.ModalBody(
                        [
                            *[badges[0], hints[0]],
                            dbc.Row([dbc.Col(badge, width=6) for badge in badges[1:3]]),
                            *[hints[1], badges[3], hints[2]],
                        ]
                    ),
                    _DISCLAIMER_FOOTER,
                ],
                id="sentiment_barplot_modal",
                size="lg",
            ),
        ],
        style={"width": "2rem", "padding-left": "0px"},
    )


def _layout_word_clouds() -> DashComponent:
    """Compute layout for word clouds.

    Returns
    -------
    DashComponent
        Layout component.
    """
    main = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        html.Img(
                            id=f"sentiment_word_cloud_{sentiment}",
                            style=dict(width="100%"),
                        )
                    )
                ),
                width=5,
            )
            for sentiment in ["positive", "negative"]
        ],
        justify="center",
        align="center",
    )

    return html.Div(
        [dbc.Collapse(main, id="sentiment_word_clouds_collapse", is_open=True)],
        id="sentiment_word_clouds_div",
    )


@memoize
def details(
    aspect_type: str = "",
    aspect_subtype: str = "",
    sort: str = "extreme",
    filter_user_types: list[str] = [],
    n: int = 20,
    always_open: bool = False,
) -> list[DashComponent]:
    """Compute layout with details about mentions and predicted sentiments.

    Parameters
    ----------
    aspect_type : str, optional
        Which aspect type (aspect_type column) to filter to, by default "", i.e. no
        filtering.
    aspect_subtype : str, optional
        Which aspect subtype (aspect_subtype column) to filter to, by default "", i.e.
        no filtering.
    sort : str, optional
        How to sort the dataset. Options are "", i.e. sort by "sentence_index" column
        "positive" - by ascending positive sentiment probability
        ("positive" column), "negative" - by ascending negative sentiment probability
        ("negative" column), or "extreme" by ascending maximum of positive and negative
        sentiment probability. The default is "extreme".
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.
    n : int, optional
        Number of mentions to include, by default 20.
    always_open : bool, optional
        Whether to allow for the opening of more than one accordion item at a time, by
        default False.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    sentiments = sentiment_utils.load_sentiment(
        aspect_type=aspect_type,
        aspect_subtype=aspect_subtype,
        sort=sort,
        filter_user_types=filter_user_types,
    )
    all_sentiments = sentiment_utils.load_sentiment(filter_=False)

    button_args = (
        dict(style={"margin-top": "0.5cm"})
        if len(sentiments) > n
        else layouts.STYLE_INVISIBLE
    )
    sentiments = sentiments.iloc[:n]

    return [
        *layouts.accordion(
            sentiments,
            title=_details_single_title,
            text=partial(layouts.accordion_windowed_text, sentiments=all_sentiments),
            always_open=always_open,
        ),
        dbc.Button(
            "Load more",
            id="sentiment_details_load_more",
            className="mm-a",
            **button_args,
        ),
    ]


def _details_single_title(sentence: pd.Series) -> str:
    """Compute the title of a single mention accordion item.

    Parameters
    ----------
    sentence : pd.Series
        Row of the sentiment dataset. Needs "organization", "sentence_index",
        "sentiment", "aspect_type" and "aspect_subtype" labels.

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
    org = utils.cutoff_text(org, 45).ljust(46)

    index = f"#{str(sentence['sentence_index'])}".rjust(7)

    sentiment = f"{sentence[sentence['sentiment']]:.0%}".rjust(4)
    sentiment_icons = {"positive": "🖒", "negative": "🖓", "neutral": "😐"}
    sentiment_icon = sentiment_icons[sentence["sentiment"]]

    aspect_nice = sentiment_utils.aspect_nice_format(
        sentence["aspect_type"], sentence["aspect_subtype"]
    ).ljust(15)

    return f"[{sentiment} {sentiment_icon}]  {index}   on {aspect_nice} by {org}"


def _details_sort_modal() -> list[DashComponent]:
    """Compute the layout for a modal to change the sorting of the details.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    positive_ = layouts.button_badge(
        "Positive", "sentiment_sort_positive", checked=False
    )
    negative_ = layouts.button_badge(
        "Negative", "sentiment_sort_negative", checked=False
    )
    extreme_ = layouts.button_badge("Extreme", "sentiment_sort_extreme", checked=True)
    buttons = [
        layouts.make_changes(button, className="fw-10 mm-a")
        for button in [positive_, negative_, extreme_]
    ]
    hints = [
        "Sort by ascending positive sentiment probability.",
        "Sort by ascending negative sentiment probability.",
        "Sort by ascending maximum of negative and positive sentiment probabilities.",
    ]
    hints = [html.Center(layouts.hint_row(hint), className="fw-10") for hint in hints]

    modal = html.Div(
        [
            dbc.Button(
                id="sentiment_sort_modal_button",
                style={"width": "80%", "max-width": "5cm"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle("How should the mentions be sorted?")
                    ),
                    dbc.ModalBody(html.Center(list(it.chain(*zip(buttons, hints))))),
                    _DISCLAIMER_FOOTER,
                ],
                id="sentiment_sort_modal",
                is_open=False,
                size="lg",
            ),
        ]
    )
    return [modal]


def _dummy_components(aspect_type: str, aspect_subtype: str) -> list[DashComponent]:
    """Compute the layout of invisible dummy components.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        html.Div(children=aspect_type, id="aspect_type", **layouts.STYLE_INVISIBLE),
        html.Div(
            children=aspect_subtype, id="aspect_subtype", **layouts.STYLE_INVISIBLE
        ),
        html.Div(children="20", id="sentiment_details_n", **layouts.STYLE_INVISIBLE),
        html.Div(id="sentiment_page_load_trigger", **layouts.STYLE_INVISIBLE),
    ]


_DISCLAIMER_FOOTER = dbc.ModalFooter(
    html.Div(
        "Changes will be applied as soon as you close this window!",
        style={"font-size": "small"},
    )
)