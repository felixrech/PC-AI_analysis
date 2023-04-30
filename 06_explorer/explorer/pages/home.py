"""Module providing the home page with links to everything else."""

import pandas as pd

import dash
import flask
from dash import html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.dependencies import Component as DashComponent

from utils import utils
from layouts import navbar


# Register page for multi-page setup
dash.register_page(__name__, path_template="/")


def layout(**kwargs) -> list[DashComponent]:
    """Compute the main layout. Ignores any arguments.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        navbar.get_navbar("Home"),
        *_layout_news(),
        html.H1("Topic & sentiment explorer"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        *_layout_tm_links(),
                        html.Br(),
                        *_layout_sentiment_links(),
                        html.Br(),
                        *_layout_other_links(),
                    ],
                    style={
                        "margin-left": "max(0.25cm, 5%)",
                        "width": "calc(42% - max(0.25cm, 5%))",
                    },
                ),
                dbc.Col(
                    [*_layout_info()],
                    style={
                        "margin-right": "max(0.25cm, 5%)",
                        "width": "calc(52% - max(0.25cm, 5%))",
                    },
                ),
            ],
        ),
    ]


def _layout_news() -> list[DashComponent]:
    """Create some alerts which may be filled with news later.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    alerts = [
        dbc.Alert(
            "",
            id=dict(type="news_", index=i),
            is_open=False,
            color="primary",
            dismissable=True,
            className="fw-m2",
        )
        for i in range(10)
    ]
    return [html.Div(id="news", children=alerts)]


def _layout_tm_links() -> list[DashComponent]:
    """Create links to the topic modeling pages.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    links: list[DashComponent] = [
        dbc.Button("Topic model overview", href="/overview/"),
        dbc.Button("Topic details", href="/topic_details/"),
        dbc.Button("Embedding", href="/embedding/"),
    ]

    return _layout_card("Topic Modeling Components", links)


def _layout_sentiment_links() -> list[DashComponent]:
    """Create links to the sentiment pages.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    links: list[DashComponent] = [
        dbc.Button("Aspects", href="/aspects/"),
        dbc.Button("Sentiments", href="/sentiment/"),
    ]

    return _layout_card("Sentiment Components", links)


def _layout_other_links() -> list[DashComponent]:
    """Create links to other pages.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    links: list[DashComponent] = [
        dbc.Button("Search dataset", href="/search/"),
        dbc.Button("Legal stuff", href="/legal/"),
    ]

    return _layout_card("Other Things", links)


def _layout_info() -> list[DashComponent]:
    """Small information box with an explanation of what this tool is all about.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    separator = html.Hr(style={"margin-top": "30px", "margin-bottom": "10px"})
    info: list[DashComponent] = [
        html.Div(
            "This explorer is a dashboard to interactively explore the feedback "
            "collected in the public consultation on the AI Act's draft. It may be "
            "used both as a standalone tool or in combination with the Master's thesis "
            "that this web app was developed for."
        ),
        separator,
        html.B("Topic Modeling Components"),
        html.Div(
            (
                "The tools in this section enable a topic modeling-based analysis. "
                '"Topic model overview" gives an overview of the selected topic model, '
                "which may help identify interesting topics, and allows for the "
                'comparison of topic models. "Topic details" allows the user to '
                "deep-dive into a single topic and also shows the actual documents "
                'associated with each topic. "Embedding" then shows the topic model '
                "projected into lower dimensions, e.g. using TSNE or hierarchical "
                "clustering (e.g. helpful for determining similarity between topics)."
            ),
            className="fs-sm",
        ),
        separator,
        html.B("Sentiment Components"),
        html.Div(
            (
                "The tools in this section offer an analysis based on aspect-based "
                'sentiment analysis. "Aspects" gives an overview of all aspects and '
                "can be used to identify outliers or otherwise interesting aspects. "
                "For details, including the actual mentions of each aspect, see the "
                '"Sentiments" link.'
            ),
            className="fs-sm",
        ),
        separator,
        html.B("Other things"),
        html.Div(
            (
                "This is where you can find the search function and attribution for "
                "icons, graphics, and other libraries used throughout this tool."
            ),
            className="fs-sm",
        ),
    ]

    return [
        dbc.Card(
            [dbc.CardHeader("What is this all about?"), dbc.CardBody(info)],
            className="mw-30",
        )
    ]


def _layout_card(header: str, links: list[DashComponent]) -> list[DashComponent]:
    """Create equally sized cards with given header and content.

    Parameters
    ----------
    header : str
        Content in the card header.
    links : list[DashComponent]
        Content in the card body.

    Returns
    -------
    list[DashComponent]
        Layout component(s).
    """
    return [
        dbc.Card(
            [
                dbc.CardHeader(header),
                dbc.CardBody(html.Div(links, className="d-grid gap-2 col-6 mx-auto")),
            ],
            className="mw-30",
        )
    ]


@callback(
    [Output(dict(type="news_", index=i), "children") for i in range(10)]
    + [Output(dict(type="news_", index=i), "is_open") for i in range(10)],
    Input("news", "style"),
)
def update_news_items(_) -> list[str]:
    """Fill news alerts based on the contents of notifications.json and which of them
    have already been dismissed by the user.

    Returns
    -------
    list[str]
        News alerts' texts.
    """
    news = pd.DataFrame(utils.get_news(), columns=["id", "msg"])
    df = pd.DataFrame(dict(id=map(str, range(10))))
    df = pd.merge(df, news, how="outer")

    df["msg"] = df["msg"].fillna("")
    df["is_open"] = df["msg"].str.len() > 0
    df["is_open"] = df.apply(
        lambda x: f"news_{x['id']}_dismissed" not in flask.request.cookies.keys()
        if x["is_open"]
        else False,
        axis="columns",
    )

    return df["msg"].to_list() + df["is_open"].to_list()
