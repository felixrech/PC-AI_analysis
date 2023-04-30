"""Module providing layout and callback for the /absa_annotation/ page."""

import numpy as np
import pandas as pd
from typing import Any
import dash_bootstrap_components as dbc

import dash
from dash.dependencies import Component as DashComponent
from dash import html, callback, Input, Output, State, ctx

from layouts import navbar, layouts
from utils import sentiment as sentiment_utils


# Register page for multi-page setup
dash.register_page(__name__, path_template="/absa_annotation/")


def layout(**kwargs) -> list[DashComponent]:
    """Layout for the /absa_annotation/ page. Ignores any arguments (search
    parameters).

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    labels = ["positive", "neutral", "negative", "conflict"]
    buttons = html.Div(
        children=[
            dbc.Button(
                label.capitalize(),
                id=f"absa_annotation_{label}",
                style=dict(width="7rem"),
                className="mm-a",
            )
            for label in labels
        ],
        style={"max-width": "15cm"},
    )

    main = [
        html.Div(
            id="absa_annotation_text",
            style={"max-width": "30cm", "margin-bottom": "0.7cm"},
        ),
        buttons,
    ]

    return [
        navbar.get_navbar("ABSA annotator"),
        html.Br(),
        dbc.Row(
            dbc.Col(main, align="center"), align="center", style=dict(height="70vh")
        ),
        html.Div(id="absa_annotation_sentence_index", **layouts.STYLE_INVISIBLE),
    ]


@callback(
    [
        Output("absa_annotation_text", "children"),
        Output("absa_annotation_sentence_index", "children"),
    ],
    [
        Input(f"absa_annotation_{label}", "n_clicks")
        for label in ["positive", "neutral", "negative", "conflict"]
    ],
    State("absa_annotation_sentence_index", "children"),
)
def update_annotation(
    _a: Any, _b: Any, _c: Any, _d: Any, sentence_index: str
) -> tuple[list[DashComponent] | str, str]:
    """Update text and details about current aspect, as well as its index.

    Parameters
    ----------
    _a : Any
        Ignored (only for triggering callbacks).
    _b : Any
        Ignored (only for triggering callbacks).
    _c : Any
        Ignored (only for triggering callbacks).
    _d : Any
        Ignored (only for triggering callbacks).
    sentence_index : str
        "Sentence_index" of the current mention.

    Returns
    -------
    tuple[list[DashComponent], str]
        Layout of current mention and its index.
    """
    del _a, _b, _c, _d

    df = pd.read_csv("data/annotation.csv").sample(frac=1)
    all_sentiments = sentiment_utils.load_sentiment(filter_=False)

    if ctx.triggered_id is not None and sentence_index is not None:
        df["sentiment"] = np.where(
            df["sentence_index"] == int(sentence_index),
            ctx.triggered_id.split("_")[-1],
            df["sentiment"],
        )
        df["sentiment"] = df["sentiment"].replace({"nan": None})
        df.to_csv("data/annotation.csv", index=False)

    if len(unanswered := df.query("sentiment.isnull()")) == 0:
        return "All done", ""

    new_ = unanswered.iloc[0]["sentence_index"]  # noqa[F841]
    new = all_sentiments.query("sentence_index == @new_").iloc[0]

    info = [
        *layouts.accordion_info_buttons(new).children[0].children[-1].children,
        dbc.Button(
            f"{len(df)-len(unanswered)}/{len(df)}", className="mm", color="info"
        ),
    ]
    accordion = html.Div(
        [
            html.Div(info),  # type: ignore
            html.Br(),
            layouts.accordion_windowed_text(new, sentiments=all_sentiments),
        ]
    )
    return [accordion], new["sentence_index"]
