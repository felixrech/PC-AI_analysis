"""Module containing the index page listing open and finished evaluations."""

import dash
from dash import html

import os
import pandas as pd
from typing import Iterable, Union
from itertools import compress, chain


# Register page for multi-page setup
dash.register_page(__name__, path="/")


def get_evaluations() -> Union[Iterable[Union[html.A, html.Br]], html.Span]:
    """Search the local evaluations/ folder for csv's of the correct format and return
    a list of html elements linking to them.

    Returns
    -------
    Union[Iterable[Union[html.A, html.Br]], html.Span]
        Evaluation list html elements.
    """
    # Read all csv's in the evaluations folder into dataframes
    evaluations = sorted([f for f in os.listdir("evaluations") if f.endswith(".csv")])
    dfs = [pd.read_csv("evaluations/" + f) for f in evaluations]

    # Filter to dataframes with the necessary columns
    columns = {"id", "task", "content", "source", "result"}
    valid = [len(columns & set(df.columns)) == len(columns) for df in dfs]
    evaluations, dfs = list(compress(evaluations, valid)), compress(dfs, valid)
    evaluations = [f[:-4] for f in evaluations]

    # If there are no evaluations return a message
    if len(evaluations) == 0:
        return html.Span("No evaluations 😥")

    # Separate dataframes into finished and unfinished evaluations
    finished = [df["result"].isna().sum() == 0 for df in dfs]
    open_evals = [
        html.A(f"⏳ {evaluation}", href=f"/eval/{evaluation}")
        for evaluation in compress(evaluations, [not f for f in finished])
    ] + [html.Span()]
    closed_evals = [
        html.A(f"✔️ {evaluation} ", href=f"/eval/{evaluation}")
        for evaluation in compress(evaluations, finished)
    ]

    # Return evaluations separated by newlines
    return sum([[a, html.Br()] for a in chain(open_evals, closed_evals)], [])[:-1]


def layout() -> dash.html.Center:
    """Compute the layout: a list of unfinished and finished evaluations.

    Returns
    -------
    html.Center
        Layout dash.html object.
    """
    return html.Center(
        html.Div(
            children=[
                html.H1(children="Evaluations"),
                html.Div(children=get_evaluations()),
            ],
            style={"text-align": "left", "max-width": "500px"},
        )
    )
