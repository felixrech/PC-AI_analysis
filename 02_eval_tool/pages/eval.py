"""Module containing the evaluation page: reading an evaluation CSV file, rendering a
page based on the contents, and updating the file based on user input."""

import dash
from dash import html, dcc, ctx, callback
from dash_extensions import EventListener
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input, State

import datetime
import pandas as pd
from typing import Tuple, Union, Any
from collections import defaultdict


# Register page for multi-page setup
dash.register_page(__name__, path_template="/eval/<evaluation>")


def layout(**kwargs):
    """Compute the layout for the evaluation page. Any arguments are ignored."""
    return html.Center(
        html.Div(
            className="wrapper",
            children=[
                html.Div(
                    className="content",
                    children=[
                        html.Div(
                            html.Span(
                                id="ui_message",
                                children="Press an arrow key to start!",
                                style={"font-weight": "700"},
                            ),
                        ),
                        html.Br(),
                        html.Div(
                            children=[
                                html.Span(id="id", style={"display": "none"}),
                                html.Iframe(
                                    id="content",
                                    sandbox="",
                                    srcDoc="",
                                    style={"max-height": "80vh"},
                                ),
                            ],
                        ),
                        EventListener(
                            dcc.Textarea(
                                id="textarea",
                                value="Test",
                                style={"opacity": 0, "height": "0px"},
                                autoFocus="autofocus",
                            ),
                            events=[
                                {
                                    "event": "keydown",
                                    "props": [
                                        "key",
                                        "altKey",
                                        "ctrlKey",
                                        "shiftKey",
                                        "metaKey",
                                        "repeat",
                                    ],
                                }
                            ],
                            logging=True,
                            id="event_keyboard",
                        ),
                        html.Div(
                            [
                                html.Span("", id="progress"),
                                html.Button(
                                    "👎",
                                    id="button_negative",
                                    n_clicks=0,
                                    style={"margin-left": "25%"},
                                ),
                                html.Button(
                                    "👌",
                                    id="button_neutral",
                                    n_clicks=0,
                                ),
                                html.Button(
                                    "👍",
                                    id="button_positive",
                                    n_clicks=0,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Aside(
                    className="sidebar",
                    children=html.Iframe(
                        id="pdf_viewer",
                    ),
                ),
                dcc.Location(id="url", refresh=False),
            ],
        ),
    )


@callback(
    [
        Output("ui_message", "children"),
        Output("id", "children"),
        Output("content", "srcDoc"),
        Output("pdf_viewer", "src"),
        Output("pdf_viewer", "style"),
        Output("progress", "children"),
    ],
    [
        Input("event_keyboard", "n_events"),
        Input("button_positive", "n_clicks"),
        Input("button_neutral", "n_clicks"),
        Input("button_negative", "n_clicks"),
    ],
    [
        State("event_keyboard", "event"),
        State("id", "children"),
        State("url", "pathname"),
    ],
)
def react_event(
    _, __, ___, ____, keyboard_event, id: str, pathname: str
) -> Tuple[Union[str, list], str, str, str, dict, str]:
    """Callback to keyboard or button event on evaluation page.

    Parameters
    ----------
    _ : Any
        Ignored.
    __ : Any
        Ignored.
    ___ : Any
        Ignored.
    ____ : Any
        Ignored.
    keyboard_event
        Keyboard event capture.
    id : str
        Current task id.
    pathname : str
        Current URL.

    Returns
    -------
    Tuple[Union[str, list], str, str, str, dict, str]
        Updated page elements.

    Raises
    ------
    PreventUpdate
        In case of None keyboard_event.
    """
    triggered_id = ctx.triggered_id

    if triggered_id == "event_keyboard":
        # Skip if initial empty event
        if keyboard_event is None:
            raise PreventUpdate()

        # Map keys to results
        results = defaultdict(lambda: "ERROR")
        for k, v in {
            "ArrowLeft": "NEGATIVE",
            "ArrowUp": "NEUTRAL",
            "ArrowDown": "NEUTRAL",
            "ArrowRight": "POSITIVE",
        }.items():
            results[k] = v
        result = results[keyboard_event["key"]]
    elif triggered_id == "button_negative":
        result = "NEGATIVE"
    elif triggered_id == "button_neutral":
        result = "NEUTRAL"
    elif triggered_id == "button_positive":
        result = "POSITIVE"
    else:
        result = "ERROR"

    # Save current result to csv and load a new task
    evaluation = pathname[6:]
    return report_and_renew(id, result, evaluation)


def report_and_renew(
    id: str, result: str, evaluation: str
) -> Tuple[Union[str, list], str, str, str, dict]:
    """Save the current result to csv and load a new task.

    Parameters
    ----------
    id : str
        Current task id.
    result : str
        Result of the current task.
    evaluation : str
        Current evaluation.

    Returns
    -------
    Tuple[Union[str, list], str, str, str, dict]
        Updated page elements.
    """
    # Read the csv corresponding to URL
    filename = f"evaluations/{evaluation}.csv"
    df = pd.read_csv(filename)

    # Write current result to csv if it's not an error
    if result == "ERROR":
        # print("Error in result!")
        pass
    elif len(df.query("id == @id")) == 1:
        log(evaluation, id, result)
        df.loc[df["id"] == id, "result"] = result
        df.to_csv(filename, index=False)
    else:
        log(evaluation, id, "Id not found/unique")

    # Edge case: No new tasks
    unfinished = df.query("result.isnull()")
    if len(unfinished) == 0:
        home = html.Div(
            children=[
                html.Span("No new tasks found! You can go back "),
                html.A("home", href="/"),
                html.Span("."),
            ]
        )
        return home, "", "", "", {"display": "none"}, ""

    # Return new task
    row = unfinished.sample(1).iloc[0]
    return (
        row.task,
        row.id,
        row.content,
        dash.get_asset_url(
            row.source
            + ("&" if "#" in row.source else "#")
            + "zoom=FitB&view=FitV&toolbar=0&navpanes=0&pagemode=none"
        ),
        {"width": "100%"},
        f"{len(df)-len(unfinished)} / {len(df)}",
    )


def log(evaluation: str, id: Any, result: str) -> None:
    """Logs the evaluation to both stdout and a logfile.

    Parameters
    ----------
    evaluation : str
        Evaluation (filename of the evaluation without its .csv file ending).
    id : Any
        ID of the task.
    result : str
        Result of the evaluation.
    """
    log_string = (
        f"[{datetime.datetime.now().isoformat(timespec='seconds')}]"
        + f"   {evaluation.ljust(20)} - {str(id).ljust(20)}: {result}"
    )
    print(log_string)
    with open("log.txt", "a") as f:
        f.write(log_string + "\n")
