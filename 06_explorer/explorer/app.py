"""Module containing the main runnable of the evaluation tool."""

import dash
from flask import Flask
from dash._callback import NoUpdate
from dash import html, callback, ctx
from dash._utils import AttributeDict
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, ClientsideFunction, ALL

from callbacks import initialize_caching


# Basic setup for a multi-page app
server = Flask(__name__)
app = dash.Dash(
    server=server,  # type: ignore [correct but dash doesn't have type hints]
    use_pages=True,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
        "https://cdn.jsdelivr.net/gh/lipis/flag-icons@6.6.6/css/flag-icons.min.css",
    ],
)
app._favicon = "assets/favicon.ico"
app.layout = html.Center(
    html.Div(
        [
            dash.page_container,
            html.Div(id="news_dummy_output", style=dict(display="none")),
        ],
        style=dict(width="100vw", margin="0 auto"),
    ),
    className="container",
)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="update_search"),
    Output("dummy_2", "children"),
    [Input("dummy_1", "children")],
)


@callback(
    Output("news_dummy_output", "children"),
    Input({"type": "news_", "index": ALL}, "is_open"),
)
def set_news_dismissed_cookies(_) -> str | NoUpdate:
    """Set cookies to remember dismissed news.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback).

    Returns
    -------
    str | NoUpdate
        Dummy output.
    """
    if type(ctx.triggered_id) is AttributeDict and ctx.triggered_id["type"] == "news_":
        if ctx.triggered_id["index"] == 0:
            return dash.no_update

        dash.ctx.response.set_cookie(
            f"news_{ctx.triggered_id['index']}_dismissed", "true", max_age=int(1e10)
        )
    return ""


initialize_caching.do()


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True)
