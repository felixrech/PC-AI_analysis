"""Module containing the main runnable of the evaluation tool."""

import dash
from dash import html


# Basic setup for a multi-page app
app = dash.Dash(use_pages=True)
app.layout = html.Div([dash.page_container])


if __name__ == "__main__":
    app.run_server(host="0.0.0.0")
