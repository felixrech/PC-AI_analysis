"""Module to access a navigation bar to add to any page."""

from dash import html
import dash_bootstrap_components as dbc

from layouts import layouts


def get_navbar(current: str) -> dbc.NavbarSimple:
    """Create a navigation bar that can be added to page layout.

    Parameters
    ----------
    current : str
        String that will be used as the title.

    Returns
    -------
    dbc.NavbarSimple
        Navigation bar dash component.
    """
    links = (
        [
            html.Span(
                dbc.NavItem(
                    dbc.NavLink(
                        layouts.HOUSE_ICON,
                        href="/",
                    ),
                ),
                title="Return to overview page",
            ),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(
                        "Topic model overview",
                        href="/overview/",
                        external_link=True,
                        disabled=current == "Topic overview",
                    ),
                    dbc.DropdownMenuItem(
                        "Topic details",
                        href="/topic_details/",
                        external_link=True,
                        disabled=current.startswith("Topic details"),
                    ),
                    dbc.DropdownMenuItem(
                        "Embedding",
                        href="/embedding/",
                        external_link=True,
                        disabled=current == "Embedding",
                    ),
                    html.Hr(style=dict(margin="")),
                    dbc.DropdownMenuItem(
                        "Aspects",
                        href="/aspects/",
                        external_link=True,
                        disabled=current == "Aspects",
                    ),
                    dbc.DropdownMenuItem(
                        "Sentiments",
                        href="/sentiment/",
                        external_link=True,
                        disabled=current.startswith("Sentiment"),
                    ),
                    html.Hr(style=dict(margin="")),
                    dbc.DropdownMenuItem(
                        "Search",
                        href="/search/",
                        external_link=True,
                        disabled=current == "Search",
                    ),
                    dbc.DropdownMenuItem(
                        "Legal stuff",
                        href="/legal/",
                        external_link=True,
                        disabled=current == "Legal stuff",
                    ),
                ],
                nav=True,
                in_navbar=True,
                label="More",
                align_end=True,
            ),
        ]
        if current != "Home"
        else []
    )
    return dbc.NavbarSimple(
        brand=current,
        children=links,
        color="#344b63",
        dark=True,
        fluid=True,
        style={
            "padding": "0",
            "margin-bottom": "0.4cm",
        },
    )
