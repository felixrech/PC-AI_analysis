"""Module providing the PDF viewer page."""

import dash
from dash import html
from dash.dependencies import Component as DashComponent


# Register page for multi-page setup
dash.register_page(__name__, path_template="/pdf/")


def layout(
    reminder: str = "",
    href: str = "https://images.huffingtonpost.com/2014-03-24-5-thumb.gif",
    **kwargs
) -> list[DashComponent]:
    """Compute the PDF viewer page layout.

    Parameters
    ----------
    reminder : str, optional
        Short (<= 3 chars) string to show on the lower right corner of the screen,
        by default "".
    href : _type_, optional
        Link to the background iframe, by default
        "https://images.huffingtonpost.com/2014-03-24-5-thumb.gif".

    Returns
    -------
    list[DashComponent]
        Layout components.
    """
    return [
        html.Div(reminder, className="iframe_overlay") if reminder != "" else None,
        html.Iframe(src=href, style=dict(width="100vw", height="100vh")),
    ]  # type: ignore
