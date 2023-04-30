"""Module providing the legal stuff page."""

import itertools as it

import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from layouts import layouts, navbar


# Register page for multi-page setup
dash.register_page(__name__, path_template="/legal")


def layout(**kwargs) -> list[DashComponent]:
    """Compute the main layout. Ignores any arguments.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        navbar.get_navbar("Legal stuff"),
        *_layout_favicon(),
        html.Br(),
        *_layout_icons(),
        html.Br(),
        *_layout_flags(),
        html.Br(),
        *_layout_ldavis(),
        html.Br(),
        *_layout_software(),
    ]


def _layout_favicon() -> list[DashComponent]:
    """Returns a (list with a) card component with infos about the favicon used.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    favicon_image = html.Img(src="/assets/favicon.ico")

    favicon = [
        html.Span("Copyright 2020 Twitter, Inc and other contributors"),
        html.Br(),
        html.Span(
            [
                "Graphics licensed under CC-BY 4.0: ",
                html.A(
                    "https://creativecommons.org/licenses/by/4.0/",
                    href="https://creativecommons.org/licenses/by/4.0/",
                    target="_blank",
                ),
            ]
        ),
        html.Br(),
        html.Span(
            [
                "For more information see: ",
                html.A(
                    "https://twemoji.twitter.com/",
                    href="https://twemoji.twitter.com/",
                    target="_blank",
                ),
            ]
        ),
    ]

    return _get_card("Favicon", favicon_image, favicon)


def _layout_icons() -> list[DashComponent]:
    """Returns a (list with a) card component with information about icons used.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    icons_source = [
        html.Span(
            "Copyright 2022 Mark Otto, Jacob Thornton, and remaining Bootstrap "
            "community"
        ),
        html.Br(),
        html.Span(
            [
                "Icons licensed under the MIT license: ",
                html.A(
                    "https://github.com/twbs/bootstrap/blob/main/LICENSE",
                    href="https://github.com/twbs/bootstrap/blob/main/LICENSE",
                    target="_blank",
                ),
            ]
        ),
        html.Br(),
        html.Span(
            [
                "For more information see: ",
                html.A(
                    "https://icons.getbootstrap.com/",
                    href="https://icons.getbootstrap.com/",
                    target="_blank",
                ),
            ]
        ),
    ]
    icons_examples = [
        layouts.make_changes(layouts.LEFT_ARROW, style={"margin-left": "0.2cm"}),
        layouts.make_changes(layouts.RIGHT_ARROW, style={"margin-left": "0.2cm"}),
        layouts.make_changes(layouts.CLOCK_ICON, style={"margin-left": "0.2cm"}),
        layouts.make_changes(layouts.PDF_ICON, style={"margin-left": "0.2cm"}),
        layouts.make_changes(layouts.HOUSE_ICON, style={"margin-left": "0.2cm"}),
        layouts.make_changes(layouts.DOUBLE_CHECK_ICON, style={"margin-left": "0.2cm"}),
        html.Br(),
        html.Span("... and more"),
    ]

    return _get_card("Icons", icons_examples, icons_source)


def _layout_flags() -> list[DashComponent]:
    """Returns a (list with a) card component with information about flags used.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    flag_source = [
        html.Span("Copyright 2022 Panayiotis Lipiridis"),
        html.Br(),
        html.Span(
            [
                "Icons licensed under the MIT license: ",
                html.A(
                    "https://github.com/lipis/flag-icons/blob/main/LICENSE",
                    href="https://github.com/lipis/flag-icons/blob/main/LICENSE",
                    target="_blank",
                ),
            ]
        ),
        html.Br(),
        html.Span(
            [
                "For more information see: ",
                html.A(
                    href := "https://flagicons.lipis.dev/", href=href, target="_blank"
                ),
            ]
        ),
    ]
    flag_examples = [
        layouts.make_changes(
            html.Span(className="fi fi-eu"), style=(style := {"margin-left": "0.1cm"})
        ),
        layouts.make_changes(html.Span(className="fi fi-de"), style=style),
        layouts.make_changes(html.Span(className="fi fi-fr"), style=style),
        layouts.make_changes(html.Span(className="fi fi-dk"), style=style),
        layouts.make_changes(html.Span(className="fi fi-it"), style=style),
        html.Br(),
        html.Span("... and more"),
    ]

    return _get_card("Flags", flag_examples, flag_source)


def _layout_ldavis() -> list[DashComponent]:
    """Returns a (list with a) card component with information about using pyLDAvis.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    image_url = (
        "https://web.archive.org/web/20221226160456/"
        "https://opengraph.githubassets.com/"
        "63f451dd091f540a8db46eb2fbe48ef293c3f00a9a21d932dee929b175534a43/"
        "bmabey/pyLDAvis"
    )
    image = html.Img(src=image_url, style=dict(width="100%"))

    license = [
        html.Span(
            "An exported analysis containing some of the pyLDAvis code is used to "
            "display the LDAvis visualization on the embedding page."
        ),
        html.Br(),
        html.Br(),
        html.Span("Copyright 2022 Ben Mabey and other contributors"),
        html.Br(),
        html.Span(
            'PyLDAvis is licensed under BSD 3-Clause "New" or "Revised" License: '
        ),
        html.Br(),
        html.A(
            "https://github.com/bmabey/pyLDAvis/blob/master/LICENSE",
            href="https://github.com/bmabey/pyLDAvis/blob/master/LICENSE",
            target="_blank",
        ),
    ]

    return _get_card("LDAvis", image, license)


def _layout_software() -> list[DashComponent]:
    """Returns a (list with a) card component with information about software used.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    code_image = layouts.make_changes(layouts.CODE_ICON, style={"font-size": "250%"})

    intro: list[DashComponent] = [
        html.Div("Built using lots of open-source projects like:")
    ]
    software_used = (
        "dash,numpy,pandas,gensim,pyarrow,gunicorn,pycountry,scikit-learn,"
        "flask_caching,more_itertools,dash_bootstrap_components"
    ).split(",")
    template = "https://pypi.org/project/{}/"
    links = [
        [html.A(link, href=template.format(link), target="_blank"), html.Span(", ")]
        for link in software_used
    ]
    others: list[DashComponent] = [
        html.Span("... and a lot more", title="e.g. the above's dependencies")
    ]
    references = intro + list(it.chain(*links)) + others

    return _get_card("Software", code_image, references)


def _get_card(
    header: str,
    left_col: DashComponent | list[DashComponent],
    right_col: DashComponent | list[DashComponent],
) -> list[DashComponent]:
    """Create a card with given header and two-column body layout.

    Parameters
    ----------
    header : str
        String to use as card header.
    left_col : DashComponent | list[DashComponent]
        Component(s) to use in the left (25% width) part of the card body.
    right_col : DashComponent | list[DashComponent]
        Component(s) to use in the right (25% width) part of the card body.

    Returns
    -------
    list[DashComponent]
        List containing the combined card component.
    """
    return [
        dbc.Card(
            [
                dbc.CardHeader(header),
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(left_col, width=3, align="center"),
                            dbc.Col(right_col, width=9, style={"text-align": "left"}),
                        ]
                    )
                ),
            ],
            className="mw-30",
        )
    ]
