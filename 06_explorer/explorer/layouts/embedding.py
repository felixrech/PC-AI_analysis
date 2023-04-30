"""Module providing the layout for the /embedding/ page."""

import flask
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from utils import utils
from graphs import graphs
from layouts import navbar
from layouts import layouts


H, W, texts = utils.load_tm()
tsne = utils.load_tsne()
topics = utils.get_topics(H)
H_norm, W_norm = utils.normalize_H(H), utils.normalize_W(W)


def layout(**kwargs) -> list[DashComponent]:
    """Compute the complete page layout. Ignores any arguments.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        navbar.get_navbar("Embedding"),
        html.Br(),
        *_layout_tsne_embeddings(),
        *_layout_dendrogram(),
        *_layout_ldavis(),
        *_layout_dummy(),
    ]


def _layout_tsne_embeddings() -> list[DashComponent]:
    """Compute the layout for the TSNE embeddings.

    Returns
    -------
    list[DashComponent]
        List of dash components.
    """
    alert = dbc.Alert(
        "Limit the plots to a single user type or dominant topic by "
        "double-clicking the respective legend entries!",
        id="alert_note_double_click",
        color="primary",
        dismissable=True,
        style=dict(width="calc(100vw - 0.4cm)", margin="0.2cm"),
        is_open="alert_note_double_click_dismissed" not in flask.request.cookies.keys(),
    )

    plots = dbc.Row(
        [
            _layout_tsne_embedding_single(
                "By user type", "embedding_user_type", "user_type_name"
            ),
            _layout_tsne_embedding_single(
                "By dominant topic", "embedding_topic", "dominant_topic"
            ),
        ],
        style=dict(width="100vw"),
    )

    return [alert, plots]


def _layout_dendrogram() -> list[DashComponent]:
    """Compute the dendrogram layout.

    Returns
    -------
    list[DashComponent]
        Layout components.
    """
    graph = dcc.Graph(
        id="embedding_dendrogram",
        figure=graphs.dummy_fig(),
        style=dict(width="97%"),
        **graphs.GRAPH_CONF,
    )

    hint = layouts.cbl(
        dcc.Markdown(
            "The dendrogram above is created as a hierarchical clustering of the "
            "term-topic matrix $W$.\n\n"
            "Details:\n"
            "- Agglomerative clustering\n"
            "- Average linkage\n"
            "- Euclidean distance measure\n"
            "- $W$ by-topic normalized",
            mathjax=True,
        )
    )
    hint = layouts.hint_accordion(hint, "How was this figure created?")

    return [
        dbc.Card(
            dbc.CardBody([graph, html.Br(), hint]),
            style=dict(width="calc(100vw - 0.4cm - 1px)", margin="0.2cm"),
        )
    ]


def _layout_tsne_embedding_single(header: str, id: str, legend: str) -> DashComponent:
    """Compute the layout of a single TSNE embedding scatter plot.

    Parameters
    ----------
    header : str
        Column header to show above the plot.
    id : str
        HTML id for the graph.
    legend : str
        What to use for coloring the scatter plot, either "user_type_name" or
        "dominant_topic".

    Returns
    -------
    DashComponent
        Card component.
    """
    header_h5 = html.H5(header)
    graph = dcc.Graph(
        id=id,
        # figure=graphs.tsne_embedding(legend=legend),
        figure=graphs.dummy_fig(),
        **graphs.GRAPH_CONF_UNRESPONSIVE,
    )

    hint = layouts.cbl(
        dcc.Markdown(
            "The plot above is created based on a TSNE embedding of the topic-document "
            "matrix $H$.\n\n"
            "Details:\n"
            "- Using the sklearn implementation: "
            "`TSNE(n_components=2, init='pca', learning_rate='auto')`\n"
            "- $H$ by-document normalized\n"
            f"- Colored by {legend.replace('_', ' ')}",
            mathjax=True,
        )
    )
    hint = layouts.hint_accordion(hint, "How was this figure created?")

    return dbc.Card(
        dbc.CardBody([header_h5, graph, hint]),
        style=layouts.fixed_width_with_margin("50vw"),
    )


def _layout_ldavis() -> list[DashComponent]:
    """Compute the LDAvis layout.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    iframe = html.Iframe(
        srcDoc="",
        sandbox="allow-scripts",
        className="lda_iframe",
        id="ldavis_iframe",
    )

    hint = layouts.cbl(
        dcc.Markdown(
            "The figure above was created using the pyLDAvis package, available on "
            "[PyPI](https://pypi.org/project/pyLDAvis/).\n\n"
            "For more information, see...\n"
            "- The original paper: [LDAvis: A method for visualizing and interpreting "
            "topics (Sievert & Shirley, 2014)](https://web.archive.org/web/"
            "20221026074119/https://aclanthology.org/W14-3110.pdf)\n"
            "- The original R package's [vignette]"
            "(https://web.archive.org/web/20220902090932/"
            "https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)\n"
            "- Other literature using LDAvis, e.g. "
            "[Open data visualizations and analytics as tools for policy-making "
            "(Hagen et al., 2019)](https://doi.org/10.1016/j.giq.2019.06.004) "
            "(PDF available on Sci-Hub)"
        )
    )
    hint = html.Center(
        layouts.hint_accordion(hint, "How was this figure created?"),
        style={"margin-bottom": "0.2cm"},
    )

    return [
        dbc.Card(
            [iframe, html.Br(), hint],
            style=dict(width="calc(100vw - 0.4cm - 1px)", margin="0.2cm"),
        )
    ]


def _layout_dummy() -> list[DashComponent]:
    """Layout consisting of invisible dummy components.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [html.Div(id="embedding_page_load", **layouts.STYLE_INVISIBLE)]
