"""Module containing the callbacks for the /aspects/ page."""

from typing import Any
import plotly.graph_objects as go

import dash
import diskcache
from dash._callback import NoUpdate
from dash._utils import AttributeDict
from dash.dependencies import Input, Output, ALL
from dash.dependencies import Component as DashComponent
from dash import dcc, ctx, DiskcacheManager

from graphs import graphs
from layouts import layouts
from utils import utils, caching


cache = diskcache.Cache("/tmp/topic_explorer_cache")
background_callback_manager = DiskcacheManager(cache)


def _update_popularity_dominant(_: Any) -> go.Figure:
    """Update the topic popularity as "dominant" barplot.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    return graphs.topics_popularity()[0]


def _update_popularity_relevant(_: Any) -> go.Figure:
    """Update the topic popularity as "relevant" barplot.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback on page load).

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    return graphs.topics_popularity()[1]


def _update_comparison_user_types(
    *_,
) -> tuple[bool | NoUpdate, bool | NoUpdate, list[go.Figure] | list[NoUpdate]]:
    """Update the layout of barplots of which topics each of the different user types
    talk about.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callbacks).

    Returns
    -------
    tuple[bool | NoUpdate, bool | NoUpdate, list[go.Figure] | list[NoUpdate]]
        Share and dominant buttons' disabled property and user types barplots figures.
    """

    if ctx.triggered_id is None:
        figs = graphs.user_types_to_topics(variant="share")
        return dash.no_update, dash.no_update, figs

    variant = str(ctx.triggered_id).replace("comparison_user_types_buttons_", "")
    figs = graphs.user_types_to_topics(variant=variant)
    return variant == "share", variant == "dominant", figs


def _update_tm(*_) -> tuple[bool, bool] | tuple[NoUpdate, NoUpdate]:
    """Update the topic model comparison's method buttons' active status based on user
    selection.

    Returns
    -------
    tuple[bool, bool] | tuple[NoUpdate, NoUpdate]
        Disabled status of the two buttons.
    """
    if ctx.triggered_id is None:
        return dash.no_update, dash.no_update

    hellinger = ctx.triggered_id == "tm_topics_heatmap_type_button_hellinger"
    return hellinger, not hellinger


def _update_other_tm_selected(_) -> str | NoUpdate:
    """Update internal dummy component based on which topic model the user selected for
    comparison.

    Parameters
    ----------
    _ : Any
        Ignored (only for triggering callback).

    Returns
    -------
    str | NoUpdate
        Currently selected topic model.
    """
    if (
        type(ctx.triggered_id) is AttributeDict
        and ctx.triggered_id["type"] == "other_tm_selected_"
    ):
        return f"Other topic model: {ctx.triggered_id['index']}"

    return dash.no_update


def _update_tm_comparison_rand(tm_selected: str) -> go.Figure:
    """Update the rand index barplot for comparing topic models.

    Parameters
    ----------
    tm_selected : str
        Topic model selected for comparison.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    main, other = "nmf_page", tm_selected[19:]
    R = utils.rand_index(main, other)

    return graphs.rand_index_bar(R)


def _update_tm_comparison_heatmap(
    tm_selected: str, type_hellinger: bool
) -> tuple[go.Figure, DashComponent]:
    """Update the topic model comparison heatmap.

    Parameters
    ----------
    tm_selected : str
        Topic model selected for comparison
    type_hellinger : bool
        Whether to use Hellinger distance (True) or top terms overlap (False).

    Returns
    -------
    tuple[go.Figure, DashComponent]
        Heatmap figure and heatmap explanation hint.
    """
    main, other = "nmf_page", tm_selected[19:]

    hint_text = (
        "- The heatmap above shows the similarity between topics from the active "
        "topic model (i.e. NMF) and the topic model selected for comparison (e.g. "
        "LDA). \n"
        "- Hellinger: For each combination of topics, the Hellinger distance between "
        "the columns of the term-topic matrices is computed (and turned into a proper "
        "similarity measure). Similarity values lie in $[0, 1]$, where $1$ "
        "represents maximum similarity and $0$ no similarity whatsoever. "
        "This approach was inspired by Gensim's [How to Compare LDA Models]("
        "https://radimrehurek.com/gensim/auto_examples/howtos/run_compare_lda.html)"
        " guide.\n"
        "- The 'top terms overlap' method is inspired by the computation of topic "
        "coherence measures like $C_V$ that only utilize the top $n$ terms in each "
        "topic (with $n$ between $10$ and $20$ common - see [Röder et al., 2015]("
        "https://web.archive.org/web/20221221132820/http://svn.aksw.org/papers/2015/"
        "WSDM_Topic_Evaluation/public.pdf) and [Gensim documentation]("
        "https://web.archive.org/web/20221221132852/"
        "https://radimrehurek.com/gensim/models/coherencemodel.html)). "
        "We adapt this to compare topics by how many of their top $20$ terms overlap. "
        "Normalizing this count by dividing by $20$ provides a similarity measure on "
        "$[0, 1]$, where $1$ represents maximum similarity."
    )
    hint = layouts.cbl(
        dcc.Markdown(
            hint_text,
            mathjax=True,
        )
    )
    hint = layouts.hint_accordion(hint, "How was this figure created?")

    fig = graphs.tm_topics_heatmap(main, other, type_hellinger)

    return fig, hint


def initialize_caching() -> None:
    """Initialize caching of figures and layouts to speed up callbacks.

    Currently cached:
    - Number of mentions area plot (complete)
    - Sentiment area plot (complete)
    """
    for variant in caching.progress_bar(["share", "dominant"], "/overview/"):
        graphs.topics_popularity()
        graphs.user_types_to_topics(variant=variant)


def setup_callbacks() -> None:
    """Setup the callbacks for element on the /sentiment/ page."""
    dash.get_app().callback(
        Output("overview_popularity_dominant", "figure"),
        Input("overview_page_load", "children"),
    )(_update_popularity_dominant)
    dash.get_app().callback(
        Output("overview_popularity_relevant", "figure"),
        Input("overview_page_load", "children"),
    )(_update_popularity_relevant)

    dash.get_app().callback(
        [
            Output("comparison_user_types_buttons_share", "disabled"),
            Output("comparison_user_types_buttons_dominant", "disabled"),
            Output({"type": "overview_user_type_barplots", "index": ALL}, "figure"),
        ],
        [
            Input("comparison_user_types_buttons_share", "n_clicks"),
            Input("comparison_user_types_buttons_dominant", "n_clicks"),
        ],
    )(_update_comparison_user_types)

    dash.get_app().callback(
        [
            Output("tm_topics_heatmap_type_button_hellinger", "disabled"),
            Output("tm_topics_heatmap_type_button_top_terms", "disabled"),
        ],
        [
            Input("tm_topics_heatmap_type_button_hellinger", "n_clicks"),
            Input("tm_topics_heatmap_type_button_top_terms", "n_clicks"),
        ],
    )(_update_tm)

    dash.get_app().callback(
        Output("other_tm_selected", "label"),
        Input({"type": "other_tm_selected_", "index": ALL}, "n_clicks"),
    )(_update_other_tm_selected)

    dash.get_app().callback(
        Output("rand_index_pie", "figure"),
        Input("other_tm_selected", "label"),
        background=True,
        running=[
            (
                Output("rand_index_container", "style"),
                {"opacity": "0.5"},
                {"opacity": "1"},
            ),
            (
                Output("rand_index_pie", "figure"),
                graphs.rand_index_bar(0),
                graphs.rand_index_bar(0),
            ),
            (
                Output("other_tm_selected", "disabled"),
                True,
                False,
            ),
        ],
        manager=background_callback_manager,
    )(_update_tm_comparison_rand)

    dash.get_app().callback(
        [
            Output("tm_topics_heatmap", "figure"),
            Output("tm_topics_heatmap_explanation", "children"),
        ],
        [
            Input("other_tm_selected", "label"),
            Input("tm_topics_heatmap_type_button_hellinger", "disabled"),
        ],
    )(_update_tm_comparison_heatmap)
