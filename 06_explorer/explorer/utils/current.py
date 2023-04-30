"""Utilities for information about the currently selected document."""

import pandas as pd

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from utils import utils
from layouts import layouts


def info_table(
    current: pd.Series,
    topic_selected: str,
    doc_topic_index: int,
    H_norm: pd.DataFrame,
) -> DashComponent:
    """Show an information table about the currently selected document.

    Parameters
    ----------
    current : pd.Series
        Series containing information about currently selected document, including "id",
        "page", "organization", "user_type", "internal_index", and topic_selected
        labels.
    topic_selected : str
        Currently selected topic.
    doc_topic_index : int
        Index of the document in the selected topic.
    H_norm : pd.DataFrame
        (Row-) Normalized topic-document matrix H.

    Returns
    -------
    DashComponent
        Table component.
    """
    id, url, page = current["id"], utils.hys_url(current["id"]), current["page"]

    link = [
        html.A(href=url, children=id, target="_blank"),
        dcc.Clipboard(
            target_id="id_div",
            title="copy",
            style={
                "display": "inline-block",
                "fontSize": 10,
                "padding": 0,
                "padding-left": 8,
            },
        ),
    ]
    organization = (
        current["organization"]
        if current["organization"] is not None
        else "(Anonymous submission)"
    )
    user_type = utils.user_type_to_name(current["user_type"])
    is_dom = utils.current_is_dominant_topic(current, H_norm, topic_selected)
    topic_share = [
        html.Span(str(round(current[topic_selected] * 100, 1)) + "%"),
        layouts.make_changes(
            layouts.DOUBLE_CHECK_ICON if is_dom else layouts.X_MARK_ICON,
            title=(
                f"'{topic_selected}' is {'' if is_dom else 'not '}the dominant topic!"
            ),
            className="px-2",  # Add some (left) padding
            additive=True,
        ),
    ]

    if utils.local_attachment_exists(id):
        reminder = str(doc_topic_index)
        reminder += utils.user_type_to_abbreviation(current["user_type"])
        href = utils.local_attachment_href(id, page=page, reminder=reminder)
        link += [
            html.Br(),
            html.A("PDF" if page > 0 else "(PDF)", href=href, target="_blank"),
        ]

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Feedback", style=dict(width="15%")),
                        html.Th("Topic share", style=dict(width="15%")),
                        html.Th("User type", style=dict(width="20%")),
                        html.Th("Organization"),
                    ]
                )
            ),
            html.Tbody(
                html.Tr(
                    [
                        html.Td(link),
                        html.Td(topic_share),
                        html.Td(user_type),
                        html.Td(organization),
                    ]
                )
            ),
        ]
    )
    return table


def text(
    current: pd.Series, topic_top_terms: pd.DataFrame, topic_selected: str
) -> DashComponent:
    """Turn the current document's text into a dash component.

    Parameters
    ----------
    current : pd.Series
        Information about currently selected document with "tokenized_text" and
        "tokenized_lemmas" labels.
    topic_top_terms : pd.DataFrame
        Extract of an arbitrary number of lemmas with highest lemma share (of normalized
        term-topic matrix W), with "lemma" and topic_selected labels.
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    DashComponent
        Div component.
    """
    # Compute which tokens to highlight and the frequency of common terms
    is_important = pd.Series(current["tokenized_lemmas"]).isin(
        set(topic_top_terms["lemma"])
    )
    freq = {
        lemma: round(
            topic_top_terms.query("lemma == @lemma").iloc[0][topic_selected] * 100, 1
        )
        for lemma in topic_top_terms["lemma"]
    }

    # Properly denote terms without associated lemma
    lemmas = [
        lemma if lemma is not None else "filtered out (no lemma)"
        for lemma in current["tokenized_lemmas"]
    ]

    # Transform into html components
    spans = [
        _token_to_span(token, lemma, is_important, freq)
        for token, lemma, is_important in zip(
            current["tokenized_text"], lemmas, is_important
        )
    ]
    return html.Div(spans)


def _token_to_span(
    term: str, lemma: str, is_important: bool, freq: dict[str, float]
) -> DashComponent:
    """Turn a term into an dash span, including hover information about its lemma and
    lemma's topic share and bold font for important terms.

    Parameters
    ----------
    term : str
        Token before lemmatization.
    lemma : str
        Token's lemma.
    is_important : bool
        Whether token is important (bold).
    freq : dict[str, float]
        Frequencies of lemmas.

    Returns
    -------
    DashComponent
        Dash span or b component.
    """
    # Quotes are not automatically translated by dash
    term_html, lemma_html = term.replace('"', "&quot;"), lemma.replace('"', "&quot;")

    # Add hover information
    title = f"Token: {term_html}\nLemma: {lemma_html}"
    if is_important:
        title += f"\nFrequency: {freq[lemma]}%"

    # Build the output span
    span = html.Span(
        term_html,
        title=title,
        className="opacity-100" if is_important else "opacity-75",
    )
    return html.B(span) if is_important else span
