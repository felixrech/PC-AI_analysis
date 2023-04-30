"""Module providing the layout for the /topic_details/ page."""

import pandas as pd
import itertools as it

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Component as DashComponent

from graphs import graphs
from utils import utils, topics
from utils.caching import memoize
from layouts import layouts, navbar


# Load the data
H, _, texts = utils.load_tm()
similarities = utils.load_similarities()
H_norm = utils.normalize_H(H)
topics_ = utils.get_topics(H)


def layout(
    document_selected: int = topics.most_relevant_doc(
        texts, H_norm, utils.get_topics(H_norm)[0]
    ),
    topic_selected: str = utils.get_topics(H_norm)[0],
    **kwargs,
) -> list[DashComponent]:
    """Compute the layout for the evaluation page. Any additional arguments are ignored.

    Parameters
    ----------
    document_selected : int, optional
        Internal index of the currently selected document, by default
        topics.most_relevant_doc(texts, H_norm, utils.get_topics(H_norm)[0]).
    topic_selected : str, optional
        Currently selected topic, by default utils.get_topics(H_norm)[0].

    Returns
    -------
    list[DashComponent]
        List of dash components.
    """
    current = texts.iloc[int(document_selected)]

    navbar_ = html.Div(
        id="navbar", children=navbar.get_navbar(f"Topic details: {topic_selected}")
    )

    l2_topic_plots = html.Div(
        _layout_left_column_modals(document_selected, topic_selected)
    )
    c2_current_text = html.Div(
        id="current_text",
        style={"max-width": "90vw", "text-align": "left"},
    )
    r2_topic_top_terms = dcc.Graph(
        id="topic_top_terms_plot",
        figure=graphs.dummy_fig(),
        config=dict(displayModeBar=False),
    )
    layout = [
        navbar_,
        html.Div(
            id="topic_selected",
            children=topic_selected,
            style=dict(display="none"),
        ),
        _layout_row(
            _layout_topic_doc_controls(document_selected),
            html.Span(id="current_info_table"),
            html.Div(id="topic_info_table"),
        ),
        html.Br(),
        _layout_row(l2_topic_plots, c2_current_text, r2_topic_top_terms),
        *_layout_dummy_elements(current["id"]),
    ]
    return layout


def _layout_topic_doc_controls(document_selected: int) -> list[DashComponent]:
    """Compute the layout of controls to select a document.

    Parameters
    ----------
    document_selected : int
        (Internal index of the) Currently selected topic.

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    # Dropdown menu to select the topic
    topic_buttons = [
        dbc.DropdownMenuItem(
            topic,
            id=dict(type="topic_selected_", index=topic),
        )
        for topic in topics_
    ]
    topic_selector = dbc.DropdownMenu(
        label="Change topic",
        children=topic_buttons,
        className="m-1",
    )

    # (Editable) currently selected document's internal_index
    doc_index = dbc.Input(
        id="document_selected",
        value=document_selected,
        type="number",
        min=0,
        max=len(H_norm) - 1,
        debounce=True,
        style={
            "background": "black",
            "color": "white",
            "width": "40%",
        },
    )

    # Buttons to switch to previous and next document (& current doc index)
    b_style = dict(n_clicks=0, class_name="me-1")
    buttons = [
        dbc.Button(layouts.LEFT_ARROW, id="document_selected_previous", **b_style),
        dbc.Button(
            layouts.LOADING_ICON,
            id="document_selected_topic_index",
            disabled=True,
            class_name="me-1",
        ),
        dbc.Button(
            layouts.RIGHT_ARROW, id="document_selected_next", color="primary", **b_style
        ),
    ]

    return [
        topic_selector,
        html.Div(
            children=[
                doc_index,
                *buttons,
            ],
            className="mx-auto flex_center",
        ),
    ]


def _layout_left_column_modals(
    document_selected: int, topic_selected: str
) -> list[DashComponent]:
    """Compute the layout of modals giving additional topic details.

    Parameters
    ----------
    document_selected : int
        (Internal index of the) Currently selected topic.
    topic_selected : str
        Currently selected topic.

    Returns
    -------
    list[DashComponent]
        List of layout components
    """
    modal_args = dict(size="lg", is_open=False)

    user_types_graphs = [
        dcc.Graph(figure=graphs.dummy_fig(), config=dict(displayModeBar=False), id=id)
        for id in [
            "topic_user_types_plot",
            "topic_user_types_dominant_plot",
            "topic_user_types_read",
            "topic_user_types_read_50",
        ]
    ]
    user_types_texts = [
        dbc.ModalBody(text)
        for text in [
            "Topic accounts, on average, for ... of documents",
            "Dominant topic for ...",
            "Top 20 documents in this topic are from ...",
            "Top 50 documents in this topic are from ...",
        ]
    ]
    user_types = [
        dbc.Button(
            "How common is this topic among different user types?",
            id="topic_user_types_button",
            className="fwp-m1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        f"How common is {topic_selected}?",
                        id="topic_user_types_modal_header",
                    )
                ),
                *it.chain(*zip(user_types_texts, user_types_graphs)),
            ],
            id="topic_user_types_modal",
            **modal_args,
        ),
    ]

    doc_numbers = [
        dbc.Button(
            "Document numbers by user type",
            id="topic_user_types_nums_button",
            className="fwp-m1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Document numbers by user type")),
                dbc.ModalBody(id="topic_user_types_doc_numbers"),
            ],
            id="topic_user_types_nums_modal",
            **modal_args,
        ),
    ]

    similarities = [
        dbc.Button(
            "What are similar documents?", id="similar_docs_button", className="fwp-m1"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle(f"What are similar docs to #{document_selected}?")
                ),
                dbc.ModalBody(html.Div(id="similar_docs")),
            ],
            id="similar_docs_modal",
            **modal_args,
        ),
    ]
    return [*user_types, *doc_numbers, *similarities]


def _similar_doc_title(similar_doc: pd.Series) -> str:
    """Extract the title for a similar document accordion item.

    Parameters
    ----------
    similar_doc : pd.Series
        Row of the dataframe with similar documents. Needs to contain "internal_index",
        "similarity", and "organization" columns.

    Returns
    -------
    str
        Title.
    """
    sim, index = similar_doc["similarity"], similar_doc["internal_index"]
    org = (
        similar_doc["organization"]
        if similar_doc["organization"] is not None
        else "<Anonymous>"
    )
    org = utils.cutoff_text(org, 43)
    return f"[{sim:.2f}]  #{str(index).rjust(4)} by {org}"


def _similar_doc_text(similar_doc: pd.Series) -> DashComponent:
    """Compute the text preview shown in the accordion item for a similar document.

    Parameters
    ----------
    similar_doc : pd.Series
        Row of the dataframe with similar documents. Needs to contain a "text" column.

    Returns
    -------
    DashComponent
        Layout.
    """
    return html.Div(utils.cutoff_text(similar_doc["text"], 1000), className="text")


@memoize
def layout_similar_docs(
    document_selected: int | str, method: str = "levenshtein"
) -> list[DashComponent]:
    """Compute the layout for the similar documents modal.

    Parameters
    ----------
    document_selected : int | str
        (Internal index of the) Currently selected topic.
    method : str, optional
        Similarity measure, either "tm", "levenshtein", or "embedding". By default
        "levenshtein".

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    intro = html.Span("Similarity measured by ...")
    buttons = [
        ("Topic model", "tm"),
        ("Levenshtein", "levenshtein"),
        ("SBERT", "embedding"),
    ]
    buttons = [
        dbc.Button(
            title,
            id=f"similar_docs_type_{id}",
            className="mm-a w-25",
            disabled=method == id,
        )
        for title, id in buttons
    ]
    hints = dict(
        tm=(
            "Topic model similarity measure: Cosine similarity between the topic "
            "distributions of current and compared document."
        ),
        levenshtein=(
            "Levenshtein: Similarity measure based on ratio of Levenshtein distance "
            "and document lengths."
        ),
        embedding=(
            "SBERT: Cosine similarity between the SBERT embeddings (transformer-based "
            "768-dimensional document embeddings) of current and compared document."
        ),
    )
    hint = layouts.info_hint(hints[method])
    similarity_types = [intro, html.Br(), *buttons, html.Br(), hint]

    df = pd.merge(
        utils.most_similar_docs(similarities, int(document_selected), method=method),
        texts,
    )
    accordion = layouts.accordion(df, title=_similar_doc_title, text=_similar_doc_text)

    return [html.Center(similarity_types), html.Br(), *accordion]


def _layout_dummy_elements(id: str | int) -> list[DashComponent]:
    """Create a layout of invisible dummy elements for accessing or storing information.

    Parameters
    ----------
    id : str | int
        Id of the currently selected document.

    Returns
    -------
    list[DashComponent]
        List of layout components.
    """
    return [
        dcc.Location(id="url", refresh=True),
        html.Div(children=id, id="id_div", **layouts.STYLE_INVISIBLE),
        html.Div(id="dummy_1", children="NO UPDATE", **layouts.STYLE_INVISIBLE),
        html.Div(id="dummy_2", **layouts.STYLE_INVISIBLE),
        html.Div(
            children="levenshtein", id="similar_docs_type", **layouts.STYLE_INVISIBLE
        ),
        html.Div(id="topic_details_page_load", **layouts.STYLE_INVISIBLE),
    ]


def _layout_row(
    left: DashComponent | list[DashComponent],
    center: DashComponent | list[DashComponent],
    right: DashComponent | list[DashComponent],
) -> DashComponent:
    """Make a row of three components.

    Parameters
    ----------
    left : DashComponent | list[DashComponent]
        Arbitrary dash component.
    center : DashComponent | list[DashComponent]
        Arbitrary dash component.
    right : DashComponent | list[DashComponent]
        Arbitrary dash component.

    Returns
    -------
    DashComponent
        Row component.
    """
    return dbc.Row(
        [
            html.Span(
                dbc.Card(
                    dbc.CardBody(html.Div(left, style={"margin-top": "0.4cm"})),
                    style={"height": "100%"},
                ),
                style=dict(width="25vw"),
            ),
            html.Span(
                dbc.Card(dbc.CardBody(center), style=dict(height="100%")),
                style=dict(width="50vw"),
            ),
            html.Span(
                dbc.Card(
                    dbc.CardBody(html.Div(right, style={"margin-top": "0.4cm"})),
                    style={"height": "100%"},
                ),
                style=dict(width="25vw"),
            ),
        ],
        style={"width": "100vw"},
    )
