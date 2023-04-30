"""Module for some very general layouts."""

import copy
import pycountry
import pandas as pd
import itertools as it
from typing import Callable

import dash
from dash._callback import NoUpdate
import dash_bootstrap_components as dbc
from dash import html, dcc, callback_context as ctx
from dash.dependencies import Component as DashComponent

from graphs import graphs
from utils import utils, sentiment


def fixed_width_with_margin(width: str) -> dict:
    """Create a style dictionary with fixed total width and 0.2cm margin.

    Parameters
    ----------
    width : str
        Total width including the margin.

    Returns
    -------
    dict
        Style dictionary.
    """
    return dict(width=f"calc({width} - 0.4cm - 1px)", margin="0.2cm")


def hint_accordion(
    hint: str | DashComponent | list[str] | list[DashComponent], title: str
) -> DashComponent:
    """Create a accordion with single accordion item containing hint and named title.

    Parameters
    ----------
    hint : str | DashComponent | list[str] | list[DashComponent]
        Accordion item content.
    title : str
        Accordion item label.

    Returns
    -------
    DashComponent
        Accordion component.
    """
    return dbc.Accordion(
        dbc.AccordionItem(hint, title=title),
        className="hint_accordion",
        start_collapsed=True,
    )


def hint_row(
    hint: str | list[str] | DashComponent | list[DashComponent],
) -> DashComponent:
    """Compute the layout for a informational hint.

    Includes a information "i" icon on the left and a configurable hint on the right.

    Parameters
    ----------
    hint : str | list[str] | DashComponent | list[DashComponent]
        Arbitrary hint.

    Returns
    -------
    DashComponent
        Layout component.
    """
    return dbc.Row(
        [
            dbc.Col(INFO_ICON_LARGE, width=1, align="center"),
            dbc.Col(hint, width=11, className="fs-sm text-left"),
        ],
        style={"margin-bottom": "0.5cm"},
    )


def cbl(component: DashComponent) -> DashComponent:
    """Center component vertically while left-aligning its text.

    Parameters
    ----------
    component : DashComponent
        Arbitrary dash component.

    Returns
    -------
    DashComponent
        Component wrapped in two divs to achieve alignments.
    """
    old_style = component.__dict__["style"] if "style" in component.__dict__ else dict()
    component.__dict__["style"] = old_style | {"text-align": "left"}

    return html.Div(
        html.Div(
            component,
            style=dict(display="inline-block"),
        ),
        style={"text-align": "center"},
    )


def accordion(
    df: pd.DataFrame,
    title: str | Callable[[pd.Series], str],
    text: str | Callable[[pd.Series], str | DashComponent | list[DashComponent]],
    always_open: bool = False,
    start_collapsed: bool = True,
) -> list[DashComponent]:
    """Accordion with information about different documents/mentions.

    Parameters
    ----------
    df : pd.DataFrame
        Information about documents/mentions. Needs to have "id", "date_feedback",
        "user_type", and "country" columns. Optional columns are: "sentiment",
        "positive", "neutral", "negative". If these are present, sentiment buttons will
        be shown on the left of each accordion item's contents. Might require additional
        columns depending on the values of the "title" and "text" arguments of this
        function.
    title : str | Callable[[pd.Series], str]
        Title of each accordion item, either as a constant string or as a function for
        each row of "df".
    text : str | Callable[[pd.Series], str  |  DashComponent  |  list[DashComponent]]
        Text to be shown within each accordion item, either as a constant string or as a
        function of each row of "df".
    always_open : bool, optional
        Whether to allow for the opening of more than one accordion item at a time, by
        default False.
    start_collapsed : bool, optional
        Whether to start the accordion in a collapsed state, by default True.

    Returns
    -------
    list[DashComponent]
        Layout component.
    """
    mentions = [
        dbc.AccordionItem(
            title=title if type(title) is str else title(sentence),  # type: ignore
            children=[
                accordion_info_buttons(sentence),
                text if type(text) is str else text(sentence),  # type: ignore
            ],
            className="accordion_highlighted",
        )
        for _, sentence in df.iterrows()
    ]

    return [
        dbc.Accordion(
            mentions, start_collapsed=start_collapsed, always_open=always_open
        ),
    ]


def accordion_info_buttons(sentence: pd.Series) -> DashComponent:
    """Strip to be shown below the accordion item button containing information about
    a document or mention.

    Parameters
    ----------
    sentence : pd.Series
        Information about a document/mention. Needs to have "id", "date_feedback",
        "user_type", and "country" labels. Optional labels are: "sentiment", "positive",
        "neutral", "negative". If these are present, sentiment buttons will be shown on
        the left.

    Returns
    -------
    DashComponent
        Layout component.
    """
    # Theming/layout
    args = dict(color="info", className="mm")
    href_args = args | dict(target="_blank", external_link=True)
    normal: dict[str, dict] = dict()
    underemphasized = dict(
        style={"background-color": "#63b9f2", "border-color": "#63b9f2"}
    )

    # Left buttons: percentages of each sentiment (if sentiment present)
    if "sentiment" in sentence.index:
        lbs_text = [
            html.Span(f"{sentence[sentiment]:.0%} ")
            for sentiment in ["positive", "neutral", "negative"]
        ]
        lbs_icons = [THUMBS_UP_ICON, NEUTRAL_ICON, THUMBS_DOWN_ICON]
        lbs_titles = ["Positive sentiment", "Neutral sentiment", "Negative sentiment"]
        lbs_args = [
            args | (normal if sentence["sentiment"] == sentiment else underemphasized)
            for sentiment in ["positive", "neutral", "negative"]
        ]
        left_buttons = [
            dbc.Button([text, icon], title=title, **args)
            for (text, icon, title, args) in zip(
                lbs_text, lbs_icons, lbs_titles, lbs_args
            )
        ]
    else:
        left_buttons = []

    # Center buttons: information about mention like user type or COO
    published_on = [
        CLOCK_ICON,
        html.Span(" " + pd.to_datetime(sentence["date_feedback"]).strftime("%d.%m")),  # type: ignore # noqa [E501]
    ]
    center_buttons = [
        dbc.Button(
            utils.user_type_to_name(sentence["user_type"]), title="User type", **args
        ),
        dbc.Button(get_country(sentence["country"]), title="Country of origin", **args),
        dbc.Button(published_on, title="Published on", **args),
    ]

    # Right buttons: links to feedback on HYS and local PDF attachment
    right_buttons = [
        dbc.Button(
            html.Span(className="fi fi-eu"),
            href=utils.hys_url(sentence["id"]),
            title="Open this feedback on Have your Say",
            **href_args,
        ),
    ]
    if utils.local_attachment_exists(sentence["id"]):
        right_buttons.append(
            dbc.Button(
                PDF_ICON,
                href=utils.local_attachment_href(sentence["id"]),
                title="PDF attachment of this feedback",
                **href_args,
            )
        )

    # Information buttons: Layout the above in three columns (non-fixed width)
    info_buttons = [
        dbc.Stack(
            [
                html.Div(left_buttons),
                html.Div(center_buttons, className="mx-auto"),
                html.Div(right_buttons),
            ],
            direction="horizontal",
            gap=3,
        )
    ]

    return html.Div(info_buttons, className="accordion_info_buttons")


def accordion_windowed_text(
    sentence: pd.Series,
    sentiments: pd.DataFrame,
    window_size: int = 2,
    main_column: str = "aspect",
) -> DashComponent:
    """Compute layout for text extract of a single sentiment/mention.

    Parameters
    ----------
    sentence : pd.Series
        Row of the sentiment dataset. Needs "sentence_index", "sentence_before",
        "sentence_after", and "aspect" labels.
    sentiments : pd.DataFrame
        Sentiments dataset as generated by the 07_sentiment module. Needs to be loaded
        with filter_ set to False.
    window_size : int, optional
        Size of the symmetric context window as the number of sentences before and
        after, by default 2.
    main_column : str, optional
        Column of the highlighted part of the main sentence, by default "aspect".

    Returns
    -------
    DashComponent
        Layout component.
    """
    # Collect the context before and after the highlighted sentence/mention
    left_context, right_context = [
        sentiment.context_string(sentiments, sentence, direction, window_size)
        for direction in ["before", "after"]
    ]

    # Complete text - including context before and after highlighted sentence/mention
    text = [
        html.Span(left_context + " ", className="text-muted"),
        html.Span(sentence["sentence_before"] + " "),
        html.Strong(
            html.Span(sentence[main_column]), style={"text-decoration": "underline"}
        ),
        html.Span(" " + sentence["sentence_after"]),
        html.Span(" " + right_context, className="text-muted"),
    ]
    return html.Div(text, className="text")


def topics_modal(
    id_prefix: str,
    bottom_margin: bool = True,
    modal_additions: list[DashComponent] = [],
) -> list[DashComponent]:
    """Compute the layout for a modal to show the relation of selected sentences to
    topics.

    Parameters
    ----------
    id_prefix : str
        Prefix of the component's ids, e.g. "search" or "sentiment".
    bottom_margin : bool, optional
        Add some space below the button badge, by default True.
    modal_additions : list[DashComponent], optional
        What to add below the modal body, e.g. some kind of disclaimer footer. By
        default empty list (i.e. nothing).

    Returns
    -------
    list[DashComponent]
        Layout containing button and modal.
    """
    # Button to open the modal
    modal_button = dbc.Button(
        "Relation to topics",
        id=f"{id_prefix}_topic_modal_button",
        style={"width": "80%", "max-width": "5cm"}
        | ({"margin-bottom": "0.5cm"} if bottom_margin else dict()),
    )

    # Button badges to select plot variant
    positive_ = button_badge(
        "Dominant topic", f"{id_prefix}_topic_dominant", checked=True
    )
    negative_ = button_badge("Average share", f"{id_prefix}_topic_share", checked=False)
    variant_buttons = [
        make_changes(button, className="fw-10 mm-a")
        for button in [positive_, negative_]
    ]
    variant_buttons = dbc.Row([dbc.Col(badge, width=6) for badge in variant_buttons])

    # Modal layout
    modal = dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Which topics do these mentions belong to?")
            ),
            dbc.ModalBody(
                [
                    html.Center(variant_buttons),
                    html.Br(),
                    dcc.Graph(
                        figure=graphs.dummy_fig(),
                        id=f"{id_prefix}_topic_fig",
                        **graphs.GRAPH_CONF_UNRESPONSIVE,
                    ),
                ]
            ),
        ]
        + modal_additions,
        id=f"{id_prefix}_topic_modal",
        **dict(is_open=False, size="xl"),
    )

    # Complete layout
    return [html.Div([modal_button, modal])]


def filter_user_types_modal(
    df: pd.DataFrame, id_prefix: str, disclaimer: bool = True
) -> list[DashComponent]:
    """Compute the layout for a modal to filter the mentions in the details by user
    type.

    Parameters
    ----------
    df : pd.DataFrame
        Arbitrary dataframe with a "user_type" column.
    id_prefix : str
        Prefix of the component's ids, e.g. "search" or "sentiment".
    disclaimer : bool
        Include a disclaimer about details updating only when closing modal.

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    user_types = utils.user_types_present(df["user_type"]).values
    buttons = [
        make_changes(
            button_badge(
                name,
                dict(type=f"{id_prefix}_filter_badge", index=user_type),
                checked=True,
            ),
            style={"width": "10cm", "max-width": "100%"},
            className="mm-a",
        )
        for (name, user_type) in user_types
    ]
    buttons = list(it.chain(*zip(buttons, it.repeat(html.Br(), len(buttons) - 1))))

    modal = html.Div(
        [
            dbc.Button(
                id=f"{id_prefix}_filter_modal_button",
                style={"width": "80%", "max-width": "5cm"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle("What user types do you want to filter to?")
                    ),
                    dbc.ModalBody(html.Center(buttons)),
                ]
                + ([_DISCLAIMER_FOOTER] if disclaimer else []),
                id=f"{id_prefix}_filter_modal",
                is_open=False,
                size="lg",
            ),
        ]
    )
    return [modal]


def filter_user_types_extract(filter_badge_children: list[list[dict]]) -> list[str]:
    """Extract list of selected filter types from their respective button badges.

    Parameters
    ----------
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.

    Returns
    -------
    list[str]
        List of user types, e.g. ["company", "business_association"].
    """
    texts = [button_badge_get_text(badge) for badge in filter_badge_children]
    checked = [button_badge_get_checked(badge) for badge in filter_badge_children]

    filter_user_types = [text for text, checked_ in zip(texts, checked) if checked_]
    return filter_user_types if len(filter_user_types) < len(texts) else []


def filter_user_types_modal_callback(
    is_open: bool, filter_badge_children: list[list[dict]]
) -> str | NoUpdate:
    """Update the text of the button to open the filter options modal.

    Parameters
    ----------
    is_open : bool
        Whether the filter options modal is open or not.
    filter_badge_children : list[list[dict]]
        "Children" properties of the filter user types button badges.

    Returns
    -------
    str | NoUpdate
        Button text, e.g. "Filter: all types" or "Filter: 3 types".
    """
    # Delay button update until the options modal is closed
    if is_open:
        return dash.no_update

    checked = [button_badge_get_checked(badge) for badge in filter_badge_children]

    if all(checked) or not any(checked):
        return "Filter: all types"
    else:
        return f"Filter: {sum(checked)} types"


def make_changes(
    component: DashComponent, additive: bool = False, **kwargs
) -> DashComponent:
    """Apply additional arguments as changes to a dash component.

    Parameters
    ----------
    component : DashComponent
        Arbitrary dash component.
    additive : bool, optional
        If the attribute currently exists and is a dict or string, merge with the
        argument value.

    Returns
    -------
    DashComponent
        Dash component with changes applied.

    Examples
    --------
    >>> make_changes(html.I(), className="bi bi-check2-all")
    I(className='bi bi-check2-all')
    """
    for attribute, value in kwargs.items():
        if (
            additive
            and hasattr(component, attribute)
            and type(getattr(component, attribute)) is dict
        ):
            setattr(component, attribute, getattr(component, attribute) | value)
        elif (
            additive
            and hasattr(component, attribute)
            and type(getattr(component, attribute)) is str
        ):
            setattr(component, attribute, getattr(component, attribute) + f" {value}")
        else:
            setattr(component, attribute, value)
    return component


def info_hint(text: str) -> DashComponent:
    """Layout for a hint ("i" icon with text to the right of it).

    Parameters
    ----------
    text : str
        Arbitrary text.

    Returns
    -------
    DashComponent
        Layout component.
    """
    return html.Div(
        [
            make_changes(INFO_ICON, className="fs-sm", additive=True),
            html.Span(text, style={"font-size": "small"}),
        ],
        style={"line-height": "0.35rem"},
    )


def plotly_hint(
    legend_element: str = "user type", separator: bool = True
) -> list[DashComponent]:
    """Compute the layout for a hint about Plotly - ability to limit plot to legend
    elements by double clicking on them.

    Parameters
    ----------
    legend_element : str, optional
        Name of label element (does not have to match plot's legend title), by default
        "user type".
    separator : bool, optional
        Whether to add a separating line before the hint, by default True.

    Returns
    -------
    list[DashComponent]
        Layout components.
    """
    line = html.Hr(style={"margin-top": "1cm", "margin-bottom": "0px"})
    hint = html.Div(
        info_hint(
            f"Double-click on a {legend_element} in the legend to limit the barplot to "
            f"that user (same for going back to all {legend_element}s)."
        ),
        className="flex_right_align",
    )
    return [line, hint] if separator else [hint]


def toggle_modal(n_clicks: None | int, is_open: bool) -> bool:
    """Toggle whether a modal is open.

    Parameters
    ----------
    n_clicks : None | int
        Number of clicks on the modal button.
    is_open : bool
        Whether the modal is currently open.

    Returns
    -------
    bool
        Whether a modal is open.
    """
    if n_clicks:
        return not is_open
    return is_open


def toggle_modal_multiple(*args) -> bool:
    """Toggle whether a modal is open. Can deal with multiple modal opener buttons.

    Parameters: "n_clicks" attribute of any number of buttons, followed by modal's
    current "is_open" state.

    Returns
    -------
    bool
        Whether a modal is open.
    """
    if any(args[:-1]):
        return not args[-1]
    return bool(args[-1])


TOGGLE_MODAL = """
function(n_clicks, is_open) {
    if ( n_clicks == null ) {
        return is_open;
    }
    return !is_open;
}
"""


def button_badge(
    text: str, id: str | dict[str, str], checked: bool = True, fake: bool = True
) -> DashComponent:
    """Create a button badge.

    Parameters
    ----------
    text : str
        Text to display within the button badge.
    id : str | dict[str, str]
        Dash/HTML id of the button badge.
    checked : bool, optional
        Whether the badge is checked (True) or not, by default True.
    fake : bool, optional
        Add invisible fake checkmark/cross to accurately center text, by default True.

    Returns
    -------
    DashComponent
        Button badge layout.
    """
    return dbc.Button(
        button_badge_children(text=text, checked=checked, fake=fake),
        id=id,
        color="primary",
        style=dict(width="100%"),
    )


def button_badge_children(
    text: str, checked: bool = True, fake: bool = True
) -> list[DashComponent]:
    """Layout/children of a button badge.

    Parameters
    ----------
    text : str
        Text to show within the button badge.
    checked : bool, optional
        Whether or not the button badge is currently checked, by default True.
    fake : bool, optional
        Add invisible fake checkmark/cross to accurately center text, by default True.

    Returns
    -------
    list[DashComponent]
        Layout component(s).
    """
    checkmark, cross = (
        dbc.Badge(
            html.I(
                className=className,
                style={"font-size": "large", "font-weigh": "bold"},
            ),
            color="white",
            text_color="primary",
            className="ms-1",
            style={
                "padding": "2px",
                "margin-top": "0.1cm",
                "display": "inline-block",
                "float": "right",
            },
        )
        for className in ["bi bi-check-lg", "bi bi-x-lg"]
    )
    fake_checkmark, fake_cross = (
        make_changes(
            copy.deepcopy(component),
            additive=True,
            style=dict(float="left", visibility="hidden"),
        )
        for component in (checkmark, cross)
    )

    return [
        *([fake_checkmark if checked else fake_cross] if fake else []),
        html.Span(
            text,
            style={
                "display": "inline-block",
                "float": "center",
            },
        ),
        checkmark if checked else cross,
    ]


def button_badge_get_checked(badge_children: list[dict]) -> bool:
    """Extract whether a button badge is currently checked.

    Parameters
    ----------
    badge_children : list[dict]
        Button badge's "children" property.

    Returns
    -------
    bool
        True if checked, False otherwise.
    """
    return bool(
        badge_children[-1]["props"]["children"]["props"]["className"]
        == "bi bi-check-lg"
    )


def button_badge_get_text(badge_children: list[dict]) -> str:
    """Extract the text of a button badge.

    Parameters
    ----------
    badge_children : list[dict]
        Button badge's "children" property.

    Returns
    -------
    str
        Text content.
    """
    return str(badge_children[-2]["props"]["children"])


def update_buttons_badge(
    n_clicks: None | int, badge_children: list[dict]
) -> list[DashComponent] | NoUpdate:
    """Callback to update a button badge's checkmark based on clicks on it.

    Parameters
    ----------
    n_clicks : None | int
        Number of clicks on the button badge.
    badge_children : list[dict]
        "Children" property of the button badge.

    Returns
    -------
    list[DashComponent] | NoUpdate
        Updated "children" of the button badge.
    """
    # Suppress page load callback
    if not n_clicks:
        return dash.no_update

    currently_checked = button_badge_get_checked(badge_children)
    text = button_badge_get_text(badge_children)
    checked = not currently_checked

    return button_badge_children(text, checked, fake=len(badge_children) == 3)


def update_buttons_badges_mutex(
    *args, **kwargs
) -> list[list[DashComponent] | NoUpdate]:
    """Update multiple button badges at the same time. Only one button badge may be
    active at a time.

    Important: This callback needs to be initialized with the names argument, containing
    the end (anything after the last "_") of each button badge's id.

    Returns
    -------
    list[DashComponent]
        List of (updated) button badge "children".
    """
    names = kwargs["names"]

    # Ignore initial callbacks
    if not any(args[: len(names)]):
        return list(it.repeat(dash.no_update, len(names)))

    # Checked button badge = clicked button badge
    new_checked = str(ctx.triggered_id).split("_")[-1]
    checked = [name == new_checked for name in names]

    texts = [button_badge_get_text(children) for children in args[len(names) :]]
    return [
        button_badge_children(text, checked) for text, checked in zip(texts, checked)
    ]


def get_country(alpha_3: str) -> DashComponent:
    """Transforms a country's ISO alpha 3 abbreviation into a string. For example,
    "NLD" is transformed into "🇳🇱 Netherlands".

    Parameters
    ----------
    alpha_3 : str
        A country's ISO alpha 3 abbreviation, e.g. "NLD"

    Returns
    -------
    DashComponent
        String with flag and country's name, e.g. HTML equivalent of "🇳🇱 Netherlands".
    """
    country = pycountry.countries.get(alpha_3=alpha_3)
    return html.Span(
        [
            html.Span(className=f"fi fi-{country.alpha_2.lower()}"),
            html.Span(f" {country.name}"),
        ]
    )


def multiselect_button(id_prefix: str) -> list[DashComponent]:
    """Create a button badge layout that allows for keeping multiple mention detail
    accordion items open at the same time.

    Parameters
    ----------
    id_prefix : str
        Prefix of the component's ids, e.g. "search" or "sentiment".

    Returns
    -------
    list[DashComponent]
        Layout.
    """
    return [
        make_changes(
            button_badge(
                "Multiselect", f"{id_prefix}_multiselect", checked=False, fake=False
            ),
            style={"width": "80%", "max-width": "5cm"},
        )
    ]


_DISCLAIMER_FOOTER = dbc.ModalFooter(
    html.Div(
        "Changes will be applied as soon as you close this window!",
        style={"font-size": "small"},
    )
)


STYLE_INVISIBLE = dict(style=dict(display="none"))


LEFT_ARROW = html.I(className="bi bi-caret-left-fill m-0")
RIGHT_ARROW = html.I(className="bi bi-caret-right-fill m-0")
CLOCK_ICON = html.I(className="bi bi-clock m-0")
PDF_ICON = html.I(className="bi bi-file-earmark-pdf m-0")
HOUSE_ICON = html.I(className="bi bi-house m-0")
TOOLS_ICON = html.I(className="bi bi-tools m-0")
DOUBLE_CHECK_ICON = html.I(className="bi bi-check2-all m-0")
X_MARK_ICON = html.I(className="bi bi-x-circle m-0")
CODE_ICON = html.I(className="bi bi-code-square m-0")
THUMBS_UP_ICON = html.I(className="bi bi-hand-thumbs-up-fill m-0")
THUMBS_DOWN_ICON = html.I(className="bi bi-hand-thumbs-down-fill m-0")
NEUTRAL_ICON = html.I(className="bi bi-emoji-neutral-fill m-0")
INFO_ICON = html.I(className="bi bi-info-lg m-0")
INFO_ICON_LARGE = html.I(className="fs-lg bi bi-info-lg m-0")
DANGER_ICON = html.I(className="bi bi-exclamation-triangle-fill m-0")
SINGLE_CHECK = html.I(className="bi bi-check-lg m-0")
X_MARK_NO_CIRC_ICON = html.I(className="bi bi-x-lg m-0")
LOADING_ICON = html.I(className="bi bi-arrow-clockwise m-0")
