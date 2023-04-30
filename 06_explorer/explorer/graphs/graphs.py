"""Module containing graphs without association to one of the other utils modules."""

import numpy as np
import pandas as pd
import itertools as it
from functools import partial
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.graph_objs.layout import XAxis

from utils import topics, utils
from utils.caching import memoize


@memoize
def tsne_embedding(
    legend: str = "user_type_name",
    user_type_selected: list[str] = [],
    topic_selected: list[str] = [],
) -> go.Figure:
    """Visualize the TSNE embedding.

    Parameters
    ----------
    legend : str, optional
        What should be used to color points - either "user_type_name" or
        "dominant_topic", by default "user_type_name".
    user_type_selected : list[str], optional
        User types to filter to, by default [] (i.e. no filtering).
    topic_selected : list[str], optional
        Topics to filter to, by default [] (i.e. no filtering).

    Returns
    -------
    go.Figure
        Figure.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)
    tsne = utils.load_tsne()

    dominant_topics = utils.document_dominant_topic(H_norm)
    cdf_embedded = pd.concat((dominant_topics, tsne, texts), axis=1).copy()
    cdf_embedded["user_type_name"] = cdf_embedded["user_type"].map(
        utils.user_type_to_name
    )

    range_x = (cdf_embedded["PC1"].agg(["min", "max"]) + [-10, 10]).to_list()
    range_y = (cdf_embedded["PC2"].agg(["min", "max"]) + [-10, 10]).to_list()

    max_legend = min(
        max(
            cdf_embedded["user_type_name"].str.len().max(),
            cdf_embedded["dominant_topic"].str.len().max(),
        ),
        26,
    )

    fix_length = partial(utils.cutoff_text, max_length=max_legend)
    cdf_embedded["user_type_name_"] = cdf_embedded["user_type_name"].map(fix_length)
    cdf_embedded["dominant_topic_"] = cdf_embedded["dominant_topic"].map(fix_length)
    cdf_embedded["value_"] = cdf_embedded["value"].map(
        lambda x: f"{round(100 * x, 1)}%"
    )

    if len(user_type_selected) > 0:
        cdf_embedded = cdf_embedded.query("user_type_name_.isin(@user_type_selected)")
    if len(topic_selected) > 0:
        cdf_embedded = cdf_embedded.query("dominant_topic_.isin(@topic_selected)")

    fig = px.scatter(
        cdf_embedded,
        x="PC1",
        y="PC2",
        color=legend + "_",
        opacity=cdf_embedded["value"],
        range_x=range_x,
        range_y=range_y,
        custom_data=["id", "organization", "dominant_topic", "value_"],
        labels={"user_type_name_": "User type", "dominant_topic_": "Dominant topic"},
    )

    fig.update_traces(
        hovertemplate="""Feedback: %{customdata[1]} (%{customdata[0]})<br>
Dominant topic: %{customdata[2]} (%{customdata[3]})<br>
Projection: (%{x:.1f}, %{y:.1f})"""
    )
    layout = {
        "xaxis_visible": False,
        "yaxis_visible": False,
        "legend_yanchor": "top",
        "legend_font_family": "Hack",
    }
    fig.update_layout(COMMON_PLOT_LAYOUT | layout)
    return fig


@memoize
def tm_dendrogram() -> go.Figure:
    """Compute a dendrogram based on a normalized term-topic matrix W and average
    linkage hierarchical clustering.

    Returns
    -------
    go.Figure
        Dendrogram.
    """
    W_norm = utils.normalize_W(utils.load_tm()[1])
    W_arr = W_norm.drop(columns="lemma").values.T

    fig = ff.create_dendrogram(
        W_arr,
        labels=W_norm.columns[1:],
        orientation="left",
        linkagefun=partial(linkage, method="average"),  # type: ignore
        colorscale=["white"] * 8,  # Color does not work properly anyways, so let's
        color_threshold=10,  # just fix it to white
    )
    layout = dict(autosize=True, xaxis_mirror=False, yaxis_mirror=False)
    fig.update_layout(COMMON_PLOT_LAYOUT | layout)
    return fig


@memoize
def topics_popularity() -> list[go.Figure]:
    """Barplot of how often a topic is dominant or relevant for a document.

    Returns
    -------
    list[go.Figure]
        List of two barplots, one for dominant and one for relevant.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)

    df_dominant, df_relevant = topics.popularity(H_norm, texts)
    for df in [df_dominant, df_relevant]:
        df["topic_share"] *= 100
        df["user_type_topic_share"] *= 100

    fig_dominant, fig_relevant = (
        px.bar(
            df,
            x="n_docs",
            y="topic",
            color="user_type_name",
            labels={"n_docs": "", "topic": "", "user_type_name": "User type"},
            custom_data=[
                "user_type_name",
                "topic_total",
                "topic_share",
                "user_type_topic_share",
            ],
        )
        for df in [df_dominant, df_relevant]
    )
    fig_dominant_2, fig_relevant_2 = (
        px.scatter(
            df.drop_duplicates(subset=["topic"]),
            x="topic_share",
            y="topic",
            opacity=0,
            custom_data=["topic_total"],
        )
        for df in [df_dominant, df_relevant]
    )

    output = []
    for fig, fig_2 in [(fig_dominant, fig_dominant_2), (fig_relevant, fig_relevant_2)]:
        fig.update_traces(
            hovertemplate=(
                "Topic: %{y}<br>User type: %{customdata[0]}<br>"
                "#docs: %{x} (%{customdata[3]:.1f}%)<br>"
                "Topic total: %{customdata[1]} (%{customdata[2]:.1f}%)"
                "<extra></extra>"
            ),
        )
        fig_2.update_traces(
            hovertemplate=("Topic: %{y}<br>Topic total: %{customdata[0]} (%{x:.1f}%)"),
        )

        layout = go.Layout(
            xaxis=XAxis(),
            xaxis2=XAxis(overlaying="x", side="top"),
            yaxis=dict(),
        )
        combined_fig = go.Figure(layout=layout)
        combined_fig.add_traces(fig.data + fig_2.update_traces(xaxis="x2").data)  # type: ignore # noqa[E501]

        fig_layout = {
            "barmode": "stack",
            "yaxis_categoryorder": "total ascending",
            "xaxis1_showgrid": False,
            "xaxis2_ticksuffix": "%",
            "xaxis2_showgrid": False,
        }
        combined_fig.update_layout(COMMON_PLOT_LAYOUT | fig_layout)

        output.append(combined_fig)

    return output


@memoize
def user_types_to_topics(variant: str = "share") -> list[go.Figure]:
    """Barplots of how common topics are for each user type (based on average share and
    for which percentage of documents each topic is dominant).

    Parameters
    ----------
    variant : str, optional
        Whether to use the average topic share or for which percentage a topic is
        dominant, by default "share"

    Returns
    -------
    list[go.Figure]
        List of barplots, one for each user type.
    """
    H, _, texts = utils.load_tm()
    H_norm = utils.normalize_H(H)

    if variant == "share":
        df = pd.merge(texts[["id", "organization", "country", "user_type"]], H_norm)
        df = (
            df.groupby("user_type")
            .agg({t: "mean" for t in utils.get_topics(H_norm)})
            .reset_index("user_type")
        )
        df = df.melt(id_vars="user_type", var_name="topic", value_name="share")

    elif variant == "dominant":
        df = topics.popularity(H_norm, texts)[0][["user_type", "topic", "n_docs"]]
        user_total = df.groupby("user_type")["n_docs"].transform("sum")
        df["share"] = df["n_docs"] / user_total

        fake_df = pd.DataFrame(
            it.product(texts["user_type"].unique(), utils.get_topics(H_norm)),
            columns=["user_type", "topic"],
        )
        fake_df["share"] = 0
        df = pd.concat((df, fake_df)).drop_duplicates(keep="first")

    else:
        raise ValueError(
            f"Argument 'variant' has to be 'share' or 'dominant'. You used '{variant}'"
        )

    df["topic_abbr"] = df["topic"].map(partial(utils.cutoff_text, max_length=25))
    df["interest"] = df["user_type"].map(utils.user_type_to_interest)

    plots = []
    topic_order = list(
        map(partial(utils.cutoff_text, max_length=25), utils.get_topics(H_norm))
    )
    for interest in ["corporate", "public", "other"]:
        df_interest = df.query("interest == @interest")
        for j, user_type in enumerate(df_interest["user_type"].unique()):
            df_user = df_interest.query("user_type == @user_type")
            fig = px.bar(
                df_user,
                x="topic_abbr",
                y="share",
                barmode="group",
                labels=dict(share="", topic_abbr=""),
                custom_data=["topic", df_user["share"] * 100],
                title=utils.user_type_to_name(user_type),
                category_orders=dict(topic_abbr=topic_order),
            )
            layout = {
                "xaxis_tickangle": 35,
                "margin_t": 25,
                "yaxis_tickformat": ".0%",
            }
            fig.update_layout(COMMON_PLOT_LAYOUT | layout)

            tmp = (
                "Topic: %{customdata[0]}<br>Avg. share: %{customdata[1]:.1f}%"
                if variant == "share"
                else "Topic: %{customdata[0]}<br>Share: %{customdata[1]:.1f}%"
            )
            fig.update_traces(hovertemplate=tmp)
            plots.append(fig)
    return plots


@memoize
def rand_index_bar(R: float) -> go.Figure:
    """Barplot visualizing how large a rand index is.

    Parameters
    ----------
    R : float
        Rand index, i.e. value in [0, 1].

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    df = pd.DataFrame([dict(x=R, y="foo")])
    fig = px.bar(
        df,
        x="x",
        y="y",
        range_x=(0, 1.01),
        labels=dict(x="", y=""),
        range_y=(-0.41, 0.41),
    )
    fig.update_traces(
        hovertemplate=(
            "How is this metric computed?<br>"
            "1. View topic model as clustering of documents (by dominant topic).<br>"
            "2. Compute the Rank index between the two clusterings,<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "this is a measure between 0 (completely different) and 1 (the same)."
        )
    )
    fig.update_layout(
        COMMON_PLOT_LAYOUT | dict(yaxis_visible=False) | {"height": 80, "width": 600}
    )
    fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    fig.add_annotation(
        x=R / 2,
        y=0,
        text=f"Rand index: {round(R, 2)}" if R > 0 else "",
        showarrow=False,
        font=dict(size=20, color="white"),
    )
    return fig


@memoize
def tm_topics_heatmap(
    main: str, other: str, type_hellinger: bool, text_auto: bool | str = False
) -> go.Figure:
    """Draw a heatmap of topic similarity between two topic models based on Hellinger
    distance or top 20 terms overlap.

    Parameters
    ----------
    main : str
        File name of the active topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    other : str
        File name of the compared topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    type_hellinger : bool
        Whether to use Hellinger distance (True) or top 20 terms overlap (False).
    text_auto : bool | str, optional
        text_auto parameter of the plotly.express.imshow method. See its documentation
        for details, by default False.


    Returns
    -------
    go.Figure
        Heatmap figure.
    """
    if type_hellinger:
        return tm_topics_heatmap_hellinger(main, other, text_auto=text_auto)
    else:
        return tm_topics_heatmap_terms(main, other, text_auto=text_auto)


@memoize
def tm_topics_heatmap_hellinger(
    main: str, other: str, text_auto: bool | str = False
) -> go.Figure:
    """Draw a heatmap of Hellinger distances between topics.

    Parameters
    ----------
    main : str
        File name of the active topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    other : str
        File name of the compared topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    text_auto : bool | str, optional
        text_auto parameter of the plotly.express.imshow method. See its documentation
        for details, by default False.


    Returns
    -------
    go.Figure
        Heatmap figure.
    """
    W_norm_main, W_norm_other = _get_Ws(main, other)

    df = utils.tm_similarity_matrix_df(W_norm_main, W_norm_other)
    return _tm_topics_heatmap(df, text_auto=text_auto)


@memoize
def tm_topics_heatmap_terms(
    main: str,
    other: str,
    n_terms: int = 20,
    normalize: str = "n_terms",
    text_auto: bool | str = False,
) -> go.Figure:
    """Draw a heatmap of similarities based on degree of term overlap between topics.

    Parameters
    ----------
    main : str
        File name of the active topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    other : str
        File name of the compared topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    n_terms : int, optional
        The n_terms terms with highest topic-specific probability will be used.
    normalize : str, optional
        How to normalize the similarities to [0, 1]. The default, "n_terms", divides by
        maximum number of overlapping terms, while "total_terms" divides by the number
        of terms in either topic.
    text_auto : bool | str, optional
        text_auto parameter of the plotly.express.imshow method. See its documentation
        for details, by default False.

    Returns
    -------
    go.Figure
        Heatmap figure.
    """
    W_norm_main, W_norm_other = _get_Ws(main, other)

    df = utils.tm_comparison_terms(W_norm_main, W_norm_other, n_terms, normalize)

    # Plot similarity as heatmap
    df = df.pivot(index="topic_x", columns="topic_y", values="similarity").loc[
        utils.get_topics(W_norm_main), utils.get_topics(W_norm_other)
    ]
    return _tm_topics_heatmap(df, text_auto=text_auto)


def _get_Ws(main: str, other: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize term-topic matrices.

    Parameters
    ----------
    main : str
        File name of the active topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.
    other : str
        File name of the compared topic model's .indepth file without the file extension
        and relative to the data folder, by default DEFAULT_TM.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Normalized W for main and other topic model.
    """
    _, W_main, _ = utils.load_tm(main)
    _, W_other, _ = utils.load_tm(other)

    return utils.normalize_W(W_main), utils.normalize_W(W_other)


def _tm_topics_heatmap(df: pd.DataFrame, text_auto: bool | str = False) -> go.Figure:
    """Draw a heatmap based on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of similarities. Values represent similarity values with the index
        being the active topic model's topic labels and column names being the compared
        topic model's topic labels.
    text_auto : bool | str, optional
        text_auto parameter of the plotly.express.imshow method. See its documentation
        for details, by default False.


    Returns
    -------
    go.Figure
        Heatmap figure.
    """
    fig = px.imshow(df, zmin=0, text_auto=text_auto)  # type: ignore [Plotly isn't properly type hinted...] # noqa[E401]

    layout = dict(
        xaxis_tickmode="linear",
        yaxis_tickmode="linear",
        height=1000,
        width=1200,
        xaxis_title="Topic model selected for comparison",
        yaxis_title="Active topic model",
        legend_xanchor="right",
        legend_x=-2,
        hoverlabel_font_family="Hack",
    )
    fig.update_layout(COMMON_PLOT_LAYOUT | layout)
    fig.update_xaxes(tickmode="linear")
    fig.update_traces(
        hovertemplate=(
            "Active TM: "
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "%{y}<br>"
            "Comparison TM: "
            "%{x}<br>"
            "Similarity: "
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "%{z:.2f}"
            "<extra></extra>"
        )
    )

    fig.add_annotation(
        text="⇦	1 is the maximum similarity possible",
        xref="paper",
        yref="paper",
        x=1.037,
        y=0.988,
        showarrow=False,
        textangle=90,
    )
    fig.add_annotation(
        text="0 is the minimum similarity possible ⇨",
        xref="paper",
        yref="paper",
        x=1.037,
        y=0.012,
        showarrow=False,
        textangle=90,
    )
    return fig


def confusion_matrix(
    df: pd.DataFrame, labels: list[str], gold: str = "gold", pred: str = "pred"
) -> go.Figure:
    """Visualize a confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with one row for each observation, has to contain columns with the
        names specified in gold and pred arguments.
    labels : list[str]
        Possible sentiment labels and in the correct order.
    gold : str, optional
        Name of the column containing the gold sentiment labels, by default "gold".
    pred : str, optional
        Name of the column containing the sentiment predictions, by default "pred".

    Returns
    -------
    go.Figure
        Confusion matrix figure.
    """
    tmp = pd.DataFrame(
        sklearn_confusion_matrix(df[gold], df[pred], labels=labels),
        columns=labels,
        index=labels,
    )

    fig = px.imshow(tmp, zmin=0, text_auto=True, width=300, height=300)
    layout = dict(
        xaxis_title="Predicted label",
        yaxis_title="Gold label",
        xaxis_title_font_size=17,
        yaxis_title_font_size=17,
        xaxis_tickfont_size=15,
        yaxis_tickfont_size=15,
        font_size=15,
        coloraxis_showscale=False,
    )
    fig.update_layout(COMMON_PLOT_LAYOUT | layout)
    return fig


def format_axis_percent(fig: go.Figure, axis: str = "x") -> None:
    """Format given figure axis as percentage.

    Parameters
    ----------
    fig : go.Figure
        Arbitrary plotly figure.
    axis : str, optional
        Axis to format, by default "x".
    """
    axis = fig.layout.xaxis if axis == "x" else fig.layout.yaxis
    axis.tickformat = ",.1%"  # type: ignore


@memoize
def dummy_fig(text: str = "Loading") -> go.Figure:
    """Make an empty figure containing only text as centered annotation.

    Parameters
    ----------
    text : str, optional
        Arbitrary string, by default "Loading".

    Returns
    -------
    go.Figure
        Empty figure containing only text.
    """
    fig = px.imshow(np.array([[]]))
    fig.add_annotation(x=0, y=0, text=text, showarrow=False)
    layout = dict(
        xaxis_visible=False,
        yaxis_visible=False,
        xaxis_range=(-0.01, 0.01),
        yaxis_range=(-0.01, 0.01),
    )
    fig.update_layout(COMMON_PLOT_LAYOUT | layout)
    return fig


def export_fig(
    fig: go.Figure,
    intermediate_svg: bool = False,
    show_x_grid: bool = False,
    show_y_grid: bool = False,
    fig_name: str = "fig",
) -> None:
    """Export plotly figure to pdf.

    Parameters
    ----------
    fig : go.Figure
        Arbitrary plotly figure.
    intermediate_svg : bool, optional
        Whether to save to an intermediate svg and convert it to pdf using inkscape (has
        to be installed locally), by default False.
    show_x_grid : bool, optional
        Whether to a grid for the x axis, by default False.
    show_y_grid : bool, optional
        Whether to a grid for the y axis, by default False.
    fig_name : str, optional
        Name of the figure, used like f"06_explorer/images/{fig_name}.pdf". By default
        "fig".
    """
    from pathlib import Path
    import os, copy, subprocess  # noqa[E401]

    file_path = Path(__file__).parents[1] / "images/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    svg_file_name = file_path / f"{fig_name}.svg"
    pdf_file_name = file_path / f"{fig_name}.pdf"

    if not intermediate_svg:
        fig2 = dummy_fig("dummy")
        fig2.write_image(pdf_file_name)

    fig = copy.deepcopy(fig)
    fig.update_layout(EXTERNAL_LAYOUT)
    if show_x_grid:
        fig.update_layout({"xaxis_gridcolor": "gray"})
    if show_y_grid:
        fig.update_layout({"yaxis_gridcolor": "gray"})

    if intermediate_svg:
        fig.write_image(svg_file_name)
        subprocess.check_output(
            f"inkscape {svg_file_name} --export-area-drawing --batch-process "
            f"--export-type=pdf --export-filename={pdf_file_name} 2>&1",
            shell=True,
        )
        os.remove(svg_file_name)
    else:
        fig.write_image(pdf_file_name)


def display_fig(fig: go.Figure) -> None:
    """Display figure with whatever means are available.

    Changes figure to have a white background and black-colored axes and text.

    Parameters
    ----------
    fig : go.Figure
        Arbitrary figure.
    """
    import copy

    fig = copy.deepcopy(fig)
    fig.update_layout(EXTERNAL_LAYOUT | dict(paper_bgcolor="white"))
    fig.show()


COMMON_PLOT_LAYOUT = {
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    "margin_l": 0,
    "margin_r": 0,
    "margin_t": 0,
    "margin_b": 0,
    "xaxis_color": "white",
    "yaxis_color": "white",
    "legend_font_color": "white",
    "font_color": "white",
}

EXTERNAL_LAYOUT = dict(
    xaxis_color="black",
    yaxis_color="black",
    legend_font_color="black",
    font_color="black",
)

GRAPH_CONF = dict(config=dict(displayModeBar=False), responsive=True)
GRAPH_CONF_UNRESPONSIVE = dict(config=dict(displayModeBar=False))

SENTIMENT_COLOR_MAP = {
    "positive": "forestgreen",
    "neutral": "deepskyblue",
    "negative": "orangered",
}
INTEREST_MARKER_MAP = dict(corporate="/", other="", public="\\")
