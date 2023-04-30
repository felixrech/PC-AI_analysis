"""Module containing graphs specific to the /aspects/ page."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graphs import graphs

from utils.caching import memoize
from utils import utils, sentiment


@memoize
def n_mentions_area_plot(
    normalize: bool = False,
    filter_: bool = True,
    aspect: str = "article",
    area_plot: bool = True,
    annotate_total: bool = False,
    annotate_all: bool = False,
) -> go.Figure:
    """Area plot about the number of mentions for each aspect.

    Parameters
    ----------
    normalize : bool, optional
        Whether to normalize for each aspect, by default False.
    filter_ : bool, optional
        Whether to filter to aspects with at least 30 mentions (matching the sentiment
        area plot), by default True.
    aspect : str, optional
        How to filter, i.e. plot the number of mentions of "article"s, the "others"
        aspects, or "all", by default "article".
    area_plot : bool, optional
        Whether to use an area plot (the default) or a bar plot. The former is a nicer
        interactive visualization but the latter (combined with annotate=True) is better
        for printing purposes.
    annotate_total : bool, optional
        Whether or not to add bar total annotations, by default False.
    annotate_all : bool, optional
        Whether or not to add annotations for bar segments, by default False.

    Returns
    -------
    go.Figure
        Area plot figure.
    """
    df = sentiment.load_sentiment(aspect=aspect)
    tmp = (
        sentiment.aggregate_mentions_by_user(df)
        .reset_index()
        .assign(
            # Pretty x-axis labels
            aspect_nice=_get_aspect_nice,
            # Bar segments represent share, not (logarithmic) absolute numbers
            mentions_share_cum=lambda tmp: tmp.sort_values("user_type_interest")
            .groupby(["aspect_type", "aspect_subtype"])["mentions_share"]
            .transform("cumsum"),
            mentions_share_prev=lambda tmp: tmp.sort_values("user_type_interest")
            .groupby(["aspect_type", "aspect_subtype"])["mentions_share_cum"]
            .transform("shift")
            .fillna(0),
            segment_height=lambda tmp: tmp.eval(
                "10**(log10(total_mentions) * mentions_share_cum) "
                "- 10**(log10(total_mentions) * mentions_share_prev)"
            ),
        )
    )
    tmp = tmp if not filter_ else tmp.query("total_mentions >= 30")
    max_ = tmp["total_mentions"].max()
    y_ticks = [10 ** (i + 1) for i in range(int(np.floor(np.log10(max_))))]
    y_ticks += [3 * n for n in (y_ticks[:-1] if max_ < 2 * y_ticks[-1] else y_ticks)]

    f = px.area if area_plot else px.bar
    fig = f(
        tmp,
        x=(x := "aspect_subtype" if aspect == "article" else "aspect_nice"),
        y="segment_height" if not normalize else "mentions_share",
        color="user_type_interest",
        color_discrete_map=INTEREST_COLOR_MAP,
        labels={
            "n_mentions": "Number of mentions",
            "total_mentions": "Number of mentions from all user types",
            "segment_height": "Number of mentions (log scale)",
            "mentions_share": "Fraction of mentions",
            "aspect_subtype": "Article",
            "user_type_interest": "Interest type",
            "details": "Details",
            "aspect_nice": "",
        },
        hover_data=dict(
            aspect_subtype=False,
            aspect_nice=False,
            segment_height=False,
            n_mentions=True,
            details=True,
            total_mentions=True,
        ),
        **dict(text="n_mentions" if not normalize else "mentions_share")
        if annotate_all
        else dict(),
    )

    if annotate_all:
        fig.update_traces(textposition="inside", insidetextanchor="middle", textangle=0)
    if annotate_total:
        for _, row in tmp.drop_duplicates(
            subset=["aspect_type", "aspect_subtype", "total_mentions"]
        ).iterrows():
            fig.add_annotation(
                x=row[x],
                y=np.log10(row["total_mentions"])
                + (0.035 if area_plot else 0.03) * np.log10(max_)
                if not normalize
                else 1.04,
                text=row["total_mentions"],
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                opacity=0.75,
            )

    # Adjust layout further
    if area_plot:
        fig.update_traces(mode="markers+lines")
    if normalize:
        graphs.format_axis_percent(fig, axis="y")
    layout = dict(
        hovermode="x unified",
        hoverlabel_bgcolor="gray",
        xaxis_type="category",
        legend_traceorder="reversed",
        **dict(yaxis_type="log", yaxis_tickmode="array", yaxis_tickvals=y_ticks)
        if not normalize
        else dict(),
    )
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    return fig


@memoize
def sentiment_area_plot(
    aspect: str = "article",
    probs: bool = False,
    area_plot: bool = True,
    annotate: bool = False,
    separate_interest_types: bool = False,
) -> go.Figure:
    """Area plot that shows the sentiment for each aspect.

    Parameters
    ----------
    aspect : str, optional
        How to filter, i.e. plot the number of mentions of "article"s, the "others"
        aspects, or "all", by default "article".
    probs : bool
        Whether to use average predicted probabilities (True) or fraction of predicted
        sentiment (False).
    area_plot : bool, optional
        Whether to use an area plot (the default) or a bar plot. The former is a nicer
        interactive visualization but the latter (combined with annotate=True) is better
        for printing purposes.
    annotate : bool, optional
        Whether or not to add bar total annotations, by default False.
    separate_interest_types : bool
        Whether to separate sentiment into three interest types.

    Returns
    -------
    go.Figure
        Area plot figure.
    """
    # Filter to aspects with at least thirty mentions
    keys = ["aspect_type", "aspect_subtype"]
    tmp = (
        sentiment.load_sentiment(aspect=aspect)
        .copy()
        .assign(
            total_mentions=lambda df: df.groupby(keys)["sentiment"].transform("size"),
            interest_type=lambda df: df["user_type"].map(utils.user_type_to_interest),
        )
        .query("total_mentions >= 30")
    )

    # Aggregate sentiment using chosen method
    tmp = sentiment.aggregate_sentiments(
        tmp, probs=probs, separate_interest_types=separate_interest_types
    ).assign(
        aspect_nice=_get_aspect_nice,
        share_nice=lambda df: df["share"].map(lambda s: f"{s:.0%}"),
    )

    # Create the area plot
    y_label = "Average predicted probabilities" if probs else "Percentage of mentions"
    if not separate_interest_types:
        f = px.area if area_plot else px.bar
        fig = f(
            tmp,
            x="aspect_subtype" if aspect == "article" else "aspect_nice",
            y="share",
            color="sentiment",
            labels={
                "aspect_subtype": "Article",
                "share": y_label,
                "sentiment": "Sentiment",
                "aspect_nice": "",
            },
            hover_data={"sentiment": True, "share": ":.0%", "aspect_subtype": False},
            color_discrete_map=graphs.SENTIMENT_COLOR_MAP,
            category_orders={"sentiment": ["negative", "neutral", "positive"]},
            **dict() if area_plot or not annotate else dict(text="share_nice"),
        )
    else:
        fig = go.Figure(
            layout=go.Layout(
                height=450,
                width=1200,
                barmode="relative",
                yaxis2=go.layout.YAxis(
                    visible=False,
                    matches="y",
                    overlaying="y",
                    anchor="x",
                ),
                yaxis3=go.layout.YAxis(
                    visible=False,
                    matches="y",
                    overlaying="y",
                    anchor="x",
                ),
                margin=dict(b=0, t=10, l=0, r=10),
            )
        )
        for i, interest_type in enumerate(["corporate", "other", "public"]):
            for sentiment_ in ["negative", "neutral", "positive"]:
                df = tmp.query(
                    "interest_type == @interest_type and sentiment == @sentiment_"
                )
                fig.add_bar(
                    x=df[
                        x := "aspect_subtype" if aspect == "article" else "aspect_nice"
                    ],
                    y=df["share"],
                    yaxis=f"y{i+1}",
                    offsetgroup=str(i),
                    offset=(i - 1) * 0.25,
                    width=0.25,
                    name=f"{interest_type}, {sentiment_}",
                    legendgroup="Legend",
                    legendgrouptitle_text="Legend",
                    marker_color=graphs.SENTIMENT_COLOR_MAP[sentiment_],
                    marker_pattern_shape=graphs.INTEREST_MARKER_MAP[interest_type],
                    marker_line=dict(width=1.5),
                    **dict(text=df["share_nice"]) if annotate else dict(),
                )
                for _, row in df.query("support < 30").iterrows():
                    fig.add_annotation(
                        x=row[x],
                        y=1.015,
                        xshift=(i - 1) * 10 + 5,
                        text="⚠️",
                        showarrow=False,
                        font=dict(size=5),
                    )
        if annotate:
            fig.update_traces(
                textposition="inside",
                insidetextanchor="middle",
                textangle=90,
                textfont=dict(color="white"),
            )

    # Adjust layout further
    if area_plot:
        fig.update_traces(mode="markers+lines")
    fig.update_traces(
        hovertemplate=(
            f"{'Article %{x}' if aspect == 'article' else 'Aspect: %{x}'}<br>"
            f"{y_label}: %{{y:.0%}}"
        ),
    )
    layout = {
        "hovermode": "x",
        "xaxis_type": "category",
        "legend_traceorder": "reversed",
    }
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    graphs.format_axis_percent(fig, "y")
    return fig


@memoize
def aspect_interest_share_barplot(annotate: bool = False) -> go.Figure:
    """Create a barplot of the share of sentences containing mentions of articles or
    annexes among all sentences, with one bar for each interest type.

    Parameters
    ----------
    annotate : bool, optional
        Whether to annotate using exact fractions, by default False.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    df = sentiment.aspect_mentions_share()

    fig = px.bar(
        tmp := df.drop_duplicates("interest_type").sort_values("interest_type"),
        x="interest_share",
        y="interest_type",
        labels=dict(
            interest_type="Interest type",
            interest_share="Share of sentences mentioning an article or annex",
        ),
        hover_data=dict(interest_type=False, interest_share=":.1%"),
        color="interest_type",
        color_discrete_map=INTEREST_COLOR_MAP,
        text_auto=annotate,
        range_x=[
            tmp["interest_share"].min()
            - (tmp["interest_share"].max() - tmp["interest_share"].min()),
            tmp["interest_share"].max()
            + 0.01 * (tmp["interest_share"].max() - tmp["interest_share"].min()),
        ],
    )

    layout = dict(yaxis_autorange="reversed", yaxis_ticksuffix="  ", showlegend=False)
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    graphs.format_axis_percent(fig)
    if annotate:
        fig.update_traces(textposition="inside", insidetextanchor="end", textangle=0)
    return fig


@memoize
def aspect_user_difference_barplot(
    user_type_name: str,
    variant: str = "sentiment",
    include_title: bool = True,
    annotate: bool = False,
) -> go.Figure:
    """Compute barplot that shows the difference between a user type's
    mentions/sentiment and overall number of mentions/sentiment percentages.

    Caution: only aspect with 30 mentions of any user type and 15 mentions by selected
    user type are displayed.

    Parameters
    ----------
    user_type_name : str
        Name of a user type, e.g. "Business Association".
    variant : str, optional
        Whether to visualize difference to fraction of sentiments or relative difference
        from expected number of mentions, either "n_mentions" or "sentiment", by default
        "sentiment".
    include_title : bool, optional
        Whether to include a title in the figure, by default True
    annotate : bool, optional
        Whether to annotate graph with numbers, e.g. "+38%".

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    # Raise error if argument value invalid
    if variant not in ["sentiment", "n_mentions"]:
        raise ValueError(f'Value of argument variant ("{variant}") unknown!')

    # Get the data
    df = sentiment.aspect_sentiment_user_difference(user_type_name=user_type_name)
    if not (v := variant == "sentiment"):
        df = df.drop_duplicates(subset=["aspect_type", "aspect_subtype"])

    # Prepare some aspects of the figure
    if include_title and variant == "sentiment":
        title = dict(title="Sentiment")
    elif include_title and variant == "n_mentions":
        title = dict(title="Number of mentions")
    else:
        title = dict()
    hover_data: dict[str, str | bool] = (
        dict(aspect=False, total_user=True)
        if v
        else dict(aspect=False, total_user=True, expected_total=":.1f")
    )

    # Bar annotations (e.g. "+10%" or "-3%")
    y = "share_diff" if v else "expected_rel_diff"
    df["annotation"] = df[y].map(lambda x: f"{'+' if x > 0 else ''}{x:.0%}")

    # Create figure
    fig = px.bar(
        df,
        x="aspect",
        y=y,
        **dict(color="sentiment") if v else dict(),
        barmode="group",
        labels=dict(
            sentiment="Sentiment",
            aspect="",
            share_diff="Difference to overall average",
            total_user="Support (# mentions by user)" if v else "Number of mentions",
            expected_total="Expected number of mentions",
            expected_rel_diff="Relative deviation from expectation",
        ),
        color_discrete_map=graphs.SENTIMENT_COLOR_MAP,
        hover_data=hover_data,
        **title,
        **dict(text="annotation") if annotate else dict(),
    )
    graphs.format_axis_percent(fig, axis="y")
    layout = dict(margin_t=30) if include_title else dict()
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    return fig


@memoize
def interest_readability_barplot(annotate: bool = False) -> go.Figure:
    """Create a barplot of median Dale-Chall readability scores for each interest type.

    Parameters
    ----------
    annotate : bool, optional
        Whether to annotate each bar with the exact score, by default False.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    df = sentiment.get_readability()

    fig = px.bar(
        tmp := df.query("score == 'dale_chall'"),
        y="interest_type",
        x="median",
        labels=dict(
            interest_type="Interest type", median="Median Dale-Chall readability score"
        ),
        hover_data=dict(interest_type=False, median=":.2f"),
        color="interest_type",
        color_discrete_map=INTEREST_COLOR_MAP,
        text_auto=annotate,
        range_x=[
            tmp["median"].min() - (tmp["median"].max() - tmp["median"].min()),
            tmp["median"].max() + 0.01 * (tmp["median"].max() - tmp["median"].min()),
        ],
    )

    layout = dict(yaxis_autorange="reversed", yaxis_ticksuffix="  ", showlegend=False)
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    if annotate:
        fig.update_traces(textposition="inside", insidetextanchor="end", textangle=0)
    return fig


def _get_aspect_nice(df: pd.DataFrame) -> pd.Series:
    """Apply sentiment.aspect_nice_format_series to the correct columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with "aspect_type" and "aspect_subtype" columns.

    Returns
    -------
    pd.Series
        Series of nice aspect texts.
    """
    return df[["aspect_type", "aspect_subtype"]].apply(
        sentiment.aspect_nice_format_series, axis="columns"
    )


INTEREST_COLOR_MAP = dict(
    public="aqua",
    corporate="deepskyblue",
    other="dodgerblue",
)
