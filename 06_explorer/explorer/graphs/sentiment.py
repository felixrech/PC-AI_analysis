"""Module containing graphs specific to the /sentiment/, /aspects/, and /search/
pages."""

import pandas as pd
import itertools as it
import plotly.express as px
import plotly.graph_objects as go

from graphs import graphs
from utils.caching import memoize
from utils import utils, sentiment


@memoize
def sentiment_overview_pie_chart(aspect_type: str, aspect_subtype: str) -> go.Figure:
    """Pie chart giving an overview of the sentiments in the sentiment dataset.

    Parameters
    ----------
    aspect_type : str
        General part of the aspect, e.g. "article".
    aspect_subtype : str
        Specific part of the aspect, e.g. "5".

    Returns
    -------
    go.Figure
        Pie chart figure.
    """
    df = sentiment.load_sentiment(
        aspect_type=aspect_type, aspect_subtype=aspect_subtype
    )

    df = (
        df["sentiment"]
        .value_counts()
        .to_frame("count")
        .reset_index(names="sentiment")
        .sort_values("sentiment")
    )
    df["color"] = df["sentiment"].map(graphs.SENTIMENT_COLOR_MAP)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["sentiment"],
                values=df["count"],
                sort=False,
                hole=0.33,
                marker_colors=df["color"],
                direction="clockwise",
            )
        ]
    )

    fig.update_traces(
        marker=dict(colors=df["color"]),
        textposition="inside",
        textinfo="label+percent",
        showlegend=False,
    )
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT)

    n_mentions = df["count"].sum()
    fig.update_layout(
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                text=f"<b>NUMBER<br>OF MENTIONS:<br>{n_mentions}</b>",
                font=dict(family="Arial", size=12),
                showarrow=False,
            )
        ]
    )
    return fig


@memoize
def sentiment_overview_barplot(
    aspect_type: str = "",
    aspect_subtype: str = "",
    normalized: bool = False,
    probs: bool = False,
    limit_pos_neg: bool = False,
) -> go.Figure:
    """Plot a barplot of sentiments for each user type.

    Parameters
    ----------
    aspect_type : str, optional
        General part of the aspect, e.g. "aspect" - by default "".
    aspect_subtype : str, optional
        Specific part of the aspect, e.g. "5" - by default "".
    normalized : bool, optional
        Normalize each bar's segments by dividing by the bar's total length, by default
        False.
    probs : bool, optional
        Instead of using the number of or percentage of dominant sentiments, use average
        sentiment probabilities, by default False.
    limit_pos_neg : bool, optional
        Limit to positive and negative sentiments, by default False.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    df = sentiment.load_sentiment(
        aspect_type=aspect_type, aspect_subtype=aspect_subtype
    )

    if not probs:
        df = (
            df.groupby(["user_type", "sentiment"])
            .size()
            .reset_index()
            .rename(columns={0: "share"})
            .assign(support=lambda df: df["share"])
        )
    else:
        df = (
            df.groupby(["user_type"])
            .agg(
                {
                    "positive": "mean",
                    "neutral": "mean",
                    "negative": "mean",
                    "sentiment": "size",
                }
            )
            .reset_index()
            .rename(columns={"sentiment": "support"})
            .melt(
                id_vars=["user_type", "support"],
                var_name="sentiment",
                value_name="share",
            )
        )

    defaults = it.product(
        [
            "academic_research_institution",
            "business_association",
            "company",
            "consumer_organisation",
            "eu_citizen",
            "ngo",
            "public_authority",
            "standardizing_body",
            "trade_union",
        ],
        ["negative", "neutral", "positive"],
    )
    defaults_df = pd.DataFrame(defaults, columns=["user_type", "sentiment"]).assign(
        share=0
    )

    df = (
        pd.concat((df, defaults_df))
        .drop_duplicates(["user_type", "sentiment"])
        .sort_values("sentiment", ascending=False)
    )

    if limit_pos_neg:
        df = df.query("sentiment.isin(['positive', 'negative'])").copy()
    if not probs:
        df["support"] = df.groupby("user_type")["support"].transform("sum")

    df["user_type_name"] = df["user_type"].map(utils.user_type_to_name)
    df["user_type_total"] = df.groupby("user_type")["share"].transform("sum")
    df["share_normalized"] = df["share"] / df["user_type_total"]

    fig = px.bar(
        df,
        y="user_type_name",
        x=(share_col := "share" if not normalized else "share_normalized"),
        color="sentiment",
        color_discrete_map={
            "positive": "forestgreen",
            "neutral": "deepskyblue",
            "negative": "orangered",
        },
        category_orders=dict(y=utils.user_type_order),
        labels=dict(user_type_name="", share="", share_normalized=""),
    )

    if probs or normalized:
        max_ = df.groupby("user_type")[share_col].sum()
        for _, row in (
            df.sort_values("support")
            .drop_duplicates(["user_type"])
            .dropna()
            .query("support < 30")
            .iterrows()
        ):
            fig.add_annotation(
                y=row["user_type_name"],
                x=max_[row["user_type"]] + 0.025 * max_.max(),
                text=f"⚠️<br>(n={row['support']:.0f})",
                showarrow=False,
            )

    for user_type in (
        df.groupby("user_type_name")["support"]
        .max()
        .to_frame("no_mentions")
        .query("no_mentions.isnull() or no_mentions == 0")
        .index
    ):
        fig.add_annotation(
            y=user_type,
            x=0.5 if normalized or probs else 0.1,
            xref="paper",
            text="🛑 no mentions"
            + (" of positive or negative sentiment" if limit_pos_neg and probs else ""),
            showarrow=False,
            bgcolor="#303030",
        )

    fig.update_traces(hovertemplate="%{x}")
    layout = dict(
        showlegend=False,
        yaxis_categoryorder="array",
        yaxis_categoryarray=utils.user_type_order[::-1],
        **dict(xaxis_tickformat=".0%") if normalized or probs else dict(),
    )
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    return fig


@memoize
def sentiment_topics(
    aspect_type: str | None = None,
    aspect_subtype: str | None = None,
    search_term: str | None = None,
    regex: bool = False,
    filter_user_types: list[str] = [],
    variant: str = "dominant",
) -> go.Figure:
    """Create a barplot of which topics the mentions of the selected aspect or search
    term results belong to.

    Parameters
    ----------
    aspect_type : str, optional
        General part of the aspect, e.g. "aspect" - by default None (unspecified).
        Either aspect_type and aspect_subtype or search_term needs to be specified. If
        both are specified, search_term is used.
    aspect_subtype : str, optional
        Specific part of the aspect, e.g. "5" - by default None (unspecified).
    search_term : str, optional
        Search term to use, e.g. "face" - by default None (unspecified). Either
        aspect_type and aspect_subtype or search_term needs to be specified. If both
        are specified, search_term is used.
    regex : bool, optional
        Whether the search term uses regular expressions, by default False.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.
    variant : str, optional
        Whether to use the fraction of mentions with X as the topic ("dominant") or the
        average topic share ("avg_share"), by default "dominant"

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    # Remove invalid argument values/combinations
    if variant not in ["dominant", "avg_share"]:
        raise ValueError("Please use either 'dominant' or 'avg_share' as the variant!")
    if (aspect_type is None or aspect_subtype is None) and search_term is None:
        raise ValueError(
            "Either aspect_type and aspect_subtype, or search_term arguments have to "
            "be specified!"
        )

    # Load the data we need
    df = sentiment.sentiment_topic_modeling(
        aspect_type=aspect_type,
        aspect_subtype=aspect_subtype,
        search_term=search_term,
        regex=regex,
        filter_user_types=filter_user_types,
    )
    topics = utils.get_topics(utils.load_tm()[0])

    # Fallback figure if there are no search results/mentions of that aspect
    if len(df) == 0:
        return graphs.dummy_fig("No mentions selected!")

    # Adapt data to selected type
    if variant == "avg_share":
        tmp = df[topics].agg("mean").to_frame("share").reset_index(names="topic")
    else:
        tmp = (
            df.groupby("dominant_topic")  # type: ignore[operator]
            .agg("size")
            .to_frame("count")
            .reset_index()
            .assign(share=lambda df: df["count"] / df["count"].sum())
            .rename(columns=dict(dominant_topic="topic"))
        )

    # Plot the figure
    share_label = (
        "Average topic share<br>"
        if variant == "avg_share"
        else "Percentage of sentences<br>for which topic is dominant"
    )
    fig = px.bar(
        tmp,
        x="topic",
        y="share",
        labels=dict(topic="", share=share_label),
        category_orders=dict(topic=topics),
    )
    graphs.format_axis_percent(fig, axis="y")
    fig.update_traces(hovertemplate="%{y:.1%}")
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | dict(xaxis_tickangle=35))
    return fig


@memoize
def search_aspects(
    search_term: str,
    regex: bool,
    filter_user_types: list[str] = [],
    annotate: bool = False,
) -> go.Figure:
    """Create a barplot of aspects among the search results.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex : bool
        Whether the search term uses regular expressions.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.
    annotate : bool, optional
        Whether to add bar totals as text annotation, by default False.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    results = sentiment.search_sentiment(
        search_term, regex=regex, filter_user_types=filter_user_types
    )
    if len(results) == 0:
        return graphs.dummy_fig("No mentions selected!")

    results = results.dropna(subset=["aspect_type"])
    if len(results) == 0:
        return graphs.dummy_fig("No aspects in selected mentions!")

    aspect = ["aspect_type", "aspect_subtype"]
    df = sentiment.sort_by_aspect(
        results.assign(
            aspect_nice=lambda df: df[aspect].apply(
                sentiment.aspect_nice_format_series, axis=1
            ),
            count=lambda df: df.groupby(aspect)["aspect_nice"].transform("count"),
            share=lambda df: df["count"] / results["sentence_index"].nunique(),
        ).drop_duplicates(subset=aspect)
    ).sort_values("count", kind="stable", ascending=False)

    fig = px.bar(
        df.iloc[:20],
        y="aspect_nice",
        x="count",
        labels=dict(
            aspect_nice="Aspect",
            count=(count := "Number of matches"),
            share="Share among matches",
        ),
        hover_data=dict(share=":.1%"),
        **(dict(text="count") if annotate else dict()),
    )

    disclaimer = (
        "" if len(df) <= 20 else "<br><i>(showing only the 20 most common aspects)</i>"
    )
    layout = dict(
        xaxis_title=count + disclaimer,
        yaxis_title=None,
        yaxis_autorange="reversed",
        yaxis_ticksuffix="   ",
    )
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    if annotate:
        fig.update_traces(textposition="outside", overwrite=True)
    return fig


@memoize
def search_user_types(
    search_term: str,
    regex: bool,
    filter_user_types: list[str] = [],
    relative: bool = False,
    annotate: bool = False,
) -> go.Figure:
    """Create a barplot of the user types among the search results.

    Parameters
    ----------
    search_term : str
        Search term to use, e.g. "face".
    regex : bool
        Whether the search term uses regular expressions.
    filter_user_types : list[str], optional
        Which user types ("user_type" column) to filter to, by default [], i.e. no
        filtering.
    relative : bool, optional
        Whether to use absolute numbers (False) or share of matches among all sentences
        by user type (True), by default False.
    annotate : bool, optional
        Whether to add bar totals as text annotation, by default False.

    Returns
    -------
    go.Figure
        Barplot figure.
    """
    results = sentiment.search_sentiment(
        search_term, regex=regex, filter_user_types=filter_user_types
    )
    if len(results) == 0:
        return graphs.dummy_fig("No mentions selected!")

    # Compute the number and share of user type
    df = (
        results["user_type"]
        .value_counts()
        .to_frame("count")
        .reset_index(names="user_type")
        .assign(
            user_type_name=lambda df: df["user_type"].map(utils.user_type_to_name),
            share=lambda df: df["count"] / results["sentence_index"].nunique(),
        )
    )

    # User type totals
    df = (
        sentiment.load_sentiment(filter_=False, filter_user_types=filter_user_types)[
            "user_type"
        ]
        .value_counts()
        .to_frame("total")
        .reset_index(names="user_type")
        .merge(df)
        .assign(total_share=lambda df: df["count"] / df["total"])
    )

    # Order in common user type order
    user_type_names = pd.Series(utils.user_type_order).to_frame("user_type_name")
    df = user_type_names.merge(df)

    # Create figure
    fig = px.bar(
        df,
        y="user_type_name",
        x="count" if not relative else "total_share",
        labels=dict(
            user_type_name="",
            count="Number of matches",
            share="Share among matches",
            total_share="Share of matches among all sentences by user type",
        ),
        hover_data=dict(
            share=":.1%" if not relative else False, user_type_name=False, count=True
        ),
        **(dict(text="count") if annotate else dict()),
    )

    # Adapt plot layout
    layout = dict(
        yaxis_title=None,
        yaxis_autorange="reversed",
        yaxis_ticksuffix="   ",
    )
    fig.update_layout(graphs.COMMON_PLOT_LAYOUT | layout)
    if relative:
        graphs.format_axis_percent(fig)
    if annotate:
        fig.update_traces(textposition="outside", overwrite=True)
    return fig
