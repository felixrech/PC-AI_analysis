"""Module containing some figures' code for the embedding_clustering.ipynb notebook.
Caution: Undocumented and without type hints (as it is unused experimental code)."""

import re
import textwrap
import plotly.graph_objects as go


def _get_color(sentiment):
    if sentiment == "positive":
        return "rgba(0, 204, 150, 1)"
    elif sentiment == "negative":
        return "rgba(239, 85, 59, 1)"

    return "rgba(99, 184, 249, 0.75)"


def _get_marker(row):
    s = (
        "circle;circle;circle;square;diamond;cross;x;triangle-up;triangle-down;"
        "triangle-left;triangle-right;triangle-ne;triangle-se;triangle-sw;triangle-nw;"
        "pentagon;hexagon;hexagon2;octagon;star;hexagram;star-triangle-down;"
        "star-triangle-up;star-square;star-diamond;diamond-tall"
    )
    initial = dict(positive=0, negative=2, neutral=1)[row["sentiment"]]
    return s.split(";")[initial::3][row["cluster"] + 1]


def _get_size(sentiment):
    if sentiment == "positive":
        return 9
    elif sentiment == "negative":
        return 9

    return 5


def _get_text_main(row):
    return "<br>".join(
        textwrap.wrap(
            re.sub(
                f"(.*)({row['aspect']})(.*)",
                "\\1<b>\\2</b>\\3",
                row["text"],
                0,
                re.DOTALL,
            )
        )
    )


def _get_text_source(row):
    source = "<br>".join(
        textwrap.wrap(
            f"by {row['organization']} ({row['user_type']})"
        )  # TODO: utils.?? for user_type
    )
    return (
        f"{row['positive']:.1%} positive, {row['negative']:.1%} negative<br>" + source
    )


def _get_text_clustering(row):
    return f"Cluster: {row['cluster']}"


def _prep_scatter_plot(df):
    return df.assign(
        fig_color=lambda df: df["sentiment"].map(_get_color),
        fig_marker=lambda df: df.apply(_get_marker, axis=1),
        fig_size=lambda df: df["sentiment"].map(_get_size),
        fig_text_main=lambda df: df.apply(_get_text_main, axis=1),
        fig_text_source=lambda df: df.apply(_get_text_source, axis=1),
        fig_text_extra=lambda df: df.apply(_get_text_clustering, axis=1),
        fig_text=lambda df: (
            (df["fig_text_main"] + "<br><br>")
            + (df["fig_text_source"] + "<br>")
            + df["fig_text_extra"]
        ),
    )


def embedding_scatter_plot(df):
    df = _prep_scatter_plot(df)

    fig = go.Figure(
        go.Scatter(
            x=df["PC1"],
            y=df["PC2"],
            text=df["fig_text"],
            hoverinfo="text",
            mode="markers",
            marker_color=df["fig_color"],
            marker_symbol=df["fig_marker"],
            marker_size=df["fig_size"],
        )
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(
        common_plot_layout
        | {
            "xaxis_visible": False,
            "yaxis_visible": False,
        }
    )
    return fig


common_plot_layout = {
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
