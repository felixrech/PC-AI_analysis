"""Module containing code for some plots to make the notebook less cluttered."""

import geopandas
import pandas as pd
from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utils


def plot_responses_gdp_scatter(gdp_df: pd.DataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of responses per capita against GDP per capita using a
    scatterplot.

    Parameters
    ----------
    gdp_df : pd.DataFrame
        Dataframe with country-aggregated responses. Needs to have "gdp_per_cap",
        "responses_per_ten_million", and "is_eu" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots()
    palette = {
        True: [0.03137255, 0.30191465, 0.58840446, 1.0],
        False: [0.51058824, 0.73230296, 0.85883891, 1.0],
    }
    g = sns.scatterplot(
        x="gdp_per_cap",
        y="responses_per_ten_million",
        hue="is_eu",
        palette=palette,
        data=gdp_df,
        ax=ax,
    )

    ax.set_xscale("log")
    ax.set_xticks([1e4, 5e4, 1e5])
    y_cutoff = 11
    ax.set_ylim((-1, y_cutoff))

    ax.set_xlabel("GDP per capita (current USD)")
    ax.set_ylabel("Number of responses\nper ten million citizens")
    ax.legend(title="EU Member", loc="upper left")

    for _, row in gdp_df.query("responses_per_ten_million > @y_cutoff").iterrows():
        ax.annotate(
            f"{row.country_name}, {round(row.responses_per_ten_million, 1)}",
            (row.gdp_per_cap, y_cutoff),
            xytext=(row.gdp_per_cap, y_cutoff - 0.85),
            ha="center",
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc"},
            color=palette[row.is_eu],
        )

    utils.theme_plot(g.axes)

    return fig, ax


def plot_responses_gdp(gdp_df: pd.DataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of responses per capita against GDP per capita using a
    scatterplot with each point being the .

    Parameters
    ----------
    gdp_df : pd.DataFrame
        Dataframe with country-aggregated responses. Needs to have "gdp_per_cap",
        "responses_per_ten_million", and "is_eu" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots()
    palette = {
        True: [0.03137255, 0.30191465, 0.58840446, 1.0],
        False: [0.51058824, 0.73230296, 0.85883891, 1.0],
    }
    y_cutoff = 11

    for _, row in gdp_df.query("responses_per_ten_million <= @y_cutoff").iterrows():
        plt.annotate(
            row.country,
            (row.gdp_per_cap, row.responses_per_ten_million),
            ha="center",
            va="center",
            fontsize="7.5",
            color=palette[row.is_eu],
        )

    plt.xscale("log")
    plt.xlim(gdp_df["gdp_per_cap"].min(), gdp_df["gdp_per_cap"].max() + 10000)
    plt.xticks([1e4, 5e4, 1e5], ["$10K", "$50K", "$100K"])
    plt.ylim((-1, y_cutoff))

    plt.xlabel("GDP per capita (current USD, log. axis)")
    plt.ylabel("Responses per 10M citizens")
    legend = plt.legend(
        title="EU Member",
        handles=[
            mpatches.Patch(color=palette[True], label="Yes"),
            mpatches.Patch(color=palette[False], label="No"),
        ],
        loc="upper left",
        labelcolor="#555555",
    )

    for _, row in gdp_df.query("responses_per_ten_million > @y_cutoff").iterrows():
        plt.annotate(
            f"{row.country_name}, {round(row.responses_per_ten_million, 1)}",
            (row.gdp_per_cap, y_cutoff),
            xytext=(row.gdp_per_cap, y_cutoff - 0.75),
            ha="center",
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc"},
            fontsize="7.5",
            color=palette[row.is_eu],
        )

    utils.theme_plot(ax)

    return fig, ax


def map_responses(countries: geopandas.GeoDataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of responses on a map of Europe (i.e. using a "choropleth map").

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe with country-aggregated number of responses. Needs to have "country",
        "n_responses", "continent", and "name" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots(1, figsize=(30, 10), constrained_layout=True)

    # Set up a colorbar
    sm = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=plt.Normalize(
            *utils._get_europe(countries)["n_responses"].agg(("min", "max"))
        ),
    )
    sm.set_array([])
    fig.colorbar(sm, shrink=0.7, anchor=(8.5, 0.9), ax=ax)

    # Plot the data to a map
    utils._get_europe(countries).plot(
        column="n_responses",
        cmap="Blues",
        linewidth=0.8,
        ax=ax,
        edgecolor=["0.7"] * utils._get_threshold(countries)
        + ["0"] * (len(utils._get_europe(countries)) - utils._get_threshold(countries)),
    )

    # Crop to utils._get_europe(countries), now axis labeling
    ax.axis("off")
    ax.set_xlim(-11.5, 35)
    ax.set_ylim(34, 71.5)

    # Add text marker for missing countries
    ax.annotate(
        utils._get_america(countries, column="n_responses"),
        xy=(-11, 46),
        fontsize=15,
        color="#555555",
    )
    ax.annotate(
        utils._get_asia(countries, column="n_responses"),
        xy=(30, 38),
        fontsize=15,
        color="#555555",
    )

    # Add numbers
    for _, row in utils._get_europe(countries).iterrows():
        if row["n_responses"] > 0:
            plt.annotate(
                row["n_responses"],
                xy=row["geometry"].representative_point().coords[0]
                if row["name"] != "Norway"
                else (9, 61.5),
                ha="center",
                va="center",
                color="white" if row["n_responses"] > 50 else "black",
                fontsize=12 if row["name"] != "Cyprus" else 8,
            )

    return fig, ax


def map_relative_responses(
    countries: geopandas.GeoDataFrame,
) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of responses per capita on a map of Europe (i.e. using a
    "choropleth map").

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe with country-aggregated number of responses. Needs to have "country",
        "responses_per_ten_million", "continent", and "name" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    belgium = utils._get_europe(countries).query("name == 'Belgium'")
    europe_ex_bel = utils._get_europe(countries).query("name != 'Belgium'")

    # Create a fake europe dataset containing 11 as the maximum
    fake_max = 11
    scale_end = countries.query("name == 'Japan'").copy()
    scale_end["responses_per_ten_million"] = fake_max
    europe_fake = pd.concat((europe_ex_bel, scale_end))

    fig, ax = plt.subplots(1, figsize=(30, 10), constrained_layout=True)

    # Set up a colorbar
    min_, max_ = (
        utils._get_europe(countries)
        .query("name != 'Belgium'")["responses_per_ten_million"]
        .agg(("min", "max"))
    )
    sm = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=plt.Normalize(min_, fake_max),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, shrink=0.7, anchor=(8.5, 0.9), ax=ax)
    cb.set_ticks((min_, 2, 4, 6, 8, 10, fake_max))
    cb.set_ticklabels(("0", "2", "4", "6", "8", "10", "$\gg10$"))

    # Plot the data to a map
    belgium.plot(
        color=[0.03137255, 0.18823529, 0.41960784, 1.0],
        linewidth=0.8,
        ax=ax,
        edgecolor="0.7",
    )
    europe_fake.plot(
        column="responses_per_ten_million",
        cmap="Blues",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.7",
    )

    # Crop to Europe, now axis labeling
    ax.axis("off")
    ax.set_xlim(-11.5, 35)
    ax.set_ylim(34, 71.5)

    # Add text marker for missing countries
    ax.annotate(
        utils._get_america(countries, column="responses_per_ten_million"),
        xy=(-11, 46),
        fontsize=15,
        color="#555555",
    )
    ax.annotate(
        utils._get_asia(countries, column="responses_per_ten_million"),
        xy=(30, 38),
        fontsize=15,
        color="#555555",
    )

    # Add numbers
    for _, row in utils._get_europe(countries).iterrows():
        if row["responses_per_ten_million"] > 0:
            plt.annotate(
                round(row["responses_per_ten_million"], 1),
                xy=row["geometry"].representative_point().coords[0]
                if row["name"] != "Norway"
                else (9, 61.5),
                ha="center",
                va="center",
                color="white" if row["responses_per_ten_million"] > 8 else "black",
                fontsize=12 if row["name"] != "Cyprus" else 8,
            )

    return fig, ax


def dataset_interest_groups(feedbacks: pd.DataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of feedback submissions for each user type using a bar plot.

    Parameters
    ----------
    feedbacks : pd.DataFrame
        Dataframe with one row per feedback and containing the "user_type_name" and
        "interest_type" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    # Create a count plot (bar chart with counts as values)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        y="user_type_name",
        order=utils.user_type_order,
        hue="interest_type",
        palette=user_type_palette,
        data=feedbacks,
        dodge=False,
        ax=ax,
    )

    # Add percentage annotations
    for i, row in enumerate(
        feedbacks["user_type_name"].value_counts()[utils.user_type_order]
    ):
        plt.annotate(
            f"{round(100 * row / len(feedbacks))}%",
            xy=(row + 1, i),
            va="center",
            ha="left",
            color="#555555",
            fontsize=9,
        )

    # Theme plot
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend().set_visible(False)
    ax.tick_params(axis="y", labelsize=15)
    utils.theme_plot(ax)

    return fig, ax


def interest_group_attachment_lengths(
    feedbacks: pd.DataFrame,
) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of feedback submissions for each user type using a violin plot.

    Parameters
    ----------
    feedbacks : pd.DataFrame
        Dataframe with one row per feedback and containing the "user_type_name" and
        "interest_type" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        y=feedbacks.query("filename.notnull()")["user_type_name"],
        x="n_pages",
        data=feedbacks.query("filename.notnull()"),
        hue="interest_type",
        dodge=False,
        cut=0,
        ax=ax,
        palette=user_type_palette,
        order=utils.user_type_order,
    )

    # plt.xlabel("Number of pages in attachment")
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend().set_visible(False)
    plt.xlim(0, 35)
    ax.yaxis.set_ticklabels([])

    utils.theme_plot(ax)

    return fig, ax


def interest_groups_total_pages(n_p: pd.DataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the total number of pages in the attachments against user types using a bar
    plot.

    Parameters
    ----------
    n_p : pd.DataFrame
        Dataframe with user_type-aggregated page counts, containing "user_type_name",
        "total_pages", and "interest_type" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    # Create a bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        y="user_type_name",
        x="total_pages",
        order=utils.user_type_order,
        hue="interest_type",
        palette=user_type_palette,
        data=n_p,
        dodge=False,
        ax=ax,
    )

    # Add percentage annotations
    for i, row in n_p.loc[utils.user_type_order].reset_index(drop=True).iterrows():
        plt.annotate(
            row["total_pages"],
            xy=(row["total_pages"] - (0 if i == 8 else 5), i),
            va="center",
            ha="right",
            color="white",
            fontsize=9,
        )

    # Theme plot
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend().set_visible(False)
    ax.tick_params(axis="y", labelsize=15)
    utils.theme_plot(ax)

    return fig, ax


def plot_english_usage(europe: geopandas.GeoDataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the fraction of English submissions on a map of Europa (i.e. use a
    "choropleth map").

    Parameters
    ----------
    europe : geopandas.GeoDataFrame
        Dataframe containing "share" and "name" columns.

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots(1, figsize=(30, 10), constrained_layout=True)

    # Set up a colorbar
    min_, max_ = europe["share"].agg(("min", "max"))
    sm = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=plt.Normalize(min_, max_),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, shrink=0.7, anchor=(8.5, 0.9), ax=ax)

    # Plot the data to a map
    europe.plot(
        column="share",
        cmap="Blues",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.7",
    )

    # Crop to Europe, now axis labeling
    ax.axis("off")
    ax.set_xlim(-11.5, 35)
    ax.set_ylim(34, 71.5)

    # Add numbers
    for _, row in europe.iterrows():
        plt.annotate(
            round(row["share"], 2),
            xy=row["geometry"].representative_point().coords[0]
            if row["name"] != "Norway"
            else (9, 61.5),
            ha="center",
            va="center",
            color="white" if row["share"] > 0.7 else "black",
            fontsize=12 if row["name"] != "Cyprus" else 8,
        )

    return fig, ax


def submissions_over_time(feedbacks: pd.DataFrame) -> Tuple[plt.figure, plt.axis]:
    """Plot the number of submissions against time.

    Parameters
    ----------
    feedbacks : pd.DataFrame
        Dataframe containing one feedback submission per row, with a "date_feedback"
        column

    Returns
    -------
    Tuple[plt.figure, plt.axis]
        Figure and axis of the plot.
    """
    fig, ax = plt.subplots()
    sns.histplot(x=pd.to_datetime(feedbacks["date_feedback"]), ax=ax)

    ax.set_xticks(["2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01"])
    ax.set_xticks(
        pd.Timestamp("2021-04-24 00:00") + pd.timedelta_range("0W", "15W", freq="7D"),
        minor=True,
    )
    ax.set_xlabel(None)

    utils.theme_plot(ax)
    return fig, ax


dark = [0.51058824, 0.73230296, 0.85883891, 1.0]
light = [0.21568627, 0.52941176, 0.75424837, 1.0]
user_type_palette = {
    "corporate": light,
    "academia": light,
    "public": dark,
    "citizen": dark,
    "standardizing_body": light,
}
