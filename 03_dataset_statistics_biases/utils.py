"""Module that contains some utility functions and other small things to make the 
notebook less cluttered."""

import fitz
import scipy
import geopandas
import numpy as np
import pandas as pd
from typing import Any, Union, Tuple

import matplotlib.pyplot as plt


def pdf_page_count(filename: str) -> int:
    """Counts the number of pages a PDF file has.

    Parameters
    ----------
    filename : str
        Filename of the PDF file, e.g. "../attachments/my_file.pdf".

    Returns
    -------
    int
        Page count.
    """
    return fitz.open(filename).page_count if type(filename) is str else 0


def by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the number of responses by country of origin and adds population and
    GDP numbers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a "country" column and with each row representing a
        response.

    Returns
    -------
    pd.DataFrame
        Dataframe with "country", "country_name", "n_responses", "is_eu", "2021_pop",
        "2021_gdp", and "responses_per_ten_million" columns.
    """
    # Compute responses per country
    df_by_country = df["country"].value_counts().to_frame("n_responses")
    df_by_country = df_by_country.reset_index(names="country")

    # Add dummy values for EU countries without any submissions
    df_by_country["is_eu"] = df_by_country["country"].isin(EU_COUNTRIES)
    remaining = list(set(EU_COUNTRIES) - set(df_by_country["country"]))
    remaining = pd.DataFrame(
        {
            "country": remaining,
            "n_responses": [0] * len(remaining),
            "is_eu": [True] * len(remaining),
        }
    )
    df_by_country = pd.concat((df_by_country, remaining))

    # Read in the two external datasets
    # Source: https://data.worldbank.org/indicator/SP.POP.TOTL
    pop_df = pd.read_csv("datasets/pop_worldbank.csv", skiprows=4)
    # Source: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
    gdp_df = pd.read_csv("datasets/gdp_worldbank.csv", skiprows=4)

    # Merge with context into a single dataframe
    context = pd.merge(pop_df, gdp_df, on="Country Code", suffixes=("_pop", "_gdp"))
    df_by_country = pd.merge(
        context, df_by_country, left_on="Country Code", right_on="country", how="right"
    )

    # Select relevant columns and rename them appropriately
    df_by_country = df_by_country[
        ["country", "Country Name_pop", "2021_pop", "2021_gdp", "n_responses", "is_eu"]
    ].rename(columns={"Country Name_pop": "country_name"})

    df_by_country["gdp_per_cap"] = df_by_country["2021_gdp"] / df_by_country["2021_pop"]
    df_by_country["responses_per_ten_million"] = df_by_country["n_responses"] / (
        df_by_country["2021_pop"] / 1e7
    )

    return df_by_country


def hstack(*args: pd.DataFrame) -> pd.DataFrame:
    """Concatenate a number of dataframes horizontally.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe.
    """
    return pd.concat(args, axis=1)


def pprint(*args: Any) -> None:
    """Semi-prettily prints a list of arbitrary objects with:
    - a newline after every two elements and at least three spaces between them
    - every odd element being left aligned
    - every even element being right aligned
    """
    left, right = args[::2], args[1::2]

    max_ = max(len(s) for s in left) + 3
    left = [s.ljust(max_) for s in left]

    floats, nums = (float, np.float64), (int, np.int64, float, np.float64)
    right = [x if type(x) not in floats else round(x, 3) for x in right]
    max_ = max([len(str(s)) for s in right if type(s) in nums], default=5)
    right = [(f"{s:.3f}" if type(s) in floats else str(s)).rjust(max_) for s in right]

    print("\n".join(map(lambda t: f"{t[0]}{t[1]}", zip(left, right))))


def spearman_test(
    df: pd.DataFrame, column_a: Any, column_b: Any, print_: bool = True
) -> Union[None, Tuple[float, float, int]]:
    """Conduct a Spearman rank correlation test between two columns of a dataframe and
    either print the results (rho, p, n) or return them.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract the columns from.
    column_a : Any
        Name of the first column.
    column_b : Any
        Name of the second column.
    print_ : bool, optional
        Whether to print the results (True) or to return them instead (False). By
        default True.

    Returns
    -------
    Union[None, Tuple[float, float, int]]
        None if the results were printed or a tuple containing rho, p, and n otherwise.
    """
    res = scipy.stats.spearmanr(df[column_a], df[column_b])
    corr, p, n = res.correlation, res.pvalue, len(df)
    if print_:
        p = str(round(p, 3)).ljust(5) if p > 0.0005 else "{:.2e}".format(p)
        print(f"ρ={str(round(corr, 2)).ljust(4)}   p={p}   n={n}")
    else:
        return corr, p, n


def mannwhitneyu_test(
    df: pd.DataFrame,
    factor_column: Any,
    value_column: Any,
    print_: bool = True,
    alternative: str = "two-sided",
) -> Union[None, Tuple[float, float, int]]:
    """Conduct a Mann-Whitney-U-test between two columns of a dataframe and
    either print the results (U, p, n) or return them.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract the columns from.
    factor_column : Any
        Name of the column containing the binary variable.
    value_column : Any
        Name of the column containing the continuous variable.
    print_ : bool, optional
        Whether to print the results (True) or to return them instead (False). By
        default True.
    alternative : str, optional
        Alternative to use, by default "two-sided".

    Returns
    -------
    Union[None, Tuple[float, float, int]]
        None if the results were printed or a tuple containing U, p, and n otherwise.
    """
    res = scipy.stats.mannwhitneyu(
        df[value_column][~df[factor_column]],
        df[value_column][df[factor_column]],
        alternative=alternative,
    )
    U, p, n = res.statistic, res.pvalue, len(df)
    if print_:
        p = str(round(p, 3)).ljust(5) if p > 0.0005 else "{:.2e}".format(p)
        print(f"U={str(round(U, 2)).ljust(4)}   p={p}   n={n}")
    else:
        return U, p, n


def theme_plot(ax: plt.axis) -> None:
    """Apply common theme to a plot.

    Parameters
    ----------
    ax : plt.axis
        Axis of the plot to theme.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="x", which="both", colors="#999999")
    ax.tick_params(axis="y", colors="#999999")
    ax.set_xlabel(ax.get_xlabel(), color="#555555")
    ax.set_ylabel(ax.get_ylabel(), color="#555555")
    plt.xticks(color="#444444")
    plt.yticks(color="#444444")
    if ax.get_legend() is not None:
        plt.setp(ax.get_legend().get_title(), color="#444444")


def _get_europe(countries: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Extract European countries from a dataframe.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe with the "continent" and "country" columns.

    Returns
    -------
    geopandas.GeoDataFrame
        Same dataframe but with rows of European countries.
    """
    return countries.query("continent == 'Europe' or country == 'CYP'").copy()


def _get_america(
    countries: geopandas.GeoDataFrame, column: str = ""
) -> Union[geopandas.GeoDataFrame, str]:
    """Extract American countries from a dataframe.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe with the "continent" and "country" columns.
    column : str, optional
        Instead of returning rows, return a string based on this column.

    Returns
    -------
    geopandas.GeoDataFrame
        Same dataframe but with rows of American countries.
    """
    america = countries.query("continent == 'North America'")
    if column == "":
        return america
    return "⇽" + "\n".join(
        america["country"] + ", " + america[column].round(1).astype(str)
    )


def _get_asia(
    countries: geopandas.GeoDataFrame, column: str = ""
) -> Union[geopandas.GeoDataFrame, str]:
    """Extract Asian countries from a dataframe.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe with the "continent" and "country" columns.
    column : str, optional
        Instead of returning rows, return a string based on this column.

    Returns
    -------
    geopandas.GeoDataFrame
        Same dataframe but with rows of Asian countries.
    """
    asia = countries.query("continent == 'Asia' and not country == 'CYP'")
    if column == "":
        return asia
    asia = "\n".join(asia["name"] + ", " + asia[column].round(1).astype(str))
    return asia.replace("\n", "⇾\n", 1)


def _get_threshold(countries: geopandas.GeoDataFrame) -> int:
    """Compute the threshold of the number of countries from which at least 50% of all
    submissions come from.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Dataframe containing "continent", "country", and "n_responses" columns.

    Returns
    -------
    int
        Threshold.
    """
    cum_responses = _get_europe(countries)["n_responses"][::-1].cumsum()
    half_responses = countries["n_responses"].sum() * (1 - 0.5)
    return len(_get_europe(countries)) - np.argmax(cum_responses > half_responses) - 1


# fmt: off
EU_COUNTRIES = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC",
    "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE"
]

# Post socialist countries (Slovenia and Croatia are successors to Yugoslavia);
# Source: DOI 10.4467/2543859XPKG.20.019.12787   
EU_POST_SOCIALIST = [
    "BGR", "CZE", "HRV", "HUN", "POL", "ROU", "SVK", "SVN", "EST", "LVA", "LTU"
]

EU_SOUTH = ["PRT", "ESP", "MLT", "GRC", "CYP", "ITA", "SVN", "HRV", "ROU", "BGR"]

# Accession to the EU in 2004, 2007, or 2013;  Source: DOI 10.1007/978-3-662-45320-9_1
EU_YOUNG = [
    "CYP", "CZE", "EST", "HUN", "LVA", "LTU", "MLT", "POL", "SVK", "SVN",
    "BGR", "ROU", "HRV"
]

# fmt: on
interest_type_map = {
    "company": "corporate",
    "business_association": "corporate",
    "academic_research_institution": "academia",
    "ngo": "public",
    "trade_union": "public",
    "public_authority": "public",
    "consumer_organisation": "public",
    "eu_citizen": "citizen",
    "standardizing_body": "standardizing_body",
}
user_type_map = {
    "company": "Company",
    "business_association": "Business Association",
    "academic_research_institution": "Academia",
    "ngo": "NGO",
    "trade_union": "Trade Union",
    "public_authority": "Public Authority",
    "consumer_organisation": "Consumer Organisation",
    "eu_citizen": "EU Citizen",
    "standardizing_body": "Standardizing Body",
}
# fmt: off
user_type_order = [
    "Company", "Business Association", "NGO", "Trade Union", "Public Authority",
    "Consumer Organisation", "Academia", "EU Citizen", "Standardizing Body"
]  # fmt: on
