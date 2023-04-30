"""Module to create and apply regex patterns."""

import re
import string
import numpy as np
import pandas as pd

ROMAN_NUMBERS = (
    "i;ii;iii;iv;v;vi;vii;viii;ix;x;xi;xii;xiii;xiv;xv;xvi;xvii;xviii;xix;xx"
).split(";")
END_ARABIC = r"(?!\d)"
END_BOTH = r"(?![0-9iIvVxX])"
END_WORD = r"(?!\w)"
END_ARTICLE = (
    r"(?!\s*(?:\(\d\)\s*(?:\([a-z]\)\s*)?)?"
    r"(?:of\s+the\s+)?(?:GDPR|General\s+Data\s+Protection\s+Regulation))"
    r"(?!\s*(?:WP|(?:Data\s+Protection\s+)?Working\s+Party))"
)


def prep_regex(pattern: str) -> str:
    """Transform a regex pattern into something more usable:

    - Convert any (lowercase) alphabetic characters to both an equivalent of
      re.IGNORECASE, e.g. "a" -> "[aA]".
    - Convert any spaces into arbitrary whitespace, i.e. " " -> "\\s+".

    Parameters
    ----------
    pattern : str
        Arbitrary regex pattern.

    Returns
    -------
    str
        Transformed pattern.
    """
    for lower, upper in zip(string.ascii_lowercase, string.ascii_uppercase):
        pattern = pattern.replace(lower, f"[{lower}{upper}]")
    return pattern.replace(" ", "\\s+")


def get_roman_regex(n: int) -> str:
    """Create regex pattern to match any (roman numeral) ints in {1, 2, ..., n}.

    Parameters
    ----------
    n : int
        Maximum integer.

    Returns
    -------
    str
        Regex pattern.
    """
    if n > 20:
        raise ValueError("This function only works for values of n up to 20.")

    return "|".join(ROMAN_NUMBERS[:n])


def get_arabic_regex(n: int) -> str:
    """Create regex pattern to match any (arabic number) ints in {1, 2, ..., n}.

    Parameters
    ----------
    n : int
        Maximum integer to match.

    Returns
    -------
    str
        Regex pattern.
    """
    ARABIC_NUMBERS = [str(n) for n in range(1, n + 1)]
    return "|".join(ARABIC_NUMBERS)


AI_ACT = (
    prep_regex(
        r"(?:ai[ -]act|artificial intelligence act|the(?: proposed)? act|the ai regulation)"
    )
    + END_WORD
)
EAIB = (
    prep_regex(r"(?:eaib|(?:european )?(?:artificial intelligence|ai) board")  # Uncased
    + r"|[tT]he\s+Board)"  # Cased part
) + END_WORD

TITLE = prep_regex(f"title (?:{get_roman_regex(12)}|{get_arabic_regex(12)})") + END_BOTH
TITLE_DIGIT = TITLE.replace("?:", "")

ARTICLE = prep_regex(f"article (?:{get_arabic_regex(85)})") + END_ARABIC + END_ARTICLE
ARTICLE_DIGIT = ARTICLE.replace("?:", "")

ANNEX = prep_regex(f"annex (?:{get_roman_regex(9)}|{get_arabic_regex(9)})") + END_BOTH
ANNEX_DIGIT = ANNEX.replace("?:", "")

ALL = "|".join([AI_ACT, EAIB, TITLE, ARTICLE, ANNEX])
PATTERN = f"(.*)({ALL})(.*)"


def classify_aspect(string: str) -> pd.Series:
    """Classify aspect into its type and subtype.

    Example: "Article 35" -> "article", "35".

    Parameters
    ----------
    string : str
        Aspect string.

    Returns
    -------
    pd.Series
        Pandas series with "aspect_type" and "aspect_subtype" labels.
    """
    out: tuple[str | None, str | None]  # Specify type to avoid type errors

    if string is None or string is np.nan:
        out = None, None
    elif article_digit := re.search(ARTICLE_DIGIT, string, flags=re.IGNORECASE):
        out = "article", str(article_digit.group(1))
    elif title_digit := re.search(TITLE_DIGIT, string, flags=re.IGNORECASE):
        out = "title", str(_convert_to_int(title_digit.group(1)))
    elif annex_digit := re.search(ANNEX_DIGIT, string, flags=re.IGNORECASE):
        out = "annex", str(_convert_to_int(annex_digit.group(1)))
    elif re.search(EAIB, string, flags=re.IGNORECASE):
        out = "eaib", ""
    elif re.search(AI_ACT, string, flags=re.IGNORECASE):
        out = "ai_act", ""
    else:
        out = None, None

    return pd.Series(out, index=["aspect_type", "aspect_subtype"])


def _convert_to_int(string: str) -> int:
    """Convert a string to integer.

    Parameters
    ----------
    string : str
        Integer, either as a arabic number, or roman number (limited to 1 to 20).

    Returns
    -------
    int
        Converted integer.
    """
    if string.lower() in ROMAN_NUMBERS:
        return ROMAN_NUMBERS.index(string.lower()) + 1

    return int(string)
