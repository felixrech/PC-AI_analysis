"""Module for resolving footnote references to the actual footnotes."""

import re
import numpy as np
import pandas as pd
from fitz.fitz import Page
from itertools import compress
from typing import List, Tuple, Dict, Any

from text_extraction import line_to_text


FITZ_SUPERSCRIPT_FLAG = 1


def span_is_footnote_reference(span: dict) -> bool:
    """Check whether a span is a footnote reference, i.e. both superscript and Arabic
    number of Roman numeral.

    Parameters
    ----------
    span : dict
        An arbitrary span dictionary.

    Returns
    -------
    bool
        True if span is a footnote reference, False otherwise.
    """
    if (
        # Flagged as superscript by PyMuPDF
        span["flags"] & FITZ_SUPERSCRIPT_FLAG
        # Positive match to Arabic numbers or Roman numerals
        and re.match(r"\s*[\divx][\s\divx,]*$", span["text"]) is not None
        # Negative match to common false positives to make sure
        and re.match(r"\s+$|\s*rd\s*|\s*st\s*|\s*th\s*|\s*\.\s*", span["text"]) is None
    ):
        return True

    # Default to False (no footnote reference)
    return False


def page_y_cutoff(
    page: Page, footnotes: List[Dict[str, Any]] = [], cutoff: float = 0.6
) -> float:
    """Computes a vertical axis cutoff point.

    If footnote is unspecified/an empty list, computes the cutoff percentile of the
    vertical axis of the given page. If footnote is specified, picks maximum between
    the former and the vertically lowest footnote candidate.

    Parameters
    ----------
    page : fitz.fitz.Page
        Arbitrary page of a PDF document.
    footnotes : List[Dict[str, Any]], optional
        List of footnote references, by default [].
    cutoff : float, optional
        Which percentile of the vertical axis to use, by default 0.6.

    Returns
    -------
    float
        Resulting cutoff point.
    """
    # Compute the cutoff percentile between the vertical axis's start and end
    page_upper, page_lower = page.bound().y0, page.bound().y1
    page_two_thirds = page_upper + cutoff * (page_lower - page_upper)

    # If no footnotes are given, default to page-based cutoff
    if len(footnotes) == 0:
        return page_two_thirds

    # Find out the y value of the vertically lowest footnote
    footnotes_max_y = np.max([s["bbox"][3] for s in footnotes])

    # And return the maximum of both criteria
    return max(page_two_thirds, footnotes_max_y)


def page_median_font_size(page: Page, lines: List[List[Dict[str, Any]]]) -> float:
    """Compute the median font size on the upper half of the page.

    Parameters
    ----------
    page : fitz.fitz.Page
        The fitz page object that the lines are part of.
    lines : List[List[Dict[str, Any]]]
        List of lines (each a list of span dicts).

    Returns
    -------
    float
        Median font size.
    """
    # Limit page to lines that are in the upper half
    y_cutoff = page_y_cutoff(page, cutoff=1 / 2)
    lines = list(
        compress(
            lines,
            [
                any([span for span in line if span["bbox"][1] < y_cutoff])
                for line in lines
            ],
        )
    )

    # Compute the median of the spans in the remaining lines
    return float(np.median([s["size"] for l in lines for s in l]))


def lines_to_footnote_references(
    lines: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Convert a list of a page's (chunk's or document's) lines into a list of
    footnotes references.

    Parameters
    ----------
    lines : List[List[Dict[str, Any]]]
        List of lines.

    Returns
    -------
    List[Dict[str, Any]]
        List of footnote references.
    """
    return [s for l in lines for s in l if span_is_footnote_reference(s)]


def lines_to_footnote_candidates(
    page: Page, lines: List[List[Dict[str, Any]]], footnotes: List[Dict[str, Any]]
) -> List[bool]:
    """For each line of a page, check whether the line might be a footnote.

    Footnotes are determined to be on the lower 40% of a page, lower than any of the
    footnote references, and later in the reading order than any of the footnote
    references.

    Parameters
    ----------
    page : fitz.fitz.Page
        An arbitrary page of a document.
    lines : List[List[Dict[str, Any]]]
        Lines of that page.
    footnotes : List[Dict[str, Any]]
        List of footnote reference candidates.

    Returns
    -------
    List[bool]
        List of booleans: for each line, could that line be a footnote?
    """
    # Footnotes should not have font size larger than median font size of page
    font_size_cutoff = page_median_font_size(page, lines)

    # Footnotes should not be above (in terms of y-axis) the last footnote reference
    y_cutoff = page_y_cutoff(page, footnotes)

    # Compute which lines mach the above two conditions
    candidates = pd.Series(
        [
            any(
                span["bbox"][1] > y_cutoff and span["size"] <= font_size_cutoff
                for span in line
            )
            for line in lines
        ]
    )

    # Footnotes should not come before footnote reference in the reading order
    contains_footnote = pd.Series(  # Lines that contain a footnote
        [any(span in footnotes for span in line) for line in lines]
    )
    # Number of line that contains the last footnote reference
    min_index = contains_footnote[contains_footnote].index[-1] + 1  # type: ignore
    # Convert to bool series w/ False for all lines until last ft. reference, then True
    after_footnotes = pd.Series([False] * min_index + [True] * (len(lines) - min_index))

    # Combine both criteria
    return (candidates & after_footnotes).to_list()


def page_resolve_footnotes(
    page: Page, lines: List[List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[dict, List[int]]]]:
    """Find footnote references on the given page and resolve them to their footnotes.

    Returned resolutions are dicts with the key 'footnote' and 'replacement', where the
    former is a span and the latter a list of lines.

    Parameters
    ----------
    page : Page
        Arbitrary page of a document.
    lines : List[List[Dict[str, Any]]]
        List of lines on the page.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Tuple[dict, List[int]]]]
        Tuple containing the footnote references (list of spans) and a list of
        resolutions (list of dicts).
    """
    # Detect all footnote references
    footnotes = lines_to_footnote_references(lines)

    # Exit early if the page does not contain any footnotes in the first place
    if len(footnotes) == 0:
        return footnotes, []

    # Which lines could be footnotes?
    candidates = lines_to_footnote_candidates(page, lines, footnotes)
    replacements = []

    def is_match(footnote: int, line: int) -> bool:
        """Compute whether a line starts with a given footnote.

        Parameters
        ----------
        footnote : int
            Index of the footnote in the footnotes.
        line : int
            Index of the line in the lines.

        Returns
        -------
        bool
            Whether the line starts with the footnote.
        """
        line_text = line_to_text(lines[line], clean=True)
        footnote_text = line_to_text([footnotes[footnote]])
        return line_text.startswith(footnote_text + " ") or line_text == footnote_text

    # Initial values: remaining footnotes and first candidate line
    remaining_footnotes = set(range(len(footnotes)))
    line = min(compress(range(len(candidates)), candidates), default=0)

    while line in range(len(lines)):
        # If the current line is not a candidate, skip
        if not candidates[line]:
            line += 1
            continue

        for footnote in remaining_footnotes:
            if line in range(len(lines)) and is_match(footnote, line):
                # Define lines that will replace footnote and footnotes after current
                repl = []
                remaining_footnotes = remaining_footnotes - set([footnote])

                # While line does not start with one of the remaining footnotes...
                while (
                    line in range(len(lines))
                    and candidates[line]
                    and not any(is_match(f, line) for f in remaining_footnotes)
                ):
                    repl.append(line)
                    line += 1

                # Current footnote is done and should not be looked at anymore
                replacements.append((footnotes[footnote], repl))

                # Line will be too large after the regular increment after the for loop
                line -= 1
                break
        line += 1

    return footnotes, replacements
