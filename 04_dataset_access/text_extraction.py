"""Module with mostly helpers for extracting text from PDF documents. To actually
convert .pdf files into .txt files, look for the clean_extract module instead."""

import re
import fitz
import pandas as pd
from itertools import compress
from typing import Any, List, Tuple, Dict


def line_to_text(line: List[Dict[str, Any]], clean: bool = False) -> str:
    """Extracts the texts of a line's spans.

    Parameters
    ----------
    line : List[Dict[str, Any]]
        Line to extract text from.
    clean : bool, optional
        Whether to remove unnecessary whitespace from the line, by default False

    Returns
    -------
    str
        Extracted text.
    """
    # Concatenate the spans' text
    text = "".join(span["text"] for span in line)

    if clean:
        # Replace any number of whitespace characters with a single space
        text = re.sub(r"\s+", " ", text)
        # Replace any whitespace before the first non-whitespace character
        text = re.sub(r"^\s*(\S.*)", r"\g<0>", text)

    return text


def line_not_empty(line: List[Dict[str, Any]]) -> bool:
    """Check whether a line is empty, i.e. contains not a single alphanumerical
    character.

    Parameters
    ----------
    line : List[Dict[str, Any]]
        A single line (= list of span dicts).

    Returns
    -------
    bool
        False if line is empty, true otherwise.
    """
    return re.search(r"\w", line_to_text(line)) is not None


def line_is_content(line: List[Dict[str, Any]]) -> bool:
    """Check whether the line is content or not. Non-content is currently classified
    as either a) whitespace or b) string of the form "Page X( of Y)?".

    Parameters
    ----------
    line : List[Dict[str, Any]]
        Any line.

    Returns
    -------
    bool
        True if line is body-text and False if it doesn't contain any real content.
    """
    cleaned = line_to_text(line, clean=True)
    if re.match(r"Page \d+( ?/ ?| of )\d+ ?$", cleaned) is not None:
        return False
    return line_not_empty(line)


def page_to_lines(page: fitz.Page) -> List[List[Dict[str, Any]]]:
    """Convert a fitz page object to a list of lines, each a list of span dictionaries.

    Parameters
    ----------
    page : fitz.fitz.Page
        Page of any PDF document.

    Returns
    -------
    List[List[Dict[str, Any]]]
        List of lines, each of which is a list of span dictionaries.
    """
    # Get page text as list of dicts, flags to preserve whitespace and de-hyphenate
    text = page.get_text("dict", flags=16 + 2)  # type: ignore
    lines = [l["spans"] for b in text["blocks"] for l in b["lines"]]

    # Filter out some non-body text lines
    lines = [l for l in lines if line_is_content(l)]

    # If the (in terms of y) lowest line is a single number, filter it out
    lines_y = [min([s["bbox"][1] for s in line], default=-1) for line in lines]
    for line in compress(lines, [y == max(lines_y, default=-1) for y in lines_y]):
        if re.match(r" ?\d{1,2} ?$", line_to_text(line, clean=True)) is not None:
            del lines[lines.index(line)]

    return lines


def page_remove_duplicate_lines(
    pages: List[List[List[Dict[str, Any]]]],
) -> List[List[List[Dict[str, Any]]]]:
    """Removes lines from pages that occur as an exact copy at least twice and at least
    half the number of pages often.

    Parameters
    ----------
    pages : List[List[List[Dict[str, Any]]]]
        List of pages (each a list of lines).

    Returns
    -------
    List[List[List[Dict[str, Any]]]]
        List of pages (each a potentially smaller list of lines than before).
    """
    # Convert the lines on all pages into (cleaned) string format
    lines = [line_to_text(line, clean=True) for page in pages for line in page]

    # If there are no lines, just return
    if len(lines) == 0:
        return pages

    # Count how often each line appears
    lines = pd.Series(lines).value_counts()

    # Filter out lines that occur on at least two and on at least "half the pages"
    duplicates = lines[lines > max(len(pages) // 2, 2)].index
    pages = [
        [line for line in page if not line_to_text(line, clean=True) in duplicates]
        for page in pages
    ]
    return pages


def extract_pdf_with_doc(
    filename: str,
) -> Tuple[fitz.Document, List[List[List[Dict[str, Any]]]]]:
    """Extract text from a PDF file (and also returns the fitz document object).

    Parameters
    ----------
    filename : str
        Filename of the PDF file.

    Returns
    -------
    Tuple[fitz.Document, List[List[List[Dict[str, Any]]]]]
        fitz doc object and list of pages - each a list of lines, i.e. a list of lists
        of spans (=dictionaries).
    """
    doc = fitz.Document(filename)  # type: ignore
    pages = [page_to_lines(page) for page in doc]

    pages = page_remove_duplicate_lines(pages)

    return doc, pages


def extract_pdf(filename: str) -> List[List[List[Dict[str, Any]]]]:
    """Extract text from a PDF file.

    Parameters
    ----------
    filename : str
        Filename of the PDF file.

    Returns
    -------
    List[List[List[Dict[str, Any]]]]
        List of pages - each a list of lines, i.e. a list of lists of spans
        (=dictionaries).
    """
    _, pages = extract_pdf_with_doc(filename)

    return pages
