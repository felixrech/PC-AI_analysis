"""CAUTION: This module only remains for replicability of the pdf_extractors
evaluations. However, later on, it was slightly restructured and fully documented. So
if you are looking for how text is extracted from the dataset, please refer to the
proper text extraction module (04_dataset_access), instead.
"""

import re
import fitz
import numpy as np
import pandas as pd
from itertools import compress


SUP = 1


def span_is_footnote_reference(span: dict) -> bool:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    if (
        # Flagged as superscript by PyMuPDF
        span["flags"] & SUP
        # Positive match to Arabic numbers or Roman numerals
        and re.match(r"\s*[\divx][\s\divx,]*$", span["text"]) is not None
        # Negative match to common false positives to make sure
        and re.match(r"\s+$|\s*rd\s*|\s*st\s*|\s*th\s*|\s*\.\s*", span["text"]) is None
    ):
        return True

    # Default to False (no footnote reference)
    return False


def page_y(
    page: fitz.fitz.Page, footnotes: list[dict] = [], cutoff: float = 0.6
) -> float:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    page_upper, page_lower = page.bound().y0, page.bound().y1
    page_two_thirds = page_upper + cutoff * (page_lower - page_upper)

    if len(footnotes) == 0:
        return page_two_thirds

    footnotes_max_y = np.max([s["bbox"][3] for s in footnotes])

    return max(page_two_thirds, footnotes_max_y)


def page_median_font_size(page: fitz.fitz.Page, lines: list[list[dict]]) -> float:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    # Limit page to lines that are in the upper half
    y_cutoff = page_y(page, cutoff=1 / 2)
    lines = compress(
        lines,
        [any([span for span in line if span["bbox"][1] < y_cutoff]) for line in lines],
    )

    # Compute the median of the spans in the remaining lines
    return np.median([s["size"] for l in lines for s in l])


def line_to_text(line: list[dict], clean: bool = False) -> str:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    # Concatenate the spans' text
    text = "".join(span["text"] for span in line)

    if clean:
        # Replace any number of whitespace characters with a single space
        text = re.sub(r"\s+", " ", text)
        # Replace any whitespace before the first non-whitespace character
        text = re.sub(r"^\s*(\S.*)", "\g<0>", text)

    return text


def line_not_empty(line: list[dict]) -> bool:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    return re.search(r"\w", line_to_text(line)) is not None


def line_is_content(line: list[dict]):
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    cleaned = line_to_text(line, clean=True)
    if re.match(r"Page \d+( ?/ ?| of )\d+ ?$", cleaned) is not None:
        return False
    return line_not_empty(line)


def page_to_lines(page: fitz.fitz.Page) -> list[list[dict]]:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    # Get page text as list of dicts, flags to preserve whitespace and de-hyphenate
    text = page.get_text("dict", flags=16 + 2)
    lines = [l["spans"] for b in text["blocks"] for l in b["lines"]]

    # TODO: Check for false positives
    # Filter out some non-body text lines
    lines = [l for l in lines if line_is_content(l)]

    # TODO: Check for false positive
    # If the (in terms of y) lowest line is a single number, filter it out
    lines_y = [min([s["bbox"][1] for s in line], default=-1) for line in lines]
    for line in compress(lines, [y == max(lines_y, default=-1) for y in lines_y]):
        if re.match(r" ?\d{1,2} ?$", line_to_text(line, clean=True)) is not None:
            del lines[lines.index(line)]

    return lines


def page_remove_duplicate_lines(
    pages: list[list[list[dict]]],
) -> list[list[list[dict]]]:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    # Convert the lines on all pages into (cleaned) string format
    lines = [line_to_text(line, clean=True) for page in pages for line in page]

    # Count how often each line appears
    lines = pd.Series(lines).value_counts()

    # Filter out lines that occur on at least two and on at least "half the pages"
    duplicates = lines[lines > max(len(pages) // 2, 2)].index
    pages = [
        [line for line in page if not line_to_text(line, clean=True) in duplicates]
        for page in pages
    ]
    return pages


def lines_to_footnote_references(lines: list[list[dict]]) -> list[dict]:
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    return [s for l in lines for s in l if span_is_footnote_reference(s)]


def lines_to_footnote_candidates(page, lines, footnotes):
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    # Footnotes should not have font size larger than median font size of page
    font_size_cutoff = page_median_font_size(page, lines)

    # Footnotes should not be above (in terms of y-axis) the last footnote reference
    y_cutoff = page_y(page, footnotes)

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
    min_index = contains_footnote[contains_footnote].index[-1] + 1
    # Convert to bool series w/ False for all lines until last ft. reference, then True
    after_footnotes = pd.Series([False] * min_index + [True] * (len(lines) - min_index))

    # Combine both criteria
    return (candidates & after_footnotes).to_list()


def parse_footnotes(page, lines):
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    footnotes = lines_to_footnote_references(lines)
    if len(footnotes) == 0:
        return lines, footnotes, []

    candidates = lines_to_footnote_candidates(page, lines, footnotes)
    replacements = []

    def is_match(footnote, line):
        """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
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
                replacements.append(
                    dict(footnote=footnotes[footnote], replacement=repl)
                )
                line -= 1
                break
        line += 1

    return lines, footnotes, replacements


def extract_pdf(filename):
    """CAUTION: UNDOCUMENTED & UNMAINTAINED, SEE 04_dataset_access FOR FINAL VERSION."""
    doc = fitz.open(filename)
    pages = [page_to_lines(page) for page in doc]

    pages = page_remove_duplicate_lines(pages)

    return [parse_footnotes(page, lines) for page, lines in zip(doc, pages)]
