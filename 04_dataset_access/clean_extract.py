"""Module for extracting cleaned text from input PDF files."""

import os
import re
import pandas as pd
import multiprocessing
from itertools import chain, repeat
from typing import Any, List, Tuple, Dict

from footnote_resolution import page_resolve_footnotes
from text_extraction import extract_pdf_with_doc, line_to_text


def convert_attachments_to_txt(
    source_folder: str = "",
    source_df: pd.DataFrame = pd.DataFrame(),
    target_folder: str = "",
    resolve_footnotes: bool = True,
    n_jobs: int = multiprocessing.cpu_count(),
) -> None:
    """Convert PDF files (=attachments in our context) to .txt files with optionally
    resolved footnotes.

    Note: Either source_folder or source_df has to be specified!

    Parameters
    ----------
    source_folder : str, optional
        Convert any .pdf files in this folder, by default "", i.e. use source_df instead.
    source_df : pd.DataFrame, optional
        Convert any files mentioned in the "filename" column of this dataframe, by
        default pd.DataFrame(), i.e. use source_folder instead.
    target_folder : str, optional
        Which folder to use for output .txt files, by default "", i.e. inferring from
        the source filenames instead. (Using the input filename but replacing .pdf with
        .txt or appending .txt to non-.pdf filenames.)
    resolve_footnotes : bool, optional
        Whether to resolve/replace footnote references to/with the actual footnotes they
        refer to, by default True.
    n_jobs : int, optional
        Number of processes to use, by default the number of CPU cores.

    Raises
    ------
    ValueError
        If neither source_folder nor source_df has been specified.
    """
    if source_folder != "":
        source_files = [
            source_folder + f for f in os.listdir(source_folder) if f.endswith(".pdf")
        ]
    elif not source_df.empty:
        source_files = source_df["filename"].to_list()
    else:
        raise ValueError(
            "At least one of source_folder and source_df have to be specified!"
        )

    # Compute the target filenames
    target_filenames = []
    for source_file in source_files:
        target_folder_ = (
            target_folder if target_folder != "" else os.path.dirname(source_file) + "/"
        )
        source_basename = os.path.basename(source_file)
        if source_file.endswith(".pdf"):
            target_filenames.append(target_folder_ + source_basename[:-4] + ".txt")
        else:
            target_filenames.append(target_folder_ + source_basename + ".txt")

    # convert_pdf_to_txt(source_file, target_filename, resolve_footnotes)

    if n_jobs == 1:
        [
            convert_pdf_to_txt(source_file, target_filename, resolve_footnotes)
            for source_file, target_filename in zip(source_files, target_filenames)
        ]
    else:
        pool = multiprocessing.Pool(n_jobs)
        pool.starmap(
            convert_pdf_to_txt,
            zip(source_files, target_filenames, repeat(resolve_footnotes)),
        )
        del pool


def convert_pdf_to_txt(
    source_filename: str, target_filename: str = "", resolve_footnotes: bool = True
) -> None:
    """Extract text from source file and write it to target file, optionally also
    resolving footnotes.

    Parameters
    ----------
    source_filename : str
        Filename of the source file, e.g. "attachments/my_file.pdf".
    target_filename : str, optional
        Filename of the output file, by default "", which will replace the file
        extension with '.txt' if the source_filename ends with '.pdf', otherwise simply
        append '.txt' to the source_filename.
    resolve_footnotes : bool, optional
        Whether to resolve/replace footnote references to/with the actual footnotes they
        refer to, by default True.
    """
    if target_filename == "":
        if source_filename.endswith(".pdf"):
            target_filename = source_filename[:-4] + ".txt"
        else:
            target_filename = source_filename + ".txt"

    text = extract_text_from_pdf(source_filename, resolve_footnotes)

    if len(text) == 0:
        print(
            f"Warning: Not writing file {target_filename}, as extracted text is empty!"
        )
        return

    with open(target_filename, "w") as file:
        file.write(text)


def extract_text_from_pdf(filename: str, resolve_footnotes: bool = True) -> str:
    """Extract the text from a PDF file and optionally resolve/replace footnotes
    references with actual footnotes.

    Parameters
    ----------
    filename : str
        Filename of the source file, e.g. "attachments/my_file.pdf".
    resolve_footnotes : bool, optional
        Whether to resolve/replace footnote references to/with the actual footnotes they
        refer to, by default True.

    Returns
    -------
    str
        Text content of the PDF file. Page breaks encoded as "\n\n" and resolved
        footnotes surrounded by "\u200b" (UTF-8 ZERO WIDTH SPACE).
    """
    # Start by extracting the document contents
    doc, pages = extract_pdf_with_doc(filename)

    # Limit to non-empty pages
    len_before = len(pages)
    pages = [page for page in pages if len(page) > 0]
    if len_before > len(pages):
        print(
            f"Warning: Removed {len_before - len(pages)} empty page(s) from extracted "
            f"text for {filename}!"
        )
    output = ""

    for i, lines in enumerate(pages):
        # Resolve the footnote references and get list of all lines that will be moved
        page = doc[i]
        footnotes, replacements = page_resolve_footnotes(page, lines)
        replacement_lines = _replacements_to_line_numbers(replacements)

        for j, line in enumerate(lines):
            # If we are resolving footnote references, skip lines that have been moved
            if resolve_footnotes and j in replacement_lines:
                continue

            for k, span in enumerate(line):
                if span in footnotes:
                    if resolve_footnotes and _has_replacement(span, replacements):
                        # Footnote ref. before sentence punctuation: Move punctuation
                        if k + 1 < len(line) and line_to_text(
                            [line[k + 1]], clean=True
                        ).startswith("."):
                            output += ". "

                        # Footnote reference after sentence punctuation: Add space
                        if re.search(r"[\.!?]$", output) is not None:
                            output += " "

                        # Add footnote resolution to output
                        output += _replace_footnote_reference(span, replacements, lines)

                        # Add some whitespace after footnote
                        if not output.endswith(" "):
                            output += " "

                    else:
                        # If not resolving footnotes references, drop them from output
                        pass

                else:
                    # Footnote reference before sentence punctuation: already in output
                    if (
                        resolve_footnotes
                        and k > 0
                        and line[k - 1] in footnotes
                        and line_to_text([span], clean=True).startswith(".")
                    ):
                        output += line_to_text([span], clean=True)[1:]

                    # Predecessor was no footnote reference or we're not resolving them
                    else:
                        output += line_to_text([span], clean=True)

            # Add newline after each line
            output += "\n"
        # Double newline between pages
        output += "\f"

    # Encode page boundaries
    output = re.sub(r"\n+", "\n", output).replace("\f", "\n")

    # Drop the tailing two newlines at the end of the output
    return output[:-2]


def _replace_footnote_reference(
    footnote_reference: Dict[str, Any],
    replacements: List[Tuple[dict, List[int]]],
    lines: List[List[Dict[str, Any]]],
) -> str:
    """Replace a footnote reference with the footnote line(s) it refers to.

    Parameters
    ----------
    footnote_reference : Dict[str, Any]
        Footnote reference span.
    replacements : List[Tuple[dict, List[int]]]
        List of all replacements, in the format that is outputted by
        `footnote_resolution.page_resolve_footnotes`.
    lines : List[List[Dict[str, Any]]]
        List of lines on the same page.

    Returns
    -------
    str
        Text that the footnote reference resolves to, surrounded by "\u200b".
    """
    output = ""
    for replacement in _find_replacement(footnote_reference, replacements):
        output += line_to_text(lines[replacement], clean=True)
        output += "\n"

    footnote_reference_text = line_to_text([footnote_reference], clean=True)

    if output.startswith(footnote_reference_text + " "):
        output = output[len(footnote_reference_text) + 1 : -1]
    else:
        output = output[len(footnote_reference_text) : -1]

    return "\u200b" + output + "\u200b"


def _replacements_to_line_numbers(
    replacements: List[Tuple[dict, List[int]]]
) -> List[int]:
    """From a list of replacements, compute the line numbers of all replacements.

    Parameters
    ----------
    replacements : List[Tuple[dict, List[int]]]
        List of all replacements, in the format that is outputted by
        `footnote_resolution.page_resolve_footnotes`.

    Returns
    -------
    List[int]
        Indices of all replacements in the lines of the page.
    """
    return list(chain(*[replacement[1] for replacement in replacements]))


def _has_replacement(
    footnote_reference: Dict[str, Any], replacements: List[Tuple[dict, List[int]]]
) -> bool:
    """Check whether the given footnote reference has a replacement.

    Parameters
    ----------
    footnote_reference : Dict[str, Any]
        Footnote reference span.
    replacements : List[Tuple[dict, List[int]]]
        List of all replacements, in the format that is outputted by
        `footnote_resolution.page_resolve_footnotes`.

    Returns
    -------
    bool
        True if the footnote reference has a replacement, False if not.
    """
    try:
        _find_replacement(footnote_reference, replacements)
        return True
    except IndexError:
        return False


def _find_replacement(
    footnote_reference: Dict[str, Any], replacements: List[Tuple[dict, List[int]]]
) -> List[int]:
    """Extract the replacement of a given footnote reference from all replacements.

    Parameters
    ----------
    footnote_reference : Dict[str, Any]
        Footnote reference span.
    replacements : List[Tuple[dict, List[int]]]
        List of all replacements, in the format that is outputted by
        `footnote_resolution.page_resolve_footnotes`.

    Returns
    -------
    List[int]
        Indices of footnote reference replacements in the lines of the page.

    Raises
    ------
    IndexError
        If the given footnote reference has no replacements.
    """
    for (footnote_reference_, replacement) in replacements:
        if footnote_reference == footnote_reference_:
            return replacement

    raise IndexError("No replacement for given footnote reference found!")
