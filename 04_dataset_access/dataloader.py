"""Module for easily accessing a dataset of already converted attachments, now in text
file format."""

import os
import re
import spacy
import fasttext
import numpy as np
import ftlangdetect
import pandas as pd
import multiprocessing
from typing import List, Any, Union
from itertools import chain, repeat


class Dataloader:
    """Data loader that can read an existing dataset, tokenize it, and add additional
    context.
    """

    def __init__(
        self,
        chunk_strategy: str = "page",
        detect_lang: bool = True,
        limit_english: bool = True,
        tokenize: bool = True,
        full: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """Initialize a new dataloader.

        Parameters
        ----------
        chunk_strategy : str, optional
            Whether and how to split each document into smaller parts. Available options
            are "line", "sentence", "page" and "document", by default "page".
        detect_lang : bool, optional
            Whether to detect the language of the text chunks, by default True. Adds a
            "language_detected" column to the output dataframes. limit_english set to
            True will overwrite this argument.
        limit_english : bool, optional
            Limit tokenization to English texts, by default True. This should probably
            be kept on, as tokenizers and lemmatization differ between languages. Adds a
            "language_detected" column to the output dataframes.
        tokenize : bool, optional
            Whether to tokenize (and lemmatize) the documents, by default True. Adds a
            "tokenized" column to the output dataframes.
        full : bool, optional
            Whether to include the tokenized text and 'None' lemmas, by default False.
            Only effective if tokenize is True.
        n_jobs : int, optional
            Degree of multiprocessing as the number of processes to use, by default -1,
            i.e. using as many processes as there are CPU cores.
        """
        self.chunk_strategy = chunk_strategy
        if self.chunk_strategy == "sentence":
            self.sentencizer = Sentencizer(n_jobs=n_jobs)

        self.detect_lang = detect_lang
        self.limit_english = limit_english
        if limit_english or detect_lang:
            self.langdetector = Langdetector(n_jobs=n_jobs)

        self.tokenize = tokenize
        if tokenize:
            self.tokenizer = Tokenizer(n_jobs=n_jobs, full=full)
        self.full = full

    def from_folder(
        self, folder: str, context: Union[str, pd.DataFrame] = ""
    ) -> pd.DataFrame:
        """Read a dataset from folder and add context from a dataframe.

        Parameters
        ----------
        folder : str
            Folder the dataset (containing .txt files) sits in. For example
            "ai_act/attachments/".
        context : Union[str, pd.DataFrame], optional
            Context to add to the dataset, either as a dataframe or the filepath to a
            CSV file, by default "".

        Returns
        -------
        pd.DataFrame
            DataFrame containing "id" and "text" columns. Additional columns depending
            on initialization options.
        """
        # Find all the .txt files in specified folder
        txt_files = [
            folder + f for f in sorted(os.listdir(folder)) if f.endswith(".txt")
        ]

        # Get ids and file content for .txt files
        ids = map(self._id_from_filename, txt_files)
        texts = map(self._txt_from_file, txt_files)

        # Transform file content and their ids into dataframe
        texts = pd.DataFrame(zip(ids, texts), columns=["id", "text"])
        texts["id"] = texts["id"].astype(int)

        # If context is specified as a filename, read it in
        if type(context) is str and context != "":
            context = pd.read_csv(context)
        if type(context) is not pd.DataFrame:
            raise TypeError(
                "Argument 'context' needs to be either the filename of a dataframe "
                + f"or a dataframe! (Passed: {type(context)})"
            )

        # Put feedback string into correct format (double newlines encode page breaks)
        context["feedback"] = (
            context["feedback"]
            .str.replace(r"\n+", "\n", regex=True)
            .str.replace(r"\s+$", "", regex=True)
        )

        # Merge attachment texts with context dataframe
        texts = pd.merge(texts, context)
        texts["text"] = texts["feedback"] + "\n\n" + texts["text"]

        # Add those that only have feedback text and no attachment
        no_pdf = context[~context["id"].isin(texts["id"])].copy()
        no_pdf["text"] = no_pdf["feedback"]
        texts = (
            pd.concat((texts, no_pdf))
            .sort_values("id", ascending=False)
            .reset_index(drop=True)
        )

        # Chunk the text
        texts["text"] = texts["text"].map(self._chunk_text)
        texts = texts.explode("text").dropna(subset=["text"]).reset_index(drop=True)

        # Compute the source of each chunk (i.e. feedback or attachment)
        texts["text_cumlen"] = texts.groupby("id", group_keys=False)["text"].apply(
            lambda x: (x + " ").astype(str).str.len().cumsum() - 1
        )
        texts["source"] = np.where(
            texts["text_cumlen"] > texts["feedback"].str.len(),
            "attachment" if self.chunk_strategy != "document" else "both",
            "feedback",
        )
        texts = texts.drop(columns="text_cumlen")

        if self.tokenize:
            # To limit to English texts, first detect the language texts are in
            if self.limit_english:
                langs = self.langdetector.transform(texts["text"]).to_frame(
                    "language_detected"
                )
                texts = pd.concat((texts, langs), axis=1)
            else:
                texts["language_detected"] = "en"  # Set language to en to unify code

            # Tokenize the english texts and recombine with rest of dataframe
            tokenized = self.tokenizer.transform(
                texts.query("language_detected == 'en'")["text"]
            )
            remaining = pd.Series(
                repeat(None, len(texts.query("language_detected != 'en'")))
            )
            tokenized = pd.concat((tokenized, remaining), ignore_index=True, axis=0)
            texts = pd.concat(
                (
                    texts.query("language_detected == 'en'"),
                    texts.query("language_detected != 'en'"),
                ),
                ignore_index=True,
            )  # type: ignore
            texts["tokenized"] = tokenized

            # Remove language column if only fake
            if not self.limit_english:
                texts.drop(columns=["language_detected"])

        if (
            # Detect language if we have skipped tokenization and therefore also
            # language detection or ...
            (not self.tokenize and self.detect_lang)
            or
            # if we did tokenization but still skipped language detection
            (self.tokenize and not self.limit_english and self.detect_lang)
        ):
            texts["language_detected"] = self.langdetector.transform(texts["text"])

        return texts

    def _id_from_filename(self, filename: str) -> str:
        """Use a regex pattern to extract the id from a filename.

        Parameters
        ----------
        filename : str
            Filename, for example "ai_act/attachments/<id>.txt".

        Returns
        -------
        str
            Id of the feedback submission.
        """
        return re.findall(r"(\w+).txt$", filename)[0]

    def _txt_from_file(self, filename: str) -> str:
        """Read a file and return the text contained in it.

        Parameters
        ----------
        filename : str
            Filename, for example "ai_act/attachments/<id>.txt".

        Returns
        -------
        str
            File content as a string.
        """
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk a text using the chunking strategy set.

        Parameters
        ----------
        text : str
            Arbitrary text as a string. Should follow the text extraction style of
            encoding page breaks as double newlines.

        Returns
        -------
        List[str]
            List of chunks.
        """
        if self.chunk_strategy == "document":
            return [text]
        elif self.chunk_strategy == "page":
            return [s for s in text.split("\n\n") if len(s) > 0]
        elif self.chunk_strategy == "line":
            return [s for s in text.split("\n") if len(s) > 0]
        elif self.chunk_strategy == "sentence":
            return self.sentencizer.transform(pd.Series([text])).explode().to_list()
        else:
            raise ValueError(f"Chunking strategy {self.chunk_strategy} unknown!")


class TextPrepocessor(object):
    """Superclass for multiprocessed text processors."""

    def __init__(self, n_jobs=-1) -> None:
        """Generic initialization of a TextProcessor.

        Parameters
        ----------
        n_jobs : int, optional
            Number of processes to use, by default -1 representing the number of CPU
            cores.
        """
        if n_jobs <= 0:
            print(f"As n_jobs={n_jobs} <= 0, enabling multiprocessing with ", end="")
            n_jobs = multiprocessing.cpu_count()
            print(f"{n_jobs} cores!")
        self.n_jobs = n_jobs

    def transform(self, X: Union[list, pd.Series]) -> pd.Series:
        """Transforms each document in the given corpus using the _transform method.
        Uses multiprocessing if n_jobs > 1 (or <= 0 during object initialization).

        Parameters
        ----------
        X : Union[list, pd.Series]
            Corpus of documents. Each document should be a list of strings.

        Returns
        -------
        pd.Series
            Pandas series of transformed documents.
        """
        parts = self._partition_X(X)

        # If specified, use a multiprocessing pool to process the input
        if self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            transformed_X = pool.map(self._transform, parts)
            del pool  # Make sure the pool gets garbage cleaned
        else:
            transformed_X = map(self._transform, parts)

        # Combine the individual parts into one
        transformed_X = chain(*transformed_X)
        return pd.Series(transformed_X)

    def _transform(self, X: Any) -> Any:
        """Method to transform corpus partitions."""
        # Should be implemented by subclasses
        raise NotImplementedError()

    def _partition_X(self, X: Union[list, pd.Series]) -> list:
        """Transforms the input into a Pandas series if a list and splits it into
        n_jobs equally sized parts to be processed in parallel.

        Parameters
        ----------
        X : Union[list, pd.Series]
            Corpus, i.e. iterable over documents.

        Returns
        -------
        list
            List of corpus partitions, each a Pandas series.
        """
        # Compute length when splitting X into n_jobs many equally sized parts
        part_length = len(X) // self.n_jobs

        # Unify datatype into pandas series
        if not type(X) is pd.Series:
            X = pd.Series(X)

        # Split X into equally sized parts and return it
        return [
            X.iloc[i * part_length : (i + 1) * part_length]
            if i < self.n_jobs - 1
            else X.iloc[i * part_length :]  # Last part might have different length
            for i in range(self.n_jobs)
        ]


class Tokenizer(TextPrepocessor):
    """TextProcessor that tokenizes and lemmatizes its input."""

    def __init__(self, n_jobs: int = -1, full: bool = False) -> None:
        """Initialize a new tokenizer.

        Parameters
        ----------
        n_jobs : int, optional
            Number of processes to use, by default -1 representing the number of CPU
            cores.
        full : bool, optional
            Whether to include the tokenized text and 'None' lemmas, by default False.
        """
        # Create a Spacy NLP object only once (as loading takes a second)
        self.nlp = spacy.load("en_core_web_sm")
        self.full = full

        super().__init__(n_jobs=n_jobs)

    def _transform(self, texts: pd.Series) -> pd.Series:
        """Transforms a series of texts into their tokenized and lemmatized forms.

        Parameters
        ----------
        texts : pd.Series
            Series of documents, each as a string.

        Returns
        -------
        pd.Series
            Series of tokenized and lemmatized documents, each as a list of strings, if
            self.full is False. Otherwise series of tuples, the first entry with the
            tokenized text, the second containing each tokens lemma (potentially None if
            filtered out).
        """
        # Replace some of the characters we inserted manually and remove some whitespace
        texts = texts.str.replace("[\n\u200b]", " ", regex=True).fillna("")
        texts = texts.str.replace(r"\s{2,}", " ", regex=True).fillna("")

        # Limit the batch_size depending on text size - If left alone, Spacy will set
        # this too high and end up using 100% RAM and causing crashes
        batch_size = 25 if texts.str.len().mean() > 5000 else 250

        tokenized_texts, tokenized_lemmas = [], []

        # Process using nlp.pipe (faster)
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            if self.full:
                tokenized_texts.append([t.text_with_ws for t in doc])
            tokenized_lemmas.append(
                [
                    # Use lowercase lemmas for tokenization
                    t.lemma_.lower()
                    # Remove punctuation, email addresses, or URLs
                    if not (t.is_punct or t.like_email or t.like_url)
                    # Limit to numbers or words
                    and (t.is_alpha or t.like_num)
                    # Use None to denote tokens that were filtered out
                    else None
                    for t in doc
                ]
            )

        # Convert to Pandas series
        if self.full:
            return pd.Series(
                zip(tokenized_texts, tokenized_lemmas), dtype="object"
            )  # dtype suppresses pandas warnings

        # Filter out None if we are not returning every token
        tokenized_lemmas = [
            [lemma for lemma in text if lemma is not None] for text in tokenized_lemmas
        ]
        return pd.Series(
            tokenized_lemmas, dtype="object"
        )  # dtype suppresses pandas warnings


class Sentencizer(TextPrepocessor):
    """TextProcessor that sentencizes its input."""

    def __init__(self, n_jobs: int = -1) -> None:
        """Initialize a new sentencizer.

        Parameters
        ----------
        n_jobs : int, optional
            Number of processes to use, by default -1 representing the number of CPU
            cores.
        """
        # Create a Spacy NLP object only once (as loading takes a second)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        super().__init__(n_jobs=n_jobs)

    def _transform(self, texts: pd.Series) -> pd.Series:
        """Transforms a series of texts into their sentencizer forms.

        Parameters
        ----------
        texts : pd.Series
            Series of documents, each as a string.

        Returns
        -------
        pd.Series
            Series of sentencized documents, each as a list of strings.
        """
        # Replace some of the characters we inserted manually and remove some whitespace
        texts = texts.str.replace("[\n\u200b]", " ", regex=True)
        texts = texts.str.replace(r"\s{2,}", " ", regex=True)

        # Limit the batch_size depending on text size - If left alone, Spacy will set
        # this too high and end up using 100% RAM and causing crashes
        batch_size = 25 if texts.str.len().mean() > 5000 else 250

        sentencized_texts = []

        # Process using nlp.pipe (faster)
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            sentencized_texts.append([s.text for s in doc.sents])

        sentencized_texts = pd.Series(
            sentencized_texts, dtype="object"
        )  # dtype suppresses pandas warnings

        return sentencized_texts


class Langdetector(TextPrepocessor):
    """TextProcessor that detects the language a text is written in."""

    def _detect(self, x: str) -> str:
        """Detects the language of a string.

        Parameters
        ----------
        x : str
            Arbitrary string.

        Returns
        -------
        str
            Language of the string, e.g. "en", "de", etc.
        """
        # https://stackoverflow.com/a/66401601
        fasttext.FastText.eprint = lambda x: None

        # Ftlangdetect doesn't like newlines, so remove them
        return str(ftlangdetect.detect(x.replace("\n", " "))["lang"])

    def _transform(self, X: pd.Series) -> pd.Series:
        """Detects the languages of a series of texts.

        Parameters
        ----------
        texts : pd.Series
            Series of documents, each as a string.

        Returns
        -------
        pd.Series
            Series of detected languages, e.g. "en", "de", etc.
        """
        return X.map(self._detect)
