"""Module containing some utilities for the embedding_clustering.ipynb notebook.
Caution: Undocumented and without type hints (as it is unused experimental code)."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile
from sklearn import cluster
from sklearn.manifold import TSNE


def get_tm(texts):
    # Allow for import of dataloader from pdf_extraction folder
    for path in ["../../04_dataset_access/", "../../05_topic_modeling/"]:
        module_path = os.path.abspath(os.path.join(path))
        if module_path not in sys.path:
            sys.path.append(module_path)

    import topic_models  # type: ignore

    with ZipFile("../../06_explorer/explorer/data/nmf_page.indepth") as zip:
        with zip.open("pipeline.pkl") as f:
            pipeline = pickle.load(f)
    topics = _get_topics()

    tm = pipeline.transform(texts["tokenized"])
    H_norm = pd.DataFrame(
        tm / np.sum(tm, axis=1).reshape(-1, 1), columns=topics
    ).assign(
        dominant_topic=lambda df: list(
            map(lambda n: topics[n], np.argmax(df[topics].values, axis=1))
        )
    )
    return pd.concat((texts.reset_index(drop=True), H_norm), axis=1)


def get_tsne_embedding(texts, embeddings, aspect_type, aspect_subtype=""):
    filter_ = (texts["aspect_type"] == aspect_type) & (
        texts["aspect_subtype"] == aspect_subtype
    )
    embeddings_ = embeddings[filter_]
    texts_ = texts.loc[filter_].reset_index(drop=True).copy()

    tsne = TSNE(n_components=2, n_jobs=-1, random_state=42)
    embedding = tsne.fit_transform(embeddings_)
    return pd.concat((texts_, pd.DataFrame(embedding, columns=["PC1", "PC2"])), axis=1)


def add_clustering(df, embeddings=None, texts=None):
    sentiments = ["positive", "negative", "neutral"]
    dfs = [df.query(f"sentiment == @sentiment").copy() for sentiment in sentiments]

    if embeddings is None:
        data = [df[["PC1", "PC2"]].values for df in dfs]

    else:
        if texts is None:
            raise ValueError("When using embeddings, texts can not be None!")
        data = [
            embeddings[
                texts[["sentence_index"]]
                .reset_index(names="n")
                .merge(df[["sentence_index"]])["n"]
            ]
            for df in dfs
        ]

    clusterings = [cluster.DBSCAN(metric="cosine") for _ in data]
    data = [
        df.assign(cluster=clustering.fit_predict(data_) if len(data_) > 0 else data_)
        for df, data_, clustering in zip(dfs, data, clusterings)
    ]
    return pd.concat(data).sort_values("sentence_index")


def _get_topics():
    with ZipFile("../../06_explorer/explorer/data/nmf_page.indepth") as zip:
        with zip.open("H.arrow") as f:
            return pd.read_feather(f).columns[1:]
