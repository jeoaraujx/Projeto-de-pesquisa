"""Modelagem de tópicos: LDA (baseline probabilística), BERTopic+HDBSCAN (baseline
padrão) e BERTopic+UMAP+K-Means com k pelo Método do Cotovelo (proposta).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from hdbscan import HDBSCAN
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)

UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
K_RANGE = range(2, 31)


@dataclass
class TopicRunResult:
    labels: list[int]
    topic_words: dict[int, list[str]]
    num_topics: int
    chosen_k: int | None = None


def build_vectorizer() -> CountVectorizer:
    # lemma_tokens já vem lematizado, sem stopwords e com n-grams unidos por "_".
    return CountVectorizer(ngram_range=(1, 2), min_df=3)


def run_lda(tokenized_docs: list[list[str]], num_topics: int, seed: int) -> tuple[LdaModel, Dictionary, list, dict[int, list[str]]]:
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.6)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=10,
        chunksize=100,
    )

    topic_words = {
        topic_id: [word for word, _ in lda.show_topic(topic_id, topn=10)]
        for topic_id in range(num_topics)
    }
    return lda, dictionary, corpus, topic_words


def _extract_topic_words(topic_model: BERTopic, top_n: int = 10) -> dict[int, list[str]]:
    topic_words = {}
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            continue
        topic_words[topic_id] = [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]
    return topic_words


def run_bertopic_hdbscan_baseline(docs_for_words: list[str], embeddings: np.ndarray, seed: int) -> TopicRunResult:
    """Configuração padrão do BERTopic: UMAP + HDBSCAN (agrupamento por densidade)."""
    umap_model = UMAP(random_state=seed, **UMAP_PARAMS)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom")
    vectorizer = build_vectorizer()

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs_for_words, embeddings)

    topic_words = _extract_topic_words(topic_model)
    num_topics = len(topic_words)
    return TopicRunResult(labels=list(topics), topic_words=topic_words, num_topics=num_topics)


def compute_elbow_k(embeddings: np.ndarray, seed: int, k_range=K_RANGE) -> tuple[int, list[float]]:
    """Reduz via UMAP e determina k automaticamente pelo ponto de inflexão da distorção (inertia)."""
    umap_model = UMAP(random_state=seed, **UMAP_PARAMS)
    reduced = umap_model.fit_transform(embeddings)

    distortions = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kmeans.fit(reduced)
        distortions.append(kmeans.inertia_)

    knee = KneeLocator(list(k_range), distortions, curve="convex", direction="decreasing")
    chosen_k = knee.knee or list(k_range)[len(list(k_range)) // 3]
    logger.info("Elbow method: k escolhido=%d (seed=%d)", chosen_k, seed)
    return chosen_k, distortions


def run_bertopic_umap_kmeans(docs_for_words: list[str], embeddings: np.ndarray, seed: int, k: int) -> TopicRunResult:
    """Configuração proposta: UMAP + K-Means, com k definido pelo Método do Cotovelo."""
    umap_model = UMAP(random_state=seed, **UMAP_PARAMS)
    kmeans_model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    vectorizer = build_vectorizer()

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs_for_words, embeddings)

    topic_words = _extract_topic_words(topic_model)
    num_topics = len(topic_words)
    return TopicRunResult(labels=list(topics), topic_words=topic_words, num_topics=num_topics, chosen_k=k)
