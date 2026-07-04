"""Orquestra o pipeline completo: pré-processamento -> embeddings -> modelagem
(LDA, BERTopic+HDBSCAN, BERTopic+UMAP+K-Means) -> métricas -> resultados.

Gera em ``results/``:
- metrics_comparison.csv  (média ± desvio-padrão de 3 execuções por config)
- topics_top_words.csv    (top-10 palavras por tópico, para validação qualitativa)
- figures/elbow_curve.png, figures/umap_scatter.png, figures/metrics_comparison.png
- logs/pipeline.log
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

from src.embeddings import embed_texts, get_domain_embedder, get_generic_embedder
from src.logging_utils import setup_logging
from src.metrics import ami_between, ami_vs_category, coherence_cv, retention_rate
from src.modeling import (
    compute_elbow_k,
    run_bertopic_hdbscan_baseline,
    run_bertopic_umap_kmeans,
    run_lda,
)
from src.preprocessing import preprocess_corpus

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "arxiv_cs_raw.jsonl"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_clean.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOG_PATH = RESULTS_DIR / "logs" / "pipeline.log"

SEEDS = [42, 7, 123]
LDA_NUM_TOPICS = 15  # comparável à ordem de grandeza de tópicos do BERTopic


def load_or_build_corpus() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        logger.info("Corpus pré-processado já existe, carregando de %s", PROCESSED_PATH)
        return pd.read_csv(PROCESSED_PATH)
    logger.info("Pré-processando corpus bruto de %s", RAW_PATH)
    return preprocess_corpus(RAW_PATH, PROCESSED_PATH)


def run_lda_experiment(tokenized_docs: list[list[str]]) -> dict:
    logger.info("=== LDA (Gensim) — baseline probabilística ===")
    cv_scores = []
    for seed in SEEDS:
        start = time.time()
        lda, dictionary, corpus, topic_words = run_lda(tokenized_docs, LDA_NUM_TOPICS, seed)
        cv = coherence_cv(topic_words, tokenized_docs, dictionary)
        cv_scores.append(cv)
        logger.info("LDA seed=%d: Cv=%.4f (%.1fs)", seed, cv, time.time() - start)

    return {
        "config": "LDA (Gensim)",
        "embedding": "N/A (bag-of-words)",
        "clustering": "LDA",
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "retention_mean": 1.0,
        "retention_std": 0.0,
        "num_topics": LDA_NUM_TOPICS,
    }, topic_words


def run_bertopic_experiments(
    docs_for_words: list[str],
    embeddings: np.ndarray,
    embedding_label: str,
    categories: list[str],
    dictionary: Dictionary,
    tokenized_docs: list[list[str]],
) -> tuple[list[dict], dict, dict, dict]:
    baseline_cv, baseline_retention, proposed_cv, proposed_retention = [], [], [], []
    ami_between_scores, ami_baseline_cat, ami_proposed_cat = [], [], []
    chosen_ks = []

    last_baseline_words, last_proposed_words = {}, {}
    last_baseline_labels, last_proposed_labels = [], []
    elbow_distortions = None

    for seed in SEEDS:
        logger.info("--- %s | seed=%d ---", embedding_label, seed)

        start = time.time()
        baseline_result = run_bertopic_hdbscan_baseline(docs_for_words, embeddings, seed)
        logger.info("Baseline HDBSCAN: %d tópicos (%.1fs)", baseline_result.num_topics, time.time() - start)

        start = time.time()
        k, distortions = compute_elbow_k(embeddings, seed)
        if elbow_distortions is None:
            elbow_distortions = distortions
        proposed_result = run_bertopic_umap_kmeans(docs_for_words, embeddings, seed, k)
        logger.info("Proposta UMAP+K-Means: k=%d, %d tópicos (%.1fs)", k, proposed_result.num_topics, time.time() - start)

        baseline_cv.append(coherence_cv(baseline_result.topic_words, tokenized_docs, dictionary))
        baseline_retention.append(retention_rate(baseline_result.labels))
        proposed_cv.append(coherence_cv(proposed_result.topic_words, tokenized_docs, dictionary))
        proposed_retention.append(retention_rate(proposed_result.labels))
        chosen_ks.append(k)

        ami_between_scores.append(ami_between(baseline_result.labels, proposed_result.labels))
        ami_baseline_cat.append(ami_vs_category(baseline_result.labels, categories))
        ami_proposed_cat.append(ami_vs_category(proposed_result.labels, categories))

        last_baseline_words, last_proposed_words = baseline_result.topic_words, proposed_result.topic_words
        last_baseline_labels, last_proposed_labels = baseline_result.labels, proposed_result.labels

    rows = [
        {
            "config": f"BERTopic + HDBSCAN (baseline)",
            "embedding": embedding_label,
            "clustering": "HDBSCAN (default)",
            "cv_mean": float(np.mean(baseline_cv)),
            "cv_std": float(np.std(baseline_cv)),
            "retention_mean": float(np.mean(baseline_retention)),
            "retention_std": float(np.std(baseline_retention)),
            "ami_vs_category_mean": float(np.mean(ami_baseline_cat)),
            "num_topics": len(last_baseline_words),
        },
        {
            "config": f"BERTopic + UMAP+K-Means (proposta)",
            "embedding": embedding_label,
            "clustering": "UMAP + K-Means (cotovelo)",
            "cv_mean": float(np.mean(proposed_cv)),
            "cv_std": float(np.std(proposed_cv)),
            "retention_mean": float(np.mean(proposed_retention)),
            "retention_std": float(np.std(proposed_retention)),
            "ami_vs_category_mean": float(np.mean(ami_proposed_cat)),
            "ami_vs_hdbscan_mean": float(np.mean(ami_between_scores)),
            "chosen_k_mean": float(np.mean(chosen_ks)),
            "num_topics": len(last_proposed_words),
        },
    ]

    extra = {
        "elbow_distortions": elbow_distortions,
        "baseline_words": last_baseline_words,
        "proposed_words": last_proposed_words,
        "baseline_labels": last_baseline_labels,
        "proposed_labels": last_proposed_labels,
    }
    return rows, extra


def save_topic_words_csv(all_topic_words: dict[str, dict[int, list[str]]], path: Path) -> None:
    rows = []
    for config_name, topic_words in all_topic_words.items():
        for topic_id, words in topic_words.items():
            rows.append({"config": config_name, "topic_id": topic_id, "top_words": ", ".join(words)})
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_elbow_curve(distortions: list[float], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(list(range(2, 2 + len(distortions))), distortions, marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Distorção (inertia)")
    plt.title("Método do Cotovelo — K-Means sobre espaço reduzido por UMAP")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_umap_scatter(embeddings: np.ndarray, labels: list[int], categories: list[str], seed: int, path: Path) -> None:
    from umap import UMAP

    reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scatter0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=10)
    axes[0].set_title("Clusters (UMAP+K-Means)")

    cat_codes = pd.factorize(categories)[0]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=cat_codes, cmap="tab20", s=10)
    axes[1].set_title("Categoria primária arXiv (proxy)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(df["config"] + "\n" + df["embedding"], df["cv_mean"], yerr=df["cv_std"], capsize=4)
    axes[0].set_title("Coherence Score (Cv)")
    axes[0].tick_params(axis="x", labelrotation=45, labelsize=7)

    axes[1].bar(df["config"] + "\n" + df["embedding"], df["retention_mean"] * 100, yerr=df["retention_std"] * 100, capsize=4)
    axes[1].set_title("Taxa de Retenção de Documentos (%)")
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    setup_logging(LOG_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Iniciando pipeline ===")
    df = load_or_build_corpus()
    logger.info("Corpus: %d documentos", len(df))

    tokenized_docs = [text.split() for text in df["lemma_tokens"]]
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=3, no_above=0.6)

    docs_for_words = df["lemma_tokens"].tolist()
    clean_texts = df["clean_text"].tolist()
    categories = df["primary_category"].tolist()

    all_rows = []
    all_topic_words = {}

    lda_row, lda_words = run_lda_experiment(tokenized_docs)
    all_rows.append(lda_row)
    all_topic_words["LDA"] = lda_words

    logger.info("=== Gerando embeddings genéricos (MiniLM) ===")
    generic_model = get_generic_embedder()
    generic_embeddings = embed_texts(clean_texts, generic_model)

    logger.info("=== Gerando embeddings de domínio (SciBERT) ===")
    domain_model = get_domain_embedder()
    domain_embeddings = embed_texts(clean_texts, domain_model)

    generic_rows, generic_extra = run_bertopic_experiments(
        docs_for_words, generic_embeddings, "Genérico (MiniLM)", categories, dictionary, tokenized_docs
    )
    all_rows.extend(generic_rows)
    all_topic_words["BERTopic+HDBSCAN (MiniLM)"] = generic_extra["baseline_words"]
    all_topic_words["BERTopic+UMAP+KMeans (MiniLM)"] = generic_extra["proposed_words"]

    domain_rows, domain_extra = run_bertopic_experiments(
        docs_for_words, domain_embeddings, "Domínio (SciBERT)", categories, dictionary, tokenized_docs
    )
    all_rows.extend(domain_rows)
    all_topic_words["BERTopic+HDBSCAN (SciBERT)"] = domain_extra["baseline_words"]
    all_topic_words["BERTopic+UMAP+KMeans (SciBERT)"] = domain_extra["proposed_words"]

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    logger.info("Métricas salvas em results/metrics_comparison.csv")

    save_topic_words_csv(all_topic_words, RESULTS_DIR / "topics_top_words.csv")
    logger.info("Palavras-chave por tópico salvas em results/topics_top_words.csv")

    plot_elbow_curve(domain_extra["elbow_distortions"], FIGURES_DIR / "elbow_curve.png")
    plot_umap_scatter(domain_embeddings, domain_extra["proposed_labels"], categories, SEEDS[0], FIGURES_DIR / "umap_scatter.png")
    plot_metrics_comparison(metrics_df, FIGURES_DIR / "metrics_comparison.png")
    logger.info("Figuras salvas em results/figures/")

    logger.info("=== Pipeline finalizado com sucesso ===")


if __name__ == "__main__":
    main()
