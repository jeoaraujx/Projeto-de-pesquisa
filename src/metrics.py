"""Métricas de validação: Coherence Score (Cv), Taxa de Retenção de Documentos e
Informação Mútua Ajustada (AMI).

Usamos AMI (Adjusted Mutual Information) em vez de MI bruta porque a MI cresce
artificialmente com o número de clusters e não é corrigida ao acaso — o que
enviesaria a comparação entre configurações que produzem quantidades diferentes
de tópicos (ex.: HDBSCAN vs K-Means). AMI é o refinamento metodológico adotado
aqui sobre a métrica de MI citada na Parcial 1.
"""
from __future__ import annotations

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import adjusted_mutual_info_score


def coherence_cv(topic_words: dict[int, list[str]], tokenized_docs: list[list[str]], dictionary: Dictionary) -> float:
    """Coherence Score (c_v) via Gensim, a partir das top-N palavras de cada tópico."""
    topics = [words for words in topic_words.values() if words]
    if not topics:
        return float("nan")
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    return coherence_model.get_coherence()


def retention_rate(labels: list[int]) -> float:
    """Fração de documentos classificados em um tópico válido (não outlier, label != -1)."""
    if not labels:
        return float("nan")
    valid = sum(1 for label in labels if label != -1)
    return valid / len(labels)


def ami_between(labels_a: list[int], labels_b: list[int]) -> float:
    return adjusted_mutual_info_score(labels_a, labels_b)


def ami_vs_category(labels: list[int], categories: list[str]) -> float:
    category_codes = {cat: idx for idx, cat in enumerate(sorted(set(categories)))}
    encoded = [category_codes[cat] for cat in categories]
    return adjusted_mutual_info_score(labels, encoded)
