"""Geração de embeddings genéricos (MiniLM) e de domínio específico (SciBERT).

Ambos expostos como ``SentenceTransformer`` para uso direto no BERTopic.
SciBERT não é nativamente um modelo de sentence-embeddings (é um BERT de
domínio científico treinado para masked-LM), então é envolto com mean pooling
via os módulos ``Transformer`` + ``Pooling`` do próprio sentence-transformers
— abordagem padrão para adaptar BERT genéricos/de domínio a embeddings de
documento.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer, models

GENERIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOMAIN_MODEL_NAME = "allenai/scibert_scivocab_uncased"


def get_generic_embedder() -> SentenceTransformer:
    return SentenceTransformer(GENERIC_MODEL_NAME)


def get_domain_embedder() -> SentenceTransformer:
    transformer = models.Transformer(DOMAIN_MODEL_NAME, max_seq_length=256)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling])


def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
