"""Pré-processamento textual: limpeza via regex, lematização (spaCy) e n-grams (gensim).

Duas trilhas de texto são produzidas por documento:
- ``clean_text``: texto higienizado (sem artefatos de LaTeX/URLs), preservando
  estrutura de sentença natural — usado para gerar os embeddings, pois modelos
  transformer captam melhor texto em linguagem natural.
- ``lemma_tokens``: tokens lematizados, minúsculos, sem stopwords, com bigramas/
  trigramas técnicos unidos (ex.: "neural_network") — usado no c-TF-IDF do
  BERTopic (via CountVectorizer) e no dicionário/corpus do LDA (Gensim).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LATEX_INLINE_MATH = re.compile(r"\$[^$]*\$")
URL_PATTERN = re.compile(r"https?://\S+")
NON_ALPHA = re.compile(r"[^A-Za-z\s\-]")
MULTI_SPACE = re.compile(r"\s+")

MIN_TOKEN_LEN = 3


def basic_clean(text: str) -> str:
    """Remove artefatos de LaTeX/URLs/símbolos e normaliza espaços, preservando o texto natural."""
    text = LATEX_INLINE_MATH.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = NON_ALPHA.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    return text


def _load_spacy_pipeline():
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_documents(texts: list[str]) -> list[list[str]]:
    """Tokeniza e lematiza em lote via spaCy, removendo stopwords/pontuação/tokens curtos."""
    nlp = _load_spacy_pipeline()
    tokenized: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=64):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop and len(token.lemma_) >= MIN_TOKEN_LEN
        ]
        tokenized.append(tokens)
    return tokenized


def build_ngrams(tokenized_docs: list[list[str]], min_count: int = 5, threshold: int = 10) -> list[list[str]]:
    """Une termos compostos frequentes (bigramas e trigramas) via gensim Phrases."""
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

    bigram = Phrases(tokenized_docs, min_count=min_count, threshold=threshold, connector_words=ENGLISH_CONNECTOR_WORDS)
    trigram = Phrases(bigram[tokenized_docs], min_count=min_count, threshold=threshold, connector_words=ENGLISH_CONNECTOR_WORDS)
    return [trigram[bigram[doc]] for doc in tokenized_docs]


def preprocess_corpus(raw_jsonl_path: Path, output_csv_path: Path) -> pd.DataFrame:
    df = pd.read_json(raw_jsonl_path, lines=True)
    df["clean_text"] = (df["title"] + ". " + df["abstract"]).apply(basic_clean)

    tokenized = lemmatize_documents(df["clean_text"].tolist())
    tokenized = build_ngrams(tokenized)
    df["lemma_tokens"] = [" ".join(tokens) for tokens in tokenized]

    # Remove documentos que ficaram vazios após a limpeza (raro, mas possível).
    df = df[(df["clean_text"].str.len() > 0) & (df["lemma_tokens"].str.len() > 0)].reset_index(drop=True)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "arxiv_cs_raw.jsonl"
    processed_path = project_root / "data" / "processed" / "corpus_clean.csv"
    df = preprocess_corpus(raw_path, processed_path)
    print(f"Corpus pré-processado: {len(df)} documentos salvos em {processed_path}")


if __name__ == "__main__":
    main()
