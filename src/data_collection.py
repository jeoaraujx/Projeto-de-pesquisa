"""Coleta de artigos científicos de Computação via API do arXiv.

Usa apenas a biblioteca padrão (urllib + xml.etree) para não depender do
ambiente de ML — permite rodar a coleta em qualquer Python 3.x disponível.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

ARXIV_API_URL = "http://export.arxiv.org/api/query"
USER_AGENT = "projeto-pesquisa-uneb-bertopic/1.0 (mailto:jeostonjunior@gmail.com)"

DEFAULT_CATEGORIES = [
    "cs.AI",  # Inteligência Artificial
    "cs.CL",  # Processamento de Linguagem Natural
    "cs.CV",  # Visão Computacional
    "cs.LG",  # Aprendizado de Máquina
    "cs.CR",  # Criptografia e Segurança
    "cs.SE",  # Engenharia de Software
    "cs.DB",  # Bancos de Dados
    "cs.DC",  # Computação Distribuída
]

logger = logging.getLogger(__name__)


@dataclass
class Article:
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: str
    published: str
    query_category: str


def _fetch_page(category: str, start: int, max_results: int) -> bytes:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _parse_entries(xml_bytes: bytes, query_category: str) -> list[Article]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    articles: list[Article] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.findtext(f"{ATOM_NS}id", default="").split("/abs/")[-1]
        title = " ".join(entry.findtext(f"{ATOM_NS}title", default="").split())
        abstract = " ".join(entry.findtext(f"{ATOM_NS}summary", default="").split())
        published = entry.findtext(f"{ATOM_NS}published", default="")

        primary_el = entry.find(f"{ARXIV_NS}primary_category")
        primary_category = primary_el.get("term") if primary_el is not None else query_category

        categories = ",".join(
            cat.get("term", "") for cat in entry.findall(f"{ATOM_NS}category")
        )

        if not arxiv_id or not abstract:
            continue

        articles.append(
            Article(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                primary_category=primary_category,
                categories=categories,
                published=published,
                query_category=query_category,
            )
        )
    return articles


def collect_category(category: str, target_count: int, page_size: int = 100, delay: float = 3.0) -> list[Article]:
    """Coleta artigos de uma subárea, paginando a API do arXiv."""
    collected: list[Article] = []
    start = 0
    while len(collected) < target_count:
        remaining = target_count - len(collected)
        page = min(page_size, remaining)
        logger.info("Coletando %s: start=%d, page_size=%d", category, start, page)
        xml_bytes = _fetch_page(category, start, page)
        entries = _parse_entries(xml_bytes, category)
        if not entries:
            logger.warning("Sem mais resultados para %s em start=%d", category, start)
            break
        collected.extend(entries)
        start += page
        time.sleep(delay)
    return collected[:target_count]


def collect_dataset(
    categories: list[str] = None,
    per_category: int = 130,
    target_total: int = 900,
    delay: float = 3.0,
) -> list[Article]:
    """Coleta artigos de várias subáreas de CS e deduplica por arxiv_id.

    Coleta um pouco acima do alvo por categoria (buffer) para compensar
    artigos cross-listed que aparecem em mais de uma consulta, depois
    deduplica e corta para o total alvo mantendo distribuição balanceada.
    """
    categories = categories or DEFAULT_CATEGORIES
    all_articles: dict[str, Article] = {}

    for category in categories:
        articles = collect_category(category, per_category, delay=delay)
        for article in articles:
            all_articles.setdefault(article.arxiv_id, article)
        logger.info("Total acumulado após %s: %d artigos únicos", category, len(all_articles))

    deduped = list(all_articles.values())

    if len(deduped) <= target_total:
        return deduped

    # Corta mantendo distribuição balanceada entre primary_category.
    by_primary: dict[str, list[Article]] = {}
    for article in deduped:
        by_primary.setdefault(article.primary_category, []).append(article)

    per_bucket = max(1, target_total // len(by_primary))
    trimmed: list[Article] = []
    for bucket in by_primary.values():
        trimmed.extend(bucket[:per_bucket])

    return trimmed[:target_total]


def save_jsonl(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "raw" / "arxiv_cs_raw.jsonl"

    articles = collect_dataset()
    save_jsonl(articles, output_path)

    logger.info("Coleta finalizada: %d artigos salvos em %s", len(articles), output_path)


if __name__ == "__main__":
    main()
