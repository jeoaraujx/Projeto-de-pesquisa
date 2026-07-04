"""Configuração de logging do pipeline (console + arquivo), para rastreabilidade
e reprodutibilidade das execuções (parâmetros, tempos, erros)."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
