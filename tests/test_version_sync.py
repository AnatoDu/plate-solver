"""Ворота согласованности версии проекта.

Единственный источник истины — ``plate_solver.__version__`` (pyproject берёт
версию оттуда динамически). Витрины для цитирования и архивации —
``CITATION.cff``, ``.zenodo.json`` и ``codemeta.json`` — ОБЯЗАНЫ объявлять ту
же версию. Ворота предотвращают дрейф метаданных (исторически витрины отстали
на три минорных релиза: 0.3.1 против 0.6.x перед минтом Zenodo). Поскольку
``.zenodo.json`` читается интеграцией Zenodo в момент публикации GitHub-релиза,
рассогласование означало бы неверную версию в архивной записи и DOI.

Парсинг ``CITATION.cff`` — регуляркой (без зависимости от PyYAML): нужно лишь
одно скалярное поле ``version``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import plate_solver

ROOT = Path(__file__).resolve().parent.parent


def _read_cff_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"^version:\s*(.+?)\s*$", text, re.MULTILINE)
    assert m is not None, "в CITATION.cff нет скалярного поля `version`"
    return m.group(1).strip().strip("\"'")


def _read_json_version(name: str) -> str:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))["version"]


def test_citation_cff_version_matches_package():
    assert _read_cff_version() == plate_solver.__version__


def test_zenodo_json_version_matches_package():
    assert _read_json_version(".zenodo.json") == plate_solver.__version__


def test_codemeta_json_version_matches_package():
    assert _read_json_version("codemeta.json") == plate_solver.__version__
