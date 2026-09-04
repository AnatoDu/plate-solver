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


def test_codemeta_download_url_points_at_current_tag():
    """``downloadUrl`` codemeta обязан указывать на тег ТЕКУЩЕЙ версии.

    В волне 0.8.0 поле осталось на архиве v0.7.0: ворота сверяли только
    ``version``, и харвестеру (Zenodo/CodeMeta) достался бы архив прошлого
    релиза при верной версии в записи (аудит 0.8.0).
    """
    url = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))["downloadUrl"]
    assert f"v{plate_solver.__version__}" in url, url


def test_citation_cff_version_matches_package():
    assert _read_cff_version() == plate_solver.__version__


def test_zenodo_json_version_matches_package():
    assert _read_json_version(".zenodo.json") == plate_solver.__version__


def test_codemeta_json_version_matches_package():
    assert _read_json_version("codemeta.json") == plate_solver.__version__


def test_changelog_top_entry_matches_version_and_dates():
    """Верхняя запись CHANGELOG — ТЕКУЩАЯ версия и та же дата, что в витринах.

    Ворота версии сверяли ``version`` в трёх файлах, но не CHANGELOG: запись
    могла остаться «не выпущено» или нести чужую дату, а витрины — свою
    (в волне 0.8.0 даты пришлось править вручную).
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entries = re.findall(r"^## \[(\d+\.\d+\.\d+)\][^\n]*?—\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
                         text, re.MULTILINE)
    assert entries, "в CHANGELOG нет ни одной датированной записи вида '## [X.Y.Z] — ГГГГ-ММ-ДД'"
    version, date = entries[0]
    assert version == plate_solver.__version__, (
        f"верхняя запись CHANGELOG — {version}, версия пакета — {plate_solver.__version__}")
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"^date-released:\s*'?([0-9\-]+)'?\s*$", cff, re.MULTILINE)
    assert m is not None and m.group(1) == date, (
        f"date-released в CITATION.cff = {m and m.group(1)}, в CHANGELOG = {date}")
    meta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    assert meta["datePublished"] == date, (
        f"datePublished в codemeta.json = {meta['datePublished']}, в CHANGELOG = {date}")


def test_ci_matrix_matches_declared_python_support():
    """Матрица CI = классификаторы pyproject = ``requires-python``.

    Расхождение означало бы, что CI молча не проверяет версию, которую пакет
    обещает поддерживать (или наоборот — тратит минуты на неподдерживаемую).
    """
    import tomllib

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    m = re.search(r"python-version:\s*\[([^\]]+)\]", ci)
    assert m is not None, "в ci.yml не найдена матрица python-version"
    matrix = {v.strip().strip('"\'') for v in m.group(1).split(",")}

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = {c.rsplit(" ", 1)[-1] for c in pyproject["project"]["classifiers"]
                   if c.startswith("Programming Language :: Python :: 3.")}
    assert matrix == classifiers, (
        f"матрица CI {sorted(matrix)} ≠ классификаторы {sorted(classifiers)}")

    requires = pyproject["project"]["requires-python"]
    lowest = min(matrix, key=lambda v: tuple(int(x) for x in v.split(".")))
    assert requires.replace(" ", "") == f">={lowest}", (
        f"requires-python = {requires}, а младшая нога матрицы — {lowest}")
