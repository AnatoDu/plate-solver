r"""Ворота провенанс-блока верификации — codemeta.json vvprov: (v0.7.0).

Машиночитаемый провенанс верификации (vvprov): блок
``vvprov:verification`` в codemeta.json переносит слой
ИСТОЛКОВАНИЯ проверок в архивную запись. Свидетельство обязано быть
САМОПРОВЕРЯЕМЫМ — этот тест делает блок вычисляемой величиной:

* каждая запись несёт все семь полей акта + benchmark;
* каждый ``vvprov:artifact`` указывает на СУЩЕСТВУЮЩИЕ файлы репозитория
  (воспроизводящий сценарий/тест) — запись не может «повиснуть»;
* ``vvprov:softwareVersion`` синхронизирован с ``__version__``
  (расширение test_version_sync на провенанс);
* статусы — из фиксированного словаря vvprov (в допуске / расхождение / вне
  допуска / унаследована), ``validationStatus`` зафиксирован с источником.
"""

from __future__ import annotations

import json
from pathlib import Path

import plate_solver

_ROOT = Path(__file__).resolve().parents[1]
_FIELDS = ("vvprov:benchmark", "vvprov:referenceType", "vvprov:vvType",
           "vvprov:metric", "vvprov:achieved", "vvprov:status",
           "vvprov:softwareVersion", "vvprov:artifact")
_STATUS_PREFIXES = ("within tolerance", "model discrepancy",
                    "out of tolerance", "inherited")


def _codemeta() -> dict:
    return json.loads((_ROOT / "codemeta.json").read_text(encoding="utf-8"))


def test_vvprov_namespace_declared():
    """@context несёт пространство имён vvprov (расширение CodeMeta)."""
    cm = _codemeta()
    ctx = cm["@context"]
    assert isinstance(ctx, list)
    assert any(isinstance(c, dict) and "vvprov" in c for c in ctx)


def test_vvprov_records_complete_and_versioned():
    """Каждая запись: все поля акта непусты, версия = __version__."""
    cm = _codemeta()
    records = cm["vvprov:verification"]
    assert len(records) >= 10                       # адресный перечень, не заглушка
    for r in records:
        for f in _FIELDS:
            assert isinstance(r.get(f), str) and r[f].strip(), (f, r)
        assert r["vvprov:softwareVersion"] == plate_solver.__version__
        assert any(r["vvprov:status"].startswith(p) for p in _STATUS_PREFIXES), r


def test_vvprov_artifacts_exist():
    """Воспроизводящие артефакты записей существуют в репозитории."""
    cm = _codemeta()
    for r in cm["vvprov:verification"]:
        for part in r["vvprov:artifact"].split(";"):
            path = part.strip().split("::")[0]
            assert (_ROOT / path).is_file(), path


def test_vvprov_validation_status_with_source():
    """Статус валидации зафиксирован явно и несёт источник модели."""
    cm = _codemeta()
    vs = cm["vvprov:validationStatus"]
    assert vs.startswith("inherited")
    assert "1999" in vs                             # источник теории назван


def test_reference_publication_qualified():
    """Квалифицированная связь записи с публикациями (DOI)."""
    cm = _codemeta()
    pubs = cm["referencePublication"]
    assert isinstance(pubs, list) and pubs
    for p in pubs:
        assert p["@type"] == "ScholarlyArticle"
        assert p["@id"].startswith("https://doi.org/")


def test_vvprov_types_cover_vv_classification():
    """Перечень покрывает разные типы V&V (не одна природа эталона)."""
    cm = _codemeta()
    types = {r["vvprov:vvType"] for r in cm["vvprov:verification"]}
    assert any("MMS" in t for t in types)
    assert any("cross-code" in t for t in types)
    assert any("cross-model" in t for t in types)
    assert any("reduction" in t for t in types)
    refs = {r["vvprov:referenceType"] for r in cm["vvprov:verification"]}
    assert any(t.startswith("analytic") for t in refs)
    assert any("independent computation" in t for t in refs)


def test_vvprov_achieved_matches_actual_runs():
    """ВОРОТА (v0.8.0): заявленная точность записей vvprov не лучше фактической.

    Числа `vvprov:achieved` писались вручную и разошлись с артефактами на 2–3
    порядка (для защемлённого круга заявлялось «< 1e-5» при факте 7.1e-3).
    Здесь для записей, привязанных к case-файлу, ПЕРЕСЧИТЫВАЕТСЯ фактическая
    относительная погрешность гейтуемых эталонных строк, и проверяется, что
    заявленное число её не приукрашивает (допускается запас в 3 раза —
    кросс-платформенный разброс) и что сам гейт проходит.
    """
    import re

    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem
    from plate_solver.references import verify_result

    checked = 0
    for rec in _codemeta()["vvprov:verification"]:
        cases = [a.strip() for a in rec["vvprov:artifact"].split(";")
                 if a.strip().endswith(".toml")]
        if not cases:
            continue
        claim = re.search(r"(\d+(?:\.\d+)?)\s*e\s*-\s*(\d+)", rec["vvprov:achieved"])
        if claim is None:                      # запись без числа («regression-gated»)
            continue
        claimed = float(f"{claim.group(1)}e-{claim.group(2)}")
        report = verify_result(solve(Problem.from_toml(_ROOT / cases[0])))
        gated = [r for r in report.rows if r.gated and "инвариант" not in r.name]
        if not gated:
            continue
        actual = max(r.rel for r in gated)
        assert report.ok, cases[0]
        assert actual <= 3.0 * claimed, (
            f"{cases[0]}: заявлено {claimed:.1e}, фактически {actual:.1e}")
        checked += 1
    assert checked >= 3, f"проверено записей: {checked}"
