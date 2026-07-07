r"""Ворота конвейера воспроизводимости (scripts/reproduce_all.py, N8 v0.6.0, §9.1).

Реестр соответствия результат → case → артефакт. Быстрые ворота — логика
(обнаружение случаев, прогон одного, формат сводки, кэш); интеграция (весь ci
зелёный одним прогоном) — маркер big.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# reproduce_all.py — скрипт (не пакет): загружаем как модуль
_spec = importlib.util.spec_from_file_location(
    "reproduce_all", _ROOT / "scripts" / "reproduce_all.py")
reproduce_all = importlib.util.module_from_spec(_spec)
sys.modules["reproduce_all"] = reproduce_all
_spec.loader.exec_module(reproduce_all)


def test_discover_cases_ci_and_ladder():
    """Обнаружение: ci непуст; --with-ladder добавляет ступени; группы верны."""
    ci = reproduce_all.discover_cases(with_ladder=False)
    both = reproduce_all.discover_cases(with_ladder=True)
    assert len(ci) > 0 and all(g == "ci" for g, _ in ci)
    assert len(both) >= len(ci) and any(g == "ladder" for g, _ in both)


def test_run_case_produces_ok_row():
    """Прогон одного быстрого случая → строка со статусом OK и заполненными полями."""
    case = _ROOT / "cases" / "ci" / "circle_clamped.toml"
    row = reproduce_all.run_case("ci", case)
    assert row["status"] == "OK"
    assert row["case"] == "circle_clamped" and row["geometry"] == "circle"
    assert row["theory"] and row["w_max"]                # числа получены


def test_run_case_bad_file_reports_fail(tmp_path):
    """Битый case-файл ⇒ статус FAIL (конвейер не падает, помечает провал)."""
    bad = tmp_path / "broken.toml"
    bad.write_text("это не валидный case", encoding="utf-8")
    row = reproduce_all.run_case("ci", bad)
    assert row["status"].startswith("FAIL")


def test_build_summary_md_registry_table():
    """Сводка — markdown-таблица соответствия со всеми строками и заголовком."""
    rows = [reproduce_all.run_case("ci", _ROOT / "cases" / "ci" / "circle_clamped.toml")]
    md = reproduce_all.build_summary_md(rows, elapsed=1.0, n_fail=0)
    assert "результат → case → артефакт" in md
    assert "circle_clamped" in md and md.count("| circle_clamped ") == 1


@pytest.mark.big
def test_pipeline_all_ci_green(tmp_path):
    """Интеграция: весь ci-набор воспроизводится одним прогоном без провалов (§9.1)."""
    code = reproduce_all.main(["--out", str(tmp_path)])
    assert code == 0                                     # 0 провалов ⇒ exit 0
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    csv_path = tmp_path / "summary.csv"
    assert summary.count("| OK |") == len(reproduce_all.discover_cases())
    assert "FAIL" not in summary and csv_path.exists()
