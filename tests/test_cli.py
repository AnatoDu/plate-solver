"""Генератор шаблонов case-файлов (фаза 2, P0.4): plate-solve --new <kind>."""

from __future__ import annotations

import pytest

from plate_solver.cli import _TEMPLATE_KINDS, main, template, write_template
from plate_solver.problem import CaseError, Problem


@pytest.mark.parametrize("kind", _TEMPLATE_KINDS)
def test_template_is_valid_case(kind, tmp_path, monkeypatch):
    """Каждый шаблон — валидный case-файл: загружается и строит Config."""
    monkeypatch.chdir(tmp_path)
    path = write_template(kind)
    assert path.name == f"{kind}.toml"
    p = Problem.from_toml(path)
    assert p.geometry.kind == kind
    cfg = p.to_config()
    assert cfg.p == 10 and cfg.Q == 256          # значения из шаблона
    assert "#" in path.read_text(encoding="utf-8")  # шаблон закомментирован


def test_write_template_refuses_overwrite(tmp_path):
    out = tmp_path / "c.toml"
    write_template("circle", out)
    with pytest.raises(CaseError, match="уже существует"):
        write_template("circle", out)


def test_template_unknown_kind():
    with pytest.raises(CaseError, match="ожидалось"):
        template("triangle")


def test_main_new_and_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--new", "annulus"]) == 0
    assert (tmp_path / "annulus.toml").is_file()
    assert "шаблон записан" in capsys.readouterr().out
    # неизвестный вид — ошибка кодом 1, текст на stderr
    assert main(["--new", "hexagon"]) == 1
    assert "ожидалось" in capsys.readouterr().err
    # решение case-файлов до P4 не реализовано — код 2
    assert main(["annulus.toml"]) == 2
    assert "P4" in capsys.readouterr().err
