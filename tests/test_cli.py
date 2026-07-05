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
    # битый case-файл — человекочитаемая ошибка кодом 1
    (tmp_path / "битый.toml").write_text("[geometry]\nkind = 'triangle'\na = 1.0\n"
                                         "[bc]\ntype = 'soft_hinge'\n"
                                         "[load]\ntype = 'uniform'\nq0 = 1.0\n",
                                         encoding="utf-8")
    assert main(["битый.toml"]) == 1
    assert "CASE_SCHEMA" in capsys.readouterr().err


def test_cli_solve_verify_sweep_end_to_end(tmp_path, monkeypatch, capsys):
    """P4.1+P4.2: решение case, verify (exit 0/1), свип с артефактами."""
    from plate_solver.cli import main_verify

    monkeypatch.chdir(tmp_path)
    main(["--new", "circle"])
    fast = (tmp_path / "circle.toml").read_text(encoding="utf-8") \
        .replace("Q = 256", "Q = 96").replace("p = 10", "p = 6")
    (tmp_path / "circle.toml").write_text(fast, encoding="utf-8")

    assert main(["circle.toml", "--out", "o"]) == 0
    assert (tmp_path / "o" / "result.json").is_file()
    out = capsys.readouterr().out
    assert "w_max" in out and "cond(A)" in out

    assert main_verify(["circle.toml"]) == 0
    assert "PASS" in capsys.readouterr().out

    rc = main_verify(["circle.toml", "--sweep", "p=2:6:2", "--out", "sw"])
    assert rc == 0                                   # вердикт — по последней точке
    for name in ("sweep.md", "sweep.csv", "sweep.png"):
        assert (tmp_path / "sw" / name).is_file()

    assert main_verify(["circle.toml", "--sweep", "h=1:2:1"]) == 1
    assert "p | Q" in capsys.readouterr().err


def test_cli_ladder(tmp_path, monkeypatch, capsys):
    """plate-ladder: каталог → сводный md с провенансом, exit по всем case."""
    from plate_solver.cli import main_ladder

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "ladder"
    folder.mkdir()
    main(["--new", "circle", "--out", str(folder / "c.toml")])
    fast = (folder / "c.toml").read_text(encoding="utf-8") \
        .replace("Q = 256", "Q = 96").replace("p = 10", "p = 6")
    (folder / "c.toml").write_text(fast, encoding="utf-8")
    assert main_ladder([str(folder)]) == 0
    summary = (folder / "ladder_summary.md").read_text(encoding="utf-8")
    assert "PASS" in summary and "Провенанс" in summary
    capsys.readouterr()
    # сломанный case валит лестницу кодом 1, но сводка пишется
    (folder / "bad.toml").write_text("[geometry]\nkind = 'x'\n", encoding="utf-8")
    assert main_ladder([str(folder)]) == 1
