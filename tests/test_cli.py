"""Командная строка: шаблоны case-файлов, флаги и коды возврата.

Шаблон ``plate-solve --new`` — исполняемая документация: обещание «снять
«``#:`` » с блока — получить валидную постановку» проверяется здесь, а не
декларируется (аудит S13, S15). Прочие тесты закрывают согласованность
флагов с тем, что они на самом деле делают (S11, S12, S14, S16, S22, S26).
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

from plate_solver.cli import (
    _SIGIL,
    _TEMPLATE_KINDS,
    _template_blocks,
    _uncomment_block,
    main,
    template,
    write_template,
)
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]

#: строка, ВЫГЛЯДЯЩАЯ как ключ или заголовок таблицы TOML
_TOMLISH = re.compile(r"^(\[[A-Za-z_]|[A-Za-z_][\w.]*\s*=)")


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
    """Решение case, verify (exit 0/1), свип с артефактами."""
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


def test_check_only_validates(tmp_path, capsys):
    """--check валидирует и не считает (мгновенно, exit 0/1)."""
    from plate_solver.cli import main

    case = _ROOT / "cases" / "ci" / "circle_soft.toml"
    assert main([str(case), "--check"]) == 0
    bad = tmp_path / "bad.toml"
    bad.write_text('[geometry]\nkind = "hexagon"\n', encoding="utf-8")
    assert main([str(bad), "--check"]) == 1
    err = capsys.readouterr().err
    assert "ошибка" in err and "CASE_SCHEMA" in err   # диагностика валидатора


def test_report_smoke(tmp_path):
    """--report — одностраничный md с постановкой/числами/verify."""
    from plate_solver.cli import main

    case = _ROOT / "cases" / "ci" / "rect_mms.toml"
    out = tmp_path / "rep"
    assert main([str(case), "--out", str(out), "--report"]) == 0
    rp = out / "report.md"
    assert rp.is_file()
    txt = rp.read_text(encoding="utf-8")
    assert "## Постановка" in txt and "## Сводные числа" in txt
    assert "## Верификация" in txt and "PASS" in txt
    assert "w_max" in txt


def test_cli_theory_override(tmp_path, capsys):
    """Флаги --theory / --inplane-bc переопределяют [model] (DoD: выбор теории через CLI)."""
    from plate_solver.cli import main

    case = _ROOT / "cases" / "ci" / "circle_clamped.toml"     # классика, малая P̄
    # переопределить на karman: считается нелинейным трактом, exit 0
    assert main([str(case), "--theory", "karman", "--inplane-bc", "immovable",
                 "--out", str(tmp_path / "o")]) == 0
    # --inplane-bc без karman — ошибка постановки (exit 1)
    assert main([str(case), "--inplane-bc", "movable", "--out", str(tmp_path / "o2")]) == 1
    assert "inplane-bc" in capsys.readouterr().err


def test_new_ellipse_template(tmp_path):
    """Шаблон ellipse (v0.7.0): создаётся, проходит самопроверку и --check."""
    out = tmp_path / "ell.toml"
    path = write_template("ellipse", out)
    assert path.is_file()
    assert main([str(path), "--check"]) == 0


def test_template_text_current_schema():
    """Тексты шаблонов не несут устаревших меток v0.2 и упоминают новые ключи."""
    for kind in _TEMPLATE_KINDS:
        txt = template(kind)
        assert "в v0.2 — только soft_hinge" not in txt
        assert "(v0.2)" not in txt
        assert "expr" in txt and "supports" in txt and "winkler" in txt


# --------------------------------------------------------------------------- #
#  Шаблон как исполняемая документация (аудит S13, S15)
# --------------------------------------------------------------------------- #
def _deep_update(dst: dict, src: dict) -> dict:
    """Внести правки аннотации «требует» в разобранный шаблон (по секциям)."""
    for key, value in src.items():
        if isinstance(value, dict):
            _deep_update(dst.setdefault(key, {}), value)
        else:
            dst[key] = value
    return dst


@pytest.mark.parametrize("kind", _TEMPLATE_KINDS)
def test_every_template_block_uncomments_to_valid_case(kind):
    """Снятие сигила с ЛЮБОГО ОДНОГО блока даёт ВАЛИДНУЮ постановку (S15).

    Прежде шаблон предлагал блоки [contact]/[supports]/thermal_moment/[eigen]
    рядом с активным эталоном [verify], а валидатор такие сочетания
    отклоняет (реестр эталонов покрывает только базовую классическую
    задачу): путь «скопировать шаблон и раскомментировать» обрывался
    ошибкой. Теперь каждый блок несёт аннотацию «# требует: {…}» — правки,
    вносимые ВМЕСТЕ с ним; тест выполняет ровно её и требует валидности.
    """
    text = template(kind)
    blocks = _template_blocks(text)
    titles = " ".join(b["title"] for b in blocks)
    for name in ("[supports]", "[contact]", "[contact.zone]", "[eigen]",
                 "[model.orthotropy]", "thermal_moment", "h_expr"):
        assert name in titles, f"{kind}: в шаблоне нет блока {name}"
    for block in blocks:
        data = tomllib.loads(_uncomment_block(text, block))
        for section in block["drop"]:
            data.pop(section, None)
        _deep_update(data, block["requires"])
        Problem.from_dict(data, source=f"{kind}: {block['title']}")


@pytest.mark.parametrize("kind", _TEMPLATE_KINDS)
def test_template_has_no_fake_uncommentable_lines(kind):
    """Всё, что ВЫГЛЯДИТ раскомментируемым, помечено сигилом «#: » (S15).

    Иначе обещание шаблона снова стало бы выборочным: строка вида
    «# [contact]» приглашает снять комментарий, но проверкой выше не
    покрыта — именно так и появлялись блоки, которые валидатор отклоняет.
    Пояснения остаются прозой (начинаются не с ключа и не с таблицы).
    """
    for line in template(kind).splitlines():
        s = line.strip()
        if not s.startswith("#") or s.startswith(_SIGIL):
            continue
        assert not _TOMLISH.match(s[1:].strip()), f"{kind}: строка-ловушка {line!r}"


def test_template_numbers_are_config_defaults():
    """Числа блоков — ДЕЙСТВИТЕЛЬНО дефолты Config либо помечены (S13).

    Шапка шаблона обещала «закомментированные ключи показывают дефолты», а
    max_iter = 8000 и mor_anderson = 5 дефолтами (20000 и 0) не были.
    Значения берутся из Config программно; всё остальное обязано нести
    пометку «не дефолт».
    """
    from plate_solver.cli import _TEMPLATE_DEFAULT_KEYS
    from plate_solver.config import Config

    cfg = Config()
    txt = template("circle")
    for key in _TEMPLATE_DEFAULT_KEYS:            # эти ключи обязаны быть в шаблоне
        m = re.search(rf"^{_SIGIL} {key} = (\S+)", txt, re.M)
        assert m is not None, f"в шаблоне нет строки {key}"
        assert float(m.group(1)) == pytest.approx(float(getattr(cfg, key)))
    for line in txt.splitlines():                 # и ни одно число не лжёт
        m = re.match(rf"^{_SIGIL} ([A-Za-z_]\w*) = (\S+)", line)
        if m is None or not hasattr(cfg, m.group(1)):
            continue
        raw, expected = m.group(2).strip('"'), getattr(cfg, m.group(1))
        try:
            same = float(raw) == float(expected)
        except (TypeError, ValueError):
            same = raw == str(expected)
        assert same or "не дефолт" in line, f"{line!r}: не дефолт и не помечено"


def test_replot_profile_version_and_errors(tmp_path, capsys):
    """plate-replot/plate-profile: --version и человеческие ошибки (exit 1)."""
    from plate_solver.cli import main_profile, main_replot

    for entry in (main_replot, main_profile):
        with pytest.raises(SystemExit) as e:
            entry(["--version"])
        assert e.value.code == 0
    capsys.readouterr()
    rc = main_profile([str(tmp_path / "нет_каталога"), "--from", "0,0",
                       "--to", "1,0"])
    assert rc == 1
    assert "fields.npz" in capsys.readouterr().err


def test_verify_eigen_not_vacuous(capsys):
    """plate-verify на [eigen]: явное предупреждение вместо тихого PASS."""
    from plate_solver.cli import main_verify

    case = _ROOT / "cases" / "ci" / "eigen_buckling_circle.toml"
    assert main_verify([str(case)]) == 0          # exit-код НЕ меняется
    out = capsys.readouterr().out
    assert "эталонов для собственной задачи" in out


# --------------------------------------------------------------------------- #
#  Согласованность флагов с их действием (аудит S11, S12, S14, S16, S22, S26)
# --------------------------------------------------------------------------- #
def test_theory_flag_documents_alias_as_case_only():
    """Справка --theory не обещает алиас, которого CLI не принимает (S11).

    Прежде help сообщал «Устаревший 'ktn' = ktn_linear», а argparse отвергал
    это значение кодом 2: алиас поддержан ТОЛЬКО в case-файле (депрекация,
    docs/MIGRATION.md), и справка обязана говорить именно это.
    """
    from plate_solver.cli import _base_parser

    action = next(a for a in _base_parser("plate-solve", "")._actions
                  if "--theory" in a.option_strings)
    assert "ktn" not in action.choices                # алиаса в CLI нет
    assert "только в case-файле" in action.help       # и справка не обещает
    case = str(_ROOT / "cases" / "ci" / "circle_soft.toml")
    with pytest.raises(SystemExit) as e:              # argparse: недопустимое
        main([case, "--theory", "ktn"])
    assert e.value.code == 2


def test_fig_format_strips_spaces_and_rejects_unknown(tmp_path, capsys):
    """--fig-format: пробелы отбрасываются, неизвестный формат — CaseError (S12).

    Запись оболочки ``--fig-format "png, pdf"`` доходила до savefig и давала
    сырую трассировку ValueError matplotlib — уже ПОСЛЕ расчёта.
    """
    from plate_solver.cli import _parse_fig_formats, main_replot

    assert _parse_fig_formats("png, pdf") == ("png", "pdf")
    assert _parse_fig_formats(" .PDF ") == ("pdf",)
    with pytest.raises(CaseError, match="ожидалось из"):
        _parse_fig_formats("png, pngg")
    with pytest.raises(CaseError, match="пустой список"):
        _parse_fig_formats(" , ")
    # сквозь команды: код 1 и понятный текст ДО расчёта (plate-solve) и до
    # чтения полей (plate-replot — каталог заведомо без fields.npz)
    case = str(_ROOT / "cases" / "ci" / "circle_soft.toml")
    assert main([case, "--fig-format", "png, растр", "--out", str(tmp_path / "o")]) == 1
    assert "fig-format" in capsys.readouterr().err
    assert main_replot([str(tmp_path), "--fig-format", "png, растр"]) == 1
    assert "fig-format" in capsys.readouterr().err


def test_check_uses_the_same_overrides_as_a_run(capsys):
    """--check подтверждает ТУ ЖЕ постановку, что посчитает plate-solve (S14).

    Прежде флаги --theory/--inplane-bc/--grid до валидатора не доходили:
    «постановка валидна» относилось к другой постановке, а прогон с теми же
    флагами падал ошибкой.
    """
    case = str(_ROOT / "cases" / "ci" / "circle_clamped.toml")
    assert main([case, "--check", "--inplane-bc", "movable"]) == 1
    assert "inplane-bc" in capsys.readouterr().err
    assert main([case, "--check", "--grid", "1"]) == 1     # та же ограда схемы
    assert "grid_n" in capsys.readouterr().err
    # валидное переопределение: код 0, эффективная постановка в отчёте, а
    # снятие «чужого» эталона (v0.8.0) уходит в stderr и проверке не мешает
    assert main([case, "--check", "--theory", "ktn_full", "--grid", "24"]) == 0
    out, err = capsys.readouterr()
    assert "ktn_full" in out and "24" in out
    assert "верификация в этом прогоне отключена" in err


def test_verify_rejects_inert_figure_flags(tmp_path, capsys):
    """plate-verify отклоняет флаги фигур, которые ничего не делали (S16).

    Верификация печатает таблицу эталонов и не сохраняет ни результата, ни
    фигур: --figures/--fig-format/--surface принимались молча и терялись.
    Флаг --out действует лишь на артефакты свипа — без --sweep он больше не
    молчит, но и код возврата не меняет.
    """
    from plate_solver.cli import main_verify

    case = str(_ROOT / "cases" / "ci" / "circle_soft.toml")
    for flag in (["--figures"], ["--fig-format", "png"], ["--surface", "top"]):
        assert main_verify([case, *flag]) == 1
        assert "не применяются" in capsys.readouterr().err
    assert main_verify([case, "--out", str(tmp_path / "o")]) == 0
    assert "не используется" in capsys.readouterr().err


def test_thickness_line_only_for_refined_theories(tmp_path, capsys):
    """Строка «толщина (КТН)» — только уточнённые теории (S22).

    У Кармана поправок сдвига и обжатия нет ПО ПОСТРОЕНИЮ (h_ψ² = h_*² = 0,
    theory.karman), а печатались физические лицевые параметры (ν, h) —
    строка выглядела свойством решения, которым не была.
    """
    case = str(_ROOT / "cases" / "ci" / "circle_clamped.toml")
    assert main([case, "--theory", "karman", "--out", str(tmp_path / "k")]) == 0
    assert "толщина (КТН)" not in capsys.readouterr().out
    assert main([case, "--theory", "ktn_linear", "--out", str(tmp_path / "l")]) == 0
    assert "толщина (КТН)" in capsys.readouterr().out


def test_sweep_outputs_keep_user_backend(tmp_path, monkeypatch):
    """Свип не переопределяет бэкенд matplotlib жёстко (S26).

    Приём один на весь CLI — MPLBACKEND по умолчанию (уважает явный выбор
    пользователя); прежде рядом с этим обещанием стоял matplotlib.use("Agg").
    """
    import matplotlib

    from plate_solver.cli import _write_sweep_outputs

    monkeypatch.setenv("MPLBACKEND", "Agg")
    calls: list = []
    monkeypatch.setattr(matplotlib, "use", lambda *a, **kw: calls.append(a))
    rows = [{"p": 4, "w_max": 1.0, "cond_A": 1.0e3},
            {"p": 6, "w_max": 1.1, "cond_A": 2.0e3}]
    _write_sweep_outputs(rows, ["p"], tmp_path / "sw", do_verify=False)
    assert calls == []                            # выбор пользователя не тронут
    assert os.environ["MPLBACKEND"] == "Agg"
    for name in ("sweep.md", "sweep.csv", "sweep.png"):
        assert (tmp_path / "sw" / name).is_file()
