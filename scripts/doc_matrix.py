#!/usr/bin/env python3
"""doc_matrix.py — матрица возможностей → docs/FEATURES.md.

Автоматический реестр трёх видов возможностей:

* КЛЮЧИ СХЕМЫ case-файлов — извлекаются из вызовов ``_require_keys`` в
  src/plate_solver/problem.py (единственный источник допустимых ключей);
* ФЛАГИ CLI — из фактических argparse-парсеров трёх команд;
* ПУБЛИЧНЫЕ ФУНКЦИИ — из ``__all__`` модулей пакета.

Для каждой строки проверяется: «где описано» (вхождение в docs/*.md,
README.md) и «чем покрыто» (case-файлы cases/**, examples/, notebooks/,
tests/). Пустая клетка = задача документации; тест
tests/test_doc_matrix.py требует нулевого числа дыр и актуальности
docs/FEATURES.md (перегенерация не меняет файл).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "plate_solver"

#: модули с публичным API (плоский пакет, фасад — __init__)
API_MODULES = ("analytic", "analytic_auto", "benchmarks", "clamped", "cli",
               "config", "contact", "dispatch", "faces", "geometry", "ktn",
               "ktn_full", "ktn_solver", "ladder", "membrane", "plate", "poisson",
               "problem", "references", "theory", "verify_fem", "viz")


# --------------------------------------------------------------------------- #
#  Сбор реестров
# --------------------------------------------------------------------------- #
def schema_keys() -> list[tuple[str, str]]:
    """(секция, ключ) из вызовов _require_keys(problem.py)."""
    tree = ast.parse((SRC / "problem.py").read_text(encoding="utf-8"))
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_require_keys" and len(node.args) >= 3):
            sec, keys = node.args[0], node.args[2]
            if isinstance(sec, ast.Constant) and isinstance(keys, ast.Set):
                sec_name = re.sub(r"\[\d+\]", "", str(sec.value))
                for el in keys.elts:
                    if isinstance(el, ast.Constant):
                        out.add((sec_name, str(el.value)))
    return sorted(out)


def cli_flags() -> list[tuple[str, str]]:
    """(команда, флаг) из фактических парсеров."""
    sys.path.insert(0, str(SRC.parent))
    from plate_solver import cli

    out = []
    for prog, factory in (("plate-solve", lambda: cli._base_parser("plate-solve", "")),
                          ("plate-verify", lambda: cli._base_parser("plate-verify", ""))):
        parser = factory()
        if prog == "plate-solve":
            parser.add_argument("--new")
            parser.add_argument("--check", action="store_true")
            parser.add_argument("--report", action="store_true")
        for act in parser._actions:
            for opt in act.option_strings:
                if opt.startswith("--"):
                    out.append((prog, opt))
    out.append(("plate-ladder", "--out"))
    out.append(("plate-ladder", "--version"))
    return sorted(set(out))


def public_functions() -> list[tuple[str, str]]:
    """(модуль, имя) из __all__ модулей API_MODULES."""
    out = []
    for mod in API_MODULES:
        text = (SRC / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in tree.body:
            if (isinstance(node, ast.Assign) and node.targets
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__"):
                for el in ast.literal_eval(node.value):
                    out.append((mod, str(el)))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
#  Поиск описаний и покрытий
# --------------------------------------------------------------------------- #
def _read_all(paths) -> dict:
    return {p: p.read_text(encoding="utf-8", errors="replace") for p in paths}


def build_matrix() -> tuple[str, int]:
    # FEATURES.md исключён из пула «где описано»: само-ссылка делала
    # генерацию неидемпотентной (результат зависел от прошлого файла)
    docs = _read_all([q for q in (ROOT / "docs").glob("*.md")
                      if q.name != "FEATURES.md"] + [ROOT / "README.md"])
    cases = _read_all(list((ROOT / "cases").rglob("*.toml")))
    usage = _read_all(list((ROOT / "examples").glob("*.py"))
                      + list((ROOT / "notebooks").glob("*.ipynb"))
                      + list((ROOT / "tests").glob("*.py"))
                      + list((ROOT / "scripts").glob("*.py")))
    pkg = _read_all(list(SRC.glob("*.py")))

    def where_doc(term: str) -> str:
        hits = [p.name for p, txt in docs.items() if term in txt]
        return ", ".join(sorted(hits)[:3])

    def where_use(term: str, pool: dict) -> str:
        hits = [p.name for p, txt in pool.items() if term in txt]
        return ", ".join(sorted(hits)[:3])

    holes = 0
    lines = ["# Матрица возможностей (автогенерация: scripts/doc_matrix.py)",
             "",
             "Каждая возможность обязана быть описана в документации и покрыта",
             "случаем/примером/тестом. Пустая клетка — задача (гейт",
             "tests/test_doc_matrix.py).",
             ""]

    lines += ["## Ключи схемы case-файлов", "",
              "| секция.ключ | описано | покрыто (cases) |", "|---|---|---|"]
    for sec, key in schema_keys():
        doc = where_doc(key)
        cov = where_use(key, cases) or where_use(key, usage)
        holes += (not doc) + (not cov)
        lines.append(f"| {sec}.{key} | {doc or '—'} | {cov or '—'} |")

    lines += ["", "## Флаги CLI", "",
              "| команда | флаг | описано | покрыто (тесты/доки) |",
              "|---|---|---|---|"]
    for prog, flag in cli_flags():
        doc = where_doc(flag)
        cov = where_use(flag, usage) or doc
        holes += (not doc) + (not cov)
        lines.append(f"| {prog} | `{flag}` | {doc or '—'} | {cov or '—'} |")

    lines += ["", "## Публичные функции и классы", "",
              "| модуль | имя | описано | покрыто (examples/notebooks/tests) |",
              "|---|---|---|---|"]
    for mod, name in public_functions():
        doc = where_doc(name)
        cov = where_use(name, usage)
        if not cov:
            # внутрипакетное использование: в чужом модуле >= 1 вхождение,
            # в определяющем — >= 3 (определение + __all__ + использование)
            own = SRC / f"{mod}.py"
            hits = [q.name for q, txt in pkg.items()
                    if (q != own and name in txt)
                    or (q == own and txt.count(name) >= 3)]
            cov = ", ".join(sorted(hits)[:3])
        holes += (not doc) + (not cov)
        lines.append(f"| {mod} | `{name}` | {doc or '—'} | {cov or '—'} |")

    lines += ["", f"Дыр (пустых клеток): **{holes}**", ""]
    return "\n".join(lines), holes


def main() -> int:
    text, holes = build_matrix()
    out = ROOT / "docs" / "FEATURES.md"
    out.write_text(text, encoding="utf-8")
    print(f"docs/FEATURES.md: {text.count(chr(10))} строк, дыр: {holes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
