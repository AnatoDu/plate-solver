"""Ворота матрицы возможностей и смок API-сирот.

FEATURES.md обязан быть актуален (перегенерация не меняет файл) и БЕЗ
пустых клеток: каждая возможность описана и покрыта. Смоки ниже дают
содержательное покрытие функциям, которые не использовались нигде
за пределами определения (сироты выявлены самой матрицей).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))


def test_features_matrix_current_and_no_holes():
    from doc_matrix import build_matrix

    text, holes = build_matrix()
    assert holes == 0, "матрица возможностей содержит пустые клетки"
    on_disk = (_ROOT / "docs" / "FEATURES.md").read_text(encoding="utf-8")
    assert text == on_disk, ("docs/FEATURES.md устарел — перегенерируйте: "
                             "python scripts/doc_matrix.py")


# --------------------------------------------------------------------------- #
#  Смоки API-сирот (осмысленные тождества, не заглушки)
# --------------------------------------------------------------------------- #
def test_circle_point_soft_moment_matches_numeric_derivatives():
    """M-поле расщепления (P/2π)·ln(a/r) ≡ −D·Δw численно (M = −D·Δw)."""
    from plate_solver.analytic import circle_point_soft, circle_point_soft_moment

    a, P, D = 1.0, 5.0, 100.0
    r = np.array([0.3, 0.5, 0.7])
    h = 1e-6

    def w(rr):
        return np.asarray(circle_point_soft(rr, a, P, D), float)

    d1 = (w(r + h) - w(r - h)) / (2 * h)
    d2 = (w(r + h) - 2 * w(r) + w(r - h)) / h**2
    m_num = -D * (d2 + d1 / r)                       # M = −D·Δw (осесимметрия)
    m_ref = np.asarray(circle_point_soft_moment(r, a, P), float)
    assert np.allclose(m_num, m_ref, rtol=5e-4)


def test_disk_poisson_uniform_center_identity():
    """Центр мембраны: u(0) = q a²/4 и совпадает с полем в нуле."""
    from plate_solver.analytic import disk_poisson_uniform, disk_poisson_uniform_center

    a, q = 1.0, 4.0
    c = float(disk_poisson_uniform_center(a, c=q))
    assert c == pytest.approx(q * a**2 / 4, rel=1e-14)
    assert c == pytest.approx(float(disk_poisson_uniform(0.0, a, c=q)), rel=1e-14)


def test_rect_sin_exact_consistency():
    """Точное поле синус-нагрузки согласовано со своим w_max в центре."""
    from plate_solver.ladder import rect_sin_exact, rect_sin_wmax

    Lx = Ly = 1.0
    D, q0 = 100.0, 4.0
    w_c = float(rect_sin_exact(Lx / 2, Ly / 2, Lx, Ly, D, q0))
    assert w_c == pytest.approx(float(rect_sin_wmax(Lx, Ly, D, q0)), rel=1e-14)


# --------------------------------------------------------------------------- #
#  Полнота схемы: ключ обязан стоять В ТАБЛИЦЕ ключей, а не «где-то в тексте»
# --------------------------------------------------------------------------- #
def _key_table_cells(schema: str) -> list[str]:
    """Первые ячейки строк раздела «## Таблица ключей» (без заголовка и разделителя)."""
    lines = schema.splitlines()
    assert "## Таблица ключей" in lines, "в docs/CASE_SCHEMA.md нет «## Таблица ключей»"
    cells = []
    for line in lines[lines.index("## Таблица ключей") + 1:]:
        if line.startswith("## "):
            break                                    # конец раздела
        if not line.startswith("|"):
            continue
        cell = line.split("|")[1].strip()
        if not cell or set(cell) <= set("-: ") or cell == "Ключ":
            continue                                 # разделитель и заголовок
        cells.append(cell)
    return cells


def _declared_keys(cell: str) -> set[tuple[str, str]]:
    """(секция, ключ) из ячейки таблицы: перечисления и диапазоны.

    Разбираются все формы, встречающиеся в таблице: ``model.theory``,
    ``model.E, nu, h`` (перечисление в одной секции), ``geometry.x1..y2``
    (диапазон координат прямоугольника), ``contact.gap.r_curv, cx, cy, apex``
    (вложенная таблица), ``[plate2] bc, load`` (секция в скобках).
    """
    import re

    cell = cell.strip().strip("`").replace("[[", "").replace("]]", "")
    m = re.match(r"^\[([\w.]+)\]\s*(.+)$", cell)          # «[plate2] bc, load»
    if m:
        section, listed = m.group(1), m.group(2)
    else:
        head = cell.split(",")[0].strip().split("..")[0]  # «geometry.x1..y2» → «geometry.x1»
        if "." not in head:
            return set()
        section = head.rsplit(".", 1)[0]
        listed = cell[len(section) + 1:]
    keys: set[str] = set()
    for token in (t.strip().strip("`") for t in listed.split(",")):
        if not token:
            continue
        if ".." not in token:
            keys.add(token)
            continue
        lo, hi = token.split("..")                        # «x1..y2» — оси × индексы
        keys |= {lo, hi}
        a, b = re.match(r"^([a-z])(\d)$", lo), re.match(r"^([a-z])(\d)$", hi)
        if a and b:
            keys |= {f"{c}{d}" for c in (a.group(1), b.group(1))
                     for d in (a.group(2), b.group(2))}
    return {(section, k) for k in keys}


def test_every_schema_key_documented_in_case_schema():
    """Ворота полноты схемы: каждый ключ problem.py стоит В ТАБЛИЦЕ ключей.

    Прежняя редакция искала имя ключа ПОДСТРОКОЙ по всему файлу — для коротких
    имён (Q, p, P, E, h, nu, tol, type) условие выполнялось случайным вхождением
    в прозе, и недокументированный ключ проходил бы молча. Теперь ключ обязан
    быть объявлен ЯЧЕЙКОЙ таблицы «## Таблица ключей» и ИМЕННО В СВОЕЙ СЕКЦИИ
    (``model.tol`` не засчитывается вхождением ``verify.tol``); секции верхнего
    уровня обязаны встречаться в виде заголовка ``[секция]``.
    """
    from doc_matrix import schema_keys

    schema = (_ROOT / "docs" / "CASE_SCHEMA.md").read_text(encoding="utf-8")
    cells = _key_table_cells(schema)
    assert len(cells) > 50, "таблица ключей не разобрана"
    documented: set[tuple[str, str]] = set()
    for cell in cells:
        declared = _declared_keys(cell)
        assert declared, f"строка таблицы не разобрана: {cell!r}"
        documented |= declared

    missing = []
    for sec, key in schema_keys():
        if sec == "case":                            # секции верхнего уровня
            if f"[{key}]" not in schema:
                missing.append(f"секция [{key}]")
        elif (sec, key) not in documented:
            missing.append(f"{sec}.{key}")
    assert not missing, f"нет в таблице ключей docs/CASE_SCHEMA.md: {missing}"

    # проверка не вакуумна: несуществующий ключ и ключ ЧУЖОЙ секции не проходят
    assert ("geometry", "radius") not in documented
    assert ("model", "tol") not in documented and "tol" in schema
