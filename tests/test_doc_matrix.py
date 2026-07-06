"""Ворота матрицы возможностей (F2.1) и смок API-сирот.

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


def test_every_schema_key_documented_in_case_schema():
    """F2.2-ворота: каждый ключ problem.py упомянут в docs/CASE_SCHEMA.md."""
    from doc_matrix import schema_keys

    schema = (_ROOT / "docs" / "CASE_SCHEMA.md").read_text(encoding="utf-8")
    missing = [f"{sec}.{key}" for sec, key in schema_keys()
               if key not in schema]
    assert not missing, f"нет в CASE_SCHEMA.md: {missing}"
