"""L-форма: геометрия + сверка изгиба через scikit-fem (тест-ворота шага).

Две колонки сверки (NOTES.md §9):
  • RFM ↔ FEM-Marcus (те же две Пуассона) — ГЕЙТ: поля совпадают в пределах
    сеточной/полиномиальной точности (~2–3 %).
  • RFM ↔ FEM-Kirchhoff (истинный Кирхгоф, Морли) — расходятся (~40–50 %, растёт
    у входящего угла): парадокс Сапонджяна–Бабушки. Фиксируем количественно.

FEM-тесты пропускаются, если scikit-fem не установлен.
"""

from __future__ import annotations

import pytest
from plates import geometry as geo
from plates.config import Config


def test_lshape_bbox_and_reentrant_corner():
    dom = geo.make_L(side=1.0, cut=0.5)
    assert dom.bbox == (0.0, 1.0, 0.0, 1.0)
    # У входящего угла (0.5, 0.5): чуть «внутрь» полос — Ω, в вырезе — вне Ω.
    assert dom.omega(0.49, 0.10) > 0
    assert dom.omega(0.60, 0.60) < 0


def test_lshape_invalid_cut_rejected():
    with pytest.raises(ValueError):
        geo.make_L(side=1.0, cut=1.0)   # cut < side нарушено


@pytest.fixture(scope="module")
def fem_cmp():
    """Сверка RFM ↔ (FEM-Marcus, FEM-Kirchhoff) на L-форме (один расчёт на модуль)."""
    pytest.importorskip("skfem")
    from plates import verify_fem as vf

    cfg = Config(nu=0.3, q0=4.0, p=10, Q=80)
    return vf.compare_rfm_vs_fem(cfg, mesh_m=16, refine=2)


def test_gate_rfm_matches_fem_marcus(fem_cmp):
    """ГЛАВНЫЕ ВОРОТА: RFM ≈ FEM-Marcus (та же модель) в норме L²."""
    assert fem_cmp.rel_marcus_pct < 5.0
    assert abs(fem_cmp.w_rfm_max - fem_cmp.w_marcus_max) / fem_cmp.w_marcus_max < 0.05


def test_sapondzhyan_paradox_documented(fem_cmp):
    """Парадокс Сапонджяна: RFM(мягкий шарнир) ≠ Кирхгоф у входящего угла."""
    assert fem_cmp.rel_kirchhoff_pct > 25.0                       # расхождение велико
    assert fem_cmp.rel_kirchhoff_pct > 8.0 * fem_cmp.rel_marcus_pct  # ≫ численной ошибки
    assert fem_cmp.w_kirchhoff_max < fem_cmp.w_marcus_max         # Кирхгоф «жёстче» у угла
