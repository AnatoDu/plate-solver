r"""Тест-ворота Части 2 — ЖЁСТКОЕ ЗАЩЕМЛЕНИЕ (структура w = ω²Φ, прямой Ритц по бигармонике).

Ключевая мысль: при защемлении расщепление КОРРЕКТНО и на криволинейной границе
(круг), и во входящем угле (L-форма) — парадокс Сапонджяна–Бабушки есть свойство
ШАРНИРНОГО опирания, а не дефект метода.

  • КРУГ: RFM (p=10) совпадает с точной формулой (4.1) с высокой точностью; ошибка
    мала ПРИ ВСЕХ p и падает с Q к нулю — НЕТ модельной погрешности (контраст с
    мягким шарниром 26.4 %). Точное решение защемления ``w = [qa²/16D]·ω²`` лежит в
    структуре уже при p=2.
  • КВАДРАТ (выпуклый): RFM↔МКЭ ничтожно мало (~0.1 %) — метод точен на выпуклых.
  • L-ФОРМА: RFM↔МКЭ-эталон (Аргирис, C¹) МАЛО и УБЫВАЕТ с p (обычная погрешность
    дискретизации у входящего угла), НЕ ~55 % и НЕ растёт — парадокса нет.

МКЭ-тесты пропускаются без scikit-fem.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from plate_solver import analytic, geometry
from plate_solver import quadrature as quad
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config

# Модуль опирается на независимую МКЭ-верификацию (scikit-fem) и квадратуры до Q=1024.
pytestmark = pytest.mark.fem

NU, Q0, E = 0.3, 4.0, 2.1e6
SOFT_HINGE_GAP = 0.2642   # модельный разрыв мягкого шарнира к (4.2) при ν=0.3 (контраст)
HINGE_PARADOX = 0.5486    # шарнирный парадокс на L-форме (золотой прогон, контраст)


def _circle_cfg(p, Q, a=1.0, h=1.0):
    return Config(a=a, q0=Q0, nu=NU, h=h, E=E, p=p, Q=Q)


def _circle_wmax(p, Q):
    dom = geometry.make_circle(1.0)
    cp = ClampedPlate.from_config(dom, _circle_cfg(p, Q))
    return float(cp.deflection(cp.solve_uniform(Q0), 0.0, 0.0))


# --------------------------------------------------------------------------- #
#  Круг: совпадение с формулой (4.1) и отсутствие модельной погрешности
# --------------------------------------------------------------------------- #
def test_clamped_circle_exact():
    """ВОРОТА: RFM (p=10, ω²Φ) ≈ формула (4.1), отн. погрешность ≪ 26 % (не как у шарнира)."""
    D = _circle_cfg(10, 1024).D
    w_exact = float(analytic.clamped_uniform_wmax(1.0, Q0, D))
    err = abs(_circle_wmax(10, 1024) - w_exact) / w_exact
    assert err < 5e-3                      # факт ~0.09 %
    assert err < 0.05 * SOFT_HINGE_GAP     # на ДВА порядка меньше модельного разрыва шарнира


def test_clamped_circle_no_model_error():
    """ВОРОТА: погрешность падает с Q к нулю — нет модельного «пола» (контраст с шарниром)."""
    D = _circle_cfg(6, 1024).D
    w_exact = float(analytic.clamped_uniform_wmax(1.0, Q0, D))
    errs = [abs(_circle_wmax(6, Q) - w_exact) / w_exact for Q in (64, 128, 256, 512)]
    assert errs[0] > errs[1] > errs[2] > errs[3]   # монотонно падает с Q (нет «пола»)
    assert errs[-1] < SOFT_HINGE_GAP / 10          # уже ≪ модельного разрыва шарнира
    # практически не зависит от p (точное решение в структуре уже при p=2):
    e2 = abs(_circle_wmax(2, 1024) - w_exact) / w_exact
    e10 = abs(_circle_wmax(10, 1024) - w_exact) / w_exact
    assert abs(e2 - e10) < 1e-3


def test_clamped_structure_satisfies_bc():
    """Структура ω²Φ: w=0 на ∂Ω точно и ∂w/∂n→0 (защемление, не мягкий шарнир)."""
    a = 1.0
    dom = geometry.make_circle(a)
    cp = ClampedPlate.from_config(dom, _circle_cfg(8, 512))
    c = cp.solve_uniform(Q0)
    # на границе ρ=a прогиб ровно ноль
    th = np.linspace(0, 2 * np.pi, 37)
    assert np.allclose(cp.deflection(c, a * np.cos(th), a * np.sin(th)), 0.0, atol=1e-14)
    # нормальная производная: w ∝ (a−ρ)² (квадратично), т.е. w/(a−ρ) → 0 у границы
    rho = np.array([0.90, 0.95, 0.99]) * a
    w = cp.deflection(c, rho, np.zeros_like(rho))
    slope = w / (a - rho)
    assert slope[2] < slope[0]                     # наклон убывает к границе ⇒ ∂w/∂n→0


# --------------------------------------------------------------------------- #
#  Выпуклый квадрат: метод точен (нет входящего угла)
# --------------------------------------------------------------------------- #
def test_clamped_square_convex_matches_fem():
    """На выпуклом квадрате RFM↔МКЭ ничтожно (контрольная проверка корректности метода)."""
    pytest.importorskip("skfem")
    from skfem import MeshTri

    from plate_solver.clamped import solve_clamped_fem

    h = 0.06
    D = E * h**3 / (12 * (1 - NU**2))
    dom = geometry.make_rectangle(0.0, 1.0, 0.0, 1.0)
    cp = ClampedPlate.from_config(dom, Config(q0=Q0, nu=NU, h=h, E=E, p=8, Q=120))
    c = cp.solve_uniform(Q0)
    fem = solve_clamped_fem(MeshTri().refined(4), D, Q0, NU)
    qn = quad.interior_nodes(dom, 120)
    keep = dom.omega(qn.x, qn.y) > 0.02
    X, Y, W = qn.x[keep], qn.y[keep], qn.w[keep]
    wr, wf = cp.deflection(c, X, Y), fem.at(X, Y)
    rel = 100 * np.sqrt(np.sum(W * (wr - wf) ** 2)) / np.sqrt(np.sum(W * wf**2))
    assert rel < 1.0                               # < 1 % на выпуклой области


# --------------------------------------------------------------------------- #
#  L-форма: нет парадокса (расхождение мало и убывает с p)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lshape_clamped_cmp():
    """RFM(защемл.) ↔ МКЭ(Аргирис) на L-форме при p=4 и p=10 (один расчёт на модуль)."""
    pytest.importorskip("skfem")
    from plate_solver.clamped import clamped_fem_lshape

    h = 0.06
    D = E * h**3 / (12 * (1 - NU**2))
    dom = geometry.make_L(1.0, 0.5)
    fem = clamped_fem_lshape(D, Q0, NU, mesh_m=16, refine=2)
    qn = quad.interior_nodes(dom, 120)
    keep = dom.omega(qn.x, qn.y) > 0.02
    X, Y, W = qn.x[keep], qn.y[keep], qn.w[keep]
    wf = fem.at(X, Y)
    wf_max = fem.w_max_on_grid(dom, grid_n=160)

    def rel_l2(p):
        cp = ClampedPlate.from_config(dom, Config(q0=Q0, nu=NU, h=h, E=E, p=p, Q=120))
        c = cp.solve_uniform(Q0)
        wr = cp.deflection(c, X, Y)
        l2 = 100 * np.sqrt(np.sum(W * (wr - wf) ** 2)) / np.sqrt(np.sum(W * wf**2))
        return l2, cp.w_max_on_grid(c, grid_n=160)

    l2_4, _ = rel_l2(4)
    l2_10, wr10 = rel_l2(10)
    return {"l2_4": l2_4, "l2_10": l2_10, "wmax_rel": abs(wr10 - wf_max) / wf_max}


def test_clamped_lshape_no_paradox(lshape_clamped_cmp):
    """ВОРОТА: RFM↔МКЭ МАЛО (единицы–десяток %), убывает с p, НЕ ~55 % — парадокса нет."""
    c = lshape_clamped_cmp
    assert c["l2_10"] < c["l2_4"]                       # убывает с p ⇒ сходятся к ОДНОМУ решению
    assert c["l2_10"] < 100 * HINGE_PARADOX / 3.0       # ≫ в разы меньше парадокса (54.86 %)
    assert c["l2_10"] < 15.0                            # десяток %, не десятки
    assert c["wmax_rel"] < 0.08                         # по w_max единицы % (факт ~5 %)


# --------------------------------------------------------------------------- #
#  Рисунки
# --------------------------------------------------------------------------- #
def test_clamped_circle_png_created(tmp_path):
    """ВОРОТА: ``clamped_circle.png`` создаётся и непустой."""
    import run_clamped_circle as rcc

    dom = geometry.make_circle(1.0)
    cp = ClampedPlate.from_config(dom, _circle_cfg(10, 256))
    data = {"plate10": cp, "c10": cp.solve_uniform(Q0)}
    out = tmp_path / "clamped_circle.png"
    path = rcc.make_figure(data, save=str(out))
    assert os.path.exists(path) and os.path.getsize(path) > 5000
