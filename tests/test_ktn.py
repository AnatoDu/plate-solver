"""Поправки КТН: сравнение классика ↔ КТН на контакте L-формы (тест-ворота).

Уточнённая теория (Карман–Тимошенко–Нагди) добавляет поперечный сдвиг и обжатие.
В контактной задаче её главный эффект — член податливости ``∝ r`` РЕГУЛЯРИЗУЕТ
сингулярность давления под жёстким штампом: реакция сглаживается и распределяется
шире. Прогиб у КТН больше (теория мягче) ⇒ поправка w_max (NOTES.md §11).

Регламент: умеренная толщина ``h=0.06`` (h/L ~ 0.1), где поправка заметна, но
ещё «поправка», а не смена режима.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending


def _roughness(r: np.ndarray) -> float:
    """Узловая шероховатость поля реакции (норма скачков соседних узлов)."""
    return float(np.sqrt(np.sum(np.diff(r) ** 2)))


@pytest.fixture(scope="module")
def classic_vs_ktn():
    dom = geometry.make_L(1.0, 0.5)
    cfg = Config(nu=0.3, q0=4.0, h=0.06, p=8, Q=36, beta=1.0, max_iter=2000, tol=1e-12, grid_n=30)
    pb = PlateBending.from_config(dom, cfg)
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    gap = 0.6 * float(pb.deflection(cw, q.x, q.y).max())
    fm = lambda x, y: (x + y) < 0.9            # noqa: E731
    classic = ContactMOR(pb, cfg, foundation_mask=fm, gap=gap).solve()
    ktn = ContactMOR(pb, cfg, foundation_mask=fm, gap=gap, ktn=KTNParams.from_config(cfg)).solve()
    return classic, ktn


def test_gate_ktn_smooths_reaction(classic_vs_ktn):
    """ГЛАВНЫЕ ВОРОТА: КТН сглаживает реакцию (ниже пик, шире контакт, глаже)."""
    c, k = classic_vs_ktn
    assert k.r_nodes.max() < 0.6 * c.r_nodes.max()                 # пик существенно ниже
    assert int((k.r_nodes > 0).sum()) > int((c.r_nodes > 0).sum())  # контакт распределён шире
    assert _roughness(k.r_nodes) < 0.7 * _roughness(c.r_nodes)      # глаже


def test_ktn_reaction_nonnegative_and_converges(classic_vs_ktn):
    _, k = classic_vs_ktn
    assert k.r_nodes.min() >= 0.0                                   # односторонняя связь
    h = k.residual_history
    assert h[-1] < h[0] / 10.0                                      # МОР сходится


def test_gate_ktn_corrects_wmax(classic_vs_ktn):
    """ГЛАВНЫЕ ВОРОТА: КТН даёт измеримую поправку w_max (теория мягче ⇒ больше)."""
    c, k = classic_vs_ktn
    assert k.w_ktn_nodes is not None
    rel = (k.w_ktn_nodes.max() - c.w_nodes.max()) / c.w_nodes.max()
    assert rel > 0.05                                              # > 5 % (при h=0.06 ~ +17 %)


def test_ktn_params_reduce_to_small_for_thin_plate():
    """Согласованность: при h→0 поправочные длины → 0 (КТН → классика)."""
    thin = KTNParams(E=2.1e6, nu=0.3, h=1e-3)
    assert thin.c_curv == pytest.approx(0.0, abs=1e-6)
    assert thin.h_psi2 > 0 and thin.h_star2 > 0
