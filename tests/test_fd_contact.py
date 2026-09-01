r"""Ворота КР+МОР (fd_contact.py, v0.7.0) — метод-дублёр контакта на прямоугольнике.

Независимая схема-дублёр: локальный (конечно-разностный) базис +
разреженная СЛАУ на каждом шаге МОР (здесь — splu-
факторизация ОДИН раз). Ворота:

* изгиб: сходимость к ТОЧНОМУ ряду Навье порядка O(h²);
* контакт: совпадение с RFM+Ритц МОР (независимая дискретизация) по прогибу
  и суммарной реакции — взаимный сертификат двух методов;
* гладкость реакции: у локального базиса профиль реакции без «звона»
  (считанные экстремумы: плато + краевые концентрации — дискретный аналог
  кольцевой реакции; мотивация модуля — уход от осцилляций глобальных базисов).
"""

from __future__ import annotations

import numpy as np

from plate_solver import dispatch
from plate_solver.fd_contact import FDPlateSS, fd_contact_foundation
from plate_solver.ladder import navier_uniform_center
from plate_solver.problem import Problem

_RECT = (0.0, 2.0, 0.0, 1.2)
_D, _Q0 = 1.0, 1.0


def test_fd_bending_converges_to_navier_second_order():
    """Изгиб SS-прямоугольника: FD → ряд Навье (точный) порядка O(h²)."""
    x1, x2, y1, y2 = _RECT
    w_ex, _mx = navier_uniform_center(x2 - x1, y2 - y1, _D, _Q0)
    rels = []
    for n in (41, 81):
        fd = FDPlateSS(x1, x2, y1, y2, int(n * (x2 - x1) / (y2 - y1)) | 1, n, _D)
        w = fd.solve(_Q0)
        ic = np.unravel_index(np.argmin((fd.X - 1.0) ** 2 + (fd.Y - 0.6) ** 2),
                              fd.X.shape)
        rels.append(abs(w[ic] - w_ex) / w_ex)
    assert rels[1] < 5e-5                                 # близко к точному
    order = np.log2(rels[0] / rels[1])
    assert 1.7 < order < 2.3                              # порядок аппроксимации ~2


def test_fd_mor_matches_rfm_mor(tmp_path):
    """Контакт с основанием: КР+МОР = RFM+Ритц МОР (независимые дискретизации).

    Взаимный сертификат: прогиб и суммарная реакция совпадают на уровне
    КР-дискретизации (~1e-3); зона непуста, r ≥ 0.
    """
    x1, x2, y1, y2 = _RECT
    w_ex, _ = navier_uniform_center(x2 - x1, y2 - y1, _D, _Q0)
    gap = 0.5 * w_ex
    fdc = fd_contact_foundation(x1, x2, y1, y2, D=_D, q0=_Q0, gap=gap,
                                nx=135, ny=81, tol=1e-7)
    assert fdc.n_contact > 0 and np.all(fdc.r >= 0.0)
    case = f"""
[geometry]
kind = "rectangle"
x1 = {x1}
x2 = {x2}
y1 = {y1}
y2 = {y2}
[bc]
type = "soft_hinge"
[load]
type = "uniform"
q0 = {_Q0}
[model]
theory = "classic"
E = {12 * (1 - 0.3**2)}
nu = 0.3
h = 1.0
[contact]
enabled = true
gap = {gap}
max_iter = 4000
tol = 1.0e-8
[discretization]
p = 12
Q = 200
grid_n = 32
[verify]
reference = "none"
"""
    p = tmp_path / "case.toml"
    p.write_text(case, encoding="utf-8")
    res = dispatch.solve(Problem.from_toml(str(p)))
    r_total_rfm = float(np.sum(res.contact.r_nodes * res._plate.quad.w))
    assert abs(fdc.w_max - res.w_max) / res.w_max < 5e-3
    assert abs(fdc.r_total - r_total_rfm) / r_total_rfm < 5e-3


def test_fd_reaction_profile_no_ringing():
    """Профиль реакции КР — БЕЗ «звона» (мотивация ухода от ряда Навье).

    Подлинная структура реакции — гладкое плато + краевые концентрации
    (дискретный аналог кольцевой реакции сертифицированного решения) — даёт
    считанные экстремумы; звон глобального базиса дал бы десятки знакопеременных
    осцилляций. Ворота: ≤ 5 локальных экстремумов + симметрия профиля.
    """
    x1, x2, y1, y2 = _RECT
    w_ex, _ = navier_uniform_center(x2 - x1, y2 - y1, _D, _Q0)
    fdc = fd_contact_foundation(x1, x2, y1, y2, D=_D, q0=_Q0, gap=0.5 * w_ex,
                                nx=135, ny=81, tol=1e-7)
    j = fdc.r.shape[0] // 2
    line = fdc.r[j, :]
    d = np.diff(line)
    n_extrema = int(np.count_nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    assert n_extrema <= 5                                  # плато + краевые пики, не звон
    assert np.allclose(line, line[::-1], rtol=1e-6, atol=1e-9 * line.max())


def test_fd_reduction_no_contact():
    """Большой зазор ⇒ реакция ≡ 0, прогиб = свободному (редукция R1)."""
    x1, x2, y1, y2 = _RECT
    fdc = fd_contact_foundation(x1, x2, y1, y2, D=_D, q0=_Q0, gap=1.0,
                                nx=67, ny=41, tol=1e-8, max_iter=200)
    fd = FDPlateSS(x1, x2, y1, y2, 67, 41, _D)
    w_free = fd.solve(_Q0)
    assert fdc.n_contact == 0 and fdc.r_total == 0.0
    assert np.allclose(fdc.w, w_free, rtol=1e-12)
