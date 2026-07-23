r"""Ньютон-ускорение итерации Пикара (Карман/КТН, §5.4, v0.6.4).

Согласованный касательный оператор ``J = dR/dc`` даёт КВАДРАТИЧНУЮ сходимость
(~5–7 итераций против десятков/сотен у Пикара, слабо зависит от уровня
нагрузки). Ключевая проверка корректности — КОНЕЧНЫЕ РАЗНОСТИ: ``J·δ`` совпадает
с ``(R(c+εδ) − R(c−εδ))/2ε`` до ~1e-10 (иначе Ньютон не сходился бы квадратично).
Плюс: Ньютон даёт ТО ЖЕ решение, что Пикар, за меньшее число итераций.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver.config import Config
from plate_solver.geometry import make_circle
from plate_solver.ktn_full import KTNPlate
from plate_solver.membrane import KarmanPlate


def _cfg(method_key, method="newton", h=1.0, Pbar=6.0, p=10, Q=128):
    return Config(E=1.0, nu=0.3, h=h, q0=Pbar, a=1.0, p=p, Q=Q, n_load_steps=1,
                  karman_tol=1e-8, karman_max_iter=300, karman_relax=1.0,
                  **{method_key: method})


def _fd_tangent_error(plate, b, c0):
    r"""Максимальная относительная невязка ``J·δ`` vs КР по нескольким направлениям."""
    forces = plate._membrane_forces(c0 @ plate._psi_x, c0 @ plate._psi_y)
    J = plate._newton_tangent(c0, forces)
    eps, errs = 1e-6, []
    rng = np.random.default_rng(0)
    for _ in range(3):
        d = rng.standard_normal(plate.basis.N)
        d /= np.linalg.norm(d)
        fd = (plate._residual(c0 + eps * d, b) - plate._residual(c0 - eps * d, b)) / (2 * eps)
        errs.append(np.linalg.norm(J @ d - fd) / max(np.linalg.norm(fd), 1e-30))
    return max(errs)


def test_karman_tangent_finite_difference():
    """Кармановский касательный оператор совпадает с КР (аналит. вывод верен)."""
    plate = KarmanPlate.from_config(make_circle(1.0), _cfg("karman_method", p=8, Q=96),
                                    bc_type="clamped", inplane_bc="immovable")
    b = plate._load_vector(np.full(plate.quad.x.size, 6.0))
    c0 = np.random.default_rng(1).standard_normal(plate.basis.N) * 1e-2
    assert _fd_tangent_error(plate, b, c0) < 1e-7


def test_ktn_tangent_finite_difference():
    """КТН касательный оператор (защемление) совпадает с КР — точен по построению."""
    plate = KTNPlate.from_config(make_circle(1.0), _cfg("ktn_method", p=8, Q=96),
                                 bc_type="clamped")
    b = plate._load_vector(np.full(plate.quad.x.size, 6.0))
    c0 = np.random.default_rng(2).standard_normal(plate.basis.N) * 1e-2
    assert _fd_tangent_error(plate, b, c0) < 1e-7


@pytest.mark.parametrize("Pbar", [3.0, 6.321, 12.0])
def test_karman_newton_matches_picard_fewer_iters(Pbar):
    """Карман-Ньютон = Пикар (то же решение), но за меньше итераций."""
    dom = make_circle(1.0)
    rn = KarmanPlate.from_config(dom, _cfg("karman_method", "newton", Pbar=Pbar),
                                 bc_type="clamped", inplane_bc="immovable").solve_uniform()
    rp = KarmanPlate.from_config(dom, _cfg("karman_method", "picard", Pbar=Pbar),
                                 bc_type="clamped", inplane_bc="immovable").solve_uniform()
    assert rn.converged
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-4
    assert rn.n_iter < rp.n_iter


def test_ktn_newton_matches_picard_fewer_iters():
    """КТН-Ньютон = Пикар (то же решение), меньше итераций; сходится где Пикар медлит."""
    dom = make_circle(1.0)
    rn = KTNPlate.from_config(dom, _cfg("ktn_method", "newton", h=1.0, Pbar=12.0),
                              bc_type="clamped").solve_uniform()
    rp = KTNPlate.from_config(dom, _cfg("ktn_method", "picard", h=1.0, Pbar=12.0),
                              bc_type="clamped").solve_uniform()
    assert rn.converged
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-3
    assert rn.n_iter < rp.n_iter


def test_newton_soft_hinge_ktn_runs():
    """КТН-Ньютон на мягком шарнире (модифиц. касат.: граничный член §3.5) — сходится."""
    dom = make_circle(1.0)
    r = KTNPlate.from_config(dom, _cfg("ktn_method", "newton", h=0.2, Pbar=0.02, p=8, Q=64),
                             bc_type="soft_hinge").solve_uniform()
    assert np.isfinite(r.w_max) and r.n_iter < 300


def test_bad_method_rejected():
    """Неизвестный метод — понятная ошибка."""
    dom = make_circle(1.0)
    with pytest.raises(NotImplementedError, match="picard | newton"):
        KarmanPlate.from_config(dom, _cfg("karman_method", "bfgs"),
                                bc_type="clamped").solve_uniform()


# --------------------------------------------------------------------------- #
#  Сквозной вход: case-файл → диспетчер → Ньютон (v0.6.4)
# --------------------------------------------------------------------------- #
def _dispatch_case(method_key, method):
    import tomllib

    from plate_solver import dispatch
    from plate_solver.problem import Problem
    theory = "karman" if method_key == "karman_method" else "ktn_full"
    toml = f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 6.0
[model]
theory = "{theory}"
inplane_bc = "immovable"
E = 1.0
nu = 0.3
h = 1.0
karman_tol = 1e-9
karman_max_iter = 300
{method_key} = "{method}"
[discretization]
p = 10
Q = 128
grid_n = 16
[verify]
reference = "none"
"""
    return dispatch.solve(Problem.from_dict(tomllib.loads(toml)))


@pytest.mark.parametrize("method_key", ["karman_method", "ktn_method"])
def test_newton_via_dispatch_matches_picard(method_key):
    """Сквозной вход case-файл→диспетчер: Ньютон = Пикар (то же поле), обе теории."""
    rn = _dispatch_case(method_key, "newton")
    rp = _dispatch_case(method_key, "picard")
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-3
