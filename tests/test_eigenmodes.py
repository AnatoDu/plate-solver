r"""Собственные задачи пластины: устойчивость и колебания (v0.6.4).

Верификация против КЛАССИЧЕСКИХ эталонов Кирхгофа (Тимошенко «Устойчивость
стержней, пластин и оболочек»; Лейсса «Vibration of Plates», NASA SP-160):

* устойчивость — безразмерный коэффициент ``k = N_cr·b²/(π²D)`` (одноосное
  сжатие) или ``N_cr·a²/D`` (радиальное сжатие круга);
* колебания — частотный параметр ``λ = ω·a²·√(ρh/D)``.

Эти эталоны СУЩЕСТВУЮТ в литературе (в отличие от КТН-контакта) — настоящая
внешняя верификация нового типа анализа.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver.config import Config
from plate_solver.eigenmodes import buckling, linear_plate, natural_frequencies
from plate_solver.geometry import make_circle, make_rectangle

_A = 1.0                                                  # характерный размер (сторона/радиус)


def _cfg(p=12, Q=72, E=1.0):
    return Config(E=E, nu=0.3, h=0.01, q0=0.0, p=p, Q=Q)


def _D():
    c = _cfg()
    return c.D


# --------------------------------------------------------------------------- #
#  Устойчивость (потеря устойчивости)
# --------------------------------------------------------------------------- #
def test_buckling_square_simply_supported():
    """Квадрат SSSS, одноосное сжатие: k = N_cr·b²/(π²D) = 4 (Тимошенко)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    ncr = buckling(plate, Nx=-1.0).values[0]
    k = ncr * _A**2 / (np.pi**2 * _D())
    assert abs(k - 4.0) / 4.0 < 1e-2


def test_buckling_square_clamped():
    """Квадрат CCCC, одноосное сжатие: k ≈ 10.07 (Тимошенко)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="clamped")
    ncr = buckling(plate, Nx=-1.0).values[0]
    k = ncr * _A**2 / (np.pi**2 * _D())
    assert abs(k - 10.07) / 10.07 < 1.5e-2


def test_buckling_circle_clamped_radial():
    """Круг CCCC, равномерное радиальное сжатие: N_cr·a²/D = 14.68 (Тимошенко)."""
    plate = linear_plate(make_circle(_A), _cfg(), bc_type="clamped")
    ncr = buckling(plate, Nx=-1.0, Ny=-1.0).values[0]
    assert abs(ncr * _A**2 / _D() - 14.68) / 14.68 < 2e-2


def test_buckling_factors_positive_sorted():
    """Множители устойчивости положительны и упорядочены по возрастанию."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    v = buckling(plate, Nx=-1.0, n_modes=4).values
    assert (v > 0).all() and np.all(np.diff(v) >= -1e-9) and v.size == 4


# --------------------------------------------------------------------------- #
#  Свободные колебания
# --------------------------------------------------------------------------- #
def test_vibration_square_simply_supported():
    """Квадрат SSSS: λ_mn = π²(m²+n²) ⇒ λ1 = 2π² = 19.74; моды (1,2)/(2,1) вырождены."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    w = natural_frequencies(plate, rho_h=1.0, n_modes=4).values
    lam = w * _A**2 * np.sqrt(1.0 / _D())                 # λ = ω·a²·√(ρh/D)
    assert abs(lam[0] - 2 * np.pi**2) / (2 * np.pi**2) < 1e-2
    assert abs(lam[1] - lam[2]) / lam[1] < 1e-2           # вырождение (1,2)=(2,1)
    assert abs(lam[1] - 5 * np.pi**2) / (5 * np.pi**2) < 2e-2


def test_vibration_circle_clamped():
    """Круг CCCC: фундаментальный частотный параметр λ1 = 10.2158 (Лейсса)."""
    plate = linear_plate(make_circle(_A), _cfg(), bc_type="clamped")
    w = natural_frequencies(plate, n_modes=3).values
    lam1 = w[0] * _A**2 * np.sqrt(1.0 / _D())
    assert abs(lam1 - 10.2158) / 10.2158 < 2e-2


def test_vibration_square_clamped():
    """Квадрат CCCC: λ1 = 35.99 (Лейсса)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="clamped")
    w = natural_frequencies(plate, n_modes=1).values
    lam1 = w[0] * _A**2 * np.sqrt(1.0 / _D())
    assert abs(lam1 - 35.99) / 35.99 < 1e-2


def test_mode_shape_on_grid():
    """Форма колебаний нормирована (max|·|=1) и обращается в нуль вне Ω (NaN)."""
    plate = linear_plate(make_circle(_A), _cfg(), bc_type="clamped")
    res = natural_frequencies(plate, n_modes=2)
    _Xg, _Yg, W = res.mode_on_grid(0, grid_n=40)
    assert np.nanmax(np.abs(W)) == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(W).any()                              # вне круга — NaN


# --------------------------------------------------------------------------- #
#  Преднапряжённые колебания: частоты под РЕАЛЬНЫМ полем усилий N(w) (v0.6.4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N0", [5e-6, 2e-5, 1e-4])
def test_vibration_prestress_matches_exact_biaxial_tension(N0):
    """SSSS-квадрат под двухосным натяжением: λ = √(4π⁴ + 2N₀π²a²/D) (точно)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    w1 = natural_frequencies(plate, rho_h=1.0, n_modes=1, prestress=(N0, N0, 0.0)).values[0]
    lam = w1 * _A**2 * np.sqrt(1.0 / _D())
    exact = np.sqrt(4 * np.pi**4 + 2 * N0 * np.pi**2 * _A**2 / _D())
    assert abs(lam - exact) / exact < 1e-3               # натяжение ужесточает — точная формула


def test_vibration_prestress_zero_is_unstressed():
    """Нулевое преднапряжение ≡ ненапряжённые частоты (машинная точность)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    w0 = natural_frequencies(plate, n_modes=3).values
    wz = natural_frequencies(plate, n_modes=3, prestress=(0.0, 0.0, 0.0)).values
    assert np.max(np.abs(w0 - wz) / w0) < 1e-12


def test_vibration_under_real_load_raises_frequency():
    """Колебания под РЕАЛЬНЫМ полем N(w) от поперечной нагрузки: частота РАСТЁТ с нагрузкой.

    Кармановское натяжение срединной поверхности (``N(w) > 0``) ужесточает пластину;
    фундаментальная частота монотонно повышается с уровнем нагрузки — физический эффект.
    """
    from plate_solver.membrane import KarmanPlate

    dom = make_circle(1.0)
    base = dict(E=1.0, nu=0.3, h=0.1, a=1.0, p=12, Q=140,
                n_load_steps=3, karman_tol=1e-9, karman_max_iter=400)
    w_unstressed = natural_frequencies(
        linear_plate(dom, Config(q0=0.0, **base), bc_type="clamped"), n_modes=1).values[0]
    freqs = []
    for P_bar in (2.0, 6.0, 12.0):
        kp = KarmanPlate.from_config(dom, Config(q0=P_bar * 0.1**4, **base),
                                     bc_type="clamped", inplane_bc="immovable")
        res = kp.solve_uniform()
        freqs.append(natural_frequencies(kp, n_modes=1, prestress=res).values[0])
    assert freqs[0] > w_unstressed                        # натяжение повышает частоту
    assert freqs[0] < freqs[1] < freqs[2]                 # монотонно с нагрузкой


# --------------------------------------------------------------------------- #
#  Масштабная инвариантность отбора собственных пар (находка аудита V02, v0.8.0)
# --------------------------------------------------------------------------- #
#  Безразмерные ``λ = ω·a²·√(ρh/D)`` и ``k = N_cr·b²/(π²D)`` ОБЯЗАНЫ не зависеть
#  от размерных масштабов ``E`` и ``ρh``: замена ``E → cE`` умножает обе части
#  ``Kφ = ω²Mφ`` на ``c`` и спектр ``ω²`` — тоже, оставляя ``λ`` на месте.
#  Прежний фильтр в ``_gen_eig`` сравнивал РАЗМЕРНЫЕ ``ω²`` с абсолютным порогом
#  ``1e-9`` и при малой ``D/ρh`` МОЛЧА терял основной тон (для квадрата SSSS при
#  ``E = 1e-5``: точное ``ω₁² = 3.57e-10`` отбрасывалось, первой возвращалась
#  мода (1,2) ⇒ λ₁ = 49.35 вместо 19.74).  Допуск на согласие с эталоном —
#  дискретизационный (p, Q); допуск на РАЗБРОС по масштабам — машинный.
_SCALES = [(1.0, 1.0), (1e-2, 1.0), (1e-4, 1.0), (1e-6, 1.0), (1.0, 1e-4), (1e-6, 1e4)]


def _lambda1_vibration(dom, bc, Q, E, rho_h, n_modes=1):
    """Безразмерный частотный параметр λ₁ = ω₁·a²·√(ρh/D) при заданных E, ρh."""
    cfg = _cfg(p=8, Q=Q, E=E)
    plate = linear_plate(dom, cfg, bc_type=bc)
    w1 = natural_frequencies(plate, rho_h=rho_h, n_modes=n_modes).values[0]
    return w1 * _A**2 * np.sqrt(rho_h / cfg.D)


def test_vibration_scale_invariant_square():
    """SSSS-квадрат: λ₁ = 2π² при E и ρh, меняющихся на 6 и 4 порядка."""
    dom = make_rectangle(0, _A, 0, _A)
    lam = np.array([_lambda1_vibration(dom, "soft_hinge", 32, E, rho) for E, rho in _SCALES])
    assert abs(lam[0] - 2 * np.pi**2) / (2 * np.pi**2) < 1e-2      # эталон (дискретизация)
    assert np.ptp(lam) / lam[0] < 1e-9                               # инвариантность (машинная)


def test_vibration_scale_invariant_circle():
    """Круг CCCC: λ₁ = 10.2158 (Лейсса) не зависит от масштабов E, ρh."""
    lam = np.array([_lambda1_vibration(make_circle(_A), "clamped", 48, E, rho)
                    for E, rho in _SCALES])
    assert abs(lam[0] - 10.2158) / 10.2158 < 1e-2
    assert np.ptp(lam) / lam[0] < 1e-9


def test_buckling_scale_invariant_square():
    """Ветвь устойчивости (B = −K_geo, знак другой): k = 4 не зависит от E."""
    dom = make_rectangle(0, _A, 0, _A)
    ks = []
    for E in (1.0, 1e-2, 1e-4, 1e-6):
        cfg = _cfg(p=8, Q=32, E=E)
        ncr = buckling(linear_plate(dom, cfg, bc_type="soft_hinge"), Nx=-1.0).values[0]
        ks.append(ncr * _A**2 / (np.pi**2 * cfg.D))
    ks = np.array(ks)
    assert abs(ks[0] - 4.0) / 4.0 < 1e-2
    assert np.ptp(ks) / ks[0] < 1e-9


@pytest.mark.parametrize("E", [1.0, 1e-6])
@pytest.mark.parametrize("dom,bc,Q", [(make_rectangle(0, _A, 0, _A), "soft_hinge", 32),
                                      (make_circle(_A), "clamped", 48)])
def test_vibration_matches_symmetric_definite_solver(dom, bc, Q, E):
    """Отобранные ω² совпадают с независимым решателем ``sla.eigh(K, M)``.

    Матрица масс ``M = ρh∫ψψ`` положительно определена, поэтому пару ``(K, M)``
    можно решить симметрично-определённым алгоритмом (разложение Холецкого) —
    он не проходит через фильтр ``_gen_eig`` и служит НЕЗАВИСИМЫМ сертификатом
    того, что фильтр не выбрасывает физические моды ни при каком масштабе.
    (Для ветви устойчивости такой сверки нет: ``B = −K_geo`` знакопеременна при
    немонотонном поле ``N``, Холецкий неприменим — потому общий тракт идёт через
    ``sla.eig``.)
    """
    import scipy.linalg as sla

    from plate_solver.eigenmodes import _bending_stiffness, _mass_matrix

    plate = linear_plate(dom, _cfg(p=8, Q=Q, E=E), bc_type=bc)
    K, M = _bending_stiffness(plate), _mass_matrix(plate, 1.0)
    w2 = natural_frequencies(plate, rho_h=1.0, n_modes=4).values**2
    ref = np.sort(sla.eigh(K, M, eigvals_only=True))[:4]
    assert np.max(np.abs(w2 - ref) / ref) < 1e-9


def test_vibration_overcritical_compression_raises():
    """Сжатие сверх наибольшего критического множителя: честный отказ, не пустой ответ.

    При ``N`` сжимающем сверх критического ``K + K_geo(N)`` теряет положительную
    определённость, положительных ``ω²`` не остаётся — колебаний около этого
    положения равновесия нет. Требуется внятная ошибка (прежде — молча пустой
    массив значений и ``IndexError`` у вызывающего).
    """
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(p=8, Q=32), bc_type="soft_hinge")
    lam_max = buckling(plate, Nx=-1.0, Ny=-1.0, n_modes=10**4).values[-1]
    with pytest.raises(ValueError, match="устойчивость"):
        natural_frequencies(plate, n_modes=1, prestress=(-2 * lam_max, -2 * lam_max, 0.0))


def test_buckling_field_matches_uniform_scalar():
    """Устойчивость под РАВНОМЕРНЫМ полем (prestress) = скалярная запись (согласованность API)."""
    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    o = np.ones(plate._W.size)
    lam_scalar = buckling(plate, Nx=-1.0, n_modes=1).values[0]
    lam_field = buckling(plate, prestress=(-o, 0.0 * o, 0.0 * o), n_modes=1).values[0]
    assert abs(lam_scalar - lam_field) / lam_scalar < 1e-10


def test_case_file_eigen_matches_benchmark():
    """Case-файл [eigen] через диспетчер даёт эталонные значения (колебания/устойчивость)."""
    from pathlib import Path

    from plate_solver import dispatch
    from plate_solver.problem import Problem

    root = Path(__file__).resolve().parents[1] / "cases" / "ci"
    r = dispatch.solve(Problem.from_toml(root / "eigen_vibration_square.toml"))
    assert r.eigen is not None and r.eigen.kind == "vibration"
    lam1 = r.eigen.values[0] * np.sqrt(1.0 / r.config.D)
    assert abs(lam1 - 2 * np.pi**2) / (2 * np.pi**2) < 1e-2

    r = dispatch.solve(Problem.from_toml(root / "eigen_buckling_circle.toml"))
    assert r.eigen is not None and r.eigen.kind == "buckling"
    assert abs(r.eigen.values[0] / r.config.D - 14.68) / 14.68 < 2e-2


def test_eigen_rejects_incompatible():
    """[eigen] несовместим с нагрузкой/контактом/смешанными КУ — понятный отказ."""
    from plate_solver.problem import CaseError, Problem

    base = ('[geometry]\nkind = "circle"\na = 1.0\n[bc]\ntype = "clamped"\n'
            '[eigen]\nkind = "vibration"\n[discretization]\np = 8\nQ = 48\n')
    # [load] вместе с [eigen]
    with_load = base.replace("[eigen]", '[load]\ntype="uniform"\nq0=1.0\n[eigen]')
    with pytest.raises(CaseError, match="eigen"):
        Problem.from_dict(_to_dict(with_load))
    # mixed КУ
    with pytest.raises(CaseError, match="clamped"):
        Problem.from_dict(_to_dict(
            '[geometry]\nkind = "rectangle"\nx1=0.0\nx2=1.0\ny1=0.0\ny2=1.0\n'
            '[bc]\ntype = "mixed"\n[[bc.sides]]\nside="x1"\ntype="clamped"\n'
            '[[bc.sides]]\nside="x2"\ntype="clamped"\n[[bc.sides]]\nside="y1"\ntype="hinge"\n'
            '[[bc.sides]]\nside="y2"\ntype="hinge"\n[eigen]\nkind="buckling"\n'
            '[discretization]\np=8\nQ=48\n'))


def _to_dict(toml_text):
    import tomllib
    return tomllib.loads(toml_text)


def test_plot_modes_renders(tmp_path):
    """viz.plot_modes рисует сетку форм и сохраняет непустой png."""
    import matplotlib

    matplotlib.use("Agg")
    from plate_solver import viz

    plate = linear_plate(make_rectangle(0, _A, 0, _A), _cfg(), bc_type="soft_hinge")
    res = buckling(plate, Nx=-1.0, n_modes=4)
    out = tmp_path / "modes.png"
    viz.plot_modes(res, n=4, grid_n=40, save=str(out))
    assert out.exists() and out.stat().st_size > 10_000
