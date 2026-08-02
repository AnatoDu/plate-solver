r"""Ворота вывода полей v0.6.6: мембранные N, σ с мембранной частью, σ_vm, схема 3.

Аудит видения «полноценный комплекс» выявил: N считались, но ТЕРЯЛИСЬ
(несовпадение сигнатуры в export.forces_on_grid, TypeError тихо глотался);
лицевые σ нелинейных теорий шли БЕЗ мембранной части T/h; σ_vm не было;
после нелинейного контакта мембрана была невосстановима; у eigen сохранялась
только первая форма. Здесь — ворота на всё исправленное.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.export import forces_on_grid
from plate_solver.ktn import stresses_faces, von_mises
from plate_solver.problem import CaseError, Problem


def _solve(tmp_path, body):
    p = tmp_path / "case.toml"
    p.write_text(body, encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


_KARMAN = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 3.0
[model]
theory = "karman"
inplane_bc = "immovable"
E = 1.0
nu = 0.3
h = 1.0
karman_tol = 1.0e-8
karman_max_iter = 200
[discretization]
p = 8
Q = 96
grid_n = 20
[verify]
reference = "none"
"""

_CONTACT = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
E = 1.0
nu = 0.3
h = 0.2
n_load_steps = 2
karman_max_iter = 100
karman_tol = 1.0e-6
[contact]
enabled = true
gap_factor = 0.5
scheme = "merged"
max_iter = 2000
tol = 3.0e-4
mor_anderson = 5
[discretization]
p = 8
Q = 96
grid_n = 20
[verify]
reference = "none"
"""


def test_membrane_forces_exported_for_bending(tmp_path):
    """N доходят до forces_on_grid у нелинейного изгиба (был тихий TypeError)."""
    res = _solve(tmp_path, _KARMAN)
    forces = forces_on_grid(res)
    assert {"Nx", "Ny", "Nxy"} <= set(forces)
    assert np.nanmax(np.abs(forces["Nx"])) > 0.0          # мембранное натяжение реально


def test_membrane_forces_recovered_after_contact(tmp_path):
    """Мембрана СОШЕДШЕГОСЯ состояния восстановима и после нелинейного контакта."""
    res = _solve(tmp_path, _CONTACT)
    forces = forces_on_grid(res)
    assert {"Nx", "Ny", "Nxy", "r"} <= set(forces)
    assert np.nanmax(np.abs(forces["Nx"])) > 0.0


def test_fields_npz_schema3_stresses_and_vm(tmp_path):
    """fields.npz схема 3: N, σ_vm; лицевые σ включают мембранную часть T/h."""
    res = _solve(tmp_path, _KARMAN)
    out = tmp_path / "out"
    res.save(out)
    d = np.load(out / "fields.npz")
    assert int(d["fields_schema"]) == 3
    assert {"Nx", "Ny", "Nxy", "svm_top", "svm_bot"} <= set(d.files)
    # мембранная часть в σ: у Кармана верх и низ НЕ антисимметричны (T/h общая)
    s_sum = np.nan_to_num(d["sx_top"] + d["sx_bot"])       # = 2N/h + обжатие
    assert np.max(np.abs(s_sum)) > 0.0
    # σ_vm согласован с шестёркой
    vm = von_mises(d["sx_top"], d["sy_top"], d["txy_top"])
    assert np.allclose(np.nan_to_num(vm), np.nan_to_num(d["svm_top"]), atol=1e-12)


def test_stresses_faces_linear_unchanged():
    """Линейный вызов (N по умолчанию 0) — прежние числа бит-точно (регресс)."""
    M = np.array([[1.0, -2.0]])
    old = {"sx_top": -6.0 * M / 0.2**2, "sx_bot": +6.0 * M / 0.2**2}
    s = stresses_faces(M, 0.5 * M, 0.1 * M, h=0.2, nu=0.3)
    assert np.allclose(s["sx_top"], old["sx_top"]) and np.allclose(s["sx_bot"], old["sx_bot"])


def test_vtk_via_output_key(tmp_path):
    """[output] vtk = true пишет result.vtk с N и r (ParaView-самообслуживание)."""
    body = _CONTACT + f'[output]\ndir = "{tmp_path / "o"}"\nvtk = true\n'
    res = _solve(tmp_path, body)
    res.save(tmp_path / "o")
    txt = (tmp_path / "o" / "result.vtk").read_text(encoding="utf-8")
    assert "SCALARS Nx" in txt and "SCALARS r" in txt


def test_replot_cli_from_npz(tmp_path):
    """plate-replot строит фигуры из ГОЛОГО fields.npz (в т.ч. membrane, von_mises)."""
    from plate_solver.cli import main_replot

    res = _solve(tmp_path, _KARMAN)
    out = tmp_path / "o"
    res.save(out)
    assert main_replot([str(out), "--fig-format", "png"]) == 0
    names = {p.name for p in out.glob("*.png")}
    assert {"membrane.png", "von_mises.png", "stress_faces.png"} <= names


def test_eigen_all_modes_saved(tmp_path):
    """У собственной задачи в npz — ВСЕ формы и значения (раньше только первая)."""
    body = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[model]
theory = "classic"
h = 0.01
[eigen]
kind = "vibration"
n_modes = 3
rho_h = 1.0
[discretization]
p = 10
Q = 72
grid_n = 20
[verify]
reference = "none"
"""
    res = _solve(tmp_path, body)
    out = tmp_path / "o"
    res.save(out)
    d = np.load(out / "fields.npz")
    assert d["eigen_modes"].shape[0] == 3
    assert d["eigen_values"].shape == (3,)
    assert np.allclose(np.nanmax(np.abs(d["eigen_modes"]), axis=(1, 2)), 1.0)


def test_eigen_nonclassic_theory_rejected(tmp_path):
    """[eigen] + нелинейная теория — отказ (раньше принималось и МОЛЧА игнорировалось)."""
    body = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[model]
theory = "karman"
h = 0.01
[eigen]
kind = "vibration"
n_modes = 2
[discretization]
p = 8
Q = 64
"""
    p = tmp_path / "case.toml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(CaseError, match="classic при \\[eigen\\]"):
        Problem.from_toml(str(p))


def test_residual_history_persisted(tmp_path):
    """Полная история сходимости МОР — в result.json (самостоятельные графики)."""
    import json

    res = _solve(tmp_path, _CONTACT)
    out = tmp_path / "o"
    res.save(out)
    j = json.loads((out / "result.json").read_text(encoding="utf-8"))
    hist = j["scalars"]["residual_history"]
    assert len(hist) == res.contact.iters
    assert hist[-1] == pytest.approx(float(res.contact.residual_history[-1]))


# --------------------------------------------------------------------------- #
#  Перерезывающие силы Q (равновесие моментов, v0.6.6)
# --------------------------------------------------------------------------- #
_CLASSIC_CIRCLE_Q = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 2.0
[model]
theory = "classic"
E = 1.0
nu = 0.3
h = 0.1
[discretization]
p = 12
Q = 200
grid_n = 129
[verify]
reference = "none"
"""


def test_shear_forces_match_axisymmetric_equilibrium(tmp_path):
    """Q_r = q·r/2 (точное осесимметричное равновесие, не зависит от КУ).

    Точность задаёт квадратурный пол маски ~1/Q, УСИЛЕННЫЙ производными
    (медиана ~1.4 % при Q=200, убывает с Q — проверено 200→1000); ворота —
    медиана < 2.5 % в кольце 0.2 < r < 0.8.
    """
    from plate_solver.export import shear_forces_on_grid

    res = _solve(tmp_path, _CLASSIC_CIRCLE_Q)
    S = shear_forces_on_grid(res)
    X, Y = res.Xg, res.Yg
    rr = np.hypot(X, Y)
    Qr = (S["Qx"] * X / np.maximum(rr, 1e-12)
          + S["Qy"] * Y / np.maximum(rr, 1e-12))
    band = np.isfinite(Qr) & (rr > 0.2) & (rr < 0.8)
    rel = np.abs(np.abs(Qr[band]) - rr[band]) / rr[band]   # q=2 ⇒ Q_r = r
    assert np.nanmedian(rel) < 2.5e-2


def test_shear_forces_local_equilibrium_rectangle(tmp_path):
    """div Q = −q во внутренности (локальное равновесие; полиномиальная граница)."""
    from plate_solver.export import shear_forces_on_grid

    body = _CLASSIC_CIRCLE_Q.replace(
        'kind = "circle"\na = 1.0',
        'kind = "rectangle"\nx1 = -1.0\nx2 = 1.0\ny1 = -0.6\ny2 = 0.6')
    res = _solve(tmp_path, body)
    S = shear_forces_on_grid(res)
    x, y = res.Xg[0, :], res.Yg[:, 0]
    divQ = (np.gradient(S["Qx"], x, axis=1) + np.gradient(S["Qy"], y, axis=0))
    inner = np.isfinite(divQ)
    inner[:6, :] = inner[-6:, :] = False
    inner[:, :6] = inner[:, -6:] = False
    err = np.abs(divQ[inner] + 2.0) / 2.0                  # q0 = 2
    assert np.nanmedian(err) < 5e-3


def test_shear_saved_and_plotted(tmp_path):
    """Qx, Qy — в fields.npz и на фигуре shear (plate-replot)."""
    from plate_solver.cli import main_replot

    res = _solve(tmp_path, _CLASSIC_CIRCLE_Q)
    out = tmp_path / "o"
    res.save(out)
    d = np.load(out / "fields.npz")
    assert {"Qx", "Qy"} <= set(d.files)
    assert main_replot([str(out), "--fig-format", "png"]) == 0
    assert (out / "shear.png").exists()


# --------------------------------------------------------------------------- #
#  Профили сечений и наложения (v0.6.6)
# --------------------------------------------------------------------------- #
def test_section_profile_and_overlay_cli(tmp_path):
    """Профиль вдоль сечения из npz + CLI-наложение двух результатов + CSV.

    Сечение по диаметру защемлённого круга: w максимален в центре, NaN вне Ω;
    наложение classic vs karman — мембранное ужесточение видно на кривой.
    """
    from plate_solver.cli import main_profile
    from plate_solver.viz import section_profile

    dirs = []
    for theory, extra in (("classic", ""),
                          ("karman", 'inplane_bc = "immovable"\n'
                                     "karman_tol = 1.0e-8\nkarman_max_iter = 200\n")):
        body = f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 3.0
[model]
theory = "{theory}"
E = 1.0
nu = 0.3
h = 1.0
{extra}[discretization]
p = 8
Q = 96
grid_n = 48
[verify]
reference = "none"
"""
        res = _solve(tmp_path, body)
        out = tmp_path / theory
        res.save(out)
        dirs.append(str(out))
    s, v = section_profile(dirs[0], "w", (-1.0, 0.0), (1.0, 0.0), n=101)
    assert np.isnan(v[0]) and np.isfinite(v[50])          # вне Ω — NaN, центр — конечен
    assert v[50] == pytest.approx(np.nanmax(v), rel=1e-6)  # максимум в центре
    csv = tmp_path / "prof.csv"
    fig = tmp_path / "prof.png"
    rc = main_profile([*dirs, "--key", "w", "--from", "-1,0", "--to", "1,0",
                       "-n", "101", "--csv", str(csv), "--fig", str(fig)])
    assert rc == 0 and csv.exists() and fig.exists()
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert data.shape == (101, 3)
    assert data[50, 2] < data[50, 1]                      # Карман жёстче классики
