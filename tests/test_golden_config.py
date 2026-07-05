"""Мост GoldenConfig → Config (P1.4): соответствие имён сведено в одном месте.

Проверяем, что to_config() воспроизводит лабораторный конфиг L-серии
(бывший run_lshape_contact.lshape_lab_config) и что переименования
mor_iter → max_iter, mor_tol → tol не разъезжаются.
"""

from __future__ import annotations

from golden_config import GoldenConfig


def test_to_config_lshape_defaults():
    """Дефолт to_config() — L-серия: h=h_ktn, Q=Q_lshape, имена МОР сведены."""
    g = GoldenConfig()
    lab = g.to_config()
    assert lab.h == g.h_ktn and lab.Q == g.Q_lshape
    assert (lab.nu, lab.q0, lab.E, lab.a) == (g.nu, g.q0, g.E, g.a)
    assert (lab.p, lab.grid_n) == (g.p, g.grid_n)
    assert lab.beta == g.beta
    assert lab.max_iter == g.mor_iter        # mor_iter → max_iter
    assert lab.tol == g.mor_tol              # mor_tol → tol
    assert lab.stop == "dr"                  # критерий останова golden — прежний
    # жёсткость L-серии согласована со свойством golden-конфига
    assert lab.D == g.D_lshape


def test_to_config_circle_overrides():
    """Круг Табл. 4.1: h и Q переопределяются, p — через overrides."""
    g = GoldenConfig()
    lab = g.to_config(h=g.h_circle, Q=g.Q_circle, p=4)
    assert lab.h == g.h_circle and lab.Q == g.Q_circle and lab.p == 4
    assert lab.D == g.D                      # жёсткость круга (h=1.0)


def test_gate_count_is_automatic():
    """P1.5: число ворот в подписи golden считается pytest'ом, а не хардкодом."""
    from run_golden import _count_gates

    n = _count_gates()
    assert n is not None and n >= 90         # в объединённой базе ≥ 90 ворот

