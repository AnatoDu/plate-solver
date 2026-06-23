"""Smoke-тест: пакет импортируется, версия определена, материал считает D."""

import plate_solver
from plate_solver import PlateMaterial, flexural_rigidity


def test_version_defined():
    assert isinstance(plate_solver.__version__, str)
    assert plate_solver.__version__


def test_flexural_rigidity_positive(steel_plate):
    assert steel_plate.D > 0
    # D = E h^3 / (12 (1 - nu^2))
    expected = flexural_rigidity(steel_plate.E, steel_plate.h, steel_plate.nu)
    assert abs(steel_plate.D - expected) < 1e-12 * expected


def test_material_validates_input():
    import pytest

    with pytest.raises(ValueError):
        PlateMaterial(E=2.0e11, nu=0.3, h=0.0).D  # нулевая толщина
