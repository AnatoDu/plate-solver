"""Общие фикстуры тестов."""

import pytest

from plate_solver import PlateMaterial


@pytest.fixture
def steel_plate():
    """Типовая стальная пластина (СИ): E=2e11 Па, nu=0.3, h=0.01 м."""
    return PlateMaterial(E=2.0e11, nu=0.3, h=0.01)
