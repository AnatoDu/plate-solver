"""Тест-ворота базиса Чебышёва: размер, форма, φ_0≡1 и ∇φ против разностей.

Производные базиса нужны для сборки Ритца (∇ψ = ∇ω·φ + ω·∇φ), поэтому ключевая
проверка — совпадение аналитических grads с конечной разностью, в т. ч. на
неединичном bbox (проверяет множители ξ_x, η_y отображения).
"""

from __future__ import annotations

import numpy as np
import pytest
from plates.basis import ChebyshevBasis


def test_size_and_indexing():
    bs = ChebyshevBasis(4, (-1.0, 1.0, -1.0, 1.0))
    assert bs.N == 25
    assert bs.index_pairs[0] == (0, 0)
    assert bs.index_pairs[-1] == (4, 4)
    assert len(bs.index_pairs) == bs.N


def test_values_shape_and_constant_mode():
    bs = ChebyshevBasis(3, (-1.0, 1.0, -1.0, 1.0))
    X = np.array([0.0, 0.3, -0.7])
    Y = np.array([0.0, -0.4, 0.2])
    V = bs.values(X, Y)
    assert V.shape == (bs.N, 3)
    # φ_0 = T_0·T_0 = 1 во всех точках (k = 0 для (i,j)=(0,0)).
    assert np.allclose(V[0], 1.0)


def test_scalar_input_shapes():
    bs = ChebyshevBasis(5, (-2.0, 2.0, -1.0, 3.0))
    V = bs.values(0.1, 0.2)
    assert V.shape == (bs.N,)
    gx, gy = bs.grads(0.1, 0.2)
    assert gx.shape == (bs.N,) and gy.shape == (bs.N,)


@pytest.mark.parametrize("bbox", [(-1.0, 1.0, -1.0, 1.0), (0.0, 2.0, 0.0, 4.0)])
def test_gate_grads_match_finite_difference(bbox):
    """∇φ совпадает с центральной разностью (учитывает множители ξ_x, η_y)."""
    bs = ChebyshevBasis(6, bbox)
    rng = np.random.default_rng(0)
    xmin, xmax, ymin, ymax = bbox
    X = rng.uniform(xmin + 0.1, xmax - 0.1, size=5)
    Y = rng.uniform(ymin + 0.1, ymax - 0.1, size=5)
    gx, gy = bs.grads(X, Y)
    eps = 1e-6
    fdx = (bs.values(X + eps, Y) - bs.values(X - eps, Y)) / (2 * eps)
    fdy = (bs.values(X, Y + eps) - bs.values(X, Y - eps)) / (2 * eps)
    np.testing.assert_allclose(gx, fdx, atol=1e-5)
    np.testing.assert_allclose(gy, fdy, atol=1e-5)
