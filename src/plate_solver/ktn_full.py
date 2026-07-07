r"""ktn_full.py — ПОЛНАЯ нелинейная теория Кармана–Тимошенко–Нагди (КТН, §3, §5).

КТН = Карман (`membrane.py`) плюс два добавочных члена во внеплоскостном
уравнении (§3.4) и лицевая постобработка (`faces.py`). Первые два уравнения
системы (§3.1) полностью определяют ``w`` и ``Φ``:

.. math::
    D\,\Delta^2 w = q_n - h_*^2\Delta q_n + (I - h_\psi^2\Delta)\,L(\Phi, w),\qquad
    \tfrac{1}{Eh}\Delta^2\Phi = -\tfrac12 L(w, w),

где ``L(\Phi, w) = N_x w_{,xx} + 2N_{xy} w_{,xy} + N_y w_{,yy}``. Относительно
решателя Кармана (``D\Delta^2 w = q_n + L``) добавляется:

* **(A) нагрузочный** ``-h_*^2\Delta q_n``: под равномерной нагрузкой ``Δq_n = 0``
  (проявляется при неравномерной нагрузке, веха N3);
* **(B) регуляризация связи** ``-h_\psi^2\,\Delta L`` — главный новый оператор.

Слабая форма (B) БЕЗ производных 4-го порядка (ключевой приём §3.5). По формуле
Грина при ЗАЩЕМЛЕНИИ (``v = ∂_n v = 0`` на ∂Ω ⇒ граничный интеграл ноль):

.. math:: -h_\psi^2\!\int_\Omega v\,\Delta L\,dA = -h_\psi^2\!\int_\Omega L\,\Delta v\,dA.

Лапласиан переносится с решения на пробную функцию (``\Delta\psi_j`` уже есть в
структуре), ``L`` — из вторых производных ``w`` и замороженных усилий ``N``.
Перенося w-зависимые члены в левую часть, на каждом шаге Пикара решается

.. math:: \big(D\,S_{bend} + K_{geo}(N) + h_\psi^2 M_2(N)\big)\,c = b - h_*^2\,b_{\Delta q},

где ``M_2[j,i] = \iint (N{:}\nabla\nabla\psi_i)\,\Delta\psi_j\,dA`` (член B),
``K_{geo}`` — мембранная связь Кармана, ``b`` — вектор нагрузки. При
``h_\psi^2 = h_*^2 = 0`` система ТОЖДЕСТВЕННО кармановская — редукция КТН→Карман
до машинной точности (Gate R1); все поправки — порядка ``O(h^2/L^2)`` (Gate R4).

Мягкий шарнир для КТН (граничный член слабой формы по полудеформационным
условиям, §3.5) — задел; в v0.5.0 КТН реализована на ЗАЩЕМЛЕНИИ. Поле сдвига
``ψ`` (§3.6) — односторонняя постобработка, ядро от неё не зависит.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .basis import ChebyshevBasis
from .faces import FaceParams
from .membrane import KarmanPlate, _w_structure
from .quadrature import interior_nodes


def _lin_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение НЕсимметричной системы (член B несимметричен) с диаг. предобусл.

    Симметричное диагональное масштабирование ``D_s A D_s`` (``s = 1/√|A_kk|``) +
    LU; при потере обусловленности — линейный МНК. Матрица КТН — малое
    (``O(h²)``) несимметричное возмущение SPD-матрицы Кармана.
    """
    d = np.abs(np.diag(A))
    if np.all(d > 0.0):
        s = 1.0 / np.sqrt(d)
        An = (A * s) * s[:, None]                    # D_s A D_s
        try:
            return sla.lu_solve(sla.lu_factor(An), b * s) * s
        except (sla.LinAlgError, np.linalg.LinAlgError):
            pass
    return np.linalg.lstsq(A, b, rcond=1e-13)[0]


class KTNPlate(KarmanPlate):
    r"""Полный нелинейный решатель КТН — слой поверх :class:`KarmanPlate` (§5.1).

    Наследует мембранную сборку, оператор ``L`` и драйвер Пикара
    (Андерсон + шаги по нагрузке) Кармана; переопределяет ТОЛЬКО внеплоскостной
    шаг, добавляя КТН-члены (A), (B) §3.4–3.5. Карман остаётся неизменным и
    служит эталоном редукции (Gate R1).

    Parameters
    ----------
    include_ktn_terms : при ``False`` КТН-члены выключены ⇒ решатель ТОЖДЕСТВЕНЕН
        ``KarmanPlate`` (для Gate R1 — редукция до машинной точности).
    """

    def __init__(self, domain, basis, quad, cfg, *, bc_type="clamped",
                 inplane_bc="immovable", include_ktn_terms=True):
        if bc_type == "soft_hinge":
            raise NotImplementedError(
                "theory = 'ktn_full' на мягком шарнире: граничный член слабой "
                "формы (B) по полудеформационным условиям (§3.5) — задел; в "
                "v0.5.0 полная КТН реализована на ЗАЩЕМЛЕНИИ (bc = clamped)")
        super().__init__(domain, basis, quad, cfg, bc_type=bc_type, inplane_bc=inplane_bc)
        self._include_ktn = bool(include_ktn_terms)
        self._faces = FaceParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
        self._h_psi_sq = self._faces.h_psi_sq
        self._h_star_sq = self._faces.h_star_sq
        # вторые производные структуры прогиба (для члена B) — в узлах квадратуры
        _, _, _, pxx, pyy, pxy = _w_structure(domain, basis, quad.x, quad.y, self._power)
        self._pxx, self._pyy, self._pxy = pxx, pyy, pxy
        self._lap_psi = pxx + pyy

    @classmethod
    def from_config(cls, domain, cfg, *, bc_type="clamped", inplane_bc="immovable",
                    include_ktn_terms=True):
        """Собрать решатель: базис степени ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg, bc_type=bc_type, inplane_bc=inplane_bc,
                   include_ktn_terms=include_ktn_terms)

    def _ktn_regularization(self, Nx, Ny, Nxy) -> np.ndarray:
        r"""Матрица члена (B): ``M_2[j,i] = ∫ (N:∇∇ψ_i)·Δψ_j dA`` (несимметрична)."""
        W = self._W
        # L, применённый к структуре i: N_x ψ_{i,xx} + 2 N_xy ψ_{i,xy} + N_y ψ_{i,yy}
        LN = self._pxx * Nx + 2.0 * self._pxy * Nxy + self._pyy * Ny
        return (self._lap_psi * W) @ LN.T

    def _picard_map(self, c, b_level, theta):
        r"""Шаг Пикара с КТН-членом (B). При выключенных членах — тождественно Карман."""
        if not self._include_ktn:
            return super()._picard_map(c, b_level, theta)
        wx = c @ self._psi_x
        wy = c @ self._psi_y
        a, b, Nx, Ny, Nxy = self._membrane_forces(wx, wy)
        A = (self.D * self._S_bend
             + self._geometric_stiffness(Nx, Ny, Nxy)
             + self._h_psi_sq * self._ktn_regularization(Nx, Ny, Nxy))
        c_raw = _lin_solve(A, b_level)
        g = (1.0 - theta) * c + theta * c_raw
        return g, (a, b, Nx, Ny, Nxy)

    # -- интроспекция и лицевые параметры ------------------------------- #
    @property
    def face_params(self) -> FaceParams:
        """Параметры лицевых величин (§6.3): h_ψ², h_*², h_c²."""
        return self._faces


__all__ = ["KTNPlate"]
