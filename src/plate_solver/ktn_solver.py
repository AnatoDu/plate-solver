r"""ktn_solver.py — единый параметрический решатель теорий (§3.6 ТЗ v0.6.0).

Один решатель полной КТН с управляемыми членами (:class:`~plate_solver.theory.TheoryParams`)
поглощает четыре теории; редукции ТОЧНЫ ПО ПОСТРОЕНИЮ (тот же путь исполнения).
Строится как обобщение :class:`~plate_solver.ktn_full.KTNPlate` флагом
``membrane`` (при выкл — мембранные усилия ``N ≡ 0`` ⇒ линейная бигармоника) и
параметрами толщины из пресета.

Соответствие пресетам (тот же код, разные параметры):

* ``classic``  — ``membrane=False``, ``h_ψ²=h_*²=0`` ⇒ линейный Кирхгоф;
* ``karman``   — ``membrane=True``,  ``h_ψ²=h_*²=0`` ⇒ Карман (== ``KarmanPlate``);
* ``ktn_full`` — ``membrane=True``, физические ``h_ψ², h_*²`` ⇒ полная КТН
  (== ``KTNPlate``);
* ``ktn_linear`` — ``membrane=False``, ``h_*²`` физич. (лицевая постобработка);
  в РЕШЕНИИ линеен (КТН-члены (A),(B) активны только в нелинейном режиме).

Непрерывный морфинг Карман↔полная КТН (``refinement_scale`` α) — единственный
путь исполнения с масштабированием ``h_ψ², h_*²`` (демонстрация §6.2). Числовое
совпадение с ``KarmanPlate``/``KTNPlate`` до машинной точности проверяется тестами
(``tests/test_unified_theory.py``, ворота R6 §7). Реализовано на ЗАЩЕМЛЕНИИ
(нелинейный тракт и контакт v0.6.0).
"""

from __future__ import annotations

import numpy as np

from .basis import ChebyshevBasis
from .ktn_full import KTNPlate
from .quadrature import interior_nodes
from .theory import TheoryParams, from_preset


class KTNSolver(KTNPlate):
    r"""Единый параметрический решатель теорий (§3): пресет → параметры одного решателя.

    Parameters
    ----------
    params : :class:`~plate_solver.theory.TheoryParams` — управляемые члены
        (нелинейность ``membrane`` × уточнение ``h_ψ², h_*²``).
    bc_type : ``clamped`` (нелинейный/КТН тракт реализован на защемлении).
    inplane_bc : ``immovable`` | ``movable`` (закрепление кромки в плане).
    """

    def __init__(self, domain, basis, quad, cfg, params: TheoryParams, *,
                 bc_type="clamped", inplane_bc="immovable"):
        super().__init__(domain, basis, quad, cfg, bc_type=bc_type,
                         inplane_bc=inplane_bc,
                         include_ktn_terms=params.solve_ktn_terms)
        self.params = params
        # параметры толщины — из пресета (могут быть масштабированы α), а не
        # физические из FaceParams; при membrane=False член A не активен (§3.4).
        self._h_psi_sq = params.h_psi_sq
        self._h_star_sq = params.h_star_sq
        self._membrane_on = params.membrane

    @classmethod
    def from_config(cls, domain, cfg, params: TheoryParams, *,
                    bc_type="clamped", inplane_bc="immovable"):
        """Собрать решатель по :class:`TheoryParams`: базис ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg, params, bc_type=bc_type, inplane_bc=inplane_bc)

    @classmethod
    def from_theory_name(cls, domain, cfg, name: str, *,
                         bc_type="clamped", inplane_bc="immovable"):
        """Собрать решатель по имени пресета (``[model] theory``; алиас ``ktn``→``ktn_linear``)."""
        params = from_preset(name, cfg.nu, cfg.h)
        return cls.from_config(domain, cfg, params, bc_type=bc_type, inplane_bc=inplane_bc)

    def _membrane_forces(self, wx, wy):
        r"""Мембранные усилия: при выключенной нелинейности ``N ≡ 0`` (линейный режим)."""
        if not self._membrane_on:
            n = self._n
            z = np.zeros(self._W.size)
            return np.zeros(n), np.zeros(n), z, z, z
        return super()._membrane_forces(wx, wy)


__all__ = ["KTNSolver"]
