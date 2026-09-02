r"""eigenmodes.py — собственные задачи пластины: устойчивость и колебания (v0.6.4).

Переиспользует ЛИНЕЙНУЮ сборку структурного Ритца (изгибная жёсткость
``K = D·S_bend``, геометрическая ``K_geo(N⁰)``, матрица масс ``M = ρh·∫ψψ``) из
решателя семейства :class:`~plate_solver.membrane.KarmanPlate` — те же матрицы,
что в задаче изгиба, только собранные в обобщённые собственные проблемы.

**Потеря устойчивости** (под действием мембранных усилий ``N⁰``):

.. math:: (K + \lambda\,K_{geo}(N^0))\,\varphi = 0,

наименьший положительный множитель ``λ_cr`` — критическая нагрузка
(усилия теряют устойчивость при ``N = λ_cr·N⁰``; сжатие — отрицательный знак).

**Свободные колебания:**

.. math:: K\,\varphi = \omega^2 M\,\varphi,\qquad M = \rho h\!\int_\Omega \psi_i\psi_j\,dA.

Обобщённые собственные значения — через :func:`scipy.linalg.eig` (терпит
ВЫРОЖДЕННУЮ правую матрицу: структурный базис Ритца избыточен, ``K_geo``/``M``
численно полуопределены, NOTES §2) с отбором конечных вещественных
положительных. Пучок предварительно НОРМИРУЕТСЯ (:func:`_gen_eig`), поэтому
отбор ведётся по БЕЗРАЗМЕРНЫМ порогам и результат не зависит от масштаба
задачи (``E``, ``h``, ``ρh``). Верификация — классические эталоны Кирхгофа
(Тимошенко, Лейсса):
квадрат SSSS ``k=4``, CCCC ``k=10.07``; круг CCCC ``N_cr·a²/D=14.68``; частоты
``λ = ω·a²·√(ρh/D)`` (см. ``tests/test_eigenmodes.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .config import Config


@dataclass
class EigenPair:
    r"""Результат собственной задачи: значения (по возрастанию) и формы.

    Attributes
    ----------
    values : собственные значения — критические множители ``λ_cr`` (устойчивость)
        или частоты ``ω`` (колебания), отсортированы по возрастанию.
    modes : коэффициенты форм ``(n_modes, N)`` — строка на форму (``w = ω^m·Σc_k T_k``).
    plate : решатель (для оценки форм на сетке).
    kind : ``"buckling"`` | ``"vibration"``.
    """

    values: np.ndarray
    modes: np.ndarray
    plate: object
    kind: str

    def mode_on_grid(self, i: int = 0, grid_n: int = 80):
        """Форма ``i`` (нормированная на max|·|=1) на фоновой сетке grid_n×grid_n."""
        dom = self.plate.domain
        x0, x1, y0, y1 = dom.bbox
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
        inside = dom.omega(Xg, Yg) > 0.0
        W = np.full(Xg.shape, np.nan)
        c = self.modes[i]
        vals = self.plate.deflection(c, Xg[inside], Yg[inside])
        peak = np.max(np.abs(vals))
        W[inside] = vals / peak if peak > 0 else vals
        return Xg, Yg, W


def linear_plate(domain, cfg: Config, *, bc_type: str = "clamped",
                 inplane_bc: str = "immovable"):
    """Собрать линейный решатель (KarmanPlate) для собственных задач."""
    from .membrane import KarmanPlate

    return KarmanPlate.from_config(domain, cfg, bc_type=bc_type, inplane_bc=inplane_bc)


def _bending_stiffness(plate) -> np.ndarray:
    """Изгибная жёсткость ``K = D·S_bend`` (полная билинейная форма)."""
    return plate.D * plate._S_bend


def _mass_matrix(plate, rho_h: float) -> np.ndarray:
    r"""Согласованная матрица масс ``M = ρh·∫ψ_iψ_j dA`` (положительно определена)."""
    return rho_h * (plate._psi * plate._W) @ plate._psi.T


_EIG_IM_TOL = 1e-6      # допуск вещественности: |Im μ| ≤ _EIG_IM_TOL·|μ| (относительный)


def _pencil_scale(M: np.ndarray) -> float:
    r"""Масштаб матрицы пучка ``‖M‖_max = max|M_ij|`` (нулевую нормируем единицей)."""
    s = float(np.max(np.abs(M))) if M.size else 0.0
    return s if s > 0.0 else 1.0


def _gen_eig(A: np.ndarray, B: np.ndarray, n_modes: int):
    r"""Обобщённые собств. значения ``Aφ = μBφ`` через ``scipy.linalg.eig``.

    Задача решается в НОРМИРОВАННОМ виде

    .. math:: \tilde A\varphi = \nu\,\tilde B\varphi,\quad
              \tilde A = A/\|A\|_{\max},\ \tilde B = B/\|B\|_{\max},\quad
              \mu = \nu\,\|A\|_{\max}/\|B\|_{\max},

    так что спектр ``ν`` БЕЗРАЗМЕРЕН: замена ``E → cE`` умножает ``A`` на ``c``,
    но ``Ã`` и ``ν`` не меняет (собственные векторы одинаковой нормировкой обеих
    частей тоже не затрагиваются). Отбор ведётся по ОДНОРОДНОЙ паре ``(α, β)``,
    ``μ = α/β`` (``homogeneous_eigvals=True``) — без деления на почти нуль;
    поскольку ``μ = α\bar β/|β|²``, знак и мнимая часть ``μ`` читаются прямо
    из ``α\bar β``:

    * регулярность: НЕ (``|α| ≤ τ`` и ``|β| ≤ τ``) — иначе пара порождена общим
      нуль-пространством пучка и ``μ`` произвольно;
    * конечность: ``|β| > τ`` (иначе ``μ = ∞`` — нуль-пространство ``B``:
      избыточность структурного базиса Ритца, NOTES §2);
    * вещественность: ``|Im α\bar β| ≤ ε_im·|α\bar β|`` (≡ ``|Im μ| ≤ ε_im·|μ|``);
    * положительность: ``Re α\bar β > 0``.

    Порог ``τ = n·ε_маш`` — уровень обратной ошибки QZ-разложения для матриц
    единичной max-нормы (стандартная оценка ``‖ΔÃ‖ ≲ p(n)·ε_маш·‖Ã‖``).

    .. note:: Пороги ОТНОСИТЕЛЬНЫЕ (v0.8.0, находка аудита V02). Прежний
       абсолютный отбор ``μ > 1e-9`` над РАЗМЕРНЫМИ ``μ`` молча терял основной
       тон при малой ``D/ρh``: для квадрата SSSS при ``E = 1e-5`` точное
       ``ω₁² = 3.57e-10`` отбрасывалось как «нуль-пространство» и первой
       возвращалась мода (1,2). Порог вида ``μ > ε·max|μ|`` НЕ годится: при
       полуопределённой ``B`` конечные ``μ`` от почти-нулевых знаменателей
       задирают его и срезают ФИЗИЧЕСКИЕ моды. Единообразная замена на
       ``scipy.linalg.eigh(A, B)`` тоже не годится: в задаче устойчивости
       ``B = −K_geo`` знакопеременна при немонотонном поле ``N`` (разложение
       Холецкого невозможно), а здесь обе ветви идут одним трактом.

    Возвращает ``(μ, φ)`` по возрастанию μ, не более ``n_modes`` штук.
    """
    sA, sB = _pencil_scale(A), _pencil_scale(B)
    ab, vecs = sla.eig(A / sA, B / sB, homogeneous_eigvals=True)
    alpha, beta = ab[0], ab[1]
    tau = alpha.size * float(np.finfo(float).eps)           # обратная ошибка QZ
    prod = alpha * np.conj(beta)                            # μ = α·β̄/|β|²
    regular = ~((np.abs(alpha) <= tau) & (np.abs(beta) <= tau))
    finite = np.abs(beta) > tau
    real = np.abs(prod.imag) <= _EIG_IM_TOL * np.abs(prod)
    keep = regular & finite & real & (prod.real > 0.0)
    mu = (alpha[keep] / beta[keep]).real * (sA / sB)        # возврат к размерным μ
    phi = vecs.real[:, keep]
    order = np.argsort(mu)
    return mu[order][:n_modes], phi[:, order][:, :n_modes].T


def _prestress_fields(plate, prestress):
    r"""Поле усилий ``(N_x, N_y, N_{xy})`` в узлах квадратуры из разных источников.

    ``prestress`` может быть: ``None`` (без преднапряжения); результат нелинейного
    решателя (`KarmanResult` — атрибуты ``Nx/Ny/Nxy``, поле ``N(w)`` РЕАЛЬНОЙ
    нагрузки); кортеж ``(Nx, Ny, Nxy)`` скаляров/массивов (равномерное или своё поле).
    """
    z = np.zeros(plate._W.size)
    if prestress is None:
        return z, z, z
    if hasattr(prestress, "Nx"):                        # результат Кармана/КТН: N(w)
        return (np.asarray(prestress.Nx, float), np.asarray(prestress.Ny, float),
                np.asarray(prestress.Nxy, float))
    Nx, Ny, Nxy = prestress
    o = np.ones(plate._W.size)
    return np.asarray(Nx, float) * o, np.asarray(Ny, float) * o, np.asarray(Nxy, float) * o


def natural_frequencies(plate, *, rho_h: float = 1.0, n_modes: int = 6,
                        prestress=None) -> EigenPair:
    r"""Собственные частоты и формы свободных колебаний пластины.

    Решает ``K_eff φ = ω² M φ`` (``K = D·S_bend``, ``M = ρh·∫ψψ``). ``rho_h`` —
    погонная масса ``ρh`` (по умолчанию 1). ``prestress`` — ПРЕДНАПРЯЖЕНИЕ:
    ``K_eff = K + K_geo(N)`` c усилиями ``N`` из нелинейного решения ``N(w)``
    (результат Кармана/КТН) либо заданным полем/скаляром ``(Nx, Ny, Nxy)``.
    Растяжение (``N > 0``, напр. мембранное натяжение от поперечной нагрузки)
    УЖЕСТОЧАЕТ пластину и ПОВЫШАЕТ частоты; сжатие — понижает (частота → 0 у
    критической нагрузки потери устойчивости). ``None`` ⇒ ненапряжённая пластина.
    """
    K = _bending_stiffness(plate)
    if prestress is not None:
        Nx, Ny, Nxy = _prestress_fields(plate, prestress)
        K = K + plate._geometric_stiffness(Nx, Ny, Nxy)         # преднапряжение (физ. знаки)
    M = _mass_matrix(plate, float(rho_h))
    w2, modes = _gen_eig(K, M, n_modes)
    if w2.size == 0:
        raise ValueError(
            "natural_frequencies: положительных ω² нет — эффективная жёсткость "
            "K + K_geo(N) не положительно определена. Так бывает при СЖИМАЮЩЕМ "
            "преднапряжении сверх критического: пластина уже потеряла устойчивость, "
            "колебаний около этого положения равновесия нет. Уменьшите сжатие либо "
            "перейдите к задаче устойчивости (buckling).")
    return EigenPair(values=np.sqrt(w2), modes=modes, plate=plate, kind="vibration")


def buckling(plate, *, Nx: float = -1.0, Ny: float = 0.0, Nxy: float = 0.0,
             n_modes: int = 6, prestress=None) -> EigenPair:
    r"""Критические множители потери устойчивости и формы выпучивания.

    Мембранное усилие-эталон ``N⁰`` (равномерное ``(Nx, Ny, Nxy)`` по умолчанию
    одноосное сжатие ``Nx = −1``, ЛИБО произвольное поле/результат ``N(w)`` через
    ``prestress``). Решает ``(K + λ K_geo(N⁰)) φ = 0`` в виде
    ``K φ = λ(−K_geo(N⁰)) φ``; ``λ_cr`` — наименьший положительный, критическая
    нагрузка = ``λ_cr·N⁰``. ``prestress`` (поле или результат) имеет приоритет над
    скалярами ``Nx/Ny/Nxy`` — устойчивость под НЕравномерным полем усилий.
    """
    K = _bending_stiffness(plate)
    if prestress is not None:
        nx, ny, nxy = _prestress_fields(plate, prestress)
    else:
        o = np.ones(plate._W.size)
        nx, ny, nxy = Nx * o, Ny * o, Nxy * o
    G = plate._geometric_stiffness(nx, ny, nxy)                 # физ. знаки
    lam, modes = _gen_eig(K, -G, n_modes)                       # −G ⪰ 0 при сжатии
    if lam.size == 0:
        raise ValueError(
            "buckling: положительного критического множителя нет — заданное поле "
            "усилий N не СЖИМАЮЩЕЕ (растяжение или ноль ⇒ пластина устойчива, терять "
            "устойчивость нечему). Для потери устойчивости задайте сжатие "
            "(например Nx < 0), либо преднапряжение с сжимающей компонентой.")
    return EigenPair(values=lam, modes=modes, plate=plate, kind="buckling")


__all__ = ["EigenPair", "linear_plate", "natural_frequencies", "buckling"]
