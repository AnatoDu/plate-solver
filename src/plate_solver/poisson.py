r"""poisson.py — решатель одной задачи Дирихле −Δv = f методом Ритца.

Решает ``−Δv = f`` в Ω, ``v = 0`` на ``∂Ω``, аппроксимация ``v = ω·Φ``,
``Φ = Σ_k c_k T_k``. Матрица Ритца ``A`` (assembler.py) собирается и факторизуется
(Холецкий) ОДИН раз при создании решателя; затем каждое решение для новой правой
части ``f`` — это лишь сборка ``b`` и две треугольные подстановки.

Зачем факторизация один раз: ``A`` общая для (P1), (P2) и всех итераций МОР
(NOTES.md §§1, 5) ⇒ контактный цикл получается дешёвым.

Кэш матриц структуры. При ``cache_fields=True`` в конструкторе один раз
сохраняются матрицы N×M в узлах квадратуры, потребляемые итерацией МОР:
``psiW = ψ·diag(w)`` (ψ = ω·T) и ``Φ`` с ``ω`` по отдельности. Тогда на каждой
итерации:

* сборка нагрузки — один GEMV: ``b = psiW @ f`` (вместо пересчёта базиса);
* значения поля в узлах квадратуры — :meth:`evaluate_at_quad`, один GEMV:
  ``v = ψᵀc = ω·(Φᵀc)`` (реализовано как ``ω·(Φᵀc)``, чтобы порядок округления
  совпадал с :meth:`evaluate` БИТ-В-БИТ — golden не должен меняться).

Матрицы ``psi``, ``psi_x``, ``psi_y`` НЕ хранятся: после сборки ``A`` у них нет
потребителей (ψ = ω·Φ восстановим из кэша), а их хранение утроило бы память.
Память кэша ~2·N·M·8 байт: для L-формы (p=10, Q=120: N·M≈1.3e6) это ~21 МБ.
Для «больших» квадратур (N·M > ``CACHE_NM_MAX`` = 5e7, т.е. кэш > ~0.8 ГБ;
напр. круг Q=1024, p=10: N·M ≈ 1e8) кэш по умолчанию ВЫКЛЮЧЕН — поведение
и арифметика прежние.

Каскад факторизации (:class:`SPDFactorization`). Матрица Ритца ``A``
положительно определена ТОЧНО, но при росте ``p`` на криволинейной границе или
во входящем угле она вырождается численно (``cond(A) ~ 1e16…1e18`` уже при
p ≈ 12 на L-форме) — тогда ``cho_factor`` бросает ``LinAlgError`` и законный
случай падает сырой ошибкой LAPACK (причём платформозависимо: исход зависит от
BLAS). Поэтому решение системы проходит ЧЕСТНЫЙ каскад с отчётностью, а не
одиночный Холецкий; см. докстринг :class:`SPDFactorization`.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .assembler import (
    assemble_load,
    assemble_stiffness,
    assemble_stiffness_from_fields,
    structure_fields,
)

# Порог включения кэша: N·M ≤ CACHE_NM_MAX (≈ 0.4 ГБ на одну матрицу float64).
CACHE_NM_MAX = 50_000_000


#: относительный допуск на асимметрию входа (выше — отказ: каскад симметричный)
ASYM_TOL = 1.0e-10

# Относительная отсечка спектра в псевдообращении: λ_k ≤ REL_CUTOFF·λ_max
# считаются шумом округления. Пол округления одиночной операции —
# ε_машинное ≈ 2.2e-16·λ_max, но сборка матрицы Ритца копит ошибку по M узлам
# квадратуры (M ~ 1e4…1e5), поэтому фактический пол на 2–3 порядка выше, и
# отсечка 1e-12 выбрана с запасом НАД ним, а не над ε.
REL_CUTOFF = 1e-12


class FactorizationWarning(RuntimeWarning):
    """Штатная факторизация Холецкого не удалась — включена ступень фолбэка.

    Отдельная категория (а не голый ``RuntimeWarning``), чтобы вызывающий код
    мог перехватить факт деградации:
    ``warnings.catch_warnings(record=True)`` + фильтр по этой категории, либо
    прямое чтение атрибутов ``fallback`` / ``n_dropped`` решателя.
    """


def _cho_try(A):
    """Попытка Холецкого: фактор или ``None`` при потере положительной определённости."""
    try:
        return sla.cho_factor(A)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        return None


class SPDFactorization:
    r"""Каскад решения симметричной системы ``A c = b`` с ЧЕСТНОЙ отчётностью.

    Ступени (следующая включается только при отказе предыдущей):

    1. **Холецкий** ``A = L Lᵀ``. Штатный путь; арифметика и результат
       ТОЖДЕСТВЕННЫ прямому ``cho_solve(cho_factor(A), b)`` (бит-в-бит) —
       свойство обязательное: замороженные эталоны не должны сдвигаться.
    2. **Симметричное масштабирование Якоби** ``Ā = S A S``,
       ``S = diag(1/\sqrt{A_{kk}})``, затем Холецкий по ``Ā``; решение
       ``c = S\,\bar c``, ``\bar b = S b``. По теореме ван дер Слёйса такое
       масштабирование минимизирует ``cond`` с точностью до множителя ``n``
       среди всех диагональных, т.е. это наилучшая дешёвая попытка спасти
       положительную определённость; иногда её хватает, когда вырождение —
       следствие разброса масштабов строк, а не ранга.
    3. **Спектральное псевдообращение**: ``A = V Λ Vᵀ`` (``eigh`` по
       симметризованной ``½(A + Aᵀ)``), отбрасываются направления
       ``λ_k ≤ τ·λ_max`` (``τ = REL_CUTOFF``), и
       ``c = V_+ Λ_+^{-1} V_+^{ᵀ} b``.

    ⚠️ Ступень 3 МЕНЯЕТ ДИСКРЕТНОЕ ПРОСТРАНСТВО. Решение ищется не во всём
    ``span{ψ_k}``, а в его подпространстве, отвечающем ``span V_+`` в
    координатах коэффициентов: ``n_dropped`` направлений с почти нулевой
    энергией из аппроксимации исключены. Это не «то же решение, посчитанное
    иначе», а решение УСЕЧЁННОЙ задачи. Отбрасывание ``λ ≤ τλ_max``
    эквивалентно возмущению оператора нормы ``≤ τ‖A‖``; отбрасываемые
    направления — почти-нулевые моды структуры ``ω·T`` (линейно зависимые с
    точностью округления при большом ``p``), их вклад в энергию ниже пола
    сборки, поэтому прогиб меняется в пределах процентов, а НЕ порядков.
    Тем не менее факт усечения обязан быть виден вызывающему: он сообщается
    предупреждением :class:`FactorizationWarning` и атрибутами.

    Attributes
    ----------
    fallback : ``None`` (штатный Холецкий) | ``"jacobi"`` | ``"pinv"``.
    n_dropped : число отсечённых спектральных направлений (0 вне ступени 3).
    warning_message : текст для журнала вызывающего (``None`` на штатном пути).
    """

    def __init__(self, A, *, label: str = "A", rel_cutoff: float = REL_CUTOFF,
                 warn: bool = True):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("SPDFactorization: ожидалась квадратная матрица, "
                             f"получено {A.shape}")
        self.label = str(label)
        self.rel_cutoff = float(rel_cutoff)
        self.n = int(A.shape[0])
        self.fallback: str | None = None
        self.n_dropped: int = 0
        self.n_negative: int = 0
        self.warning_message: str | None = None
        self._chol = None
        self._scale = None
        self._spec = None                       # (V_+, 1/λ_+) для ступени 3
        # -- ступень 1: Холецкий (штатный путь, бит-в-бит как раньше) ------ #
        chol = _cho_try(A)
        if chol is not None:
            self._chol = chol
            return
        if not np.all(np.isfinite(A)):
            raise np.linalg.LinAlgError(
                f"{self.label}: матрица содержит NaN/Inf — вырождение не "
                "численное, а следствие некорректной постановки (нулевая "
                "толщина, вырожденная геометрия, неопределённые параметры)")
        # КОНТРАКТ СИММЕТРИИ (v0.8.0). Ступени 2 и 3 работают с ½(A + Aᵀ):
        # для НЕсимметричного входа они решают симметризованную задачу, и при
        # полном ранге симметричной части это происходило МОЛЧА (аудит:
        # касательная КТН-Ньютона, ошибка шага до 5 %). Численная асимметрия
        # уровня округления (сборка квадратурой) допускается, содержательная —
        # отказ с указанием правильного инструмента.
        asym = float(np.max(np.abs(A - A.T)))
        scale = float(np.max(np.abs(A)))
        if scale > 0.0 and asym > ASYM_TOL * scale:
            raise np.linalg.LinAlgError(
                f"{self.label}: матрица НЕсимметрична (‖A−Aᵀ‖/‖A‖ = "
                f"{asym / scale:.2e} > {ASYM_TOL:.0e}) — симметричный каскад к "
                "ней неприменим: он решил бы задачу с ½(A+Aᵀ). Для "
                "несимметричных операторов (например касательной уточнённой "
                "теории) используйте ktn_full._lin_solve (LU + честный фолбэк)")
        # -- ступень 2: масштабирование Якоби ------------------------------ #
        d = np.diag(A)
        if np.all(d > 0.0):
            s = 1.0 / np.sqrt(d)
            chol = _cho_try((A * s) * s[:, None])
            if chol is not None:
                self._chol, self._scale = chol, s
                self.fallback = "jacobi"
                self._emit(warn, "Холецкий не прошёл; спасло симметричное "
                                 "масштабирование Якоби (решение полное, "
                                 "пространство не усечено)")
                return
        # -- ступень 3: спектральное псевдообращение ----------------------- #
        lam, V = np.linalg.eigh(0.5 * (A + A.T))
        lam_max = float(lam[-1])
        if not lam_max > 0.0:
            raise np.linalg.LinAlgError(
                f"{self.label}: матрица не положительно полуопределена "
                f"(λ_max = {lam_max:.3e} ≤ 0) — ошибка постановки, а не "
                "обусловленности")
        keep = lam > self.rel_cutoff * lam_max
        self.n_dropped = int(self.n - np.count_nonzero(keep))
        self.n_negative = int(np.count_nonzero(lam < -self.rel_cutoff * lam_max))
        self._spec = (np.ascontiguousarray(V[:, keep]), 1.0 / lam[keep])
        self.fallback = "pinv"
        if self.n_negative:
            # Матрица симметрична (контракт выше), но ИНДЕФИНИТНА: отрицательные
            # направления отбрасываются вместе с шумовыми, и решение —
            # ПРОЕКЦИЯ на положительный конус, а не решение исходной системы.
            # Молчать об этом нельзя: для индефинитного оператора нужен не
            # SPD-каскад, а несимметричный/знаконеопределённый решатель (v0.8.0).
            self._emit(warn, (
                f"матрица СИММЕТРИЧНА, но ИНДЕФИНИТНА: {self.n_negative} из "
                f"{self.n} собственных значений отрицательны. Спектральное "
                "решение отбрасывает эти направления — возвращается ПРОЕКЦИЯ, "
                "а не решение исходной системы. Для знаконеопределённых "
                "операторов SPD-каскад неприменим (проверьте постановку: "
                "сжимающее преднапряжение выше критического, вырожденная "
                "геометрия, ошибочная сборка)"))
            return
        if self.n_dropped == 0:
            # Вход симметричен (контракт выше), и ни одно направление не
            # отброшено ⇒ λ_min > 0: матрица положительно определена, а
            # Холецкий и масштабирование Якоби отказали по ОБУСЛОВЛЕННОСТИ
            # (округление у границы положительной определённости). Спектральное
            # решение при полном ранге ТОЧНО, дискретное пространство не
            # усечено — предупреждать не о чем (v0.8.0).
            self._emit(warn, (
                "Холецкий и масштабирование Якоби отказали по обусловленности "
                "(λ_min > 0, матрица положительно определена); решение получено "
                "спектральным разложением, НИ ОДНО направление не отброшено — "
                "решение полное, пространство не усечено"),
                       loud=False)
            return
        self._emit(warn, (
            f"Холецкий и масштабирование Якоби не прошли; включено "
            f"спектральное псевдообращение с отсечкой {self.rel_cutoff:.0e}·λ_max: "
            f"отсечено {self.n_dropped} из {self.n} направлений — ДИСКРЕТНОЕ "
            "ПРОСТРАНСТВО УСЕЧЕНО (решение спроецировано). Причина обычно — "
            "избыточная степень базиса p при данной квадратуре; уменьшите p "
            "либо увеличьте Q"))

    def _emit(self, warn: bool, msg: str, *, loud: bool = True) -> None:
        """Зафиксировать факт фолбэка: атрибут + предупреждение модуля warnings.

        ``loud=False`` — факт записывается в атрибуты (``fallback``,
        ``warning_message``), но предупреждение НЕ издаётся: так помечаются
        деградации без потери информации (полный ранг, решение точное).
        """
        self.warning_message = f"{self.label}: {msg}"
        if warn and loud:
            warnings.warn(self.warning_message, FactorizationWarning, stacklevel=4)

    @property
    def degraded(self) -> bool:
        """Сработал ли фолбэк (любая ступень, кроме штатного Холецкого)."""
        return self.fallback is not None

    @property
    def cho(self):
        """Фактор Холецкого ШТАТНОГО пути (``cho_factor``) либо ``None`` при фолбэке."""
        return self._chol if self.fallback is None else None

    def solve(self, b) -> np.ndarray:
        """Решить ``A c = b`` той ступенью каскада, что удалась при построении.

        Правая часть — вектор ``(n,)`` ЛИБО матрица ``(n, k)`` (сразу ``k``
        систем с одной факторизацией: так решается, например, производная
        ``∂N/∂c`` касательного оператора Ньютона). Диагональные множители
        ступеней 2 и 3 действуют по СТРОКАМ, поэтому для матричной правой
        части они разворачиваются в столбец: без этого numpy молча
        транслировал бы их по столбцам (иная арифметика) либо отказывал по
        несовпадению форм.
        """
        b = np.asarray(b, dtype=float)
        if b.ndim not in (1, 2) or b.shape[0] != self.n:
            raise ValueError(
                f"{self.label}: правая часть формы {b.shape} несовместима с "
                f"матрицей {self.n}×{self.n} — ожидалось ({self.n},) либо "
                f"({self.n}, k)")
        col = (slice(None),) if b.ndim == 1 else (slice(None), None)
        if self._chol is not None:
            if self._scale is None:
                return sla.cho_solve(self._chol, b)          # штатный путь
            s = self._scale[col]                             # A = S⁻¹(SAS)S⁻¹
            return sla.cho_solve(self._chol, b * s) * s
        V, inv_lam = self._spec
        return V @ ((V.T @ b) * inv_lam[col])


class PoissonSolver:
    """Решатель ``−Δv = f`` с предвычисленной факторизацией ``A``.

    Parameters
    ----------
    domain : область (Domain) — даёт ω и ∇ω.
    basis : тензорный базис Чебышёва (ChebyshevBasis).
    quad : узлы и веса квадратуры внутри Ω (QuadNodes).
    cache_fields : кэшировать ли матрицы структуры N×M (см. докстринг модуля);
        ``None`` (по умолчанию) — автоматически: True при N·M ≤ CACHE_NM_MAX.
    """

    def __init__(self, domain, basis, quad, cache_fields: bool | None = None):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        if cache_fields is None:
            cache_fields = basis.N * quad.x.size <= CACHE_NM_MAX
        self.cache_fields = bool(cache_fields)
        if self.cache_fields:
            psi, psi_x, psi_y, W = structure_fields(domain, basis, quad)
            self.A = assemble_stiffness_from_fields(psi_x, psi_y, W)
            self.psiW = psi * W                                  # для b = psiW @ f
            del psi, psi_x, psi_y                                # без потребителей — не держим
            # Φ и ω отдельно — чтобы evaluate_at_quad повторял evaluate бит-в-бит.
            self._phi_quad = basis.values(quad.x, quad.y)        # (N, M)
            self._om_quad = domain.omega(quad.x, quad.y)         # (M,)
        else:
            self.psiW = self._phi_quad = self._om_quad = None
            self.A = assemble_stiffness(domain, basis, quad)
        # Факторизация ОДИН раз — каскадом (Холецкий → Якоби → псевдообращение).
        # На штатном пути арифметика тождественна прежнему cho_factor/cho_solve.
        self._fact = SPDFactorization(self.A, label="A (Ритц, −Δ)")
        # cho-фактор оставлен публичным для совместимости; None, если Холецкий
        # не прошёл (раньше в этом случае конструктор бросал LinAlgError).
        self.chol = self._fact.cho

    @property
    def cond(self) -> float:
        """Число обусловленности cond(A) — диагностика устойчивости (NOTES.md §2)."""
        return float(np.linalg.cond(self.A))

    @property
    def fallback(self) -> str | None:
        """Ступень каскада факторизации: ``None`` | ``"jacobi"`` | ``"pinv"``."""
        return self._fact.fallback

    @property
    def n_dropped(self) -> int:
        """Число отсечённых спектральных направлений (усечение пространства)."""
        return self._fact.n_dropped

    @property
    def fallback_message(self) -> str | None:
        """Текст о деградации факторизации для журнала вызывающего (или ``None``)."""
        return self._fact.warning_message

    def solve(self, f_values) -> np.ndarray:
        """Коэффициенты ``c`` разложения Φ: решить ``A c = b(f)``.

        ``f_values`` — значения правой части в узлах квадратуры (self.quad).
        При включённом кэше сборка ``b = psiW @ f`` — один GEMV.
        """
        if self.cache_fields:
            b = self.psiW @ np.asarray(f_values, dtype=float)
        else:
            b = assemble_load(self.domain, self.basis, self.quad, f_values)
        return self._fact.solve(b)

    def load_vector(self, f_values) -> np.ndarray:
        """Вектор нагрузки ``b[k] = ∫ f ψ_k`` (для внешних правых частей, A4)."""
        if self.cache_fields:
            return self.psiW @ np.asarray(f_values, dtype=float)
        return assemble_load(self.domain, self.basis, self.quad, f_values)

    def solve_b(self, b) -> np.ndarray:
        """Решение по ГОТОВОМУ вектору нагрузки (две подстановки Холецкого; A4)."""
        return self._fact.solve(np.asarray(b, dtype=float))

    def evaluate(self, c, X, Y) -> np.ndarray:
        """Значения ``v = ω·Σ_k c_k T_k`` в точках (X, Y)."""
        Phi = self.basis.values(X, Y)               # (N, *shape)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        return self.domain.omega(X, Y) * v

    def evaluate_at_quad(self, c) -> np.ndarray:
        r"""Значения ``v = ψᵀc`` в узлах квадратуры через кэш (один GEMV).

        Тождественно ``evaluate(c, quad.x, quad.y)`` (бит-в-бит): считается как
        ``ω·(Φᵀc)`` по кэшированным Φ и ω. Без кэша — просто вызывает evaluate.
        Главный потребитель — итерация МОР (contact.py) и (P2) в plate.solve.
        """
        if not self.cache_fields:
            return self.evaluate(c, self.quad.x, self.quad.y)
        v = np.tensordot(np.asarray(c, float), self._phi_quad, axes=(0, 0))
        return self._om_quad * v

    def solve_field(self, f_values, X, Y) -> np.ndarray:
        """Удобный фасад: решить и сразу вычислить поле в точках (X, Y)."""
        return self.evaluate(self.solve(f_values), X, Y)


__all__ = [
    "PoissonSolver",
    "SPDFactorization",
    "FactorizationWarning",
    "CACHE_NM_MAX",
    "REL_CUTOFF",
]
