r"""exprfield.py — безопасный разбор выражений ``f(x, y)`` из case-файлов (v0.7.0).

Общий вычислитель для ключей схемы ``[load] expr`` (нагрузка выражением) и
``[contact] gap_expr`` (профиль зазора/штампа): строка → sympy-выражение →
значения в узлах квадратуры; для гладких нагрузок лапласиан ``Δf`` берётся
СИМВОЛЬНЫМ дифференцированием (точно — член ``−h_*²Δq`` полной КТН, §7).

ИНВАРИАНТ БЕЗОПАСНОСТИ (не ослаблять). ``sympy.parse_expr`` исполняет
произвольный Python (это eval; инъекция вида ``__import__('os')`` действительно
исполняется), поэтому разбору предшествует ТОКЕН-ОГРАДА:
строка токенизируется штатным ``tokenize``, допускаются ТОЛЬКО числа
(вещественные; комплексные литералы ``1j`` отклоняются), имена из закрытого
белого списка ``_NAMES``, арифметика ``+ - * / **``, скобки и запятая. Всё
прочее (строки, атрибуты, индексация, ``lambda``, ``;``, ``=`` …) — отказ ДО
разбора. Любое расширение алфавита или белого списка — осознанный пересмотр
безопасности, не «добавить имя в словарь».

Ошибки поднимаются ``ValueError`` с полным текстом (ключ + перечень
разрешённого); вызывающий код (``problem``/``dispatch``) оборачивает их в
``CaseError`` — модуль не импортирует ``problem`` (без циклов).
"""

from __future__ import annotations

import io
import tokenize
from functools import lru_cache

import numpy as np
import sympy as sp

_X, _Y = sp.symbols("x y")

#: закрытый белый список имён выражения (расширять только с пересмотром
#: безопасности и документации CASE_SCHEMA)
_NAMES: dict[str, object] = {
    "x": _X, "y": _Y, "pi": sp.pi,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "tanh": sp.tanh, "min": sp.Min, "max": sp.Max,
}
#: без Integer/Float/Rational ломается auto_number разборщика; Symbol нужен
#: его внутренним преобразованиям. Имена пользователю НЕ доступны (ограда).
_GLOBALS = {"Integer": sp.Integer, "Float": sp.Float,
            "Rational": sp.Rational, "Symbol": sp.Symbol}
_OPS = {"+", "-", "*", "/", "**", "(", ")", ","}
_SKIP_TOKENS = {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER,
                tokenize.INDENT, tokenize.DEDENT}

_ALLOWED_TXT = ("разрешены имена: " + ", ".join(sorted(_NAMES)) +
                "; операции + - * / **, скобки и числа")


#: ресурсные пороги ограды: sympy вычисляет целочисленную степень уже при
#: разборе — башня вида 9**9**9 подвесила бы чтение case-файла
_MAX_LEN = 20_000            # длина строки-выражения
_MAX_INT_DIGITS = 12         # целочисленный литерал (координаты — float'ы)
_MAX_POW_EXP = 64            # целый показатель сразу после '**'


def _is_int_literal(s: str) -> bool:
    t = s.lower()
    return not ("." in t or "e" in t or t.startswith(("0x", "0b", "0o")))


def _fence(key: str, s: str) -> None:
    """Токен-ограда ДО sympy: только числа, белый список имён, арифметика.

    Плюс ресурсные пороги: длина строки, размер целочисленных литералов,
    целый показатель степени ≤ 64 и запрет башен ``a**b**c`` — иначе sympy
    вычисляет гигантские целые прямо при разборе case-файла (DoS).
    """
    if len(s) > _MAX_LEN:
        raise ValueError(f"{key}: выражение длиннее {_MAX_LEN} символов")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ValueError(f"{key}: не разбирается как выражение ({exc}); "
                         f"{_ALLOWED_TXT}") from None
    sig = []                                     # значимые токены по порядку
    for t in toks:
        if t.type in _SKIP_TOKENS:
            continue
        if t.type == tokenize.NUMBER:
            if t.string.lower().endswith("j"):
                raise ValueError(f"{key}: комплексный литерал '{t.string}' "
                                 f"не допускается; {_ALLOWED_TXT}")
            if (_is_int_literal(t.string)
                    and len(t.string.replace("_", "")) > _MAX_INT_DIGITS):
                raise ValueError(f"{key}: целочисленный литерал '{t.string}' "
                                 f"длиннее {_MAX_INT_DIGITS} цифр")
            sig.append(t)
            continue
        if t.type == tokenize.NAME:
            if t.string not in _NAMES:
                raise ValueError(f"{key}: имя '{t.string}' вне белого списка; "
                                 f"{_ALLOWED_TXT}")
            sig.append(t)
            continue
        if t.type == tokenize.OP:
            if t.string not in _OPS:
                raise ValueError(f"{key}: оператор '{t.string}' не допускается; "
                                 f"{_ALLOWED_TXT}")
            sig.append(t)
            continue
        raise ValueError(f"{key}: недопустимый элемент '{t.string}' "
                         f"(строки/атрибуты/индексация запрещены); {_ALLOWED_TXT}")
    # степени: '**' с целым показателем > 64 и башни '** … **' — отказ
    for i, t in enumerate(sig):
        if t.type != tokenize.OP or t.string != "**":
            continue
        j = i + 1
        while j < len(sig) and sig[j].type == tokenize.OP \
                and sig[j].string in ("+", "-"):
            j += 1
        if (j < len(sig) and sig[j].type == tokenize.NUMBER
                and _is_int_literal(sig[j].string)
                and abs(int(sig[j].string.replace("_", ""), 0)) > _MAX_POW_EXP):
            raise ValueError(f"{key}: целый показатель степени "
                             f"{sig[j].string} > {_MAX_POW_EXP}")
        if (j + 1 < len(sig) and sig[j].type == tokenize.NUMBER
                and sig[j + 1].type == tokenize.OP and sig[j + 1].string == "**"):
            raise ValueError(f"{key}: степенная башня a**b**c не допускается "
                             "(риск гигантских целых при разборе)")


@lru_cache(maxsize=128)
def _parse_cached(s: str) -> sp.Expr:
    return sp.parse_expr(s, local_dict=_NAMES, global_dict=_GLOBALS)


def parse_field(key: str, s: str) -> sp.Expr:
    """Строка → sympy-выражение от ``x, y`` (ограда → разбор → контроль типа)."""
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{key}: ожидалась непустая строка-выражение f(x, y); "
                         f"{_ALLOWED_TXT}")
    _fence(key, s)
    try:
        e = _parse_cached(s)
    except (ValueError, SyntaxError, TypeError,
            RecursionError, MemoryError) as exc:
        # RecursionError/MemoryError — глубина/размер у CPython-парсера на
        # патологических строках: тоже диагностируемый отказ
        raise ValueError(f"{key}: синтаксическая ошибка/предел ресурсов "
                         f"выражения ({type(exc).__name__}); "
                         f"{_ALLOWED_TXT}") from None
    if not isinstance(e, sp.Expr):
        raise ValueError(f"{key}: '{s}' — не скалярное выражение "
                         f"(получен {type(e).__name__}); {_ALLOWED_TXT}")
    if not e.free_symbols <= {_X, _Y}:
        extra = ", ".join(sorted(str(v) for v in e.free_symbols - {_X, _Y}))
        raise ValueError(f"{key}: свободные символы вне x, y: {extra}")
    return e


def field_values(e: sp.Expr, xq, yq, *, key: str = "expr"):
    """Значения выражения в узлах: константа → ``float``, иначе массив формы xq.

    Скалярный возврат константы сохраняет скалярные пути вызывающего кода
    (например, ``gap_expr``-константа идёт путём скалярного ``gap`` бит-точно).
    Не-вещественные или неконечные значения на узлах — ``ValueError``.
    """
    if not e.free_symbols:
        try:
            v = float(e)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: выражение не вещественно/не конечно "
                             f"({e})") from exc
        if not np.isfinite(v):
            raise ValueError(f"{key}: выражение не конечно ({e})")
        return v
    try:
        f = sp.lambdify((_X, _Y), e, modules="numpy")
        raw = np.asarray(f(np.asarray(xq, float), np.asarray(yq, float)))
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — lambdify/печать sympy: NameError,
        # PrintMethodNotImplementedError и пр. — наружу единый ValueError
        raise ValueError(f"{key}: выражение не вычислимо численно "
                         f"({type(exc).__name__}: {exc})") from exc
    if np.iscomplexobj(raw):
        # NaN в мнимой части не должен обходить сторожа (nan > 0 ложно)
        if not np.all(np.isfinite(raw.imag)) or np.max(np.abs(raw.imag)) > 0.0:
            raise ValueError(f"{key}: выражение даёт комплексные значения на "
                             "области; проверьте sqrt/log")
        raw = raw.real
    vals = np.array(np.broadcast_to(np.asarray(raw, float), np.shape(xq)))
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{key}: выражение обязано быть конечным и вещественным "
                         "во всех узлах квадратуры Ω; проверьте деление, log и "
                         "sqrt (область определения)")
    return vals


def field_laplacian(e: sp.Expr) -> sp.Expr:
    """Символьный лапласиан ``Δf = f_xx + f_yy`` (точный; для члена −h_*²Δq).

    Для НЕГЛАДКИХ выражений (abs/min/max — изломы: Δ порождает
    DiracDelta/Heaviside/sign, невычислимые численно) — ``ValueError``:
    вызывающий код честно опускает член с пометкой.
    """
    if e.has(sp.Abs, sp.Min, sp.Max, sp.sign, sp.Heaviside):
        raise ValueError("лапласиан негладкого выражения (abs/min/max) "
                         "не определён классически — член опускается")
    lap = sp.diff(e, _X, 2) + sp.diff(e, _Y, 2)
    if lap.has(sp.DiracDelta, sp.Heaviside, sp.sign, sp.Derivative):
        raise ValueError("лапласиан выражения негладок — член опускается")
    return lap


__all__ = ["parse_field", "field_values", "field_laplacian"]
