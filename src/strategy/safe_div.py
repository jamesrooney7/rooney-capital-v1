import logging
import math
from typing import Optional, Set, Tuple

import backtrader as bt
import backtrader.functions as btfunc

EPS = 1e-12

LOGGER = logging.getLogger(__name__)
seen_denoms: Set[Tuple[str, str]] = set()


def _is_valid_denominator(den, eps: float) -> bool:
    if den is None:
        return False
    try:
        if not math.isfinite(den):
            return False
        return abs(den) >= eps
    except TypeError:
        return False


def _log_invalid_denominator(den, eps: float, zero, logger: Optional[logging.Logger]) -> None:
    if logger is None:
        return
    if den is None:
        reason = "none"
        value_repr = "None"
    else:
        try:
            if not math.isfinite(den):
                reason = "nan" if math.isnan(den) else "infinite"
                value_repr = repr(den)
            elif abs(den) < eps:
                reason = "near-zero"
                value_repr = repr(den)
            else:
                reason = "invalid"
                value_repr = repr(den)
        except TypeError:
            reason = type(den).__name__
            value_repr = repr(den)
    key = (reason, value_repr)
    if key in seen_denoms:
        return
    seen_denoms.add(key)
    logger.warning(
        "SafeDivision encountered %s denominator (%s); returning fallback %r",
        reason,
        value_repr,
        zero,
    )


class SafeDivision(btfunc.DivByZero):
    """Divide two line series with a fallback for invalid denominators."""

    def __init__(
        self,
        a,
        b,
        zero: float = 0.0,
        eps: float = EPS,
        log: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        num_line = a if hasattr(a, "lines") else bt.LineNum(a)
        den_line = b if hasattr(b, "lines") else bt.LineNum(b)
        super().__init__(num_line, den_line, zero=zero)
        self.p.zero = zero
        self.eps = eps
        self.log = log
        self.logger = (logger or LOGGER) if log else None

    def next(self):
        den = self.b[0]
        if _is_valid_denominator(den, self.eps):
            self[0] = self.a[0] / den
            return

        self[0] = self.p.zero
        if self.log:
            _log_invalid_denominator(den, self.eps, self.p.zero, self.logger)

    def once(self, start, end):
        dst = self.array
        srca = self.a.array
        srcb = self.b.array
        zero = self.p.zero
        eps = self.eps
        log = self.log
        logger = self.logger

        for i in range(start, end):
            den = srcb[i]
            if _is_valid_denominator(den, eps):
                dst[i] = srca[i] / den
            else:
                dst[i] = zero
                if log:
                    _log_invalid_denominator(den, eps, zero, logger)


def safe_div(
    num,
    den,
    zero: float = 0.0,
    orig_den=None,
    eps: float = EPS,
    log: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """Safely divide ``num`` by ``den``.

    ``zero`` is returned when the denominator is ``None``, ``NaN`` or zero.
    For :class:`~backtrader.LineSeries` inputs a :class:`SafeDivision`
    indicator is instantiated to perform the calculation.
    ``orig_den`` is accepted for backwards compatibility but ignored.
    """

    active_logger = logger or LOGGER if log else None

    if hasattr(num, "lines") or hasattr(den, "lines"):
        num_line = num if hasattr(num, "lines") else bt.LineNum(num)
        den_line = den if hasattr(den, "lines") else bt.LineNum(den)
        return SafeDivision(
            num_line,
            den_line,
            zero=zero,
            eps=eps,
            log=log,
            logger=active_logger,
        )

    if _is_valid_denominator(den, eps):
        return num / den

    if log:
        _log_invalid_denominator(den, eps, zero, active_logger)
    return zero


def monkey_patch_division(eps: float = EPS, log: bool = False) -> None:
    """Monkey patch Backtrader division operators to use :class:`SafeDivision`."""

    bt.LineSeries.__truediv__ = lambda self, other: SafeDivision(
        self,
        other,
        eps=eps,
        log=log,
    )
    bt.LineSeries.__rtruediv__ = lambda self, other: SafeDivision(
        other,
        self,
        eps=eps,
        log=log,
    )
