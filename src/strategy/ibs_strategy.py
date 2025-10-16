import bisect
import backtrader as bt
import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

try:  # pragma: no cover - optional dependency guard
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional at runtime
    pd = None

from config import COMMISSION_PER_SIDE, PAIR_MAP
from .filter_column import FilterColumn
from .safe_div import safe_div
from .contract_specs import CONTRACT_SPECS, point_value
from .feature_utils import normalize_column_name


logger = logging.getLogger(__name__)


EXECUTION_SYMBOLS: set[str] = {
    "ES",
    "NQ",
    "RTY",
    "YM",
    "GC",
    "SI",
    "HG",
    "CL",
    "NG",
    "6A",
    "6B",
    "6E",
}

REFERENCE_SYMBOLS: set[str] = {
    "PL",
    "6C",
    "6J",
    "6M",
    "6N",
    "6S",
    "TLT",
    "VIX",
}


FRIENDLY_FILTER_NAMES: dict[str, str] = {
    "enablePrevBarPct": "Prev Bar %",
    "enablePrevIBSDaily": "Prev IBS Daily",
    "enableRSIEntry2Len": "RSI Len 2",
    "enableRSIEntry14Len": "RSI Len 14",
    "enableDailyRSI2Len": "Daily RSI Len 2",
    "enableDailyRSI14Len": "Daily RSI Len 14",
    "enableBBHigh": "Bollinger High",
    "enableBBHighD": "Bollinger High Daily",
    "enableEMA8": "EMA 8",
    "enableEMA20": "EMA 20",
    "enableN7Bar": "N7 Bar",
    "enableHourlyATRPercentile": "Hourly ATR Percentile",
}


FEATURE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "enableOpenClose": ("open_close",),
    "enableDailyATRPercentile": ("daily_atr_percentile",),
    "enableHourlyATRPercentile": ("hourly_atr_percentile",),
}


CROSS_Z_INSTRUMENTS = [
    "ES",
    "NQ",
    "RTY",
    "YM",
    "GC",
    "SI",
    "HG",
    "PL",
    "6A",
    "6B",
    "6C",
    "6E",
    "6J",
    "6M",
    "6N",
    "6S",
    "CL",
    "NG",
    "TLT",
    "VIX",
]

CROSS_Z_TIMEFRAMES: list[tuple[str, str, str]] = [
    ("Hour", "hour", "Hourly"),
    ("Day", "day", "Daily"),
]

METAL_ENERGY_SYMBOLS: set[str] = {"SI", "PL", "HG", "CL", "NG"}

SYMBOL_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "6A": ("sixA", "6a"),
    "6B": ("sixB", "6b"),
    "6C": ("sixC", "6c"),
    "6E": ("sixE", "6e"),
    "6J": ("sixJ", "6j"),
    "6M": ("sixM", "6m"),
    "6N": ("sixN", "6n"),
    "6S": ("sixS", "6s"),
}

CROSS_FEED_ALIASES: dict[str, str] = {}


EXIT_PARAM_DEFAULTS: dict[str, object] = {
    "enable_stop": True,
    "stop_type": "Percent",
    "stop_perc": 1.0,
    "stop_atr_len": 14,
    "stop_atr_mult": 1.0,
    "enable_tp": True,
    "tp_type": "Percent",
    "tp_perc": 1.0,
    "tp_atr_len": 14,
    "tp_atr_mult": 0.5,
    "enable_bar_stop": True,
    "bar_stop_bars": 8,
    "enable_auto_close": True,
    "auto_close_time": 1500,
}


IBS_ENTRY_EXIT_DEFAULTS: dict[str, object] = {
    "enable_ibs_entry": True,
    "ibs_entry_low": 0.0,
    "ibs_entry_high": 0.2,
    "enable_ibs_exit": True,
    "ibs_exit_low": 0.8,
    "ibs_exit_high": 1.0,
}


def _param_prefixes(symbol: str) -> tuple[str, ...]:
    aliases = SYMBOL_PARAM_ALIASES.get(symbol)
    if aliases:
        return aliases
    lower = symbol.lower()
    return (lower,)

CROSS_Z_PARAM_DEFAULTS: dict[str, object] = {}
for _symbol in CROSS_Z_INSTRUMENTS:
    _prefixes = _param_prefixes(_symbol)
    for _tf_key, _, _ in CROSS_Z_TIMEFRAMES:
        CROSS_Z_PARAM_DEFAULTS[f"enable{_symbol}ZScore{_tf_key}"] = False
        for _prefix in _prefixes:
            CROSS_Z_PARAM_DEFAULTS[f"{_prefix}ZLen{_tf_key}"] = 20
            CROSS_Z_PARAM_DEFAULTS[f"{_prefix}ZWindow{_tf_key}"] = 252
            CROSS_Z_PARAM_DEFAULTS[f"{_prefix}ZLow{_tf_key}"] = -2.0
            CROSS_Z_PARAM_DEFAULTS[f"{_prefix}ZHigh{_tf_key}"] = 2.0

for _symbol in CROSS_Z_INSTRUMENTS:
    for _tf_key, _, _friendly in CROSS_Z_TIMEFRAMES:
        FRIENDLY_FILTER_NAMES[f"enable{_symbol}ZScore{_tf_key}"] = (
            f"{_symbol} {_friendly} Z Score"
        )


RETURN_INSTRUMENTS: list[str] = list(CROSS_Z_INSTRUMENTS)
RETURN_TIMEFRAMES = CROSS_Z_TIMEFRAMES
RETURN_PARAM_DEFAULTS: dict[str, object] = {}
for _symbol in RETURN_INSTRUMENTS:
    _prefixes = _param_prefixes(_symbol)
    for _tf_key, _, _friendly in RETURN_TIMEFRAMES:
        RETURN_PARAM_DEFAULTS[f"enable{_symbol}Return{_tf_key}"] = False
        for _prefix in _prefixes:
            RETURN_PARAM_DEFAULTS[f"{_prefix}ReturnLen{_tf_key}"] = 1
            RETURN_PARAM_DEFAULTS[f"{_prefix}ReturnLow{_tf_key}"] = -1.0
            RETURN_PARAM_DEFAULTS[f"{_prefix}ReturnHigh{_tf_key}"] = 1.0
        FRIENDLY_FILTER_NAMES[f"enable{_symbol}Return{_tf_key}"] = (
            f"{_symbol} {_friendly} Return"
        )


def line_val(line, ago: int = 0):
    """Return ``line[ago]`` only when data is present."""

    if line is None or not len(line):
        return None
    try:
        return line[ago]
    except Exception:
        return None


def _extract_timeframe(source) -> object | None:
    """Best-effort extraction of a Backtrader object's timeframe."""

    if source is None:
        return None

    tf = getattr(source, "_timeframe", None)
    if tf is not None:
        return tf

    params = getattr(source, "params", None)
    if params is not None and hasattr(params, "timeframe"):
        return params.timeframe

    data = getattr(source, "data", None)
    if data is not None and data is not source:
        tf = getattr(data, "_timeframe", None)
        if tf is not None:
            return tf

    datas = getattr(source, "datas", None)
    if datas:
        for data in datas:
            tf = getattr(data, "_timeframe", None)
            if tf is not None:
                return tf

    return None


def _is_daily_timeframe(tf: object | None) -> bool:
    if tf is None:
        return False
    if isinstance(tf, str):
        return tf.lower().startswith("d")
    try:
        return tf == bt.TimeFrame.Days
    except Exception:
        pass
    try:
        return int(tf) == int(bt.TimeFrame.Days)
    except Exception:
        return False


def _feature_timeframe_label(timeframe: object | None) -> str:
    """Return the snake_case label for a metadata timeframe."""

    if timeframe is None:
        return "timeframe"
    if isinstance(timeframe, str):
        tf = timeframe.lower()
        if tf.startswith("d"):
            return "daily"
        if tf.startswith("h"):
            return "hourly"
        return normalize_column_name(tf)
    try:
        if timeframe == bt.TimeFrame.Days or int(timeframe) == int(bt.TimeFrame.Days):
            return "daily"
    except Exception:
        pass
    try:
        if timeframe == bt.TimeFrame.Minutes and int(timeframe) == int(bt.TimeFrame.Minutes):
            return "hourly"
    except Exception:
        pass
    return normalize_column_name(str(timeframe))


def _metadata_feature_key(symbol: str, timeframe: object | None, suffix: str) -> str:
    """Build a normalised feature key for cross-instrument metrics."""

    label = _feature_timeframe_label(timeframe)
    base = f"{symbol}_{label}_{suffix}"
    return normalize_column_name(base)


class ExpandingPercentileTracker:
    """Incrementally compute expanding percentile ranks."""

    def __init__(self) -> None:
        self._sorted_values: dict[str, list[float]] = defaultdict(list)
        self._marker_history: dict[str, dict[object, float]] = defaultdict(dict)
        self._last_marker: dict[str, object] = {}
        self._last_pct: dict[str, float] = {}

    @staticmethod
    def _is_valid(value: float | int | None) -> bool:
        if value is None:
            return False
        try:
            return not math.isnan(float(value))
        except Exception:
            return False

    def update(self, key: str, value, marker: object | None) -> float | None:
        """Insert ``value`` for ``key`` and return its percentile rank."""

        if not self._is_valid(value):
            return None

        v = float(value)
        if marker is not None and self._last_marker.get(key) == marker:
            return self._last_pct.get(key)

        values = self._sorted_values[key]
        bisect.insort(values, v)
        n = len(values)
        if n == 0:
            return None

        left = bisect.bisect_left(values, v)
        right = bisect.bisect_right(values, v)
        avg_rank = (left + 1 + right) / 2.0
        pct = avg_rank / n

        self._last_marker[key] = marker
        self._last_pct[key] = pct
        if marker is not None:
            self._marker_history[key][marker] = pct
        return pct

    def get(self, key: str, marker: object | None) -> float | None:
        """Return the cached percentile for ``marker`` if available."""

        if marker is None:
            return self._last_pct.get(key)
        return self._marker_history.get(key, {}).get(marker)


def timeframed_line_val(
    line,
    *,
    data=None,
    timeframe: object | None = None,
    daily_ago: int = -1,
    intraday_ago: int = 0,
):
    """Return ``line[ago]`` based on the underlying timeframe.

    Daily inputs default to ``ago=-1`` (the last completed bar) while
    intraday inputs remain on the current bar (``ago=0``).
    """

    ago = timeframe_ago(
        data=data,
        line=line,
        timeframe=timeframe,
        daily_ago=daily_ago,
        intraday_ago=intraday_ago,
    )
    return line_val(line, ago=ago)


def timeframe_ago(
    *,
    data=None,
    line=None,
    timeframe: object | None = None,
    daily_ago: int = -1,
    intraday_ago: int = 0,
) -> int:
    """Return the default ``ago`` offset for the supplied timeframe."""

    tf = timeframe
    if tf is None:
        tf = _extract_timeframe(data)
        if tf is None:
            tf = _extract_timeframe(line)

    return daily_ago if _is_daily_timeframe(tf) else intraday_ago

# Expected data feed names:
# - "<SYMBOL>_hour": hourly bars for the given symbol
# - "<SYMBOL>_day": daily bars for the given symbol
# - "TLT_day": TLT daily bars (shared filter)
#
# CSV inputs are expected to follow the `<SYMBOL>_bt.csv` naming convention.
logging.basicConfig(level=logging.WARNING)


def clamp_period(period):
    """Ensure indicator periods are at least 2."""

    try:
        return max(2, int(period))
    except Exception:
        logging.warning("Invalid period %s; using 2", period)
        return 2


def clamp_return_len(length):
    """Ensure return lookbacks are at least 1 bar."""

    try:
        return max(1, int(length))
    except Exception:
        logging.warning("Invalid return length %s; using 1", length)
        return 1


def classify_pivots(last_high, last_low, close) -> int:
    """Classify pivot regime based on the relationship between ``close`` and
    the most recent pivot high/low.

    Returns 1 when price breaks above the last pivot high, 3 when it falls
    below the last pivot low, and 2 when it remains inside the range or when
    inputs are missing.
    """

    try:
        if close is None:
            return 2
        if last_high is not None and close > last_high:
            return 1
        if last_low is not None and close < last_low:
            return 3
    except Exception:
        pass
    return 2


class RollingMedian(bt.Indicator):
    """Rolling window median."""

    lines = ("median",)
    params = (("period", 20),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        window = self.data.get(size=self.p.period)
        if window:
            self.lines.median[0] = statistics.median(window)


class PercentReturn(bt.Indicator):
    """Percent return over a configurable lookback."""

    lines = ("pct",)
    params = (("period", 1),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        ago = -int(self.p.period)
        prev = line_val(self.data, ago=ago)
        curr = line_val(self.data)
        if prev in (None, 0) or curr is None:
            self.lines.pct[0] = float("nan")
            return
        if math.isnan(prev) or math.isnan(curr):
            self.lines.pct[0] = float("nan")
            return
        change = safe_div(curr - prev, prev, zero=None)
        if change is None or math.isnan(change):
            self.lines.pct[0] = float("nan")
        else:
            self.lines.pct[0] = change * 100.0


class IbsStrategy(bt.Strategy):
    """Intraday IBS reversal with daily trend filters.

    The optional ``ml_model``/``ml_features`` parameters enable callers to
    inject a binary classifier whose positive-class probability gates entries
    and is cached alongside each filter snapshot for downstream logging.
    """

    params = dict(
        size=1,
        symbol="ES",
        filter_columns=None,
        ml_model=None,
        ml_features=None,
        ml_threshold=None,
        trade_start=date(2010, 1, 1),
        # Session window defaults
        use_window1=True,
        start_time1="0000",
        end_time1="1500",
        use_window2=True,
        start_time2="1700",
        end_time2="2400",
        # Exit options
        **EXIT_PARAM_DEFAULTS,
        # Calendar filters
        enable_dow=False,
        allowed_dow="1,2,3,4,5",
        enable_month=False,
        allowed_month="1,2,3,4,5,6,7,8,9,10,11,12",
        enable_dom=False,
        dom_day=25,
        enable_beg_week=False,
        enable_even_odd=False,
        even_odd_sel="Even",
        enable_prev_day_pct=False,
        prev_day_pct_low=-2.0,
        prev_day_pct_high=2.0,
        enable_prev_bar_pct=False,
        prev_bar_pct_low=-1.0,
        prev_bar_pct_high=1.0,
        prev_bar_pct_tf="Hourly",
        # IBS filters
        **IBS_ENTRY_EXIT_DEFAULTS,
        enable_daily_ibs=False,
        daily_ibs_low=0.0,
        daily_ibs_high=0.2,
        enable_prev_ibs=False,
        prev_ibs_low=0.8,
        prev_ibs_high=1.0,
        enable_prev_ibs_daily=False,
        prev_ibs_daily_low=0.8,
        prev_ibs_daily_high=1.0,
        # Pair IBS filter
        enable_pair_ibs=False,
        pair_symbol="",
        pair_ibstf="Hourly",
        pair_ibs_low=0.0,
        pair_ibs_high=0.2,
        # Pair Z-Score filter
        enable_pair_z=False,
        pair_z_len=20,
        pair_z_low=-2.0,
        pair_z_high=2.0,
        pair_z_tf="Hourly",
        # Secondary value filter
        use_val_filter=False,
        val_symbol="CBOE:VIX",
        val_tf="Daily",
        val_len=1,
        val_low=0.0,
        val_high=15.0,
        # Regime filters
        enable_vix_reg=False,
        vix_mode="Risk On",
        vix_len=100,
        vix_tf="Daily",
        # RSI filters
        enable_rsi_entry=False,
        rsi_entry_len=5,
        rsi_entry_low=40,
        rsi_entry_high=60,
        rsi_entry_tf="Hourly",
        enable_rsi_entry2_len=False,
        rsi_entry2_len=2,
        rsi_entry2_low=0,
        rsi_entry2_high=40,
        rsi_entry2_tf="Hourly",
        enable_rsi_entry14_len=False,
        rsi_entry14_len=14,
        rsi_entry14_low=40,
        rsi_entry14_high=60,
        rsi_entry14_tf="Hourly",
        # Secondary RSI filter
        enable_rsi_entry2=False,
        rsi2_symbol="",
        rsi2_entry_len=14,
        rsi2_entry_low=0,
        rsi2_entry_high=40,
        rsi2_entry_tf="Hourly",
        # Daily RSI band
        enableDailyRSI=False,
        dailyRSILen=5,
        dailyRSILow=0,
        dailyRSIHigh=40,
        dailyRSITF="Daily",
        enableDailyRSI2Len=False,
        dailyRSI2Len=2,
        dailyRSI2Low=0,
        dailyRSI2High=40,
        dailyRSI2TF="Daily",
        enableDailyRSI14Len=False,
        dailyRSI14Len=14,
        dailyRSI14Low=40,
        dailyRSI14High=60,
        dailyRSI14TF="Daily",
        # Pivot filters
        entry_setting="inside range",
        signal_tf="Hourly",
        leftLenH=5,
        rightLenH=5,
        leftLenL=5,
        rightLenL=5,
        # Supply zone filter
        use_supply_zone=False,
        supplyZoneTF="Hourly",
        zoneATRLen=14,
        zoneATRLowMult=0.0,
        zoneATRHighMult=1.0,
        # ATR Z-Score
        enableATRZ=False,
        atrLen=5,
        atrWindow=252,
        atrLow=-2.0,
        atrHigh=2.0,
        atrTF="Hourly",
        # Volume Z-Score
        enableVolZ=False,
        volLen=5,
        volWindow=252,
        volLow=-2.0,
        volHigh=2.0,
        volTF="Hourly",
        # Daily ATR Z-Score
        enableDATRZ=False,
        dAtrLen=5,
        dAtrWindow=252,
        dAtrLow=-2.0,
        dAtrHigh=2.0,
        dAtrTF="Daily",
        # Daily Volume Z-Score
        enableDVolZ=False,
        dVolLen=5,
        dVolWindow=252,
        dVolLow=-2.0,
        dVolHigh=2.0,
        dVolTF="Daily",
        # Generic price Z-Score
        enableZScore=False,
        zScoreLen=20,
        zScoreLow=-2.0,
        zScoreHigh=2.0,
        zScoreTF="Hourly",
        # Distance Z filter
        enableDistZ=False,
        distZTF="Hourly",
        distZMaLen=20,
        distZStdLen=20,
        distZDir="Below",
        distZThresh=-1.5,
        # Momentum Z (3-bar) filter
        enableMom3=False,
        mom3TF="Hourly",
        mom3Len=3,
        mom3ZLen=20,
        mom3Dir="Below",
        mom3Thresh=-1.5,
        # TR/ATR ratio filter
        enableTRATR=False,
        tratrTF="Hourly",
        tratrLen=20,
        tratrDir="Above",
        tratrThresh=1.2,
        # Bearish bar count
        enableBearCount=False,
        bearTF="Hourly",
        bearMin=2,
        # Inside bar pattern
        enableInsideBar=False,
        insideBarTF="Hourly",
        # N7 bar smallest-range pattern
        enableN7Bar=False,
        n7BarTF="Hourly",
        # Bollinger Bands filter
        enableBB=False,
        bbTF="Hourly",
        bbLen=20,
        bbMult=2.0,
        bbDir="Below Lower",
        enable_bb_high=False,
        bbHighTF="Hourly",
        bbHighLen=20,
        bbHighMult=2.0,
        enable_bb_high_d=False,
        bbHighDTF="Daily",
        bbHighDLen=20,
        bbHighDMult=2.0,
        # Donchian Channel filter
        enableDonch=False,
        donchTF="Daily",
        donchLen=20,
        donchDir="Above",
        # Intraday EMA filters
        enableEMA8=False,
        ema8Len=8,
        ema8TF="Hourly",
        ema8Dir="Above",
        enableEMA20=False,
        ema20Len=20,
        ema20TF="Hourly",
        ema20Dir="Above",
        enableEMA50=False,
        ema50Len=50,
        ema50TF="Hourly",
        ema50Dir="Above",
        enableEMA200=False,
        ema200Len=200,
        ema200TF="Hourly",
        ema200Dir="Above",
        # Daily EMA filters
        enableEMA20D=False,
        ema20DLen=20,
        ema20DTF="Daily",
        ema20DDir="Above",
        enableEMA50D=False,
        ema50DLen=50,
        ema50DTF="Daily",
        ema50DDir="Above",
        enableEMA200D=False,
        ema200DLen=200,
        ema200DTF="Daily",
        ema200DDir="Above",
        # Additional filters
        enableSessions=False,
        sessionSel=1,
        enableOpenClose=False,
        openCloseSel=1,
        enableRangeCompressionATR=False,
        rcLen=14,
        rcThresh=0.5,
        rcDir="Below",
        enableDailyRangeCompression=False,
        drcLen=14,
        drcThresh=0.5,
        drcDir="Below",
        enableBullishBarCount=False,
        bullTF="Hourly",
        bullMin=2,
        enable4BarRet=False,
        ret4Low=-1.0,
        ret4High=1.0,
        enable4BarRetD=False,
        dRet4Low=-1.0,
        dRet4High=1.0,
        enableAboveOpen=False,
        aboSel="Above",
        enableDailyATRPercentile=False,
        dAtrPercLen=14,
        dAtrPercWindow=252,
        dAtrPercLow=0,
        dAtrPercHigh=100,
        enableHourlyATRPercentile=False,
        hAtrPercLen=14,
        hAtrPercWindow=252,
        hAtrPercLow=0,
        hAtrPercHigh=100,
        enableDirDrift=False,
        driftLen=50,
        driftLow=-2.0,
        driftHigh=2.0,
        enableDirDriftD=False,
        dDriftLen=20,
        dDriftLow=-2.0,
        dDriftHigh=2.0,
        enableSpiralER=False,
        serLookbacks="5,20,60",
        serWeights="0.5,0.25,0.125",
        serLow=0.0,
        serHigh=1.0,
        serTF="Hourly",
        enableTWRC=False,
        twrcFast=10,
        twrcBase=100,
        twrcThreshold=0.7,
        twrcTau=10,
        twrcTrigger=0.5,
        twrcTF="Hourly",
        enableMASlope=False,
        maSlopeLen=20,
        maSlopeShift=5,
        maSlopeAtrLen=14,
        maSlopeLow=-1.0,
        maSlopeHigh=1.0,
        enableDailySlope=False,
        dSlopeLen=20,
        dSlopeShift=5,
        dSlopeAtrLen=14,
        dSlopeLow=-1.0,
        dSlopeHigh=1.0,
        enableMASpread=False,
        maSpreadFast=20,
        maSpreadSlow=50,
        maSpreadAtrLen=14,
        maSpreadUseATR=False,
        maSpreadTF="Hourly",
        maSpreadLow=0.0,
        maSpreadHigh=1.0,
        enableDonchProx=False,
        donchProxTF="Hourly",
        donchProxLen=20,
        donchProxATRLen=14,
        donchProxLow=0.0,
        donchProxHigh=1.0,
        enableBBW=False,
        bbwTF="Hourly",
        bbwLen=20,
        bbwMult=2.0,
        bbwLow=0.0,
        bbwHigh=1.0,
        enableADX=False,
        adxLen=14,
        adxLow=20.0,
        adxHigh=40.0,
        enableDADX=False,
        dAdxLen=14,
        dAdxLow=20.0,
        dAdxHigh=40.0,
        enablePSAR=False,
        psarTF="Hourly",
        psarAF=0.02,
        psarAFmax=0.2,
        psarUseAbs=False,
        psarLow=-1.0,
        psarHigh=1.0,
    )
    params.update(CROSS_Z_PARAM_DEFAULTS)
    params.update(RETURN_PARAM_DEFAULTS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sym = (self.p.symbol or "").upper()
        if not sym:
            raise ValueError("symbol parameter must be set")
        self.p.symbol = sym
        self.can_execute = sym in EXECUTION_SYMBOLS

        self._periods: list[int] = []
        self.cross_data_cache: dict[tuple[str, str], bt.LineSeries] = {}
        self.cross_zscore_cache: dict[tuple[str, str], dict[str, object]] = {}
        self.return_indicator_cache: dict[tuple[str, str], dict[str, object]] = {}

        def get_period(val):
            cp = clamp_period(val)
            self._periods.append(cp)
            return cp

        # Data aliases for clarity
        hourly_name = f"{sym}_hour"
        try:
            self.hourly = self.getdatabyname(hourly_name)
        except KeyError:
            self.hourly = None
        if self.hourly is None or self.hourly._name != hourly_name:
            raise ValueError(f"Missing data feed {hourly_name}")

        daily_name = f"{sym}_day"
        try:
            self.daily = self.getdatabyname(daily_name)
        except KeyError:
            self.daily = None
        if self.daily is None or self.daily._name != daily_name:
            raise ValueError(f"Missing data feed {daily_name}")

        tlt_name = "TLT_day"
        try:
            self.tlt = self.getdatabyname(tlt_name)
        except KeyError:
            self.tlt = None
        if self.tlt is None or self.tlt._name != tlt_name:
            raise ValueError(f"Missing data feed {tlt_name}")

        self.available_data_names: set[str] = {
            data._name
            for data in getattr(self, "datas", [])
            if getattr(data, "_name", None)
        }
        self._available_vix_suffixes: set[str] = {
            name.split("_", 1)[1]
            for name in self.available_data_names
            if name.startswith("VIX_")
        }
        self.has_vix: bool = bool(self._available_vix_suffixes)
        if not self.has_vix:
            logging.info(
                "VIX reference feeds are unavailable; falling back to neutral defaults"
            )

        def select_base_data(tf_param):
            tf = str(tf_param).lower()
            return self.daily if tf.startswith("d") else self.hourly

        def init_bollinger(tf_param, length, mult):
            tf = str(tf_param).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is None:
                return None, None
            period = get_period(length)
            indicator = bt.indicators.BollingerBands(
                data.close, period=period, devfactor=mult
            )
            return data, indicator

        def store_zscore_pipeline(
            meta: dict[str, object],
            data_feed: bt.LineSeries,
            pipeline: dict[str, object] | None,
            length: int,
            window: int,
        ) -> None:
            if not pipeline:
                return
            meta.update(
                {
                    "line": pipeline["line"],
                    "data": data_feed,
                    "mean": pipeline["mean"],
                    "std": pipeline["std"],
                    "denom": pipeline["denom"],
                    "len": length,
                    "window": window,
                }
            )

        def store_return_pipeline(
            meta: dict[str, object],
            data_feed: bt.LineSeries,
            pipeline: dict[str, object] | None,
            lookback: int,
        ) -> None:
            if not pipeline:
                return
            meta["data"] = data_feed
            meta["lookback"] = lookback
            meta["indicator"] = pipeline.get("indicator")
            meta["line"] = pipeline.get("line")

        # Data series used for pivot calculations
        tf = str(self.p.signal_tf).lower()
        self.signal_data = self.daily if tf.startswith("d") else self.hourly
        self.last_pivot_high = self.prev_pivot_high = None
        self.last_pivot_low = self.prev_pivot_low = None

        # Discrete bar-pattern data selections
        self.inside_bar_data = select_base_data(self.p.insideBarTF)
        self.n7_bar_data = select_base_data(self.p.n7BarTF)
        self.prev_bar_pct_data = select_base_data(self.p.prev_bar_pct_tf)

        # Daily trend filters
        tlt_len = get_period(20)
        sma200_len = get_period(200)
        self.tlt_sma20 = bt.indicators.SimpleMovingAverage(
            self.tlt.close, period=tlt_len
        )
        self.sma200 = bt.indicators.SimpleMovingAverage(
            self.daily.close, period=sma200_len
        )

        self.log = logging.getLogger(__name__).info
        self.log(f"Initialized {sym} with hourly feed: {self.hourly._name}")
        self.order = None
        self.current_signal = None
        self.trades_log: list[dict] = []
        self.pending_exit: dict | None = None
        self.filter_columns: list[FilterColumn] = list(self.p.filter_columns or [])
        self.filter_keys = {column.parameter for column in self.filter_columns}
        self.filter_column_keys = {column.column_key for column in self.filter_columns}
        self.filter_columns_by_param: dict[str, list[FilterColumn]] = {}
        self.column_to_param: dict[str, str] = {}
        self.percentile_tracker = ExpandingPercentileTracker()
        for column in self.filter_columns:
            self.filter_columns_by_param.setdefault(column.parameter, []).append(column)
            self.column_to_param[column.column_key] = column.parameter

        self.ml_model = getattr(self.p, "ml_model", None)
        raw_features = getattr(self.p, "ml_features", None)
        if isinstance(raw_features, str) and raw_features:
            self.ml_features: tuple[str, ...] | None = (raw_features,)
        elif raw_features:
            try:
                self.ml_features = tuple(raw_features)
            except TypeError:
                raise ValueError("ml_features must be an iterable of feature names")
        else:
            self.ml_features = None
        if self.ml_features and len(self.ml_features) == 0:
            self.ml_features = None

        threshold_param = getattr(self.p, "ml_threshold", None)
        if threshold_param is None:
            self.ml_threshold: float | None = None
        else:
            try:
                self.ml_threshold = float(threshold_param)
            except (TypeError, ValueError) as exc:
                raise ValueError("ml_threshold must be a numeric value") from exc

        if self.ml_model is not None and not hasattr(self.ml_model, "predict_proba"):
            raise ValueError("ml_model must implement predict_proba")

        self._ml_last_score: float | None = None

        # Clamp bar and pivot lengths
        self.bar_stop_bars = get_period(self.p.bar_stop_bars)
        self.left_len_h = get_period(self.p.leftLenH)
        self.right_len_h = get_period(self.p.rightLenH)
        self.left_len_l = get_period(self.p.leftLenL)
        self.right_len_l = get_period(self.p.rightLenL)
        self._periods.append(clamp_period(self.left_len_h + self.right_len_h))
        self._periods.append(clamp_period(self.left_len_l + self.right_len_l))

        # Pre‑parsed session windows for quick comparison
        self.win1_start = int(self.p.start_time1)
        self.win1_end = int(self.p.end_time1)
        self.win2_start = int(self.p.start_time2)
        self.win2_end = int(self.p.end_time2)

        # Calendar filters
        self.allowed_dow = {
            int(x) for x in str(self.p.allowed_dow).split(",") if x
        }
        self.allowed_month = {
            int(x) for x in str(self.p.allowed_month).split(",") if x
        }
        try:
            self.dom_threshold = int(self.p.dom_day)
        except Exception:
            logging.warning(
                "Invalid dom_day %s; Day of Month filter disabled", self.p.dom_day
            )
            self.dom_threshold = None

        # RSI indicator (only constructed if used)
        self.rsi = None
        if self.p.enable_rsi_entry or "enableRSIEntry" in self.filter_keys:
            tf = str(self.p.rsi_entry_tf).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                period = get_period(self.p.rsi_entry_len)
                self.rsi = bt.indicators.RSI(data.close, period=period)

        self.rsi_len2 = None
        if self.p.enable_rsi_entry2_len or "enableRSIEntry2Len" in self.filter_keys:
            tf = str(self.p.rsi_entry2_tf).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                period = get_period(self.p.rsi_entry2_len)
                self.rsi_len2 = bt.indicators.RSI(data.close, period=period)

        self.rsi_len14 = None
        if self.p.enable_rsi_entry14_len or "enableRSIEntry14Len" in self.filter_keys:
            tf = str(self.p.rsi_entry14_tf).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                period = get_period(self.p.rsi_entry14_len)
                self.rsi_len14 = bt.indicators.RSI(data.close, period=period)

        # Cross-asset data feeds / indicators
        if not self.p.pair_symbol:
            self.p.pair_symbol = PAIR_MAP.get(sym, "")
        if not self.p.rsi2_symbol and self.p.symbol in PAIR_MAP:
            self.p.rsi2_symbol = PAIR_MAP[self.p.symbol]
        psym = self.p.pair_symbol.upper()
        if (self.p.enable_pair_ibs or "enablePairIBS" in self.filter_keys) and not psym:
            raise ValueError("pair_symbol must be set when enable_pair_ibs is True")

        self.pair_data = None
        if self.p.enable_pair_ibs or "enablePairIBS" in self.filter_keys:
            tf = str(self.p.pair_ibstf).lower()
            name = f"{psym}_day" if tf.startswith("d") else f"{psym}_hour"
            try:
                self.pair_data = self.getdatabyname(name)
            except KeyError:
                self.pair_data = None
            if self.pair_data is None or self.pair_data._name != name:
                raise ValueError(f"Missing data feed {name}")

        # Pair Z-Score indicator
        self.pair_z = None
        self.pair_z_data = None
        self.pair_z_denom = None
        if self.p.enable_pair_z or "enablePairZ" in self.filter_keys:
            if not psym:
                raise ValueError("pair_symbol must be set when enable_pair_z is True")
            tf = str(self.p.pair_z_tf).lower()
            name = f"{psym}_day" if tf.startswith("d") else f"{psym}_hour"
            try:
                pair_data = self.getdatabyname(name)
            except KeyError:
                pair_data = None
            base_data = self.daily if tf.startswith("d") else self.hourly
            if (
                pair_data is None
                or pair_data._name != name
                or base_data is None
            ):
                logging.warning("Missing data feed %s for pair Z-Score data", name)
            else:
                period = get_period(self.p.pair_z_len)
                base_close = base_data.close
                pair_close = pair_data.close
                spread = base_close - pair_close
                mean = bt.indicators.SimpleMovingAverage(spread, period=period)
                den = bt.indicators.StandardDeviation(spread, period=period)
                self.pair_z_denom = den
                den = abs(den) + 1e-12
                self.pair_z_data = pair_data
                self.pair_z = safe_div(spread - mean, den, orig_den=self.pair_z_denom)

        self.cross_zscore_meta: dict[str, dict[str, object]] = {}
        for symbol in CROSS_Z_INSTRUMENTS:
            prefixes = _param_prefixes(symbol)
            for tf_key, feed_suffix, _friendly in CROSS_Z_TIMEFRAMES:
                enable_param = f"enable{symbol}ZScore{tf_key}"
                len_aliases = [f"{prefix}ZLen{tf_key}" for prefix in prefixes]
                window_aliases = [f"{prefix}ZWindow{tf_key}" for prefix in prefixes]
                low_aliases = [f"{prefix}ZLow{tf_key}" for prefix in prefixes]
                high_aliases = [f"{prefix}ZHigh{tf_key}" for prefix in prefixes]
                meta: dict[str, object] = {
                    "symbol": symbol,
                    "timeframe": tf_key,
                    "feed_suffix": feed_suffix,
                    "line": None,
                    "data": None,
                    "mean": None,
                    "std": None,
                    "denom": None,
                    "len_param": len_aliases[0],
                    "window_param": window_aliases[0],
                    "low_param": low_aliases[0],
                    "high_param": high_aliases[0],
                    "len_aliases": tuple(len_aliases),
                    "window_aliases": tuple(window_aliases),
                    "low_aliases": tuple(low_aliases),
                    "high_aliases": tuple(high_aliases),
                }
                self.cross_zscore_meta[enable_param] = meta
                need_indicator = (
                    getattr(self.p, enable_param, False)
                    or (enable_param in self.filter_keys)
                    or symbol in METAL_ENERGY_SYMBOLS
                )
                if not need_indicator:
                    continue
                data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
                if data_feed is None:
                    continue
                default_len = getattr(self.p, meta["len_param"], 20)
                default_window = getattr(self.p, meta["window_param"], 252)
                length_raw = self._resolve_param_value(
                    meta["len_aliases"], default_len
                )
                window_raw = self._resolve_param_value(
                    meta["window_aliases"], default_window
                )
                length = get_period(length_raw)
                window = get_period(window_raw)
                pipeline = self._build_cross_zscore_pipeline(
                    symbol,
                    tf_key,
                    feed_suffix,
                    length,
                    window,
                    data_feed,
                )
                store_zscore_pipeline(meta, data_feed, pipeline, length, window)

        self.return_meta: dict[str, dict[str, object]] = {}
        for symbol in RETURN_INSTRUMENTS:
            prefixes = _param_prefixes(symbol)
            for tf_key, feed_suffix, _friendly in RETURN_TIMEFRAMES:
                enable_param = f"enable{symbol}Return{tf_key}"
                len_aliases = [f"{prefix}ReturnLen{tf_key}" for prefix in prefixes]
                low_aliases = [f"{prefix}ReturnLow{tf_key}" for prefix in prefixes]
                high_aliases = [f"{prefix}ReturnHigh{tf_key}" for prefix in prefixes]
                meta = {
                    "symbol": symbol,
                    "timeframe": tf_key,
                    "feed_suffix": feed_suffix,
                    "len_param": len_aliases[0],
                    "low_param": low_aliases[0],
                    "high_param": high_aliases[0],
                    "len_aliases": tuple(len_aliases),
                    "low_aliases": tuple(low_aliases),
                    "high_aliases": tuple(high_aliases),
                    "data": None,
                    "lookback": None,
                    "last_dt": None,
                    "last_value": None,
                    "indicator": None,
                    "line": None,
                }
                self.return_meta[enable_param] = meta
                need_value = (
                    getattr(self.p, enable_param, False)
                    or (enable_param in self.filter_keys)
                    or symbol in METAL_ENERGY_SYMBOLS
                )
                if not need_value:
                    continue
                data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
                if data_feed is None:
                    continue
                default_len = getattr(self.p, meta["len_param"], 1)
                lookback_raw = self._resolve_param_value(
                    meta["len_aliases"], default_len
                )
                lookback = clamp_return_len(lookback_raw)
                pipeline = self._build_return_pipeline(
                    symbol,
                    tf_key,
                    feed_suffix,
                    lookback,
                    data_feed,
                )
                store_return_pipeline(meta, data_feed, pipeline, lookback)

        preload_symbols = {
            "6A",
            "6B",
            "6C",
            "6E",
            "6N",
            "6S",
            "TLT",
        }
        if self.has_vix:
            preload_symbols.add("VIX")
        for symbol in preload_symbols:
            for tf_key, feed_suffix, _friendly in CROSS_Z_TIMEFRAMES:
                enable_param = f"enable{symbol}ZScore{tf_key}"
                meta = self.cross_zscore_meta.get(enable_param)
                if not meta or meta.get("line") is not None:
                    continue
                data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
                if data_feed is None:
                    continue
                default_len = getattr(self.p, meta["len_param"], 20)
                default_window = getattr(self.p, meta["window_param"], 252)
                length_raw = self._resolve_param_value(
                    meta["len_aliases"], default_len
                )
                window_raw = self._resolve_param_value(
                    meta["window_aliases"], default_window
                )
                length = get_period(length_raw)
                window = get_period(window_raw)
                pipeline = self._build_cross_zscore_pipeline(
                    symbol,
                    tf_key,
                    feed_suffix,
                    length,
                    window,
                    data_feed,
                )
                store_zscore_pipeline(meta, data_feed, pipeline, length, window)

        for symbol in preload_symbols:
            for tf_key, feed_suffix, _friendly in RETURN_TIMEFRAMES:
                enable_param = f"enable{symbol}Return{tf_key}"
                meta = self.return_meta.get(enable_param)
                if not meta or meta.get("line") is not None:
                    continue
                data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
                if data_feed is None:
                    continue
                default_len = getattr(self.p, meta["len_param"], 1)
                lookback_raw = self._resolve_param_value(
                    meta["len_aliases"], default_len
                )
                lookback = clamp_return_len(lookback_raw)
                pipeline = self._build_return_pipeline(
                    symbol,
                    tf_key,
                    feed_suffix,
                    lookback,
                    data_feed,
                )
                store_return_pipeline(meta, data_feed, pipeline, lookback)

        # Secondary value filter
        self.val_data = None
        self.val_ma = None
        if self.p.use_val_filter or "useValFilter" in self.filter_keys:
            vsym = self.p.val_symbol.upper()
            if ":" in vsym:
                vsym = vsym.split(":")[-1]
            tf = str(self.p.val_tf).lower()
            name = f"{vsym}_day" if tf.startswith("d") else f"{vsym}_hour"
            try:
                data = self.getdatabyname(name)
            except KeyError:
                data = None
            if data is None or data._name != name:
                logging.warning("Missing data feed %s for value filter", name)
            else:
                self.val_data = data
                period = get_period(self.p.val_len)
                self.val_ma = bt.indicators.SimpleMovingAverage(
                    data.close, period=period
                )

        # Regime filters
        self.vix_data = None
        self.vix_median = None
        if self.p.enable_vix_reg or "enableVixReg" in self.filter_keys:
            if not self.has_vix:
                logging.info(
                    "VIX regime filter enabled but no VIX data is available; "
                    "trades will not be gated by VIX"
                )
            else:
                tf = str(self.p.vix_tf).lower()
                name = "VIX_day" if tf.startswith("d") else "VIX_hour"
                try:
                    data = self.getdatabyname(name)
                except KeyError:
                    data = None
                if data is None or data._name != name:
                    logging.warning(
                        "Missing data feed %s for regime filter", name
                    )
                else:
                    self.vix_data = data
                    period = get_period(self.p.vix_len)
                    median = RollingMedian(data.close, period=period)
                    self.vix_median = bt.Max(median, 1e-12)

        self.rsi2 = None
        self.rsi2_data = None
        if self.p.enable_rsi_entry2 or "enableRSIEntry2" in self.filter_keys:
            rsym = self.p.rsi2_symbol.upper()
            if not rsym:
                raise ValueError(
                    "rsi2_symbol must be set when enable_rsi_entry2 is True"
                )
            tf2 = str(self.p.rsi2_entry_tf).lower()
            name2 = f"{rsym}_day" if tf2.startswith("d") else f"{rsym}_hour"
            try:
                data2 = self.getdatabyname(name2)
            except KeyError:
                data2 = None
            if data2 is None:
                if self.p.enable_rsi_entry2:
                    raise ValueError(
                        f"Missing feed for secondary RSI symbol {self.p.rsi2_symbol}"
                    )
            else:
                if data2._name != name2:
                    raise ValueError(
                        f"Expected {name2} feed for secondary RSI data"
                    )
                self.rsi2_data = data2
                period = get_period(self.p.rsi2_entry_len)
                self.rsi2 = bt.indicators.RSI(data2.close, period=period)

        # Daily RSI filter
        self.daily_rsi = None
        self.daily_rsi_data = None
        if self.p.enableDailyRSI or "enableDailyRSI" in self.filter_keys:
            tf = str(self.p.dailyRSITF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.daily_rsi_data = data
                period = get_period(self.p.dailyRSILen)
                self.daily_rsi = bt.indicators.RSI(data.close, period=period)

        self.daily_rsi2 = None
        self.daily_rsi2_data = None
        if self.p.enableDailyRSI2Len or "enableDailyRSI2Len" in self.filter_keys:
            tf = str(self.p.dailyRSI2TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.daily_rsi2_data = data
                period = get_period(self.p.dailyRSI2Len)
                self.daily_rsi2 = bt.indicators.RSI(data.close, period=period)

        self.daily_rsi14 = None
        self.daily_rsi14_data = None
        if self.p.enableDailyRSI14Len or "enableDailyRSI14Len" in self.filter_keys:
            tf = str(self.p.dailyRSI14TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.daily_rsi14_data = data
                period = get_period(self.p.dailyRSI14Len)
                self.daily_rsi14 = bt.indicators.RSI(data.close, period=period)

        # Bearish bar count data
        self.bear_data = None
        if self.p.enableBearCount or "enableBearCount" in self.filter_keys:
            tf = str(self.p.bearTF).lower()
            self.bear_data = self.daily if tf.startswith("d") else self.hourly

        # Track previous bar IBS for filter checks
        self.prev_ibs_val: float | None = None
        self.prev_daily_ibs_val: float | None = None

        # Track the last date auto close executed
        self.last_auto_close_date: date | None = None
        close_time = int(self.p.auto_close_time)
        self.close_h = close_time // 100
        self.close_m = close_time % 100

        # Track entry bar for bar-based exits
        self.bar_executed: int | None = None

        # ATR indicators for stop loss and take profit exits
        self.stop_atr = (
            bt.indicators.AverageTrueRange(
                self.hourly, period=get_period(self.p.stop_atr_len)
            )
            if self.p.enable_stop
            and self.p.stop_type.lower().startswith("atr")
            and self.hourly is not None
            else None
        )
        self.tp_atr = (
            bt.indicators.AverageTrueRange(
                self.hourly, period=get_period(self.p.tp_atr_len)
            )
            if self.p.enable_tp
            and self.p.tp_type.lower().startswith("atr")
            and self.hourly is not None
            else None
        )

        # ATR used for supply zone distance checks
        self.zone_atr = None
        self.supply_zone_data = select_base_data(self.p.supplyZoneTF)
        if self.p.use_supply_zone or "useSupplyZone" in self.filter_keys:
            if self.supply_zone_data is not None:
                period = get_period(self.p.zoneATRLen)
                self.zone_atr = bt.indicators.AverageTrueRange(
                    self.supply_zone_data, period=period
                )

        # ATR Z-Score
        self.atr_z = None
        self.atr_z_denom = None
        self.atr_z_data = None
        if self.p.enableATRZ or "enableATRZ" in self.filter_keys:
            tf = str(self.p.atrTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.atr_z_data = data
                atr_len = get_period(self.p.atrLen)
                atr_window = get_period(self.p.atrWindow)
                atr = bt.indicators.AverageTrueRange(data, period=atr_len)
                mean = bt.indicators.SimpleMovingAverage(atr, period=atr_window)
                den = bt.indicators.StandardDeviation(atr, period=atr_window)
                self.atr_z_denom = den
                den = abs(den) + 1e-12
                self.atr_z = safe_div(atr - mean, den, orig_den=self.atr_z_denom)

        # Volume Z-Score
        self.vol_z = None
        self.vol_z_denom = None
        self.vol_z_data = None
        if self.p.enableVolZ or "enableVolZ" in self.filter_keys:
            tf = str(self.p.volTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.vol_z_data = data
                vol_len = get_period(self.p.volLen)
                vol_window = get_period(self.p.volWindow)
                vol = bt.indicators.SimpleMovingAverage(
                    data.volume, period=vol_len
                )
                mean = bt.indicators.SimpleMovingAverage(
                    vol, period=vol_window
                )
                den = bt.indicators.StandardDeviation(
                    vol, period=vol_window
                )
                self.vol_z_denom = den
                den = abs(den) + 1e-12
                self.vol_z = safe_div(vol - mean, den, orig_den=self.vol_z_denom)

        # Daily ATR Z-Score
        self.datr_z = None
        self.datr_z_denom = None
        self.datr_z_data = None
        if self.p.enableDATRZ or "enableDATRZ" in self.filter_keys:
            tf = str(self.p.dAtrTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.datr_z_data = data
                d_atr_len = get_period(self.p.dAtrLen)
                d_atr_window = get_period(self.p.dAtrWindow)
                atr = bt.indicators.AverageTrueRange(data, period=d_atr_len)
                mean = bt.indicators.SimpleMovingAverage(
                    atr, period=d_atr_window
                )
                den = bt.indicators.StandardDeviation(
                    atr, period=d_atr_window
                )
                self.datr_z_denom = den
                den = abs(den) + 1e-12
                self.datr_z = safe_div(atr - mean, den, orig_den=self.datr_z_denom)

        # Daily Volume Z-Score
        self.dvol_z = None
        self.dvol_z_denom = None
        self.dvol_z_data = None
        if self.p.enableDVolZ or "enableDVolZ" in self.filter_keys:
            tf = str(self.p.dVolTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.dvol_z_data = data
                d_vol_len = get_period(self.p.dVolLen)
                d_vol_window = get_period(self.p.dVolWindow)
                vol = bt.indicators.SimpleMovingAverage(
                    data.volume, period=d_vol_len
                )
                mean = bt.indicators.SimpleMovingAverage(
                    vol, period=d_vol_window
                )
                den = bt.indicators.StandardDeviation(
                    vol, period=d_vol_window
                )
                self.dvol_z_denom = den
                den = abs(den) + 1e-12
                self.dvol_z = safe_div(vol - mean, den, orig_den=self.dvol_z_denom)

        # Generic price Z-Score
        self.price_z = None
        self.price_z_denom = None
        self.price_z_data = None
        if self.p.enableZScore or "enableZScore" in self.filter_keys:
            tf = str(self.p.zScoreTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.price_z_data = data
                period = get_period(self.p.zScoreLen)
                mean = bt.indicators.SimpleMovingAverage(
                    data.close, period=period
                )
                den = bt.indicators.StandardDeviation(
                    data.close, period=period
                )
                self.price_z_denom = den
                den = abs(den) + 1e-12
                self.price_z = safe_div(data.close - mean, den, orig_den=self.price_z_denom)

        # Distance Z filter
        self.dist_z = None
        self.dist_z_denom = None
        self.dist_z_data = None
        if self.p.enableDistZ or "enableDistZ" in self.filter_keys:
            tf = str(self.p.distZTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.dist_z_data = data
                ma_len = get_period(self.p.distZMaLen)
                std_len = get_period(self.p.distZStdLen)
                ma = bt.indicators.SimpleMovingAverage(
                    data.close, period=ma_len
                )
                dist = data.close - ma
                change = data.close - data.close(-1)
                den = bt.indicators.StandardDeviation(change, period=std_len)
                self.dist_z_denom = den
                den = abs(den) + 1e-12
                self.dist_z = safe_div(dist, den, orig_den=self.dist_z_denom)

        # Momentum Z (3-bar) filter
        self.mom3_z = None
        self.mom3_z_denom = None
        self.mom3_z_data = None
        if self.p.enableMom3 or "enableMom3" in self.filter_keys:
            tf = str(self.p.mom3TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.mom3_z_data = data
                mom_len = get_period(self.p.mom3Len)
                z_len = get_period(self.p.mom3ZLen)
                mom = bt.indicators.Momentum(
                    data.close, period=mom_len
                )
                mean = bt.indicators.SimpleMovingAverage(
                    mom, period=z_len
                )
                den = bt.indicators.StandardDeviation(
                    mom, period=z_len
                )
                self.mom3_z_denom = den
                den = abs(den) + 1e-12
                self.mom3_z = safe_div(mom - mean, den, orig_den=self.mom3_z_denom)

        # TR/ATR ratio filter
        self.tratr_ratio = None
        self.tratr_denom = None
        self.tratr_data = None
        if self.p.enableTRATR or "enableTRATR" in self.filter_keys:
            tf = str(self.p.tratrTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.tratr_data = data
                tr = bt.indicators.TrueRange(data)
                tratr_len = get_period(self.p.tratrLen)
                den = bt.indicators.AverageTrueRange(
                    data, period=tratr_len
                )
                self.tratr_denom = den
                den = abs(den) + 1e-12
                self.tratr_ratio = safe_div(tr, den, orig_den=self.tratr_denom)

        # Bollinger Bands
        self.bb_data = None
        self.bb = None
        if self.p.enableBB or "enableBB" in self.filter_keys:
            self.bb_data, self.bb = init_bollinger(
                self.p.bbTF, self.p.bbLen, self.p.bbMult
            )

        self.bb_high_data = None
        self.bb_high = None
        if self.p.enable_bb_high or "enableBBHigh" in self.filter_keys:
            self.bb_high_data, self.bb_high = init_bollinger(
                self.p.bbHighTF, self.p.bbHighLen, self.p.bbHighMult
            )

        self.bb_high_d_data = None
        self.bb_high_d = None
        if self.p.enable_bb_high_d or "enableBBHighD" in self.filter_keys:
            self.bb_high_d_data, self.bb_high_d = init_bollinger(
                self.p.bbHighDTF, self.p.bbHighDLen, self.p.bbHighDMult
            )

        # Donchian Channels
        self.donch_high = self.donch_low = None
        if self.p.enableDonch or "enableDonch" in self.filter_keys:
            tf = str(self.p.donchTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.donch_data = data
                period = get_period(self.p.donchLen)
                self.donch_high = bt.indicators.Highest(data.high, period=period)
                self.donch_low = bt.indicators.Lowest(data.low, period=period)

        # Intraday EMA filters
        self.ema8 = None
        self.ema8_data = None
        if self.p.enableEMA8 or "enableEMA8" in self.filter_keys:
            tf = str(self.p.ema8TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema8_data = data
                period = get_period(self.p.ema8Len)
                self.ema8 = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        self.ema20 = None
        self.ema20_data = None
        if self.p.enableEMA20 or "enableEMA20" in self.filter_keys:
            tf = str(self.p.ema20TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema20_data = data
                period = get_period(self.p.ema20Len)
                self.ema20 = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        self.ema50 = None
        self.ema50_data = None
        if self.p.enableEMA50 or "enableEMA50" in self.filter_keys:
            tf = str(self.p.ema50TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema50_data = data
                period = get_period(self.p.ema50Len)
                self.ema50 = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        self.ema200 = None
        self.ema200_data = None
        if self.p.enableEMA200 or "enableEMA200" in self.filter_keys:
            tf = str(self.p.ema200TF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema200_data = data
                period = get_period(self.p.ema200Len)
                self.ema200 = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        # Daily EMA filters
        self.ema20d = None
        if self.p.enableEMA20D or "enableEMA20D" in self.filter_keys:
            tf = str(self.p.ema20DTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema20d_data = data
                period = get_period(self.p.ema20DLen)
                self.ema20d = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        self.ema50d = None
        if self.p.enableEMA50D or "enableEMA50D" in self.filter_keys:
            tf = str(self.p.ema50DTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema50d_data = data
                period = get_period(self.p.ema50DLen)
                self.ema50d = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        self.ema200d = None
        if self.p.enableEMA200D or "enableEMA200D" in self.filter_keys:
            tf = str(self.p.ema200DTF).lower()
            data = self.daily if tf.startswith("d") else self.hourly
            if data is not None:
                self.ema200d_data = data
                period = get_period(self.p.ema200DLen)
                self.ema200d = bt.indicators.ExponentialMovingAverage(
                    data.close, period=period
                )

        # Range Compression ATR (hourly)
        self.rc_atr = None
        if self.p.enableRangeCompressionATR or "enableRangeCompressionATR" in self.filter_keys:
            period = get_period(self.p.rcLen)
            self.rc_atr = bt.indicators.AverageTrueRange(self.hourly, period=period)

        # Daily Range Compression ATR
        self.drc_atr = None
        if self.p.enableDailyRangeCompression or "enableDailyRangeCompression" in self.filter_keys:
            period = get_period(self.p.drcLen)
            self.drc_atr = bt.indicators.AverageTrueRange(self.daily, period=period)

        # Bullish bar count data
        self.bull_data = None
        if self.p.enableBullishBarCount or "enableBullishBarCount" in self.filter_keys:
            tf = str(self.p.bullTF).lower()
            self.bull_data = self.daily if tf.startswith("d") else self.hourly

        # Previous daily close for above/below open
        self.prev_daily_close = None

        # Daily ATR Percentile
        self.datr_pct_atr = None
        self.datr_pct_values = []
        self.datr_pct_pct = None
        self.datr_pct_last_date = None  # track last daily bar processed
        self.datr_pct_val_date = None  # track last ATR value date
        if self.p.enableDailyATRPercentile or "enableDailyATRPercentile" in self.filter_keys:
            period = get_period(self.p.dAtrPercLen)
            self.datr_pct_atr = bt.indicators.AverageTrueRange(self.daily, period=period)

        # Hourly ATR Percentile
        self.hatr_pct_atr = None
        self.hatr_pct_values = []
        self.hatr_pct_pct = None
        self.hatr_pct_last_dt = None  # track last hourly bar processed
        self.hatr_pct_val_dt = None  # track last ATR value timestamp
        if self.p.enableHourlyATRPercentile or "enableHourlyATRPercentile" in self.filter_keys:
            period = get_period(self.p.hAtrPercLen)
            self.hatr_pct_atr = bt.indicators.AverageTrueRange(self.hourly, period=period)

        # Spiral Efficiency Ratio configuration
        self.ser_lookbacks = [int(x) for x in str(self.p.serLookbacks).split(",") if x]
        self.ser_weights = [float(x) for x in str(self.p.serWeights).split(",") if x]
        self.ser_weights = self.ser_weights[: len(self.ser_lookbacks)]
        self.ser_data = select_base_data(self.p.serTF)

        # TWRC indicators and streak
        self.twrc_data = select_base_data(self.p.twrcTF)
        self.twrc_fast = None
        self.twrc_base = None
        self.twrc_streak = 0
        self.twrc_last_dt = None
        if (
            self.twrc_data is not None
            and (self.p.enableTWRC or "enableTWRC" in self.filter_keys)
        ):
            self.twrc_fast = bt.indicators.AverageTrueRange(
                self.twrc_data, period=get_period(self.p.twrcFast)
            )
            self.twrc_base = bt.indicators.AverageTrueRange(
                self.twrc_data, period=get_period(self.p.twrcBase)
            )

        # MA Slope (fast)
        self.ma_slope_ema = None
        self.ma_slope_atr = None
        if self.p.enableMASlope or "enableMASlope" in self.filter_keys:
            self.ma_slope_ema = bt.indicators.ExponentialMovingAverage(
                self.hourly.close, period=get_period(self.p.maSlopeLen)
            )
            self.ma_slope_atr = bt.indicators.AverageTrueRange(
                self.hourly, period=get_period(self.p.maSlopeAtrLen)
            )

        # Daily Slope (fast)
        self.d_slope_ema = None
        self.d_slope_atr = None
        if self.p.enableDailySlope or "enableDailySlope" in self.filter_keys:
            self.d_slope_ema = bt.indicators.ExponentialMovingAverage(
                self.daily.close, period=get_period(self.p.dSlopeLen)
            )
            self.d_slope_atr = bt.indicators.AverageTrueRange(
                self.daily, period=get_period(self.p.dSlopeAtrLen)
            )

        # MA Spread
        self.ma_spread_fast = self.ma_spread_slow = None
        self.ma_spread_atr = None
        self.ma_spread_data = select_base_data(self.p.maSpreadTF)
        if (
            self.p.enableMASpread or "enableMASpread" in self.filter_keys
        ) and self.ma_spread_data is not None:
            self.ma_spread_fast = bt.indicators.ExponentialMovingAverage(
                self.ma_spread_data.close, period=get_period(self.p.maSpreadFast)
            )
            self.ma_spread_slow = bt.indicators.ExponentialMovingAverage(
                self.ma_spread_data.close, period=get_period(self.p.maSpreadSlow)
            )
            if self.p.maSpreadUseATR:
                self.ma_spread_atr = bt.indicators.AverageTrueRange(
                    self.ma_spread_data, period=get_period(self.p.maSpreadAtrLen)
                )

        # Donchian Proximity
        self.donch_prox_high = self.donch_prox_low = None
        self.donch_prox_atr = None
        self.donch_prox_data = select_base_data(self.p.donchProxTF)
        if (
            self.p.enableDonchProx or "enableDonchProx" in self.filter_keys
        ) and self.donch_prox_data is not None:
            period = get_period(self.p.donchProxLen)
            self.donch_prox_high = bt.indicators.Highest(
                self.donch_prox_data.high, period=period
            )
            self.donch_prox_low = bt.indicators.Lowest(
                self.donch_prox_data.low, period=period
            )
            self.donch_prox_atr = bt.indicators.AverageTrueRange(
                self.donch_prox_data, period=get_period(self.p.donchProxATRLen)
            )

        # Bollinger Bandwidth
        self.bbw = None
        self.bbw_data = select_base_data(self.p.bbwTF)
        if (
            self.p.enableBBW or "enableBBW" in self.filter_keys
        ) and self.bbw_data is not None:
            period = get_period(self.p.bbwLen)
            self.bbw = bt.indicators.BollingerBands(
                self.bbw_data.close, period=period, devfactor=self.p.bbwMult
            )

        # ADX indicators
        self.adx = None
        if self.p.enableADX or "enableADX" in self.filter_keys:
            self.adx = bt.indicators.ADX(self.hourly, period=get_period(self.p.adxLen))

        self.dadx = None
        if self.p.enableDADX or "enableDADX" in self.filter_keys:
            self.dadx = bt.indicators.ADX(self.daily, period=get_period(self.p.dAdxLen))

        # Parabolic SAR
        self.psar = None
        self.psar_data = select_base_data(self.p.psarTF)
        if (
            self.p.enablePSAR or "enablePSAR" in self.filter_keys
        ) and self.psar_data is not None:
            self.psar = bt.indicators.ParabolicSAR(
                self.psar_data, af=self.p.psarAF, afmax=self.p.psarAFmax
            )

        # Finalize the warm-up period once all lookback-based indicators have
        # registered their requirements.  ``self.max_period`` represents the
        # longest lookback length among all constructed indicators.
        self.max_period = max(self._periods)
        logging.getLogger(__name__).debug(
            "Strategy warm-up period: %s", self.max_period
        )
        self.addminperiod(self.max_period)

    def prenext(self):
        """Return early until ``self.max_period`` bars are ready."""

        if len(self.data) < self.max_period:
            return
        self.next()

    def update_pivots(self) -> None:
        """Update recent pivot highs and lows on the selected timeframe."""

        data = self.signal_data

        lh, rh = self.left_len_h, self.right_len_h
        if len(data) > lh + rh:
            idx = -rh
            candidate = data.high[idx]
            if all(candidate > data.high[idx - i] for i in range(1, lh + 1)) and \
               all(candidate >= data.high[idx + i] for i in range(1, rh + 1)):
                self.prev_pivot_high = self.last_pivot_high
                self.last_pivot_high = candidate

        ll, rl = self.left_len_l, self.right_len_l
        if len(data) > ll + rl:
            idx = -rl
            candidate = data.low[idx]
            if all(candidate < data.low[idx - i] for i in range(1, ll + 1)) and \
               all(candidate <= data.low[idx + i] for i in range(1, rl + 1)):
                self.prev_pivot_low = self.last_pivot_low
                self.last_pivot_low = candidate

    def _calc_ibs(self, data, ago: int = 0) -> Optional[float]:
        """Safely compute IBS for the given data feed.

        ``ago`` specifies how many bars back to reference. Defaults to ``0``
        (the current bar). Returns ``None`` when the requested bar is not
        available.
        """

        if len(data) <= abs(ago):
            return None

        hi = line_val(data.high, ago)
        lo = line_val(data.low, ago)
        close = line_val(data.close, ago)
        if hi is None or lo is None or close is None:
            return None

        den = hi - lo
        if math.isnan(hi) or math.isnan(lo):
            ts = data.datetime.datetime(ago)
            logging.warning("NaN bar at %s for %s", ts, data._name)
            return 0.5
        if den <= 0:
            ts = data.datetime.datetime(ago)
            logging.warning("Zero-range bar at %s for %s", ts, data._name)
            return 0.5
        raw = safe_div(close - lo, den)
        return max(0, min(1, raw))

    def ibs(self):
        return self._calc_ibs(self.hourly)

    def daily_ibs(self):
        """Return IBS computed on the previous daily bar if available."""

        data = self.daily
        if len(data) < 2:
            return None
        return self._calc_ibs(data, ago=-1)

    def prev_daily_ibs(self):
        """Return IBS computed on the prior completed daily bar before yesterday."""

        data = self.daily
        if len(data) < 3:
            return None
        return self._calc_ibs(data, ago=-2)

    def prev_day_pct(self):
        """Return previous day's percent change if available."""
        data = self.daily
        if len(data) < 3:
            return None
        base_ago = timeframe_ago(data=data, intraday_ago=-1)
        prev = line_val(data.close, ago=base_ago)
        prev_prev = line_val(data.close, ago=base_ago - 1)
        if prev is None or prev_prev is None:
            return None
        prev = float(prev)
        prev_prev = float(prev_prev)
        pct = safe_div(prev - prev_prev, prev_prev, zero=None)
        return pct * 100 if pct is not None else None

    def prev_bar_pct(self):
        """Return the previous bar's percent change for the configured feed."""

        data = getattr(self, "prev_bar_pct_data", None)
        if data is None or len(data) < 3:
            return None

        base_ago = timeframe_ago(data=data)
        last_close = line_val(data.close, ago=base_ago)
        prior_close = line_val(data.close, ago=base_ago - 1)
        if None in (last_close, prior_close):
            return None
        if any(math.isnan(v) for v in (last_close, prior_close)):
            return None

        pct = safe_div(last_close - prior_close, prior_close, zero=None)
        return pct * 100 if pct is not None else None

    def _calc_return_value(self, meta: dict[str, object]) -> float | None:
        """Compute and cache the percent return for a return filter."""

        line = meta.get("line")
        data = meta.get("data")
        lookback = meta.get("lookback")

        dt_num = None
        if data is not None:
            dt_num = timeframed_line_val(
                data.datetime,
                data=data,
                timeframe=meta.get("timeframe"),
            )
            if dt_num is not None and meta.get("last_dt") == dt_num:
                cached = meta.get("last_value")
                if isinstance(cached, (int, float)) and not math.isnan(cached):
                    return cached

        if line is not None:
            val = timeframed_line_val(
                line,
                data=data,
                timeframe=meta.get("timeframe"),
            )
            if val is None:
                val = timeframed_line_val(
                    line,
                    data=data,
                    timeframe=meta.get("timeframe"),
                    daily_ago=0,
                    intraday_ago=0,
                )
            if val is None or math.isnan(val):
                return None
            if dt_num is not None:
                meta["last_dt"] = dt_num
            meta["last_value"] = float(val)
            return float(val)

        if data is None or lookback is None:
            return None

        if dt_num is None:
            dt_num = timeframed_line_val(
                data.datetime,
                data=data,
                timeframe=meta.get("timeframe"),
            )
        if dt_num is None:
            return None

        base_ago = timeframe_ago(
            data=data,
            timeframe=meta.get("timeframe"),
        )
        last_close = line_val(data.close, ago=base_ago)
        base_close = line_val(data.close, ago=base_ago - int(lookback))
        if None in (last_close, base_close):
            return None
        if any(math.isnan(v) for v in (last_close, base_close)):
            return None

        change = safe_div(last_close - base_close, base_close, zero=None)
        if change is None or math.isnan(change):
            return None
        pct = change * 100.0
        meta["last_dt"] = dt_num
        meta["last_value"] = pct
        return pct

    def _has_feed_name(self, symbol: str, feed_suffix: str) -> bool:
        names_to_try = [f"{symbol}_{feed_suffix}"]
        alt_symbol = CROSS_FEED_ALIASES.get(symbol)
        if alt_symbol:
            names_to_try.append(f"{alt_symbol}_{feed_suffix}")
        available = getattr(self, "available_data_names", set())
        return any(name in available for name in names_to_try)

    def _get_cross_feed(
        self, symbol: str, feed_suffix: str, enable_param: str
    ) -> bt.LineSeries | None:
        """Return and cache the requested cross-instrument data feed."""

        key = (symbol, feed_suffix)
        if key in self.cross_data_cache:
            return self.cross_data_cache[key]
        feed_name = f"{symbol}_{feed_suffix}"
        alt_symbol = CROSS_FEED_ALIASES.get(symbol)
        names_to_try = [feed_name]
        if alt_symbol:
            alt_name = f"{alt_symbol}_{feed_suffix}"
            if alt_name not in names_to_try:
                names_to_try.append(alt_name)
        data_feed = None
        actual_name = None
        for name in names_to_try:
            try:
                candidate = self.getdatabyname(name)
            except KeyError:
                candidate = None
            if candidate is not None and candidate._name == name:
                data_feed = candidate
                actual_name = name
                break
        if data_feed is None:
            optional_vix = symbol == "VIX" and not self._has_feed_name(
                symbol, feed_suffix
            )
            if optional_vix:
                logging.info(
                    "Optional VIX feed %s is unavailable; using neutral defaults",
                    feed_name,
                )
                self.cross_data_cache[key] = None
                return None
            message = f"Missing data feed {feed_name} for {symbol} {feed_suffix} data"
            if alt_symbol:
                message += f" (also tried {alt_symbol}_{feed_suffix})"
            if getattr(self.p, enable_param, False):
                raise ValueError(message)
            logging.warning(message)
            self.cross_data_cache[key] = None
            return None
        if actual_name and hasattr(self, "available_data_names"):
            self.available_data_names.add(actual_name)
        self.cross_data_cache[key] = data_feed
        return data_feed

    def _build_cross_zscore_pipeline(
        self,
        symbol: str,
        timeframe: str,
        feed_suffix: str,
        length: int,
        window: int,
        data_feed: bt.LineSeries | None = None,
    ) -> dict[str, object] | None:
        """Construct and cache the rolling mean/std + Z-score pipeline."""

        cache_key = (symbol, timeframe)
        cached = self.cross_zscore_cache.get(cache_key)
        if cached and cached.get("len") == length and cached.get("window") == window:
            return cached
        if data_feed is None:
            enable_param = f"enable{symbol}ZScore{timeframe}"
            data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
        if data_feed is None:
            return None
        mean = bt.indicators.SimpleMovingAverage(data_feed.close, period=length)
        std = bt.indicators.StandardDeviation(data_feed.close, period=window)
        denom = bt.Max(std, 1e-12)
        line = safe_div(data_feed.close - mean, denom, orig_den=std)
        pipeline = {
            "line": line,
            "mean": mean,
            "std": std,
            "denom": denom,
            "len": length,
            "window": window,
            "data": data_feed,
        }
        self.cross_zscore_cache[cache_key] = pipeline
        return pipeline

    def _build_return_pipeline(
        self,
        symbol: str,
        timeframe: str,
        feed_suffix: str,
        lookback: int,
        data_feed: bt.LineSeries | None = None,
    ) -> dict[str, object] | None:
        """Construct and cache the percent-return indicator pipeline."""

        cache_key = (symbol, timeframe)
        cached = self.return_indicator_cache.get(cache_key)
        if cached and cached.get("lookback") == lookback:
            return cached
        if data_feed is None:
            enable_param = f"enable{symbol}Return{timeframe}"
            data_feed = self._get_cross_feed(symbol, feed_suffix, enable_param)
        if data_feed is None:
            return None
        indicator = PercentReturn(data_feed.close, period=lookback)
        pipeline = {
            "indicator": indicator,
            "line": indicator.lines.pct,
            "lookback": lookback,
            "data": data_feed,
        }
        self.return_indicator_cache[cache_key] = pipeline
        return pipeline

    def _resolve_param_value(
        self,
        param_names: tuple[str, ...] | list[str] | str,
        default: object | None = None,
    ):
        """Return the first configured value for ``param_names``.

        Parameters may have multiple aliases (e.g., ``sixA`` and ``6a``). The
        first attribute present on ``self.p`` with a non-``None`` value is
        returned. When no values are found ``default`` is returned instead.
        """

        if isinstance(param_names, str):
            param_names = (param_names,)
        for name in param_names:
            if not name:
                continue
            try:
                value = getattr(self.p, name)
            except AttributeError:
                continue
            if value is not None:
                return value
        return default

    def _update_datr_pct(self):
        """Update and return the current daily ATR percentile."""

        if self.datr_pct_atr is None:
            return None
        # Use the ATR from the last completed daily bar
        val = line_val(self.datr_pct_atr, ago=-1)
        if val is None or math.isnan(val):
            return None

        # Determine the date of the ATR value to avoid duplicates
        dt_num = line_val(self.daily.datetime, ago=-1)
        if dt_num is None:
            return None
        dt = bt.num2date(dt_num).date()
        if self.datr_pct_val_date == dt:
            return self.datr_pct_pct

        self.datr_pct_values.append(val)
        window = self.datr_pct_values[-self.p.dAtrPercWindow :]
        if not window:
            return None
        sorted_vals = sorted(window)
        import bisect as _bisect

        idx = _bisect.bisect_left(sorted_vals, val)
        pct = 100 * idx / len(sorted_vals)
        self.datr_pct_val_date = dt
        return pct

    def _update_hatr_pct(self):
        """Update and return the current hourly ATR percentile."""

        if self.hatr_pct_atr is None:
            return None

        val = line_val(self.hatr_pct_atr)
        if val is None or math.isnan(val):
            return None

        dt_num = line_val(self.hourly.datetime)
        if dt_num is None:
            return None
        if self.hatr_pct_val_dt == dt_num:
            return self.hatr_pct_pct

        self.hatr_pct_values.append(val)
        window = self.hatr_pct_values[-self.p.hAtrPercWindow :]
        if not window:
            return None
        sorted_vals = sorted(window)
        import bisect as _bisect

        idx = _bisect.bisect_left(sorted_vals, val)
        pct = 100 * idx / len(sorted_vals)
        self.hatr_pct_val_dt = dt_num
        return pct

    def _compute_spiral_er(self):
        data = self.ser_data
        if data is None:
            return None
        base_ago = timeframe_ago(
            data=data,
            timeframe=self.p.serTF,
        )
        total = 0.0
        weight_sum = 0.0
        for lookback, weight in zip(self.ser_lookbacks, self.ser_weights):
            if len(data) <= lookback:
                continue
            c0 = line_val(data.close, ago=base_ago)
            cL = line_val(data.close, ago=base_ago - lookback)
            if None in (c0, cL) or math.isnan(c0) or math.isnan(cL):
                continue
            direct = abs(c0 - cL)
            path = 0.0
            ok = True
            for i in range(1, lookback + 1):
                c1 = line_val(data.close, ago=base_ago - i + 1)
                c2 = line_val(data.close, ago=base_ago - i)
                if None in (c1, c2) or math.isnan(c1) or math.isnan(c2):
                    ok = False
                    break
                path += abs(c1 - c2)
            if ok and path > 0:
                total += weight * direct / path
                weight_sum += weight
        if weight_sum > 0:
            return total / weight_sum
        return None

    def _compute_twrc_score(self):
        if self.twrc_fast is None or self.twrc_base is None:
            return None
        fast = timeframed_line_val(
            self.twrc_fast,
            data=self.twrc_data,
            timeframe=self.p.twrcTF,
        )
        base = timeframed_line_val(
            self.twrc_base,
            data=self.twrc_data,
            timeframe=self.p.twrcTF,
        )
        if fast is None or base in (None, 0) or math.isnan(fast) or math.isnan(base):
            return None
        ratio = safe_div(fast, base, zero=None)
        if ratio is None or math.isnan(ratio):
            return None
        threshold = self.p.twrcThreshold
        tau = self.p.twrcTau
        if threshold is None or threshold <= 0:
            raise ValueError("twrcThreshold must be > 0")
        if tau is None or tau <= 0:
            raise ValueError("twrcTau must be > 0")
        c = max(0.0, min(1.0, safe_div(threshold - ratio, threshold)))
        w = 1 - math.exp(-safe_div(self.twrc_streak, tau))
        return c * w

    def collect_filter_values(self, intraday_ago: int = 0) -> dict:
        """Map each configured ``filter_columns`` key to its current numeric value."""

        values: dict = {}

        def coerce_float(value):
            if value is None:
                return None
            try:
                numeric = float(value)
            except Exception:
                return None
            if math.isnan(numeric):
                return None
            return numeric

        def percentile_marker(
            *,
            data=None,
            line=None,
            timeframe=None,
            intraday_offset: int | None = None,
            daily_ago: int = -1,
            align_to_date: bool = False,
        ):
            marker_intraday = intraday_ago if intraday_offset is None else intraday_offset
            source = data
            if source is None and line is not None:
                source = getattr(line, "data", None)
            tf = timeframe
            if tf is None:
                tf = _extract_timeframe(source)
                if tf is None:
                    tf = _extract_timeframe(line)
            dt_line = None
            if source is not None:
                dt_line = getattr(source, "datetime", None)
            if dt_line is None and line is not None:
                dt_line = getattr(line, "datetime", None)
            if dt_line is None:
                return None
            try:
                dt_num = timeframed_line_val(
                    dt_line,
                    data=source,
                    timeframe=tf,
                    intraday_ago=marker_intraday,
                    daily_ago=daily_ago,
                )
            except Exception:
                dt_num = None
            if dt_num is None:
                return None
            marker_dt = bt.num2date(dt_num)
            if align_to_date or _is_daily_timeframe(tf):
                return marker_dt.date()
            return marker_dt

        def record_percentile(
            base_key: str,
            value,
            *,
            data=None,
            line=None,
            timeframe=None,
            intraday_offset: int | None = None,
            daily_ago: int = -1,
            align_to_date: bool = False,
            marker=None,
        ) -> float | None:
            if marker is None:
                marker = percentile_marker(
                    data=data,
                    line=line,
                    timeframe=timeframe,
                    intraday_offset=intraday_offset,
                    daily_ago=daily_ago,
                    align_to_date=align_to_date,
                )
            pct = self.percentile_tracker.update(base_key, value, marker)
            return float(pct) if pct is not None else None

        def record_value(column_key: str, value):
            values[column_key] = value
            alias = FRIENDLY_FILTER_NAMES.get(column_key)
            if alias:
                values[alias] = value
            feature_aliases = FEATURE_KEY_ALIASES.get(column_key)
            if feature_aliases:
                for alias_key in feature_aliases:
                    values[alias_key] = value

        def record_param(param_key: str, value):
            columns = self.filter_columns_by_param.get(param_key)
            if not columns:
                record_value(param_key, value)
                return
            for column in columns:
                record_value(column.column_key, value)

        def record_continuous(
            base_key: str,
            param_key: str,
            value,
            *,
            data=None,
            line=None,
            timeframe=None,
            intraday_offset: int | None = None,
            daily_ago: int = -1,
            align_to_date: bool = False,
            marker=None,
            pct_source=None,
        ):
            record_param(param_key, value)
            record_value(base_key, value)
            pct_value = pct_source if pct_source is not None else value
            pct = record_percentile(
                base_key,
                pct_value,
                data=data,
                line=line,
                timeframe=timeframe,
                intraday_offset=intraday_offset,
                daily_ago=daily_ago,
                align_to_date=align_to_date,
                marker=marker,
            )
            record_param(f"{base_key}_pct", pct)

        sig_close = line_val(self.signal_data.close, ago=intraday_ago)
        record_param(
            "fractalPivot",
            classify_pivots(
                self.last_pivot_high,
                self.last_pivot_low,
                sig_close,
            ),
        )
        if self.has_vix and self.vix_median is not None:
            raw_vix_med = line_val(self.vix_median, ago=intraday_ago)
            vix_med = float(raw_vix_med) if raw_vix_med is not None else 0.0
        else:
            vix_med = 0.0
        values["vix_med"] = vix_med
        if not self.filter_columns:
            self.ensure_filter_keys(values)
            return values
        dt_num = line_val(self.hourly.datetime, ago=intraday_ago)
        if dt_num is None:
            self.ensure_filter_keys(values)
            return values
        dt = bt.num2date(dt_num)
        dom_threshold = self.dom_threshold
        for column in self.filter_columns:
            key = column.parameter
            if isinstance(key, str) and key.endswith("_pct"):
                # Percentile columns are populated alongside their base filters.
                # Avoid overwriting the recorded percentile with ``None`` when
                # iterating over the synthetic parameter itself.
                continue
            if key in {"fractalPivot", "vix_med"}:
                continue
            if key == "allowedDOW":
                record_param(key, dt.isoweekday())
            elif key == "allowedMon":
                record_param(key, dt.month)
            elif key == "enableDOM":
                if dom_threshold is not None:
                    record_param(key, 1 if dt.day > dom_threshold else 2)
                else:
                    record_param(key, None)
            elif key == "domDay":
                if dom_threshold is not None:
                    record_param(key, 1 if dt.day > dom_threshold else 2)
                else:
                    record_param(key, None)
            elif key == "enableBegWeek":
                record_param(key, 1 if dt.weekday() <= 1 else 2)
            elif key == "enableEvenOdd":
                record_param(key, 1 if dt.day % 2 == 0 else 2)
            elif key == "enablePrevDayPct":
                raw = self.prev_day_pct()
                stored = float(raw) if raw is not None else float("nan")
                record_continuous(
                    "prev_day_pct",
                    key,
                    stored,
                    data=self.daily,
                    align_to_date=True,
                    pct_source=raw,
                )
            elif key == "enablePrevBarPct":
                raw = self.prev_bar_pct()
                stored = float(raw) if raw is not None else float("nan")
                record_continuous(
                    "prev_bar_pct",
                    key,
                    stored,
                    data=self.prev_bar_pct_data,
                    timeframe=self.p.prev_bar_pct_tf,
                    pct_source=raw,
                )
            elif key in {"enableIBSEntry", "enableIBSExit"}:
                raw = self.ibs()
                record_continuous(
                    "ibs",
                    key,
                    raw,
                    data=self.hourly,
                    intraday_offset=0,
                    pct_source=raw,
                )
            elif key == "enableDailyIBS":
                raw = self.daily_ibs()
                record_continuous(
                    "daily_ibs",
                    key,
                    raw,
                    data=self.daily,
                    align_to_date=True,
                    pct_source=raw,
                )
            elif key == "enablePrevIBS":
                raw = self.prev_ibs_val
                record_continuous(
                    "prev_ibs",
                    key,
                    raw,
                    data=self.hourly,
                    intraday_offset=0,
                    pct_source=raw,
                )
            elif key == "enablePrevIBSDaily":
                val = (
                    self.prev_daily_ibs_val
                    if self.prev_daily_ibs_val is not None
                    else self.prev_daily_ibs()
                )
                record_param(key, float(val) if val is not None else None)
            elif key == "enablePairIBS":
                if self.pair_data is not None:
                    ago = timeframe_ago(
                        data=self.pair_data,
                        timeframe=self.p.pair_ibstf,
                        intraday_ago=intraday_ago,
                    )
                    val = self._calc_ibs(self.pair_data, ago=ago)
                else:
                    val = None
                record_continuous(
                    "pair_ibs",
                    key,
                    val,
                    data=self.pair_data,
                    timeframe=self.p.pair_ibstf,
                    pct_source=val,
                )
            elif key == "enablePairZ":
                v = timeframed_line_val(
                    self.pair_z,
                    data=self.pair_z_data,
                    timeframe=self.p.pair_z_tf,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "pair_z",
                    key,
                    val,
                    data=self.pair_z_data,
                    timeframe=self.p.pair_z_tf,
                    pct_source=val,
                )
            elif key in self.cross_zscore_meta:
                meta = self.cross_zscore_meta[key]
                line = meta.get("line")
                val = (
                    timeframed_line_val(
                        line,
                        data=meta.get("data"),
                        timeframe=meta.get("timeframe"),
                        intraday_ago=intraday_ago,
                    )
                    if line is not None
                    else None
                )
                numeric = coerce_float(val)
                record_param(key, numeric)
                symbol = meta.get("symbol")
                if symbol:
                    timeframe = meta.get("timeframe")
                    record_value(
                        _metadata_feature_key(symbol, timeframe, "z_score"),
                        numeric,
                    )
                    denom_line = meta.get("denom")
                    denom_val = None
                    if denom_line is not None:
                        raw_denom = timeframed_line_val(
                            denom_line,
                            data=meta.get("data"),
                            timeframe=meta.get("timeframe"),
                            intraday_ago=intraday_ago,
                        )
                        denom_val = coerce_float(raw_denom)
                    record_value(
                        _metadata_feature_key(symbol, timeframe, "z_pipeline"),
                        denom_val,
                    )
            elif key in self.return_meta:
                meta = self.return_meta[key]
                val = self._calc_return_value(meta)
                numeric = coerce_float(val)
                record_param(key, numeric)
                symbol = meta.get("symbol")
                if symbol:
                    timeframe = meta.get("timeframe")
                    record_value(
                        _metadata_feature_key(symbol, timeframe, "return"),
                        numeric,
                    )
                    pipeline_line = meta.get("line")
                    pipeline_val = None
                    if pipeline_line is not None:
                        raw_pipeline = timeframed_line_val(
                            pipeline_line,
                            data=meta.get("data"),
                            timeframe=meta.get("timeframe"),
                            intraday_ago=intraday_ago,
                        )
                        pipeline_val = coerce_float(raw_pipeline)
                    else:
                        pipeline_val = numeric
                    record_value(
                        _metadata_feature_key(symbol, timeframe, "return_pipeline"),
                        pipeline_val,
                    )
            elif key == "useValFilter":
                v = (
                    timeframed_line_val(
                        self.val_data.close,
                        data=self.val_data,
                        timeframe=self.p.val_tf,
                    intraday_ago=intraday_ago)
                    if self.val_data is not None
                    else None
                )
                val = float(v) if v is not None else None
                record_continuous(
                    "value",
                    key,
                    val,
                    data=self.val_data,
                    timeframe=self.p.val_tf,
                    align_to_date=_is_daily_timeframe(self.p.val_tf),
                    pct_source=val,
                )
            elif key == "enableVixReg":
                if self.has_vix and self.vix_data is not None:
                    vval = timeframed_line_val(
                        self.vix_data.close,
                        data=self.vix_data,
                        timeframe=self.p.vix_tf,
                    intraday_ago=intraday_ago)
                    if vval is not None and not math.isnan(vval):
                        record_param(key, 1 if vval < vix_med else 2)
                else:
                    record_param(key, 1)
            elif key == "enableRSIEntry":
                v = line_val(self.rsi, ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "rsi",
                    key,
                    val,
                    line=self.rsi,
                    intraday_offset=intraday_ago,
                    pct_source=val,
                )
            elif key == "enableRSIEntry2Len":
                v = line_val(self.rsi_len2, ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "rsi_len2",
                    key,
                    val,
                    line=self.rsi_len2,
                    intraday_offset=intraday_ago,
                    pct_source=val,
                )
            elif key == "enableRSIEntry14Len":
                v = line_val(self.rsi_len14, ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "enableRSIEntry2":
                v = line_val(self.rsi2, ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "rsi2",
                    key,
                    val,
                    line=self.rsi2,
                    intraday_offset=intraday_ago,
                    pct_source=val,
                )
            elif key == "enableDailyRSI":
                v = timeframed_line_val(
                    self.daily_rsi,
                    data=getattr(self, "daily_rsi_data", None),
                    timeframe=self.p.dailyRSITF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "daily_rsi",
                    key,
                    val,
                    data=getattr(self, "daily_rsi_data", None),
                    timeframe=self.p.dailyRSITF,
                    align_to_date=True,
                    pct_source=val,
                )
            elif key == "enableDailyRSI2Len":
                v = timeframed_line_val(
                    self.daily_rsi2,
                    data=getattr(self, "daily_rsi2_data", None),
                    timeframe=self.p.dailyRSI2TF,
                intraday_ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "enableDailyRSI14Len":
                v = timeframed_line_val(
                    self.daily_rsi14,
                    data=getattr(self, "daily_rsi14_data", None),
                    timeframe=self.p.dailyRSI14TF,
                intraday_ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "useSupplyZone":
                v = timeframed_line_val(
                    self.zone_atr,
                    data=self.supply_zone_data,
                    timeframe=self.p.supplyZoneTF,
                intraday_ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "enableATRZ":
                v = timeframed_line_val(
                    self.atr_z,
                    data=self.atr_z_data,
                    timeframe=self.p.atrTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "atrz",
                    key,
                    val,
                    data=self.atr_z_data,
                    timeframe=self.p.atrTF,
                    pct_source=val,
                )
            elif key == "enableVolZ":
                v = timeframed_line_val(
                    self.vol_z,
                    data=self.vol_z_data,
                    timeframe=self.p.volTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "volz",
                    key,
                    val,
                    data=self.vol_z_data,
                    timeframe=self.p.volTF,
                    pct_source=val,
                )
            elif key == "enableDATRZ":
                v = timeframed_line_val(
                    self.datr_z,
                    data=self.datr_z_data,
                    timeframe=self.p.dAtrTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "datrz",
                    key,
                    val,
                    data=self.datr_z_data,
                    timeframe=self.p.dAtrTF,
                    align_to_date=_is_daily_timeframe(self.p.dAtrTF),
                    pct_source=val,
                )
            elif key == "enableDVolZ":
                v = timeframed_line_val(
                    self.dvol_z,
                    data=self.dvol_z_data,
                    timeframe=self.p.dVolTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "dvolz",
                    key,
                    val,
                    data=self.dvol_z_data,
                    timeframe=self.p.dVolTF,
                    align_to_date=_is_daily_timeframe(self.p.dVolTF),
                    pct_source=val,
                )
            elif key == "enableZScore":
                v = timeframed_line_val(
                    self.price_z,
                    data=self.price_z_data,
                    timeframe=self.p.zScoreTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "z_score",
                    key,
                    val,
                    data=self.price_z_data,
                    timeframe=self.p.zScoreTF,
                    pct_source=val,
                )
            elif key == "enableDistZ":
                v = timeframed_line_val(
                    self.dist_z,
                    data=self.dist_z_data,
                    timeframe=self.p.distZTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "dist_z",
                    key,
                    val,
                    data=self.dist_z_data,
                    timeframe=self.p.distZTF,
                    pct_source=val,
                )
            elif key == "enableMom3":
                v = timeframed_line_val(
                    self.mom3_z,
                    data=self.mom3_z_data,
                    timeframe=self.p.mom3TF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "mom3_z",
                    key,
                    val,
                    data=self.mom3_z_data,
                    timeframe=self.p.mom3TF,
                    pct_source=val,
                )
            elif key == "enableTRATR":
                v = timeframed_line_val(
                    self.tratr_ratio,
                    data=self.tratr_data,
                    timeframe=self.p.tratrTF,
                intraday_ago=intraday_ago)
                val = float(v) if v is not None else None
                record_continuous(
                    "tratr",
                    key,
                    val,
                    data=self.tratr_data,
                    timeframe=self.p.tratrTF,
                    pct_source=val,
                )
            elif key == "enableBB":
                if self.bb is not None:
                    price = (
                        timeframed_line_val(
                            self.bb_data.close,
                            data=self.bb_data,
                            timeframe=self.p.bbTF,
                        intraday_ago=intraday_ago)
                        if self.bb_data is not None
                        else None
                    )
                    upper = timeframed_line_val(
                        self.bb.lines.top,
                        data=self.bb_data,
                        timeframe=self.p.bbTF,
                    intraday_ago=intraday_ago)
                    lower = timeframed_line_val(
                        self.bb.lines.bot,
                        data=self.bb_data,
                        timeframe=self.p.bbTF,
                    intraday_ago=intraday_ago)
                    if (
                        price is not None
                        and upper is not None
                        and lower is not None
                        and not math.isnan(upper)
                        and not math.isnan(lower)
                        and upper != lower
                    ):
                        bb_val = safe_div(price - lower, upper - lower)
                    else:
                        bb_val = None
                    record_continuous(
                        "bb",
                        key,
                        bb_val,
                        data=self.bb_data,
                        timeframe=self.p.bbTF,
                        pct_source=bb_val,
                    )
            elif key == "enableBBHigh":
                if self.bb_high is not None:
                    price = (
                        timeframed_line_val(
                            self.bb_high_data.close,
                            data=self.bb_high_data,
                            timeframe=self.p.bbHighTF,
                        intraday_ago=intraday_ago)
                        if self.bb_high_data is not None
                        else None
                    )
                    upper = timeframed_line_val(
                        self.bb_high.lines.top,
                        data=self.bb_high_data,
                        timeframe=self.p.bbHighTF,
                    intraday_ago=intraday_ago)
                    lower = timeframed_line_val(
                        self.bb_high.lines.bot,
                        data=self.bb_high_data,
                        timeframe=self.p.bbHighTF,
                    intraday_ago=intraday_ago)
                    if (
                        price is not None
                        and upper is not None
                        and lower is not None
                        and not math.isnan(price)
                        and not math.isnan(upper)
                        and not math.isnan(lower)
                    ):
                        record_param(key, safe_div(price - upper, upper - lower))
            elif key == "enableBBHighD":
                if self.bb_high_d is not None:
                    price = (
                        timeframed_line_val(
                            self.bb_high_d_data.close,
                            data=self.bb_high_d_data,
                            timeframe=self.p.bbHighDTF,
                        intraday_ago=intraday_ago)
                        if self.bb_high_d_data is not None
                        else None
                    )
                    upper = timeframed_line_val(
                        self.bb_high_d.lines.top,
                        data=self.bb_high_d_data,
                        timeframe=self.p.bbHighDTF,
                    intraday_ago=intraday_ago)
                    lower = timeframed_line_val(
                        self.bb_high_d.lines.bot,
                        data=self.bb_high_d_data,
                        timeframe=self.p.bbHighDTF,
                    intraday_ago=intraday_ago)
                    if (
                        price is not None
                        and upper is not None
                        and lower is not None
                        and not math.isnan(price)
                        and not math.isnan(upper)
                        and not math.isnan(lower)
                    ):
                        record_param(key, safe_div(price - upper, upper - lower))
            elif key == "enableDonch":
                if self.donch_high is not None and self.donch_data is not None:
                    price = timeframed_line_val(
                        self.donch_data.close,
                        data=self.donch_data,
                        timeframe=self.p.donchTF,
                    intraday_ago=intraday_ago)
                    high = timeframed_line_val(
                        self.donch_high,
                        data=self.donch_data,
                        timeframe=self.p.donchTF,
                        daily_ago=-1,
                        intraday_ago=-1,
                    )
                    low = timeframed_line_val(
                        self.donch_low,
                        data=self.donch_data,
                        timeframe=self.p.donchTF,
                        daily_ago=-1,
                        intraday_ago=-1,
                    )
                    if (
                        price is not None
                        and high is not None
                        and low is not None
                        and not math.isnan(high)
                        and not math.isnan(low)
                    ):
                        if price > high:
                            record_param(key, 1)
                        elif price < low:
                            record_param(key, 2)
                        else:
                            record_param(key, 3)
            elif key == "enableEMA8":
                if self.ema8 is not None and self.ema8_data is not None:
                    price = timeframed_line_val(
                        self.ema8_data.close,
                        data=self.ema8_data,
                        timeframe=self.p.ema8TF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema8,
                        data=self.ema8_data,
                        timeframe=self.p.ema8TF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA20":
                if self.ema20 is not None and self.ema20_data is not None:
                    price = timeframed_line_val(
                        self.ema20_data.close,
                        data=self.ema20_data,
                        timeframe=self.p.ema20TF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema20,
                        data=self.ema20_data,
                        timeframe=self.p.ema20TF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA50":
                if self.ema50 is not None and self.ema50_data is not None:
                    price = timeframed_line_val(
                        self.ema50_data.close,
                        data=self.ema50_data,
                        timeframe=self.p.ema50TF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema50,
                        data=self.ema50_data,
                        timeframe=self.p.ema50TF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA200":
                if self.ema200 is not None and self.ema200_data is not None:
                    price = timeframed_line_val(
                        self.ema200_data.close,
                        data=self.ema200_data,
                        timeframe=self.p.ema200TF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema200,
                        data=self.ema200_data,
                        timeframe=self.p.ema200TF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA20D":
                if self.ema20d is not None and self.ema20d_data is not None:
                    price = timeframed_line_val(
                        self.ema20d_data.close,
                        data=self.ema20d_data,
                        timeframe=self.p.ema20DTF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema20d,
                        data=self.ema20d_data,
                        timeframe=self.p.ema20DTF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA50D":
                if self.ema50d is not None and self.ema50d_data is not None:
                    price = timeframed_line_val(
                        self.ema50d_data.close,
                        data=self.ema50d_data,
                        timeframe=self.p.ema50DTF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema50d,
                        data=self.ema50d_data,
                        timeframe=self.p.ema50DTF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableEMA200D":
                if self.ema200d is not None and self.ema200d_data is not None:
                    price = timeframed_line_val(
                        self.ema200d_data.close,
                        data=self.ema200d_data,
                        timeframe=self.p.ema200DTF,
                    intraday_ago=intraday_ago)
                    ema = timeframed_line_val(
                        self.ema200d,
                        data=self.ema200d_data,
                        timeframe=self.p.ema200DTF,
                    intraday_ago=intraday_ago)
                    if price is not None and ema is not None and not math.isnan(ema):
                        record_param(key, 1 if price > ema else 2)
            elif key == "enableBearCount":
                data = self.bear_data
                if data is not None:
                    count = 0
                    data_len = len(data)
                    for i in range(1, data_len + 1):
                        c = line_val(data.close, ago=-i)
                        o = line_val(data.open, ago=-i)
                        if (
                            c is None
                            or o is None
                            or math.isnan(c)
                            or math.isnan(o)
                        ):
                            break
                        if c < o:
                            count += 1
                        else:
                            break
                    record_continuous(
                        "bear_count",
                        key,
                        count,
                        data=data,
                        timeframe=self.p.bearTF,
                        align_to_date=_is_daily_timeframe(self.p.bearTF),
                        pct_source=count,
                    )
            elif key == "enableBullishBarCount":
                data = self.bull_data
                if data is not None:
                    count = 0
                    data_len = len(data)
                    for i in range(1, data_len + 1):
                        c = line_val(data.close, ago=-i)
                        o = line_val(data.open, ago=-i)
                        if (
                            c is None
                            or o is None
                            or math.isnan(c)
                            or math.isnan(o)
                        ):
                            break
                        if c > o:
                            count += 1
                        else:
                            break
                    record_param(key, count)
            elif key == "enableSessions":
                t = dt.hour * 100 + dt.minute
                if t < 800:
                    record_param(key, 1)
                elif 800 <= t < 1500:
                    record_param(key, 2)
                elif 1500 <= t < 1700:
                    record_param(key, 4)
                elif 1700 <= t <= 2400:
                    record_param(key, 3)
                else:
                    record_param(key, 0)
            elif key == "enableOpenClose":
                t = dt.hour * 100 + dt.minute
                value = 0
                if 800 <= t < 900:
                    value = 1
                elif 1400 <= t < 1500:
                    value = 2
                record_param(key, value)
                record_value("open_close", value)
            elif key == "enableRangeCompressionATR":
                hi = line_val(self.hourly.high, ago=intraday_ago)
                lo = line_val(self.hourly.low, ago=intraday_ago)
                atr = line_val(self.rc_atr, ago=intraday_ago)
                if None not in (hi, lo, atr) and atr not in (0, None) and not math.isnan(atr):
                    record_param(key, safe_div(hi - lo, atr))
            elif key == "enableDailyRangeCompression":
                hi = line_val(self.daily.high, ago=-1)
                lo = line_val(self.daily.low, ago=-1)
                atr = line_val(self.drc_atr, ago=-1)
                if None not in (hi, lo, atr) and atr not in (0, None) and not math.isnan(atr):
                    record_param(key, safe_div(hi - lo, atr))
            elif key == "enable4BarRet":
                data = self.hourly
                if len(data) >= 5:
                    c0 = line_val(data.close, ago=intraday_ago)
                    c4 = line_val(data.close, ago=-4)
                    if None not in (c0, c4) and not math.isnan(c0) and not math.isnan(c4) and c4 != 0:
                        record_param(key, safe_div(c0 - c4, c4) * 100)
            elif key == "enable4BarRetD":
                data = self.daily
                if len(data) >= 5:
                    c0 = line_val(data.close, ago=-1)
                    c4 = line_val(data.close, ago=-5)
                    if None not in (c0, c4) and not math.isnan(c0) and not math.isnan(c4) and c4 != 0:
                        record_param(key, safe_div(c0 - c4, c4) * 100)
            elif key == "enableAboveOpen":
                price = line_val(self.hourly.close, ago=intraday_ago)
                prev = line_val(self.daily.close, ago=-1)
                if price is not None and prev is not None and not math.isnan(price) and not math.isnan(prev):
                    record_param(key, 1 if price > prev else 2)
            elif key == "enableDailyATRPercentile":
                pct = self.datr_pct_pct
                numeric = coerce_float(pct)
                record_param(key, numeric)
                record_value("daily_atr_percentile", numeric)
            elif key == "enableHourlyATRPercentile":
                pct = self.hatr_pct_pct
                numeric = coerce_float(pct)
                record_param(key, numeric)
                record_value("hourly_atr_percentile", numeric)
            elif key == "enableDirDrift":
                n = int(self.p.driftLen)
                data = self.hourly
                if len(data) > n:
                    c0 = line_val(data.close, ago=intraday_ago)
                    cn = line_val(data.close, ago=-n)
                    if None not in (c0, cn) and not math.isnan(c0) and not math.isnan(cn) and cn != 0:
                        ret = safe_div(c0 - cn, cn)
                        rets = []
                        valid = True
                        for i in range(n):
                            c1 = line_val(data.close, ago=-i)
                            c2 = line_val(data.close, ago=-i - 1)
                            if None in (c1, c2) or math.isnan(c1) or math.isnan(c2):
                                valid = False
                                break
                            rets.append(safe_div(c1 - c2, c2, zero=0))
                        if valid and len(rets) > 1:
                            sd = statistics.stdev(rets)
                            record_param(key, safe_div(ret, sd))
            elif key == "enableDirDriftD":
                n = int(self.p.dDriftLen)
                data = self.daily
                if len(data) > n + 1:
                    c0 = line_val(data.close, ago=-1)
                    cn = line_val(data.close, ago=-n - 1)
                    if None not in (c0, cn) and not math.isnan(c0) and not math.isnan(cn) and cn != 0:
                        ret = safe_div(c0 - cn, cn)
                        rets = []
                        valid = True
                        for i in range(n):
                            c1 = line_val(data.close, ago=-i - 1)
                            c2 = line_val(data.close, ago=-i - 2)
                            if None in (c1, c2) or math.isnan(c1) or math.isnan(c2):
                                valid = False
                                break
                            rets.append(safe_div(c1 - c2, c2, zero=0))
                        if valid and len(rets) > 1:
                            sd = statistics.stdev(rets)
                            record_param(key, safe_div(ret, sd))
            elif key == "enableSpiralER":
                val = self._compute_spiral_er()
                record_param(key, float(val) if val is not None else None)
            elif key == "enableTWRC":
                score = self._compute_twrc_score()
                record_param(key, float(score) if score is not None else None)
            elif key == "enableMASlope":
                shift = int(self.p.maSlopeShift)
                base_ago = timeframe_ago(data=self.hourly, intraday_ago=intraday_ago)
                ema = line_val(self.ma_slope_ema, ago=base_ago)
                ema_prev = line_val(self.ma_slope_ema, ago=base_ago - shift)
                atr = line_val(self.ma_slope_atr, ago=base_ago)
                if None not in (ema, ema_prev, atr) and atr not in (0, None) and not math.isnan(atr):
                    record_param(key, (ema - ema_prev) / atr)
            elif key == "enableDailySlope":
                shift = int(self.p.dSlopeShift)
                base_ago = timeframe_ago(data=self.daily, intraday_ago=intraday_ago)
                ema = line_val(self.d_slope_ema, ago=base_ago)
                ema_prev = line_val(self.d_slope_ema, ago=base_ago - shift)
                atr = line_val(self.d_slope_atr, ago=base_ago)
                if None not in (ema, ema_prev, atr) and atr not in (0, None) and not math.isnan(atr):
                    record_param(key, (ema - ema_prev) / atr)
            elif key == "enableMASpread":
                e1 = timeframed_line_val(
                    self.ma_spread_fast,
                    data=self.ma_spread_data,
                    timeframe=self.p.maSpreadTF,
                intraday_ago=intraday_ago)
                e2 = timeframed_line_val(
                    self.ma_spread_slow,
                    data=self.ma_spread_data,
                    timeframe=self.p.maSpreadTF,
                intraday_ago=intraday_ago)
                if e1 is not None and e2 is not None and not math.isnan(e1) and not math.isnan(e2):
                    diff = abs(e1 - e2)
                    if self.ma_spread_atr is not None:
                        atr = timeframed_line_val(
                            self.ma_spread_atr,
                            data=self.ma_spread_data,
                            timeframe=self.p.maSpreadTF,
                        intraday_ago=intraday_ago)
                        if atr is not None and atr != 0 and not math.isnan(atr):
                            diff = diff / atr
                    record_param(key, diff)
            elif key == "enableDonchProx":
                if self.donch_prox_high is not None and self.donch_prox_data is not None:
                    price = timeframed_line_val(
                        self.donch_prox_data.close,
                        data=self.donch_prox_data,
                        timeframe=self.p.donchProxTF,
                    intraday_ago=intraday_ago)
                    high = timeframed_line_val(
                        self.donch_prox_high,
                        data=self.donch_prox_data,
                        timeframe=self.p.donchProxTF,
                        daily_ago=-1,
                        intraday_ago=-1,
                    )
                    low = timeframed_line_val(
                        self.donch_prox_low,
                        data=self.donch_prox_data,
                        timeframe=self.p.donchProxTF,
                        daily_ago=-1,
                        intraday_ago=-1,
                    )
                    if None not in (price, high, low) and not math.isnan(high) and not math.isnan(low):
                        prox = min(abs(price - low), abs(high - price))
                        if self.donch_prox_atr is not None:
                            atr = timeframed_line_val(
                                self.donch_prox_atr,
                                data=self.donch_prox_data,
                                timeframe=self.p.donchProxTF,
                                daily_ago=-1,
                                intraday_ago=-1,
                            )
                            if atr is not None and atr != 0 and not math.isnan(atr):
                                prox = prox / atr
                        record_param(key, prox)
            elif key == "enableBBW":
                if self.bbw is not None:
                    up = timeframed_line_val(
                        self.bbw.lines.top,
                        data=self.bbw_data,
                        timeframe=self.p.bbwTF,
                    intraday_ago=intraday_ago)
                    bot = timeframed_line_val(
                        self.bbw.lines.bot,
                        data=self.bbw_data,
                        timeframe=self.p.bbwTF,
                    intraday_ago=intraday_ago)
                    mid = timeframed_line_val(
                        self.bbw.lines.mid,
                        data=self.bbw_data,
                        timeframe=self.p.bbwTF,
                    intraday_ago=intraday_ago)
                    if None not in (up, bot, mid) and mid not in (0, None) and not math.isnan(mid):
                        record_param(key, safe_div(up - bot, mid))
            elif key == "enableADX":
                v = line_val(self.adx, ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "enableDADX":
                v = line_val(self.dadx, ago=intraday_ago)
                record_param(key, float(v) if v is not None else None)
            elif key == "enablePSAR":
                price = (
                    timeframed_line_val(
                        self.psar_data.close,
                        data=self.psar_data,
                        timeframe=self.p.psarTF,
                    intraday_ago=intraday_ago)
                    if self.psar_data is not None
                    else None
                )
                ps = timeframed_line_val(
                    self.psar,
                    data=self.psar_data,
                    timeframe=self.p.psarTF,
                intraday_ago=intraday_ago)
                if price is not None and ps is not None and not math.isnan(ps):
                    dist = price - ps
                    if self.p.psarUseAbs:
                        dist = abs(dist)
                    record_param(key, dist)
            elif key == "enableN7Bar":
                data = self.n7_bar_data
                result = 3
                if data is not None and len(data) >= 7:
                    high0 = line_val(data.high, ago=intraday_ago)
                    low0 = line_val(data.low, ago=intraday_ago)
                    if (
                        high0 is not None
                        and low0 is not None
                        and not math.isnan(high0)
                        and not math.isnan(low0)
                    ):
                        rng0 = high0 - low0
                        valid = True
                        is_smallest = True
                        for i in range(1, 7):
                            prev_high = line_val(data.high, ago=-i)
                            prev_low = line_val(data.low, ago=-i)
                            if (
                                prev_high is None
                                or prev_low is None
                                or math.isnan(prev_high)
                                or math.isnan(prev_low)
                            ):
                                valid = False
                                break
                            prev_range = prev_high - prev_low
                            if rng0 >= prev_range:
                                is_smallest = False
                                break
                        if valid:
                            result = 1 if is_smallest else 2
                    record_param(key, result)
                else:
                    record_param(key, result)
            elif key == "enableInsideBar":
                data = self.inside_bar_data
                if data is not None and len(data) >= 3:
                    prev_high = line_val(data.high, ago=-1)
                    prev2_high = line_val(data.high, ago=-2)
                    prev_low = line_val(data.low, ago=-1)
                    prev2_low = line_val(data.low, ago=-2)
                    if all(
                        v is not None and not math.isnan(v)
                        for v in (prev_high, prev2_high, prev_low, prev2_low)
                    ):
                        inside = prev_high < prev2_high and prev_low > prev2_low
                        record_param(key, 1 if inside else 2)
                    else:
                        record_param(key, 3)
                else:
                    record_param(key, 3)
            else:
                record_param(key, None)
        for param_key, meta in self.cross_zscore_meta.items():
            existing = values.get(param_key)
            if isinstance(existing, (int, float)) and not math.isnan(existing):
                continue
            line = meta.get("line")
            val = (
                timeframed_line_val(
                    line,
                    data=meta.get("data"),
                    timeframe=meta.get("timeframe"),
                intraday_ago=intraday_ago)
                if line is not None
                else None
            )
            numeric = coerce_float(val)
            record_param(param_key, numeric)
            symbol = meta.get("symbol")
            if symbol:
                timeframe = meta.get("timeframe")
                record_value(
                    _metadata_feature_key(symbol, timeframe, "z_score"),
                    numeric,
                )
                denom_line = meta.get("denom")
                denom_val = None
                if denom_line is not None:
                    raw_denom = timeframed_line_val(
                        denom_line,
                        data=meta.get("data"),
                        timeframe=meta.get("timeframe"),
                        intraday_ago=intraday_ago,
                    )
                    denom_val = coerce_float(raw_denom)
                record_value(
                    _metadata_feature_key(symbol, timeframe, "z_pipeline"),
                    denom_val,
                )

        for param_key, meta in self.return_meta.items():
            existing = values.get(param_key)
            if isinstance(existing, (int, float)) and not math.isnan(existing):
                continue
            val = self._calc_return_value(meta)
            numeric = coerce_float(val)
            record_param(param_key, numeric)
            symbol = meta.get("symbol")
            if symbol:
                timeframe = meta.get("timeframe")
                record_value(
                    _metadata_feature_key(symbol, timeframe, "return"),
                    numeric,
                )
                pipeline_line = meta.get("line")
                pipeline_val = None
                if pipeline_line is not None:
                    raw_pipeline = timeframed_line_val(
                        pipeline_line,
                        data=meta.get("data"),
                        timeframe=meta.get("timeframe"),
                        intraday_ago=intraday_ago,
                    )
                    pipeline_val = coerce_float(raw_pipeline)
                else:
                    pipeline_val = numeric
                record_value(
                    _metadata_feature_key(symbol, timeframe, "return_pipeline"),
                    pipeline_val,
                )

        self.ensure_filter_keys(values)
        return values

    def ensure_filter_keys(self, record: dict) -> None:
        """Ensure all dynamic filter keys are present with numeric values.

        Any missing or non-numeric entries are replaced with ``0`` so downstream
        processing (e.g., CSV export) has a consistent schema.
        """

        alias_keys = {
            alias
            for column_key, alias in FRIENDLY_FILTER_NAMES.items()
            if column_key in self.filter_column_keys
        }
        full_keys = self.filter_column_keys | alias_keys | {"vix_med", "fractalPivot"}
        fallbacks = {"enableInsideBar": 3, "enableN7Bar": 3}
        for param_key in self.cross_zscore_meta:
            fallbacks.setdefault(param_key, 0.0)
        for param_key in self.return_meta:
            fallbacks.setdefault(param_key, 0.0)
        for key in full_keys:
            if isinstance(key, str) and key.endswith("_pct"):
                record.setdefault(key, None)
                continue
            val = record.get(key)
            if not isinstance(val, (int, float)) or (
                isinstance(val, float) and math.isnan(val)
            ):
                param_key = self.column_to_param.get(key, key)
                record[key] = fallbacks.get(param_key, 0)

    def _ml_default_for_feature(self, key: str) -> float:
        if key in {"enableInsideBar", "enableN7Bar"}:
            return 3.0
        if key in self.cross_zscore_meta or key in self.return_meta:
            return 0.0
        return 0.0

    def _evaluate_ml_score(self) -> float | None:
        if self.ml_model is None or not self.ml_features:
            return None

        snapshot = self.collect_filter_values(intraday_ago=0)
        normalized_snapshot = {
            normalize_column_name(key): value for key, value in snapshot.items()
        }
        ordered: list[float] = []
        for feature in self.ml_features:
            value = normalized_snapshot.get(feature)
            if isinstance(value, float) and math.isnan(value):
                value = None
            if value is None:
                value = self._ml_default_for_feature(feature)
            try:
                ordered.append(float(value))
            except (TypeError, ValueError):
                ordered.append(float(self._ml_default_for_feature(feature)))

        ml_input: object = [ordered]
        if pd is not None:
            try:
                ml_input = pd.DataFrame([ordered], columns=self.ml_features)
            except Exception:  # pragma: no cover - fallback to list input
                logging.exception("Failed to build DataFrame for ML features")
                ml_input = [ordered]

        try:
            probabilities = self.ml_model.predict_proba(ml_input)
        except Exception:
            logging.exception("ml_model.predict_proba failed")
            return None

        try:
            row = probabilities[0]
        except (TypeError, IndexError):
            logging.error(
                "ml_model.predict_proba returned unexpected output: %r",
                probabilities,
            )
            return None

        try:
            probs = list(row)
        except TypeError:
            probs = [row]

        if not probs:
            return None

        classes = getattr(self.ml_model, "classes_", None)
        positive_index = len(probs) - 1
        if classes is not None:
            try:
                class_list = list(classes)
            except TypeError:
                class_list = None
            if class_list:
                if 1 in class_list:
                    positive_index = class_list.index(1)
                else:
                    positive_index = len(class_list) - 1

        if positive_index >= len(probs):
            positive_index = len(probs) - 1

        try:
            return float(probs[positive_index])
        except (TypeError, ValueError):
            logging.error(
                "Positive class probability was non-numeric: %r",
                probs[positive_index],
            )
            return None

    def _with_ml_score(self, snapshot: dict | None) -> dict:
        result = dict(snapshot) if snapshot else {}
        result["ml_score"] = self._ml_last_score
        return result

    def in_session(self, dt: datetime) -> bool:
        """Check whether ``dt`` falls within allowed session windows.

        Session window end times are treated as *exclusive*. For example,
        with ``start_time1="0000"`` and ``end_time1="1500"`` trading is
        allowed from 00:00 up to 14:59, but not at 15:00.
        """

        t = dt.hour * 100 + dt.minute
        win1 = self.p.use_window1 and self.win1_start <= t < self.win1_end
        win2 = self.p.use_window2 and self.win2_start <= t < self.win2_end
        return win1 or win2

    def entry_allowed(self, dt: datetime, ibs_val: float) -> bool:
        """Return ``True`` if all configured entry filters pass."""

        if (
            self.ml_model is not None
            and self.ml_threshold is not None
            and self.ml_features
        ):
            score = self._evaluate_ml_score()
            self._ml_last_score = score
            if score is None or score < self.ml_threshold:
                return False
        else:
            self._ml_last_score = None

        if not self.in_session(dt):
            return False
        if self.p.enable_dow and dt.isoweekday() not in self.allowed_dow:
            return False
        if self.p.enable_month and dt.month not in self.allowed_month:
            return False
        if self.p.enable_dom:
            if self.dom_threshold is None:
                return False
            if dt.day <= self.dom_threshold:
                return False
        if self.p.enable_beg_week and dt.weekday() > 1:
            return False
        if self.p.enable_even_odd:
            want_even = str(self.p.even_odd_sel).lower().startswith("even")
            if (dt.day % 2 == 0) != want_even:
                return False
        if self.p.enable_prev_day_pct:
            pct = self.prev_day_pct()
            if pct is None or not (self.p.prev_day_pct_low <= pct <= self.p.prev_day_pct_high):
                return False
        if self.p.enable_prev_bar_pct:
            pct = self.prev_bar_pct()
            if pct is None or not (
                self.p.prev_bar_pct_low <= pct <= self.p.prev_bar_pct_high
            ):
                return False
        if self.p.enable_daily_ibs:
            dval = self.daily_ibs()
            if dval is None or not (
                self.p.daily_ibs_low <= dval <= self.p.daily_ibs_high
            ):
                return False
        if self.p.enable_prev_ibs and self.prev_ibs_val is not None:
            if not (self.p.prev_ibs_low <= self.prev_ibs_val <= self.p.prev_ibs_high):
                return False
        if self.p.enable_prev_ibs_daily:
            dprev = (
                self.prev_daily_ibs_val
                if self.prev_daily_ibs_val is not None
                else self.prev_daily_ibs()
            )
            if dprev is None or math.isnan(dprev):
                return False
            if not (
                self.p.prev_ibs_daily_low
                <= dprev
                <= self.p.prev_ibs_daily_high
            ):
                return False
        if self.p.enable_pair_ibs and self.pair_data is not None:
            ago = timeframe_ago(
                data=self.pair_data,
                timeframe=self.p.pair_ibstf,
            )
            pval = self._calc_ibs(self.pair_data, ago=ago)
            if not (self.p.pair_ibs_low <= pval <= self.p.pair_ibs_high):
                return False
        if self.p.enable_pair_z:
            val = timeframed_line_val(
                self.pair_z,
                data=self.pair_z_data,
                timeframe=self.p.pair_z_tf,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.pair_z_low <= val <= self.p.pair_z_high):
                    return False
        for param_key, meta in self.cross_zscore_meta.items():
            if getattr(self.p, param_key):
                line = meta.get("line")
                if line is None:
                    return False
                val = timeframed_line_val(
                    line,
                    data=meta.get("data"),
                    timeframe=meta.get("timeframe"),
                )
                if val is None or math.isnan(val):
                    return False
                low = self._resolve_param_value(
                    meta.get("low_aliases", (meta["low_param"],))
                )
                high = self._resolve_param_value(
                    meta.get("high_aliases", (meta["high_param"],))
                )
                if low is not None and val < low:
                    return False
                if high is not None and val > high:
                    return False
        for param_key, meta in self.return_meta.items():
            if getattr(self.p, param_key):
                value = self._calc_return_value(meta)
                if value is None or math.isnan(value):
                    return False
                low_raw = self._resolve_param_value(
                    meta.get("low_aliases", (meta["low_param"],))
                )
                high_raw = self._resolve_param_value(
                    meta.get("high_aliases", (meta["high_param"],))
                )
                try:
                    low_val = float(low_raw) if low_raw is not None else None
                except Exception:
                    low_val = None
                try:
                    high_val = float(high_raw) if high_raw is not None else None
                except Exception:
                    high_val = None
                if low_val is not None and not math.isnan(low_val) and value < low_val:
                    return False
                if high_val is not None and not math.isnan(high_val) and value > high_val:
                    return False
        if self.p.use_val_filter:
            val_price = (
                timeframed_line_val(
                    self.val_data.close,
                    data=self.val_data,
                    timeframe=self.p.val_tf,
                )
                if self.val_data is not None
                else None
            )
            if val_price is not None and not math.isnan(val_price):
                if not (self.p.val_low <= val_price <= self.p.val_high):
                    return False
        if (
            self.p.enable_vix_reg
            and self.has_vix
            and self.vix_data is not None
            and self.vix_median is not None
        ):
            vval = timeframed_line_val(
                self.vix_data.close,
                data=self.vix_data,
                timeframe=self.p.vix_tf,
            )
            vmed = line_val(self.vix_median)
            if (
                vval is not None
                and vmed is not None
                and not math.isnan(vval)
                and not math.isnan(vmed)
            ):
                mode = str(self.p.vix_mode).lower()
                if mode.startswith("risk on"):
                    if vval >= vmed:
                        return False
                else:
                    if vval <= vmed:
                        return False
        if self.p.enable_rsi_entry:
            val = line_val(self.rsi)
            if val is not None and not math.isnan(val):
                if not (self.p.rsi_entry_low <= val <= self.p.rsi_entry_high):
                    return False
        if self.p.enable_rsi_entry2_len:
            val = line_val(self.rsi_len2)
            if val is not None and not math.isnan(val):
                if not (
                    self.p.rsi_entry2_low
                    <= val
                    <= self.p.rsi_entry2_high
                ):
                    return False
        if self.p.enable_rsi_entry14_len:
            val = line_val(self.rsi_len14)
            if val is not None and not math.isnan(val):
                if not (
                    self.p.rsi_entry14_low
                    <= val
                    <= self.p.rsi_entry14_high
                ):
                    return False
        if self.p.enable_rsi_entry2:
            val = line_val(self.rsi2)
            if val is not None and not math.isnan(val):
                if not (
                    self.p.rsi2_entry_low <= val <= self.p.rsi2_entry_high
                ):
                    return False
        if self.p.enableDailyRSI:
            val = timeframed_line_val(
                self.daily_rsi,
                data=getattr(self, "daily_rsi_data", None),
                timeframe=self.p.dailyRSITF,
            )
            if val is not None and not math.isnan(val):
                if not (
                    self.p.dailyRSILow <= val <= self.p.dailyRSIHigh
                ):
                    return False
        if self.p.enableDailyRSI2Len:
            val = timeframed_line_val(
                self.daily_rsi2,
                data=getattr(self, "daily_rsi2_data", None),
                timeframe=self.p.dailyRSI2TF,
            )
            if val is not None and not math.isnan(val):
                if not (
                    self.p.dailyRSI2Low
                    <= val
                    <= self.p.dailyRSI2High
                ):
                    return False
        if self.p.enableDailyRSI14Len:
            val = timeframed_line_val(
                self.daily_rsi14,
                data=getattr(self, "daily_rsi14_data", None),
                timeframe=self.p.dailyRSI14TF,
            )
            if val is not None and not math.isnan(val):
                if not (
                    self.p.dailyRSI14Low
                    <= val
                    <= self.p.dailyRSI14High
                ):
                    return False
        if self.p.enableBullishBarCount:
            data = self.bull_data
            data_len = len(data) if data is not None else 0
            if data_len <= self.p.bullMin:
                return False
            bull_count = 0
            i = 1
            while i <= data_len and data.close(-i) > data.open(-i):
                bull_count += 1
                i += 1
            if bull_count < self.p.bullMin:
                return False
        if self.p.enableSessions:
            t = dt.hour * 100 + dt.minute
            if t < 800:
                sess = 1
            elif 800 <= t < 1500:
                sess = 2
            elif 1500 <= t < 1700:
                sess = 4
            elif 1700 <= t <= 2400:
                sess = 3
            else:
                sess = 0
            if sess != self.p.sessionSel:
                return False
        if self.p.enableOpenClose:
            t = dt.hour * 100 + dt.minute
            if self.p.openCloseSel == 1:
                if not (800 <= t < 900):
                    return False
            elif self.p.openCloseSel == 2:
                if not (1400 <= t < 1500):
                    return False
            else:
                return False
        if self.p.enableRangeCompressionATR:
            hi = line_val(self.hourly.high)
            lo = line_val(self.hourly.low)
            atr = line_val(self.rc_atr)
            if None not in (hi, lo, atr) and atr not in (0, None) and not math.isnan(atr):
                ratio = (hi - lo) / atr
                if str(self.p.rcDir).lower().startswith("above"):
                    if ratio < self.p.rcThresh:
                        return False
                else:
                    if ratio > self.p.rcThresh:
                        return False
        if self.p.enableDailyRangeCompression:
            hi = line_val(self.daily.high, ago=-1)
            lo = line_val(self.daily.low, ago=-1)
            atr = line_val(self.drc_atr, ago=-1)
            if None not in (hi, lo, atr) and atr not in (0, None) and not math.isnan(atr):
                ratio = (hi - lo) / atr
                if str(self.p.drcDir).lower().startswith("above"):
                    if ratio < self.p.drcThresh:
                        return False
                else:
                    if ratio > self.p.drcThresh:
                        return False
        if self.p.enable4BarRet:
            data = self.hourly
            if len(data) >= 5:
                c0 = line_val(data.close)
                c4 = line_val(data.close, ago=-4)
                if None not in (c0, c4) and not math.isnan(c0) and not math.isnan(c4) and c4 != 0:
                    ret = safe_div(c0 - c4, c4) * 100
                    if not (self.p.ret4Low <= ret <= self.p.ret4High):
                        return False
        if self.p.enable4BarRetD:
            data = self.daily
            if len(data) >= 5:
                c0 = line_val(data.close, ago=-1)
                c4 = line_val(data.close, ago=-5)
                if None not in (c0, c4) and not math.isnan(c0) and not math.isnan(c4) and c4 != 0:
                    ret = safe_div(c0 - c4, c4) * 100
                    if not (self.p.dRet4Low <= ret <= self.p.dRet4High):
                        return False
        if self.p.enableAboveOpen:
            price = line_val(self.hourly.close)
            prev = line_val(self.daily.close, ago=-1)
            if price is not None and prev is not None and not math.isnan(price) and not math.isnan(prev):
                if str(self.p.aboSel).lower().startswith("above"):
                    if price <= prev:
                        return False
                else:
                    if price >= prev:
                        return False
        if self.p.enableDailyATRPercentile:
            pct = self.datr_pct_pct
            if pct is not None and not math.isnan(pct):
                if not (self.p.dAtrPercLow <= pct <= self.p.dAtrPercHigh):
                    return False
        if self.p.enableHourlyATRPercentile:
            pct = self.hatr_pct_pct
            if pct is not None and not math.isnan(pct):
                if not (self.p.hAtrPercLow <= pct <= self.p.hAtrPercHigh):
                    return False
        if self.p.enableDirDrift:
            n = int(self.p.driftLen)
            data = self.hourly
            if len(data) > n:
                c0 = line_val(data.close)
                cn = line_val(data.close, ago=-n)
                if None not in (c0, cn) and not math.isnan(c0) and not math.isnan(cn) and cn != 0:
                    ret = safe_div(c0 - cn, cn)
                    rets = []
                    valid = True
                    for i in range(n):
                        c1 = line_val(data.close, ago=-i)
                        c2 = line_val(data.close, ago=-i - 1)
                        if None in (c1, c2) or math.isnan(c1) or math.isnan(c2):
                            valid = False
                            break
                        rets.append(safe_div(c1 - c2, c2, zero=0))
                    if valid and len(rets) > 1:
                        sd = statistics.stdev(rets)
                        drift = safe_div(ret, sd)
                        if not (self.p.driftLow <= drift <= self.p.driftHigh):
                            return False
        if self.p.enableDirDriftD:
            n = int(self.p.dDriftLen)
            data = self.daily
            if len(data) > n + 1:
                c0 = line_val(data.close, ago=-1)
                cn = line_val(data.close, ago=-n - 1)
                if None not in (c0, cn) and not math.isnan(c0) and not math.isnan(cn) and cn != 0:
                    ret = safe_div(c0 - cn, cn)
                    rets = []
                    valid = True
                    for i in range(n):
                        c1 = line_val(data.close, ago=-i - 1)
                        c2 = line_val(data.close, ago=-i - 2)
                        if None in (c1, c2) or math.isnan(c1) or math.isnan(c2):
                            valid = False
                            break
                        rets.append(safe_div(c1 - c2, c2, zero=0))
                    if valid and len(rets) > 1:
                        sd = statistics.stdev(rets)
                        drift = safe_div(ret, sd)
                        if not (self.p.dDriftLow <= drift <= self.p.dDriftHigh):
                            return False
        if self.p.enableSpiralER:
            ser = self._compute_spiral_er()
            if ser is None or math.isnan(ser):
                return False
            if not (self.p.serLow <= ser <= self.p.serHigh):
                return False
        if self.p.enableTWRC:
            score = self._compute_twrc_score()
            if score is None or math.isnan(score):
                return False
            if score < self.p.twrcTrigger:
                return False
        if self.p.enableMASlope:
            shift = int(self.p.maSlopeShift)
            base_ago = timeframe_ago(data=self.hourly)
            ema = line_val(self.ma_slope_ema, ago=base_ago)
            ema_prev = line_val(self.ma_slope_ema, ago=base_ago - shift)
            atr = line_val(self.ma_slope_atr, ago=base_ago)
            if None not in (ema, ema_prev, atr) and atr not in (0, None) and not math.isnan(atr):
                val = (ema - ema_prev) / atr
                if not (self.p.maSlopeLow <= val <= self.p.maSlopeHigh):
                    return False
        if self.p.enableDailySlope:
            shift = int(self.p.dSlopeShift)
            base_ago = timeframe_ago(data=self.daily)
            ema = line_val(self.d_slope_ema, ago=base_ago)
            ema_prev = line_val(self.d_slope_ema, ago=base_ago - shift)
            atr = line_val(self.d_slope_atr, ago=base_ago)
            if None not in (ema, ema_prev, atr) and atr not in (0, None) and not math.isnan(atr):
                val = (ema - ema_prev) / atr
                if not (self.p.dSlopeLow <= val <= self.p.dSlopeHigh):
                    return False
        if self.p.enableMASpread:
            e1 = timeframed_line_val(
                self.ma_spread_fast,
                data=self.ma_spread_data,
                timeframe=self.p.maSpreadTF,
            )
            e2 = timeframed_line_val(
                self.ma_spread_slow,
                data=self.ma_spread_data,
                timeframe=self.p.maSpreadTF,
            )
            if e1 is not None and e2 is not None and not math.isnan(e1) and not math.isnan(e2):
                diff = abs(e1 - e2)
                if self.ma_spread_atr is not None:
                    atr = timeframed_line_val(
                        self.ma_spread_atr,
                        data=self.ma_spread_data,
                        timeframe=self.p.maSpreadTF,
                    )
                    if atr is not None and atr != 0 and not math.isnan(atr):
                        diff = diff / atr
                if not (self.p.maSpreadLow <= diff <= self.p.maSpreadHigh):
                    return False
        if self.p.enableDonchProx:
            if self.donch_prox_high is not None and self.donch_prox_data is not None:
                price = timeframed_line_val(
                    self.donch_prox_data.close,
                    data=self.donch_prox_data,
                    timeframe=self.p.donchProxTF,
                )
                high = timeframed_line_val(
                    self.donch_prox_high,
                    data=self.donch_prox_data,
                    timeframe=self.p.donchProxTF,
                    daily_ago=-1,
                    intraday_ago=-1,
                )
                low = timeframed_line_val(
                    self.donch_prox_low,
                    data=self.donch_prox_data,
                    timeframe=self.p.donchProxTF,
                    daily_ago=-1,
                    intraday_ago=-1,
                )
                if None not in (price, high, low) and not math.isnan(high) and not math.isnan(low):
                    prox = min(abs(price - low), abs(high - price))
                    if self.donch_prox_atr is not None:
                        atr = timeframed_line_val(
                            self.donch_prox_atr,
                            data=self.donch_prox_data,
                            timeframe=self.p.donchProxTF,
                            daily_ago=-1,
                            intraday_ago=-1,
                        )
                        if atr is not None and atr != 0 and not math.isnan(atr):
                            prox = prox / atr
                    if not (self.p.donchProxLow <= prox <= self.p.donchProxHigh):
                        return False
        if self.p.enableBBW and self.bbw is not None:
            up = timeframed_line_val(
                self.bbw.lines.top,
                data=self.bbw_data,
                timeframe=self.p.bbwTF,
            )
            bot = timeframed_line_val(
                self.bbw.lines.bot,
                data=self.bbw_data,
                timeframe=self.p.bbwTF,
            )
            mid = timeframed_line_val(
                self.bbw.lines.mid,
                data=self.bbw_data,
                timeframe=self.p.bbwTF,
            )
            if None not in (up, bot, mid) and mid not in (0, None) and not math.isnan(mid):
                bw = (up - bot) / mid
                if not (self.p.bbwLow <= bw <= self.p.bbwHigh):
                    return False
        if self.p.enableADX:
            val = line_val(self.adx)
            if val is not None and not math.isnan(val):
                if not (self.p.adxLow <= val <= self.p.adxHigh):
                    return False
        if self.p.enableDADX:
            val = line_val(self.dadx)
            if val is not None and not math.isnan(val):
                if not (self.p.dAdxLow <= val <= self.p.dAdxHigh):
                    return False
        if self.p.enablePSAR:
            price = (
                timeframed_line_val(
                    self.psar_data.close,
                    data=self.psar_data,
                    timeframe=self.p.psarTF,
                )
                if self.psar_data is not None
                else None
            )
            ps = timeframed_line_val(
                self.psar,
                data=self.psar_data,
                timeframe=self.p.psarTF,
            )
            if price is not None and ps is not None and not math.isnan(ps):
                dist = price - ps
                if self.p.psarUseAbs:
                    dist = abs(dist)
                if not (self.p.psarLow <= dist <= self.p.psarHigh):
                    return False
        if self.p.enableBearCount:
            data = self.bear_data
            data_len = len(data) if data is not None else 0
            if data_len <= self.p.bearMin:
                return False
            bear_count = 0
            i = 1
            while i <= data_len and data.close(-i) < data.open(-i):
                bear_count += 1
                i += 1
            if bear_count <= self.p.bearMin:
                return False
        if self.p.enableInsideBar:
            data = self.inside_bar_data
            data_len = len(data) if data is not None else 0
            if data is None or data_len < 3:
                return False
            if not (data.high(-1) < data.high(-2) and data.low(-1) > data.low(-2)):
                return False
        if self.p.enableN7Bar:
            data = self.n7_bar_data
            data_len = len(data) if data is not None else 0
            if data is None or data_len < 7:
                return False
            high0 = line_val(data.high)
            low0 = line_val(data.low)
            if high0 is None or low0 is None:
                return False
            rng0 = high0 - low0
            for i in range(1, 7):
                if rng0 >= data.high(-i) - data.low(-i):
                    return False
        if self.p.enable_ibs_entry:
            if not (self.p.ibs_entry_low <= ibs_val <= self.p.ibs_entry_high):
                return False
        setting = str(self.p.entry_setting).lower()
        if setting and setting != "none":
            if setting == "hh above":
                sig_high = (
                    line_val(self.signal_data.high)
                    if self.signal_data is not None
                    else None
                )
                cond = (
                    self.last_pivot_high is not None
                    and self.prev_pivot_high is not None
                    and self.last_pivot_high > self.prev_pivot_high
                    and sig_high is not None
                    and sig_high > self.last_pivot_high
                )
                if not cond:
                    return False
            elif setting == "inside range":
                sig_close = (
                    line_val(self.signal_data.close)
                    if self.signal_data is not None
                    else None
                )
                cond = (
                    self.last_pivot_high is not None
                    and self.last_pivot_low is not None
                    and sig_close is not None
                    and self.last_pivot_low <= sig_close <= self.last_pivot_high
                )
                if not cond:
                    return False
        if self.p.use_supply_zone:
            if setting in {"range", "any"}:
                return True
            pivot = self.last_pivot_high if setting == "hh above" else self.last_pivot_low
            atr = timeframed_line_val(
                self.zone_atr,
                data=self.supply_zone_data,
                timeframe=self.p.supplyZoneTF,
            )
            if pivot is None or atr is None or math.isnan(atr):
                return False
            low_band = pivot + self.p.zoneATRLowMult * atr
            high_band = pivot + self.p.zoneATRHighMult * atr
            price = (
                timeframed_line_val(
                    self.supply_zone_data.close,
                    data=self.supply_zone_data,
                    timeframe=self.p.supplyZoneTF,
                )
                if self.supply_zone_data is not None
                else None
            )
            if price is None or not (low_band <= price <= high_band):
                return False
        if self.p.enableATRZ:
            val = timeframed_line_val(
                self.atr_z,
                data=self.atr_z_data,
                timeframe=self.p.atrTF,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.atrLow <= val <= self.p.atrHigh):
                    return False
        if self.p.enableVolZ:
            val = timeframed_line_val(
                self.vol_z,
                data=self.vol_z_data,
                timeframe=self.p.volTF,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.volLow <= val <= self.p.volHigh):
                    return False
        if self.p.enableDATRZ:
            val = timeframed_line_val(
                self.datr_z,
                data=self.datr_z_data,
                timeframe=self.p.dAtrTF,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.dAtrLow <= val <= self.p.dAtrHigh):
                    return False
        if self.p.enableDVolZ:
            val = timeframed_line_val(
                self.dvol_z,
                data=self.dvol_z_data,
                timeframe=self.p.dVolTF,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.dVolLow <= val <= self.p.dVolHigh):
                    return False
        if self.p.enableZScore:
            val = timeframed_line_val(
                self.price_z,
                data=self.price_z_data,
                timeframe=self.p.zScoreTF,
            )
            if val is not None and not math.isnan(val):
                if not (self.p.zScoreLow <= val <= self.p.zScoreHigh):
                    return False
        if self.p.enableBB and self.bb is not None:
            data = self.bb_data
            price = (
                timeframed_line_val(
                    data.close,
                    data=data,
                    timeframe=self.p.bbTF,
                )
                if data is not None
                else None
            )
            upper = timeframed_line_val(
                self.bb.lines.top,
                data=data,
                timeframe=self.p.bbTF,
            )
            lower = timeframed_line_val(
                self.bb.lines.bot,
                data=data,
                timeframe=self.p.bbTF,
            )
            mid = timeframed_line_val(
                self.bb.lines.mid,
                data=data,
                timeframe=self.p.bbTF,
            )
            if None not in (price, upper, lower, mid):
                dir = str(self.p.bbDir).lower()
                if "above upper" in dir:
                    if price <= upper:
                        return False
                elif "below lower" in dir:
                    if price >= lower:
                        return False
                elif "above mid" in dir:
                    if price <= mid:
                        return False
                elif "below mid" in dir:
                    if price >= mid:
                        return False
                elif "inside" in dir:
                    if not (lower <= price <= upper):
                        return False
        if self.p.enable_bb_high and self.bb_high is not None:
            data = self.bb_high_data
            price = (
                timeframed_line_val(
                    data.close,
                    data=data,
                    timeframe=self.p.bbHighTF,
                )
                if data is not None
                else None
            )
            upper = timeframed_line_val(
                self.bb_high.lines.top,
                data=data,
                timeframe=self.p.bbHighTF,
            )
            if price is None or upper is None or math.isnan(price) or math.isnan(upper):
                return False
            if price <= upper:
                return False
        if self.p.enable_bb_high_d and self.bb_high_d is not None:
            data = self.bb_high_d_data
            price = (
                timeframed_line_val(
                    data.close,
                    data=data,
                    timeframe=self.p.bbHighDTF,
                )
                if data is not None
                else None
            )
            upper = timeframed_line_val(
                self.bb_high_d.lines.top,
                data=data,
                timeframe=self.p.bbHighDTF,
            )
            if price is None or upper is None or math.isnan(price) or math.isnan(upper):
                return False
            if price <= upper:
                return False
        if (
            self.p.enableDonch
            and self.donch_high is not None
            and self.donch_data is not None
        ):
            data = self.donch_data
            price = timeframed_line_val(
                data.close,
                data=data,
                timeframe=self.p.donchTF,
            )
            high = timeframed_line_val(
                self.donch_high,
                data=data,
                timeframe=self.p.donchTF,
                daily_ago=-1,
                intraday_ago=-1,
            )
            low = timeframed_line_val(
                self.donch_low,
                data=data,
                timeframe=self.p.donchTF,
                daily_ago=-1,
                intraday_ago=-1,
            )
            if price is not None and high is not None and low is not None:
                dir = str(self.p.donchDir).lower()
                if dir.startswith("above"):
                    if price <= high:
                        return False
                elif dir.startswith("below"):
                    if price >= low:
                        return False
                else:
                    if not (low <= price <= high):
                        return False
        if self.p.enableEMA8 and self.ema8 is not None and self.ema8_data is not None:
            price = timeframed_line_val(
                self.ema8_data.close,
                data=self.ema8_data,
                timeframe=self.p.ema8TF,
            )
            ema = timeframed_line_val(
                self.ema8,
                data=self.ema8_data,
                timeframe=self.p.ema8TF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema8Dir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA20 and self.ema20 is not None and self.ema20_data is not None:
            price = timeframed_line_val(
                self.ema20_data.close,
                data=self.ema20_data,
                timeframe=self.p.ema20TF,
            )
            ema = timeframed_line_val(
                self.ema20,
                data=self.ema20_data,
                timeframe=self.p.ema20TF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema20Dir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA50 and self.ema50 is not None and self.ema50_data is not None:
            price = timeframed_line_val(
                self.ema50_data.close,
                data=self.ema50_data,
                timeframe=self.p.ema50TF,
            )
            ema = timeframed_line_val(
                self.ema50,
                data=self.ema50_data,
                timeframe=self.p.ema50TF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema50Dir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA200 and self.ema200 is not None and self.ema200_data is not None:
            price = timeframed_line_val(
                self.ema200_data.close,
                data=self.ema200_data,
                timeframe=self.p.ema200TF,
            )
            ema = timeframed_line_val(
                self.ema200,
                data=self.ema200_data,
                timeframe=self.p.ema200TF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema200Dir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA20D and self.ema20d is not None and self.ema20d_data is not None:
            price = timeframed_line_val(
                self.ema20d_data.close,
                data=self.ema20d_data,
                timeframe=self.p.ema20DTF,
            )
            ema = timeframed_line_val(
                self.ema20d,
                data=self.ema20d_data,
                timeframe=self.p.ema20DTF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema20DDir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA50D and self.ema50d is not None and self.ema50d_data is not None:
            price = timeframed_line_val(
                self.ema50d_data.close,
                data=self.ema50d_data,
                timeframe=self.p.ema50DTF,
            )
            ema = timeframed_line_val(
                self.ema50d,
                data=self.ema50d_data,
                timeframe=self.p.ema50DTF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema50DDir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableEMA200D and self.ema200d is not None and self.ema200d_data is not None:
            price = timeframed_line_val(
                self.ema200d_data.close,
                data=self.ema200d_data,
                timeframe=self.p.ema200DTF,
            )
            ema = timeframed_line_val(
                self.ema200d,
                data=self.ema200d_data,
                timeframe=self.p.ema200DTF,
            )
            if price is not None and ema is not None:
                if math.isnan(ema):
                    return False
                dir = str(self.p.ema200DDir).lower()
                if dir.startswith("above"):
                    if price <= ema:
                        return False
                else:
                    if price >= ema:
                        return False
        if self.p.enableDistZ and self.dist_z is not None:
            val = timeframed_line_val(
                self.dist_z,
                data=self.dist_z_data,
                timeframe=self.p.distZTF,
            )
            if val is not None and not math.isnan(val):
                dir = str(self.p.distZDir).lower()
                thr = self.p.distZThresh
                if dir.startswith("above"):
                    if val <= thr:
                        return False
                else:
                    if val >= thr:
                        return False
        if self.p.enableMom3 and self.mom3_z is not None:
            val = timeframed_line_val(
                self.mom3_z,
                data=self.mom3_z_data,
                timeframe=self.p.mom3TF,
            )
            if val is not None and not math.isnan(val):
                dir = str(self.p.mom3Dir).lower()
                thr = self.p.mom3Thresh
                if dir.startswith("above"):
                    if val < thr:
                        return False
                else:
                    if val > thr:
                        return False
        if self.p.enableTRATR and self.tratr_ratio is not None:
            val = timeframed_line_val(
                self.tratr_ratio,
                data=self.tratr_data,
                timeframe=self.p.tratrTF,
            )
            if val is not None and not math.isnan(val):
                dir = str(self.p.tratrDir).lower()
                thr = self.p.tratrThresh
                if dir.startswith("above"):
                    if val < thr:
                        return False
                else:
                    if val > thr:
                        return False
        return True

    def next(self):
        self.update_pivots()
        # Refresh daily ATR percentile only once per completed daily bar
        try:
            cur_day = self.daily.datetime.date(0)
        except Exception:
            cur_day = None
        if cur_day is not None and cur_day != self.datr_pct_last_date:
            self.datr_pct_pct = self._update_datr_pct()
            self.datr_pct_last_date = cur_day
        cur_bar_num = line_val(self.hourly.datetime)
        if cur_bar_num is not None and cur_bar_num != self.hatr_pct_last_dt:
            self.hatr_pct_pct = self._update_hatr_pct()
            self.hatr_pct_last_dt = cur_bar_num
        try:
            current_data_dt = self.hourly.datetime.datetime(0)
        except Exception:
            current_data_dt = None

        if self.order is not None:
            created = self.order.info.get("created_dt") or self.order.info.get("created")
            if created is not None and current_data_dt is not None:
                if current_data_dt - created >= timedelta(hours=24):
                    # self.log(f"Canceling stale order at {current_data_dt}")
                    self.cancel(self.order)
                    self.order = None
                else:
                    return
            else:
                return

        dt = self.hourly.datetime.date()
        if dt < self.p.trade_start:
            self.prev_ibs_val = None
            self.prev_daily_ibs_val = None
            return
        dt = self.hourly.datetime.datetime()
        ibs_val = self.ibs()
        price = line_val(self.hourly.close)
        if ibs_val is not None:
            price_str = f"{price:.2f}" if price is not None else "n/a"
            logger.info(
                f"{self.p.symbol} | Bar {len(self.hourly)} | IBS: {ibs_val:.3f} | "
                f"Price: {price_str} | Time: {dt}"
            )
        if ibs_val is None:
            self.prev_ibs_val = None
            self.prev_daily_ibs_val = None
            return

        self.prev_daily_ibs_val = self.prev_daily_ibs()

        if (
            self.twrc_data is not None
            and (self.p.enableTWRC or "enableTWRC" in self.filter_keys)
        ):
            dt_num = timeframed_line_val(
                self.twrc_data.datetime,
                data=self.twrc_data,
                timeframe=self.p.twrcTF,
            )
            if dt_num is not None and dt_num != self.twrc_last_dt:
                self.twrc_last_dt = dt_num
                if self.twrc_fast is None or self.twrc_base is None:
                    self.twrc_streak = 0
                else:
                    fast = timeframed_line_val(
                        self.twrc_fast,
                        data=self.twrc_data,
                        timeframe=self.p.twrcTF,
                    )
                    base = timeframed_line_val(
                        self.twrc_base,
                        data=self.twrc_data,
                        timeframe=self.p.twrcTF,
                    )
                    ratio = None
                    if fast is not None and base is not None:
                        ratio = safe_div(fast, base, zero=None)
                    threshold = self.p.twrcThreshold
                    if threshold is None or threshold <= 0:
                        raise ValueError("twrcThreshold must be > 0")
                    if ratio is None or math.isnan(ratio):
                        self.twrc_streak = 0
                    elif ratio < threshold:
                        self.twrc_streak += 1
                    else:
                        self.twrc_streak = 0

        for name, denom in [
            ("pair_z", self.pair_z_denom),
            ("atr_z", self.atr_z_denom),
            ("vol_z", self.vol_z_denom),
            ("dist_z", self.dist_z_denom),
            ("mom3_z", self.mom3_z_denom),
            ("price_z", self.price_z_denom),
            ("datr_z", self.datr_z_denom),
            ("dvol_z", self.dvol_z_denom),
            ("tratr_ratio", self.tratr_denom),
        ]:
            if denom is not None:
                val = line_val(denom)
                if val is not None and (val == 0 or math.isnan(val)):
                    logging.debug("%s denominator was %s", name, val)
        exit_ibs = (
            self.p.enable_ibs_exit
            and ibs_val is not None
            and self.p.ibs_exit_low <= ibs_val <= self.p.ibs_exit_high
        )
        signal = None
        exit_signal = None

        if self.can_execute:
            if not self.getposition(self.hourly):
                if ibs_val is not None and self.entry_allowed(dt, ibs_val):
                    signal = "IBS entry"
                    price0 = line_val(self.hourly.close)
                    price0_str = f"{price0:.2f}" if price0 is not None else "n/a"
                    logger.info(
                        f"📊 {self.p.symbol} ENTRY SIGNAL | IBS: {ibs_val:.3f} | "
                        f"Price: {price0_str} | Time: {dt}"
                    )
                    if price0 is not None:
                        filter_snapshot = self._with_ml_score(
                            self.collect_filter_values(intraday_ago=0)
                        )
                        ml_score = filter_snapshot.get("ml_score")
                        ml_passed = filter_snapshot.get("ml_passed", False)
                        ml_score_str = (
                            f"{ml_score:.3f}"
                            if isinstance(ml_score, (int, float)) and not math.isnan(ml_score)
                            else "n/a"
                        )
                        logger.info(
                            f"🤖 {self.p.symbol} ML FILTER | Score: {ml_score_str} | "
                            f"Passed: {ml_passed} | Threshold: {self.p.ml_threshold}"
                        )
                        order = self.buy(
                            data=self.hourly,
                            size=self.p.size,
                            exectype=bt.Order.Limit,
                            price=price0,
                        )
                        order.addinfo(
                            ibs=ibs_val,
                            created=dt,
                            created_dt=dt,
                            filter_snapshot=filter_snapshot,
                        )
                        self.order = order
            else:
                price = line_val(self.hourly.close)
                if price is None:
                    return
                entry_price = self.getposition(self.hourly).price
                size = self.getposition(self.hourly).size

                if self.p.enable_stop:
                    if str(self.p.stop_type).lower().startswith("percent"):
                        stop_price = entry_price * (
                            1 - safe_div(self.p.stop_perc, 100.0) if size > 0 else 1 + safe_div(self.p.stop_perc, 100.0)
                        )
                    else:
                        atr = line_val(self.stop_atr) if self.stop_atr is not None else None
                        atr = atr if atr is not None else 0
                        stop_price = (
                            entry_price - self.p.stop_atr_mult * atr
                            if size > 0
                            else entry_price + self.p.stop_atr_mult * atr
                        )
                    if (size > 0 and price <= stop_price) or (size < 0 and price >= stop_price):
                        exit_signal = "stop loss"

                if exit_signal is None and self.p.enable_tp:
                    if str(self.p.tp_type).lower().startswith("percent"):
                        tp_price = entry_price * (
                            1 + safe_div(self.p.tp_perc, 100.0) if size > 0 else 1 - safe_div(self.p.tp_perc, 100.0)
                        )
                    else:
                        atr = line_val(self.tp_atr) if self.tp_atr is not None else None
                        atr = atr if atr is not None else 0
                        tp_price = (
                            entry_price + self.p.tp_atr_mult * atr
                            if size > 0
                            else entry_price - self.p.tp_atr_mult * atr
                        )
                    if (size > 0 and price >= tp_price) or (size < 0 and price <= tp_price):
                        exit_signal = "take profit"

                if exit_signal is None and exit_ibs:
                    exit_signal = "IBS exit"

                if (
                    exit_signal is None
                    and self.p.enable_bar_stop
                    and self.bar_executed is not None
                    and len(self.hourly) >= self.bar_executed + self.bar_stop_bars
                ):
                    exit_signal = "bar stop"

                if exit_signal is None and self.p.enable_auto_close:
                    if (
                        dt.hour == self.close_h
                        and dt.minute == self.close_m
                        and self.last_auto_close_date != dt.date()
                    ):
                        exit_signal = "auto close"
                        self.last_auto_close_date = dt.date()

                if exit_signal:
                    logger.info(
                        f"🚪 {self.p.symbol} EXIT SIGNAL | Reason: {exit_signal} | "
                        f"IBS: {ibs_val:.3f} | Price: {price:.2f}"
                    )
                    filter_snapshot = self._with_ml_score(
                        self.collect_filter_values(intraday_ago=0)
                    )
                    order = self.close(data=self.hourly)
                    order.addinfo(
                        ibs=ibs_val,
                        created=dt,
                        created_dt=dt,
                        exit_reason=exit_signal,
                        filter_snapshot=filter_snapshot,
                    )
                    self.order = order

        if signal or exit_signal:
            self.current_signal = signal or exit_signal

        # Update previous IBS for next bar checks
        self.prev_ibs_val = ibs_val

    def notify_order(self, order):
        if order.status in [bt.Order.Completed, bt.Order.Canceled, bt.Order.Rejected]:
            if order.status == bt.Order.Completed:
                action = "BUY" if order.isbuy() else "SELL"
                ibs_val = order.info.get("ibs", self.ibs())
                ibs_str = (
                    f"{ibs_val:.3f}"
                    if isinstance(ibs_val, (int, float)) and not math.isnan(ibs_val)
                    else "n/a"
                )
                logger.info(
                    f"✅ {self.p.symbol} {action} FILLED | "
                    f"Size: {order.executed.size} | Price: {order.executed.price:.2f} | "
                    f"IBS: {ibs_str}"
                )
                sma200_ready = self.sma200 is not None and len(self.sma200) > 0
                if sma200_ready:
                    dclose = timeframed_line_val(
                        self.daily.close,
                        data=self.daily,
                        timeframe=_extract_timeframe(self.daily),
                        daily_ago=-1,
                    )
                    s200 = timeframed_line_val(
                        self.sma200,
                        data=self.daily,
                        timeframe=_extract_timeframe(self.sma200),
                        daily_ago=-1,
                    )
                    sma200 = (
                        "above"
                        if dclose is not None and s200 is not None and dclose > s200
                        else "below"
                    )
                else:
                    sma200 = "unavailable"

                tlt_ready = self.tlt_sma20 is not None and len(self.tlt_sma20) > 0
                if tlt_ready:
                    tclose = timeframed_line_val(
                        self.tlt.close,
                        data=self.tlt,
                        timeframe=_extract_timeframe(self.tlt),
                        daily_ago=-1,
                    )
                    tsma = timeframed_line_val(
                        self.tlt_sma20,
                        data=self.tlt,
                        timeframe=_extract_timeframe(self.tlt_sma20),
                        daily_ago=-1,
                    )
                    tlt_sma20 = (
                        "above"
                        if tclose is not None and tsma is not None and tclose > tsma
                        else "below"
                    )
                else:
                    tlt_sma20 = "unavailable"
                ibs_val = order.info.get("ibs", self.ibs())
                if order.isbuy():
                    entry = {
                        "dt": bt.num2date(order.executed.dt),
                        "instrument": order.data._name.split("_")[0].upper(),
                        "signal": self.current_signal,
                        "price": order.executed.price,
                        "size": order.executed.size,
                        "ibs_value": ibs_val,
                        "sma200": sma200,
                        "tlt_sma20": tlt_sma20,
                    }
                    snapshot = order.info.get("filter_snapshot")
                    if snapshot is None:
                        snapshot = self.collect_filter_values(intraday_ago=-1)
                    entry.update(self._with_ml_score(snapshot))
                    self.trades_log.append(entry)
                    self.bar_executed = len(self.hourly)
                    self.current_signal = None
                else:
                    self.pending_exit = {
                        "price": order.executed.price,
                        "size": order.executed.size,
                        "ibs_value": ibs_val,
                        "sma200": sma200,
                        "tlt_sma20": tlt_sma20,
                        "exit_reason": order.info.get("exit_reason", self.current_signal),
                        "filter_snapshot": self._with_ml_score(
                            order.info.get("filter_snapshot")
                        ),
                    }
            else:
                logger.info(f"ℹ️ {self.p.symbol} ORDER {order.getstatusname()}")
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pv = point_value(self.p.symbol)
            spec = CONTRACT_SPECS[self.p.symbol]
            slippage_usd = spec["tick_value"] * abs(trade.size)
            commission_usd = 2 * COMMISSION_PER_SIDE * abs(trade.size)

            broker_pnl = getattr(trade, "pnlcomm", None)
            if broker_pnl is None:
                pnl_usd = trade.pnl * pv - commission_usd
            else:
                pnl_usd = broker_pnl
            sma200_ready = self.sma200 is not None and len(self.sma200) > 0
            if sma200_ready:
                dclose = timeframed_line_val(
                    self.daily.close,
                    data=self.daily,
                    timeframe=_extract_timeframe(self.daily),
                    daily_ago=-1,
                )
                s200 = timeframed_line_val(
                    self.sma200,
                    data=self.daily,
                    timeframe=_extract_timeframe(self.sma200),
                    daily_ago=-1,
                )
                sma200 = (
                    "above" if dclose is not None and s200 is not None and dclose > s200 else "below"
                )
            else:
                sma200 = "unavailable"
            tlt_ready = self.tlt_sma20 is not None and len(self.tlt_sma20) > 0
            if tlt_ready:
                tclose = timeframed_line_val(
                    self.tlt.close,
                    data=self.tlt,
                    timeframe=_extract_timeframe(self.tlt),
                    daily_ago=-1,
                )
                tsma = timeframed_line_val(
                    self.tlt_sma20,
                    data=self.tlt,
                    timeframe=_extract_timeframe(self.tlt_sma20),
                    daily_ago=-1,
                )
                tlt_sma20 = (
                    "above" if tclose is not None and tsma is not None and tclose > tsma else "below"
                )
            else:
                tlt_sma20 = "unavailable"
            ibs_val = self.pending_exit["ibs_value"] if self.pending_exit else self.ibs()
            price = (
                self.pending_exit["price"]
                if self.pending_exit
                else line_val(self.hourly.close)
            )
            size = (
                self.pending_exit["size"]
                if self.pending_exit
                else -self.p.size
            )
            exit_reason = (
                self.pending_exit.get("exit_reason", "exit")
                if self.pending_exit
                else "exit"
            )
            exit_entry = {
                "dt": bt.num2date(trade.dtclose),
                "instrument": trade.data._name.split("_")[0].upper(),
                "signal": exit_reason,
                "price": price,
                "size": size,
                "ibs_value": ibs_val,
                "sma200": sma200,
                "tlt_sma20": tlt_sma20,
            }
            # Round-trip slippage assumes 0.5 tick per fill → 1 tick per contract
            exit_entry["slippage_usd"] = slippage_usd
            exit_entry["commission_usd"] = commission_usd
            exit_entry["pnl"] = pnl_usd
            filter_snapshot: dict | None = None
            if self.pending_exit:
                snapshot = self.pending_exit.get("filter_snapshot")
                if isinstance(snapshot, dict):
                    filter_snapshot = dict(snapshot)
            if filter_snapshot is not None:
                filter_snapshot = self._with_ml_score(filter_snapshot)
            else:
                filter_snapshot = self._with_ml_score(
                    self.collect_filter_values(intraday_ago=-1)
                )
            exit_entry.update(filter_snapshot)
            self.pending_exit = None
            self.bar_executed = None
            self.trades_log.append(exit_entry)

    def trade_report(self):
        """Return recorded trades.

        Each entry contains the order timestamp, instrument, signal label,
        size, price, and exit metrics populated when the trade closes.
        """
        return self.trades_log

