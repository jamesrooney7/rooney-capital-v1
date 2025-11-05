from .safe_div import safe_div

CONTRACT_SPECS = {
    "ES": {"tick_size": 0.25, "tick_value": 12.50},
    "NQ": {"tick_size": 0.25, "tick_value": 5.00},
    "RTY": {"tick_size": 0.10, "tick_value": 5.00},
    "YM": {"tick_size": 1.00, "tick_value": 5.00},
    "GC": {"tick_size": 0.10, "tick_value": 10.00},
    "SI": {"tick_size": 0.005, "tick_value": 25.00},
    "PL": {"tick_size": 0.10, "tick_value": 5.00},
    "HG": {"tick_size": 0.0005, "tick_value": 12.50},
    "CL": {"tick_size": 0.01, "tick_value": 10.00},
    "NG": {"tick_size": 0.001, "tick_value": 10.00},
    "6A": {"tick_size": 0.0001, "tick_value": 10.00},
    "6B": {"tick_size": 0.0001, "tick_value": 6.25},
    "6C": {"tick_size": 0.0001, "tick_value": 5.00},
    "6E": {"tick_size": 0.00005, "tick_value": 6.25},
    "6J": {"tick_size": 0.0000005, "tick_value": 6.25},
    "6M": {"tick_size": 0.00005, "tick_value": 5.00},
    "6N": {"tick_size": 0.0001, "tick_value": 10.00},
    "6S": {"tick_size": 0.0001, "tick_value": 12.50},
    "MBT": {"tick_size": 5.00, "tick_value": 0.50},
    "TLT": {"tick_size": 0.01, "tick_value": 0.01},
    "VIX": {"tick_size": 0.05, "tick_value": 50.00},
}


def point_value(symbol: str) -> float:
    """Return the dollar value per one point move for a futures contract."""
    spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 1, "tick_value": 1})
    return safe_div(spec["tick_value"], spec["tick_size"])
