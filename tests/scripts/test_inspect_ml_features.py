from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli_module() -> object:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_ml_features.py"
    spec = importlib.util.spec_from_file_location("inspect_ml_features_cli", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to load inspect_ml_features module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_ml_features_from_heartbeat(tmp_path, capsys):
    heartbeat_payload = {
        "status": "running",
        "details": {
            "ml_features": {
                "ES": {
                    "ready": True,
                    "missing_features": [],
                    "features": {
                        "ema_21": 4300.12,
                        "rsi_14": 55.321,
                    },
                },
                "NQ": {
                    "ready": False,
                    "missing_features": ["volatility"],
                    "features": {
                        "volatility": None,
                        "momentum": 1.2345,
                    },
                },
            }
        },
    }
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(json.dumps(heartbeat_payload), encoding="utf-8")

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "contract_map: Data/Databento_contract_map.yml",
                "symbols: ['ES', 'NQ']",
                f"heartbeat_file: {heartbeat_path}",
                "databento_api_key: DUMMY",
            ]
        ),
        encoding="utf-8",
    )

    module = _load_cli_module()
    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--heartbeat",
            str(heartbeat_path),
            "--source",
            "heartbeat",
            "--show-features",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Symbol: ES" in captured.out
    assert "Ready: yes" in captured.out
    assert "Missing features: —" in captured.out
    assert "Symbol: NQ" in captured.out
    assert "Ready: no" in captured.out
    assert "Missing features: volatility" in captured.out
    assert "ema_21: 4300.12" in captured.out
    assert "rsi_14: 55.321" in captured.out
    assert "momentum: 1.2345" in captured.out
    assert "volatility: —" in captured.out
