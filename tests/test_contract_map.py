import json
from pathlib import Path

import pytest

from runner.contract_map import ContractMapError, load_contract_map


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def test_load_contract_map_parses_fixture():
    contract_map = load_contract_map(FIXTURE_ROOT / "contract_map_valid.yml")

    assert contract_map.symbols() == ("ES", "NQ")

    es_contract = contract_map.active_contract("es")
    subscription = es_contract.subscription()
    assert subscription.dataset == "GLBX.MDP3"
    assert subscription.stype_in == "product_id"
    assert subscription.codes == ("MES.FUT", "MES", "ES")

    product_map = contract_map.product_to_root(["ES"])
    assert product_map == {"MES.FUT": "ES", "MES": "ES", "ES": "ES"}

    dataset_groups = contract_map.dataset_groups()
    expected_codes = tuple(sorted(["MES.FUT", "MES", "ES", "MNQ.FUT", "MNQ", "NQ"]))
    assert dataset_groups == {("GLBX.MDP3", "product_id"): expected_codes}

    metadata = contract_map.traderspost_metadata("ES")
    assert metadata["tradovate_symbol"] == "MES"
    assert metadata["databento_dataset"] == "GLBX.MDP3"
    assert metadata["databento_feed_symbol"] == "MES"
    assert metadata["optimized"] is True

    reference = contract_map.reference_feed("tlt")
    assert reference is not None
    assert reference.dataset == "GLBX.MDP3"
    assert reference.feed_symbol == "ZB"
    assert reference.product_id == "ZB.FUT"


def test_contract_map_missing_dataset(tmp_path):
    payload = {
        "contracts": [
            {
                "symbol": "ES",
                "databento": {
                    "feed_symbol": "MES"
                }
            }
        ]
    }
    path = tmp_path / "missing_dataset.yml"
    path.write_text(json.dumps(payload))

    with pytest.raises(ContractMapError, match="missing Databento dataset"):
        load_contract_map(path)


def test_contract_map_invalid_roll_rules(tmp_path):
    payload = {
        "contracts": [
            {
                "symbol": "ES",
                "databento": {
                    "dataset": "GLBX.MDP3",
                    "feed_symbol": "MES"
                },
                "roll": ["invalid"]
            }
        ]
    }
    path = tmp_path / "invalid_roll.yml"
    path.write_text(json.dumps(payload))

    with pytest.raises(ContractMapError, match="roll rules must be a mapping"):
        load_contract_map(path)
