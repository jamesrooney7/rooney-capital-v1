from __future__ import annotations

from pathlib import Path

from runner.contract_map import load_contract_map


def test_contract_map_subscriptions_are_consistent() -> None:
    contract_map = load_contract_map(Path("Data/Databento_contract_map.yml"))

    # Smoke check that active contracts and reference feeds load correctly.
    assert set(contract_map.symbols()) >= {"ES", "NQ", "CL"}
    assert set(contract_map.reference_symbols()) >= {"TLT", "VIX"}

    es_root = contract_map.active_contract("ES")
    subscription = es_root.subscription()

    # Dataset and subscription codes should align with the metadata payload.
    assert subscription.dataset == "GLBX.MDP3"
    assert subscription.stype_in == es_root.roll.stype_in == "product_id"
    assert subscription.codes[0] == "MES.FUT"
    assert set(subscription.codes) == {"MES.FUT", "MES"}

    # TradersPost metadata should surface the tradovate and Databento details.
    metadata = es_root.traderspost_metadata()
    assert metadata["tradovate_symbol"] == "MES"
    assert metadata["tradovate_description"].startswith("Micro E-mini S&P")
    assert metadata["databento_dataset"] == "GLBX.MDP3"
    assert metadata["databento_feed_symbol"] == "MES"
    assert metadata["databento_product_id"] == "MES.FUT"
    assert metadata["optimized"] is True

    # Product codes must round-trip to the contract root across helpers.
    product_mapping = contract_map.product_to_root()
    for code in subscription.codes:
        assert product_mapping[code] == "ES"

    # Grouped datasets should key bars by dataset and stype_in.
    grouped = contract_map.dataset_groups(("ES", "NQ"))
    assert ("GLBX.MDP3", "product_id") in grouped
    assert set(grouped[("GLBX.MDP3", "product_id")]) >= {"MES.FUT", "MNQ.FUT"}

    # Reference feeds should also provide product-to-root lookups.
    reference_mapping = contract_map.reference_product_to_root()
    for symbol in contract_map.reference_symbols():
        feed = contract_map.reference_feed(symbol)
        assert feed is not None
        assert feed.dataset == "GLBX.MDP3"
        assert reference_mapping[feed.feed_symbol] == symbol
        if feed.product_id:
            assert reference_mapping[feed.product_id] == symbol
