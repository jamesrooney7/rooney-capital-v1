"""
Contract selector for choosing futures contracts based on open interest/volume.

This module queries Databento for contract statistics and selects the contract
with the highest open interest (OI) for each root symbol. This ensures we trade
the most liquid contract.

Usage:
    selector = ContractSelector(api_key="...")
    selected = selector.select_contracts(["ES", "NQ", "CL"])
    # Returns: {"ES": "ESH2026", "NQ": "NQH2026", "CL": "CLG2026"}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ContractSelector",
    "ContractSelection",
    "select_highest_oi_contract",
]


@dataclass
class ContractSelection:
    """Information about a selected contract."""
    root_symbol: str  # e.g., "ES"
    contract_symbol: str  # e.g., "ESH2026"
    open_interest: int  # Open interest
    volume: int  # Trading volume
    expiration: Optional[str] = None  # Expiration date if available
    selected_at: Optional[datetime] = None


class ContractSelector:
    """
    Selects futures contracts based on open interest from Databento.

    The selector queries Databento's definitions endpoint to find all
    active contracts for a root symbol, then selects the one with the
    highest open interest.
    """

    def __init__(
        self,
        api_key: str,
        dataset: str = "GLBX.MDP3",
        cache_ttl_hours: int = 4,
    ):
        """
        Initialize the contract selector.

        Args:
            api_key: Databento API key
            dataset: Databento dataset (default: GLBX.MDP3 for CME)
            cache_ttl_hours: How long to cache selections
        """
        self.api_key = api_key
        self.dataset = dataset
        self.cache_ttl_hours = cache_ttl_hours

        # Cache of selections: {root_symbol: (selection, timestamp)}
        self._cache: Dict[str, Tuple[ContractSelection, datetime]] = {}

        # Current selections for quick lookup
        self._selections: Dict[str, ContractSelection] = {}

    def select_contracts(
        self,
        root_symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, ContractSelection]:
        """
        Select the highest OI contract for each root symbol.

        Args:
            root_symbols: List of root symbols (e.g., ["ES", "NQ", "CL"])
            force_refresh: Force refresh even if cached

        Returns:
            Dict mapping root symbol to ContractSelection
        """
        results: Dict[str, ContractSelection] = {}

        for root in root_symbols:
            root_upper = root.strip().upper()

            # Check cache first
            if not force_refresh and root_upper in self._cache:
                selection, cached_at = self._cache[root_upper]
                if datetime.now() - cached_at < timedelta(hours=self.cache_ttl_hours):
                    results[root_upper] = selection
                    continue

            # Query Databento for contract info
            try:
                selection = self._select_contract_for_root(root_upper)
                if selection:
                    self._cache[root_upper] = (selection, datetime.now())
                    self._selections[root_upper] = selection
                    results[root_upper] = selection
                    logger.info(
                        "Selected contract for %s: %s (OI=%d, Vol=%d)",
                        root_upper,
                        selection.contract_symbol,
                        selection.open_interest,
                        selection.volume,
                    )
            except Exception as e:
                logger.error("Failed to select contract for %s: %s", root_upper, e)
                # Fall back to front month if we have a cached selection
                if root_upper in self._selections:
                    results[root_upper] = self._selections[root_upper]

        return results

    def get_contract_symbol(self, root_symbol: str) -> Optional[str]:
        """Get the selected contract symbol for a root."""
        root_upper = root_symbol.strip().upper()
        selection = self._selections.get(root_upper)
        return selection.contract_symbol if selection else None

    def get_selection(self, root_symbol: str) -> Optional[ContractSelection]:
        """Get the full selection info for a root."""
        return self._selections.get(root_symbol.strip().upper())

    def _select_contract_for_root(self, root_symbol: str) -> Optional[ContractSelection]:
        """
        Query Databento and select the highest OI contract.

        This uses the Databento Historical API to get instrument definitions
        and statistics for all active contracts of the root symbol.
        """
        try:
            import databento as db
        except ImportError:
            logger.warning("databento package not installed, using fallback contract selection")
            return self._fallback_selection(root_symbol)

        try:
            client = db.Historical(key=self.api_key)

            # Query definitions for the root symbol
            # This returns all active contracts for the root
            product_id = f"{root_symbol}.FUT"

            # Get instrument definitions to find all contracts
            definitions = client.metadata.get_dataset_range(dataset=self.dataset)

            # Query statistics (OI/volume) for each contract
            # Use timeseries.get_range with schema='statistics'
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            # Query OHLCV data with volume, and use OI from definitions
            # Note: Databento provides OI in the instrument definition response
            try:
                data = client.timeseries.get_range(
                    dataset=self.dataset,
                    symbols=[product_id],
                    stype_in="parent",
                    schema="definition",
                    start=start_date,
                    end=end_date,
                )

                # Parse the definitions to find contracts and their OI
                contracts: List[Tuple[str, int, int]] = []  # (symbol, oi, volume)

                for record in data:
                    if hasattr(record, 'raw_symbol') and hasattr(record, 'open_interest_qty'):
                        symbol = record.raw_symbol
                        oi = getattr(record, 'open_interest_qty', 0) or 0
                        vol = getattr(record, 'volume', 0) or 0
                        if oi > 0:  # Only consider contracts with OI
                            contracts.append((symbol, oi, vol))

                if not contracts:
                    logger.warning("No contracts with OI found for %s", root_symbol)
                    return self._fallback_selection(root_symbol)

                # Sort by OI descending and select the highest
                contracts.sort(key=lambda x: x[1], reverse=True)
                best_symbol, best_oi, best_vol = contracts[0]

                # Convert to full year format (ESH5 -> ESH2025)
                full_symbol = self._convert_to_full_year(best_symbol)

                return ContractSelection(
                    root_symbol=root_symbol,
                    contract_symbol=full_symbol,
                    open_interest=best_oi,
                    volume=best_vol,
                    selected_at=datetime.now(),
                )

            except Exception as e:
                logger.warning("Databento query failed for %s: %s", root_symbol, e)
                return self._fallback_selection(root_symbol)

        except Exception as e:
            logger.error("Databento client error for %s: %s", root_symbol, e)
            return self._fallback_selection(root_symbol)

    def _fallback_selection(self, root_symbol: str) -> ContractSelection:
        """
        Generate a fallback contract selection based on current date.

        Handles both quarterly (H, M, U, Z) and monthly contract cycles.
        For commodities, we try to select the contract with the most remaining
        liquidity - typically the front month until close to expiration.
        """
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day

        # Products with monthly contracts - these have specific delivery months
        # GC, SI, PL: Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)
        # CL, HG, NG: Every month
        monthly_all_months = {'CL', 'HG', 'NG'}
        monthly_even_months = {'GC', 'SI', 'PL'}  # Only even months

        # Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun
        #              N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
        month_codes = {
            1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
            7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
        }

        # Even month codes for GC, SI, PL
        even_months = {2: 'G', 4: 'J', 6: 'M', 8: 'Q', 10: 'V', 12: 'Z'}

        root_upper = root_symbol.upper()

        if root_upper in monthly_even_months:
            # Products with bi-monthly contracts (Feb, Apr, Jun, Aug, Oct, Dec)
            # Find the current or next valid month
            valid_months = [2, 4, 6, 8, 10, 12]

            # Find the current front month
            front_month = None
            front_year = year
            for m in valid_months:
                if m >= month:
                    front_month = m
                    break

            if front_month is None:
                # We're past December, roll to February next year
                front_month = 2
                front_year = year + 1

            # If we're past the 20th of the month before expiration, roll forward
            # E.g., if front_month=12 and we're in November 21+, roll to Feb
            if month == front_month - 1 and day > 20:
                # Roll to next contract
                idx = valid_months.index(front_month)
                if idx < len(valid_months) - 1:
                    front_month = valid_months[idx + 1]
                else:
                    front_month = 2
                    front_year = year + 1
            elif month == front_month and day > 1:
                # Already in expiration month, definitely roll
                idx = valid_months.index(front_month)
                if idx < len(valid_months) - 1:
                    front_month = valid_months[idx + 1]
                else:
                    front_month = 2
                    front_year = year + 1

            month_code = even_months[front_month]
            next_year = front_year

        elif root_upper in monthly_all_months:
            # Monthly contracts (CL, HG, NG) - every month
            # Roll around the 20th of the month before expiration
            if day > 20:
                # Roll to next month
                next_month = month + 1
                next_year = year
                if next_month > 12:
                    next_month = 1
                    next_year = year + 1
            else:
                # Use current month's contract (it's still the front month)
                next_month = month
                next_year = year
            month_code = month_codes[next_month]
        else:
            # Quarterly contracts (H=Mar, M=Jun, U=Sep, Z=Dec)
            quarterly_months = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

            # Find current quarter
            current_quarter = None
            for m in [3, 6, 9, 12]:
                if month <= m:
                    current_quarter = m
                    break
            if current_quarter is None:
                current_quarter = 12

            # If past mid-month of expiration month, roll to next quarter
            if month == current_quarter and day > 15:
                quarter_index = [3, 6, 9, 12].index(current_quarter)
                if quarter_index < 3:
                    next_quarter = [3, 6, 9, 12][quarter_index + 1]
                    next_year = year
                else:
                    next_quarter = 3
                    next_year = year + 1
            elif month > current_quarter:
                quarter_index = [3, 6, 9, 12].index(current_quarter)
                if quarter_index < 3:
                    next_quarter = [3, 6, 9, 12][quarter_index + 1]
                    next_year = year
                else:
                    next_quarter = 3
                    next_year = year + 1
            else:
                next_quarter = current_quarter
                next_year = year

            month_code = quarterly_months[next_quarter]

        contract_symbol = f"{root_symbol}{month_code}{next_year}"

        logger.info(
            "Using fallback contract selection for %s: %s",
            root_symbol, contract_symbol
        )

        return ContractSelection(
            root_symbol=root_symbol,
            contract_symbol=contract_symbol,
            open_interest=0,
            volume=0,
            selected_at=datetime.now(),
        )

    def _convert_to_full_year(self, symbol: str) -> str:
        """
        Convert short year format to full year.

        ESH5 -> ESH2025
        CLZ4 -> CLZ2024
        """
        if not symbol or len(symbol) < 3:
            return symbol

        # Check if last char is a digit (year)
        if not symbol[-1].isdigit():
            return symbol

        # Extract year digit
        year_digit = int(symbol[-1])

        # Determine century based on current year
        current_year = datetime.now().year
        current_decade = current_year // 10 % 10

        # If year digit is less than current decade + 5, assume current decade
        # Otherwise assume previous decade
        if year_digit >= current_decade:
            full_year = 2020 + year_digit
        else:
            full_year = 2030 + year_digit

        # Replace single digit with full year
        return symbol[:-1] + str(full_year)


def select_highest_oi_contract(
    api_key: str,
    root_symbols: List[str],
    dataset: str = "GLBX.MDP3",
) -> Dict[str, str]:
    """
    Convenience function to select highest OI contracts.

    Args:
        api_key: Databento API key
        root_symbols: List of root symbols
        dataset: Databento dataset

    Returns:
        Dict mapping root symbol to contract symbol
    """
    selector = ContractSelector(api_key=api_key, dataset=dataset)
    selections = selector.select_contracts(root_symbols)
    return {
        root: sel.contract_symbol
        for root, sel in selections.items()
    }
