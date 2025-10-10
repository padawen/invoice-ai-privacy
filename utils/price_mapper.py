"""
VAT-aware price mapper for invoice line items
Handles ambiguous OCR numbers and validates price relationships
"""
import re
import logging
import itertools
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Common Hungarian VAT rates
HUNGARIAN_VAT_RATES = [0.0, 0.05, 0.18, 0.27]

class PriceMapper:
    """Maps ambiguous price tokens to (unit_price, net, gross) triples using VAT validation"""

    def __init__(self, vat_tolerance: float = 0.5):
        """
        Args:
            vat_tolerance: Tolerance in currency units for VAT calculation (default 0.5 HUF)
        """
        self.vat_tolerance = vat_tolerance

    def extract_price_candidates(self, text: str) -> Dict[str, Any]:
        """
        Extract price tokens and VAT% from a table row

        Args:
            text: OCR text from a table row

        Returns:
            Dict with 'prices', 'vat_percent', and 'ambiguous' candidates
        """
        import re

        # Extract VAT percentage FIRST (before extracting prices)
        vat_percent = None
        vat_values_to_exclude = set()
        vat_matches = re.findall(r'(\d+)[,.]?(\d*)\s*(?:\d+)?\s*%', text)
        if vat_matches:
            integer_part, decimal_part = vat_matches[0]
            try:
                vat_str = integer_part
                if decimal_part:
                    vat_str += '.' + decimal_part
                vat_percent = float(vat_str) / 100.0
                vat_values_to_exclude.add(float(vat_str))
            except ValueError:
                pass

        # Extract all price-like tokens (number with , or .)
        price_pattern = r'-?\d+[.,]\d+'
        raw_prices = re.findall(price_pattern, text)

        # Normalize to floats and exclude VAT% values
        prices = []
        for p in raw_prices:
            try:
                normalized = float(p.replace(',', '.'))
                # Skip if this is the VAT percentage (e.g., don't include 27.0 from "27%")
                if normalized not in vat_values_to_exclude:
                    prices.append(normalized)
            except ValueError:
                continue

        # Detect ambiguous 3-digit prices that might have lost leading digit
        ambiguous_candidates = {}
        if len(prices) > 0:
            max_price = max(abs(p) for p in prices)
            for i, price in enumerate(prices):
                # If price is exactly 3 digits with decimals and there's a larger price
                if 100 <= abs(price) < 1000 and max_price >= 900:
                    # Generate candidate with leading "1"
                    sign = -1 if price < 0 else 1
                    candidate = sign * (1000 + abs(price))
                    ambiguous_candidates[i] = [price, candidate]
                    logger.debug(f"Ambiguous price detected: {price} → candidates {ambiguous_candidates[i]}")

        return {
            'prices': prices,
            'vat_percent': vat_percent,
            'ambiguous': ambiguous_candidates,
            'raw_text': text
        }

    def score_triple(self, unit: float, net: float, gross: float,
                     vat_percent: Optional[float], quantity: float = 1.0) -> float:
        """
        Score a (unit, net, gross) triple based on validity constraints

        Lower score is better. Returns float('inf') for invalid triples.
        """
        # Basic validity checks
        if unit < 0 or net < 0 or gross < 0:
            # Allow negative for discounts, but all must be negative
            if not (unit <= 0 and net <= 0 and gross <= 0):
                return float('inf')

        # Monotonicity check (with tolerance for rounding)
        tolerance = 0.01
        positives = [v for v in (unit, net, gross) if v > tolerance]
        negatives = [v for v in (unit, net, gross) if v < -tolerance]

        if positives and negatives:
            return float('inf')

        if not negatives:  # all values are zero or positive
            if unit > net + tolerance or net > gross + tolerance:
                return float('inf')
        else:  # all values are zero or negative
            if unit < net - tolerance or net < gross - tolerance:
                return float('inf')

        score = 0.0

        # Rule 1: If quantity == 1, strongly prefer unit == net
        if abs(quantity - 1.0) < 0.01:
            if abs(unit - net) < 0.01:
                score -= 100  # Large bonus
            else:
                score += abs(unit - net) * 10  # Penalty for mismatch

        # Rule 2: VAT validation
        if vat_percent is not None:
            expected_gross = round(net * (1 + vat_percent), 2)
            vat_error = abs(expected_gross - gross)

            if vat_error <= self.vat_tolerance:
                score -= 50  # Bonus for matching VAT
            else:
                score += vat_error * 20  # Penalty proportional to VAT mismatch
        else:
            # No VAT provided - try to infer from common rates
            best_vat_fit = float('inf')
            for vat_rate in HUNGARIAN_VAT_RATES:
                expected_gross = round(net * (1 + vat_rate), 2)
                vat_error = abs(expected_gross - gross)
                if vat_error < best_vat_fit:
                    best_vat_fit = vat_error

            if best_vat_fit <= self.vat_tolerance:
                score -= 30  # Bonus for matching common VAT
            else:
                score += best_vat_fit * 10

        # Rule 3: Prefer tighter gaps (unit ~ net ~ gross)
        gap_penalty = (net - unit) + (gross - net)
        score += gap_penalty * 0.1

        return score

    def find_best_triple(self, candidates: Dict[str, Any], quantity: float = 1.0) -> Optional[Dict[str, float]]:
        """
        Find best (unit_price, net, gross) triple from price candidates

        Args:
            candidates: Result from extract_price_candidates
            quantity: Item quantity (default 1.0)

        Returns:
            Dict with 'unit_price', 'net', 'gross', 'score', 'vat_used' or None
        """
        prices = candidates['prices']
        vat_percent = candidates['vat_percent']
        ambiguous = candidates['ambiguous']

        if len(prices) < 3:
            logger.warning(f"Too few prices ({len(prices)}) to map triple")
            return None

        # Generate all candidate price sets (considering ambiguous alternatives)
        price_sets = [prices.copy()]
        for idx, alternates in ambiguous.items():
            new_sets = []
            for price_set in price_sets:
                for alt_price in alternates:
                    new_set = price_set.copy()
                    new_set[idx] = alt_price
                    new_sets.append(new_set)
            price_sets = new_sets

        # Try all feasible triples from each price set
        best_triple = None
        best_score = float('inf')

        for price_set in price_sets:
            if len(price_set) == 0:
                continue

            n = len(price_set)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        values = [price_set[i], price_set[j], price_set[k]]

                        for ordering in set(itertools.permutations(values, 3)):
                            unit, net, gross = ordering
                            score = self.score_triple(unit, net, gross, vat_percent, quantity)

                            if score < best_score:
                                best_score = score
                                best_triple = {
                                    'unit_price': unit,
                                    'net': net,
                                    'gross': gross,
                                    'score': score,
                                    'vat_used': vat_percent
                                }

        if best_triple and best_score < float('inf'):
            logger.debug(f"Best triple: unit={best_triple['unit_price']}, net={best_triple['net']}, "
                        f"gross={best_triple['gross']}, score={best_score:.2f}")
            return best_triple

        logger.warning("No valid triple found")
        return None

    def validate_and_fix_item(self, item: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """
        Validate and potentially fix an invoice line item using VAT-aware mapping

        Args:
            item: Dict with 'name', 'quantity', 'unit_price', 'net', 'gross', 'currency'
            raw_text: Original OCR text for this row (optional, for re-extraction)

        Returns:
            Fixed item dict with 'warnings' field added
        """
        try:
            quantity = float(item.get('quantity', '1').replace(',', '.')) if item.get('quantity') else 1.0
            unit_price = float(item.get('unit_price', '0').replace(',', '.')) if item.get('unit_price') else 0.0
            net = float(item.get('net', '0').replace(',', '.')) if item.get('net') else 0.0
            gross = float(item.get('gross', '0').replace(',', '.')) if item.get('gross') else 0.0
        except (ValueError, AttributeError):
            logger.warning(f"Invalid numeric values in item: {item.get('name', '')}")
            item['warnings'] = ['invalid_numeric_values']
            return item

        warnings = []

        # Check if current values are valid
        current_score = self.score_triple(unit_price, net, gross, None, quantity)
        original_repeated = len({round(unit_price, 2), round(net, 2), round(gross, 2)}) <= 1

        # Calculate VAT error with current values
        current_vat_error = float('inf')
        if net > 0 and gross > 0:
            for vat_rate in HUNGARIAN_VAT_RATES:
                expected_gross = round(net * (1 + vat_rate), 2)
                vat_error = abs(expected_gross - gross)
                if vat_error < current_vat_error:
                    current_vat_error = vat_error

        # CONSERVATIVE REMAPPING POLICY:
        # Only attempt remapping if:
        # 1. All values are identical (clearly wrong), OR
        # 2. Current VAT calculation is significantly wrong (>5% of gross value)

        needs_remapping = False
        if original_repeated:
            # All values identical - definitely needs fixing
            needs_remapping = True
            logger.debug(f"Item '{item.get('name', '')[:30]}' has repeated values, attempting remap")
        elif current_vat_error > max(5.0, abs(gross) * 0.05):
            # VAT error is >5% of gross value AND >5 currency units
            needs_remapping = True
            logger.debug(f"Item '{item.get('name', '')[:30]}' has large VAT error ({current_vat_error:.2f}), attempting remap")

        # If we have raw text and remapping is needed, try to find a better mapping
        if raw_text and needs_remapping:
            candidates = self.extract_price_candidates(raw_text)

            if len(candidates['prices']) >= 3:
                best_triple = self.find_best_triple(candidates, quantity)
            else:
                best_triple = None


            if best_triple:
                new_unit = float(best_triple['unit_price'])
                new_net = float(best_triple['net'])
                new_gross = float(best_triple['gross'])
                new_score = best_triple['score']

                # Calculate VAT error with new values
                new_vat_error = float('inf')
                if new_net > 0 and new_gross > 0:
                    for vat_rate in HUNGARIAN_VAT_RATES:
                        expected_gross = round(new_net * (1 + vat_rate), 2)
                        vat_error = abs(expected_gross - new_gross)
                        if vat_error < new_vat_error:
                            new_vat_error = vat_error

                vat_used = best_triple.get('vat_used')
                vat_consistent = False
                if vat_used is not None:
                    expected_gross = round(new_net * (1 + vat_used), 2)
                    vat_consistent = abs(expected_gross - new_gross) <= self.vat_tolerance

                # Check if new values would create duplicate issue
                candidate_repeated = len({round(new_unit, 2), round(new_net, 2), round(new_gross, 2)}) <= 1

                # STRICTER CONDITIONS for remapping:
                # 1. New VAT error must be significantly better (at least 50% improvement)
                # 2. New values must not be duplicates
                # 3. New values must be from the OCR row
                # 4. Either improved score OR significantly better VAT consistency

                source_prices = candidates.get('prices', []) if isinstance(candidates, dict) else []
                all_from_row = all(
                    any(abs(value - price) <= 0.51 for price in source_prices)
                    for value in (new_unit, new_net, new_gross)
                )

                vat_improvement = current_vat_error - new_vat_error
                score_improvement = current_score - new_score

                should_replace = False

                # Case 1: Original values were all identical - accept if new values are different and valid
                if original_repeated and not candidate_repeated and all_from_row and vat_consistent:
                    should_replace = True
                    logger.debug(f"Replacing repeated values with differentiated values")

                # Case 2: VAT error improved significantly (>50% reduction AND absolute improvement >2.0)
                elif vat_improvement > max(2.0, current_vat_error * 0.5) and not candidate_repeated:
                    should_replace = True
                    logger.debug(f"VAT error improved from {current_vat_error:.2f} to {new_vat_error:.2f}")

                # Case 3: Score improved dramatically (>20 point improvement) and VAT is at least as good
                elif score_improvement > 20 and new_vat_error <= current_vat_error and not candidate_repeated:
                    should_replace = True
                    logger.debug(f"Score improved from {current_score:.2f} to {new_score:.2f}")

                if should_replace:
                    logger.info(f"Remapping prices for '{item.get('name', '')[:30]}': "
                               f"old=({unit_price}, {net}, {gross}) VAT_err={current_vat_error:.2f} -> "
                               f"new=({new_unit}, {new_net}, {new_gross}) VAT_err={new_vat_error:.2f}")

                    item['unit_price'] = f"{new_unit:.2f}"
                    item['net'] = f"{new_net:.2f}"
                    item['gross'] = f"{new_gross:.2f}"
                    warnings.append('remapped_prices_by_vat_rule')
                else:
                    logger.debug(f"Skipping remap for '{item.get('name', '')[:30]}': "
                               f"improvements insufficient (VAT: {vat_improvement:.2f}, Score: {score_improvement:.2f})")
        # Sanity checks
        if gross < net and gross > 0 and net > 0:
            warnings.append('gross_less_than_net')

        if abs(quantity - 1.0) < 0.01 and abs(unit_price - net) > 0.01 and unit_price > 0:
            warnings.append('unit_price_ne_net_when_qty_1')

        if warnings:
            item['warnings'] = warnings

        return item

# Global mapper instance
price_mapper = PriceMapper()
