def calculate_cagr(start_value: float, end_value: float, years: int) -> float | None:
    """
    Standard compound annual growth rate.
    Returns None if inputs are invalid (zero, negative, null).
    Returns as a decimal e.g. 0.082 = 8.2%
    """
    if not start_value or not end_value or years <= 0:
        return None
    if start_value <= 0 or end_value <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def _to_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def calculate_cagr_hist(row: dict, from_year: int = 2019, to_year: int = 2025) -> float | None:
    """Historical CAGR from from_year to to_year. Default: 2019→2025 (6 years)."""
    years = to_year - from_year
    return calculate_cagr(_to_float(row.get(str(from_year))), _to_float(row.get(str(to_year))), years)


def calculate_cagr_fcast(row: dict, from_year: int = 2025, to_year: int = 2030) -> float | None:
    """Forecast CAGR between two years. Default: 2025→2030 (5 years)."""
    years = to_year - from_year
    return calculate_cagr(_to_float(row.get(str(from_year))), _to_float(row.get(str(to_year))), years)


def project_future_value(current_value: float, cagr: float, years: int) -> float:
    """Compound growth projection: current_value × (1 + cagr)^years"""
    return current_value * ((1 + cagr) ** years)