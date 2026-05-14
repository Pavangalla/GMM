def apply_entitlement(
    query_params: dict,
    subscription_tier: str,
    licensed_industries: list = None,
    licensed_geographies: list = None,
) -> dict:
    """
    Enforces subscription scope on every tool call.
    Raises PermissionError if request is outside licensed scope.
    Injects allowlists into query_params for SQL filtering.
    """
    if subscription_tier == "industry" and licensed_industries:
        requested = query_params.get("industry")
        if requested and requested not in licensed_industries:
            raise PermissionError(
                f"Your subscription does not include '{requested}'. "
                f"Licensed industries: {', '.join(licensed_industries)}."
            )
        if not requested:
            query_params["_industry_allowlist"] = licensed_industries

    if subscription_tier == "country" and licensed_geographies:
        requested = query_params.get("country")
        if requested and requested not in licensed_geographies:
            raise PermissionError(
                f"Your subscription does not include '{requested}'. "
                f"Licensed geographies: {', '.join(licensed_geographies)}."
            )
        if not requested:
            query_params["_country_allowlist"] = licensed_geographies

    return query_params