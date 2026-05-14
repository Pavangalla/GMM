import sqlite3
import json
import numpy as np
from data_loader import get_conn, EMBEDDINGS_PATH
from entitlement import apply_entitlement
from cagr import calculate_cagr, project_future_value

# ── Semantic search state (lazy-loaded) ──────────────────────────────────────
_embed_model = None
_name_matrix = None   # embeddings of market name only
_def_matrix = None    # embeddings of name + definition
_embed_names = None


def _load_semantic_index():
    global _embed_model, _name_matrix, _def_matrix, _embed_names
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _name_matrix is None:
        data = np.load(EMBEDDINGS_PATH)
        _name_matrix = data["name_embeddings"].astype(np.float32)
        _def_matrix = data["def_embeddings"].astype(np.float32)
        _embed_names = data["names"].tolist()


def _string_overlap_score(query: str, market_name: str) -> float:
    """Token-level Jaccard overlap between query and market name."""
    q_tokens = set(query.lower().split())
    n_tokens = set(market_name.lower().split())
    if not q_tokens or not n_tokens:
        return 0.0
    return len(q_tokens & n_tokens) / len(q_tokens | n_tokens)


def _attach_cagr(conn, results: list[dict], country: str) -> list[dict]:
    """Joins cagr_cache onto a list of result dicts."""
    for r in results:
        row = conn.execute(
            "SELECT cagr_hist_5yr, cagr_fcast_5yr FROM cagr_cache WHERE market_name = ? AND country = ?",
            (r.get('market_name'), country)
        ).fetchone()
        if row:
            h, f = row['cagr_hist_5yr'], row['cagr_fcast_5yr']
            r['cagr_hist_5yr_pct']  = round(h * 100, 2) if h is not None else None
            r['cagr_fcast_5yr_pct'] = round(f * 100, 2) if f is not None else None
        else:
            r['cagr_hist_5yr_pct'] = None
            r['cagr_fcast_5yr_pct'] = None
    return results


def _available_geographies(conn, market_name: str, year: int) -> list[str]:
    """Return geographies that have data for this market+year."""
    rows = conn.execute(
        "SELECT DISTINCT country FROM size_data "
        "WHERE market_name = ? COLLATE NOCASE AND year = ? AND value IS NOT NULL ORDER BY country",
        (market_name, year)
    ).fetchall()
    return [r['country'] for r in rows]


# ── TOOL 1: search_markets ────────────────────────────────────────────────────

def search_markets(query: str, industry: str = None, level: int = None, limit: int = 20,
                   subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    params = apply_entitlement({"industry": industry}, subscription_tier,
                               licensed_industries, licensed_geographies)
    conn = get_conn()
    sql = """SELECT market_id, market_name, parent_market, level, level_num, industry, short_definition
             FROM taxonomy WHERE (market_name LIKE ? OR short_definition LIKE ? OR long_definition LIKE ?)"""
    args = [f"%{query}%"] * 3

    if industry:
        sql += " AND industry = ?"; args.append(industry)
    elif params.get("_industry_allowlist"):
        ph = ",".join("?" * len(params["_industry_allowlist"]))
        sql += f" AND industry IN ({ph})"; args.extend(params["_industry_allowlist"])
    if level:
        sql += " AND level_num = ?"; args.append(level)
    sql += f" LIMIT {int(limit)}"

    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── TOOL 2: get_market ────────────────────────────────────────────────────────

def get_market(market_name: str = None, market_id: str = None, country: str = "Global",
               year: int = 2025, subscription_tier="enterprise", licensed_industries=None,
               licensed_geographies=None):
    try:
        apply_entitlement({"country": country}, subscription_tier,
                          licensed_industries, licensed_geographies)
    except PermissionError as e:
        return {"error": str(e)}

    conn = get_conn()
    if market_id:
        row = conn.execute("SELECT * FROM taxonomy WHERE market_id = ?", (market_id,)).fetchone()
    else:
        row = conn.execute("SELECT * FROM taxonomy WHERE market_name = ? COLLATE NOCASE",
                           (market_name,)).fetchone()
        if not row:
            row = conn.execute("SELECT * FROM taxonomy WHERE market_name LIKE ? LIMIT 1",
                               (f"%{market_name}%",)).fetchone()
    if not row:
        conn.close()
        return {"error": f"Market not found: '{market_name or market_id}'"}

    result = dict(row)
    size_row = conn.execute(
        "SELECT value, units FROM size_data WHERE market_name = ? COLLATE NOCASE AND country = ? AND year = ?",
        (result['market_name'], country, year)
    ).fetchone()
    result['country'] = country
    result['year'] = year
    if size_row:
        result['value'] = size_row['value']
        result['units'] = size_row['units']
    else:
        result['value'] = None
        result['units'] = None
        result['_no_data_note'] = (
            f"No data for '{country}'. "
            f"Available geographies: {', '.join(_available_geographies(conn, result['market_name'], year))}"
        )

    result = _attach_cagr(conn, [result], country)[0]
    conn.close()
    return result


# ── TOOL 3: get_children ──────────────────────────────────────────────────────

def get_children(market_name: str = None, market_id: str = None, country: str = "Global",
                 year: int = 2025, include_size: bool = True,
                 subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    conn = get_conn()
    if market_id and not market_name:
        row = conn.execute("SELECT market_name FROM taxonomy WHERE market_id = ?", (market_id,)).fetchone()
        market_name = row['market_name'] if row else None

    children = conn.execute(
        "SELECT market_id, market_name, parent_market, level, level_num, industry, short_definition "
        "FROM taxonomy WHERE parent_market = ? COLLATE NOCASE", (market_name,)
    ).fetchall()

    results = [dict(c) for c in children]

    if include_size:
        for r in results:
            size_row = conn.execute(
                "SELECT value, units FROM size_data WHERE market_name = ? COLLATE NOCASE AND country = ? AND year = ?",
                (r['market_name'], country, year)
            ).fetchone()
            r['value'] = size_row['value'] if size_row else None
            r['units'] = size_row['units'] if size_row else None
            r['country'] = country
            r['year'] = year
        results = _attach_cagr(conn, results, country)

    conn.close()
    return results


# ── TOOL 4: get_parent ────────────────────────────────────────────────────────

def get_parent(market_name: str = None, market_id: str = None,
               country: str = "Global", year: int = 2025,
               subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    conn = get_conn()
    if market_id and not market_name:
        row = conn.execute("SELECT market_name FROM taxonomy WHERE market_id = ?", (market_id,)).fetchone()
        market_name = row['market_name'] if row else None

    child = conn.execute(
        "SELECT parent_market FROM taxonomy WHERE market_name = ? COLLATE NOCASE", (market_name,)
    ).fetchone()

    if not child or not child['parent_market']:
        conn.close()
        return {"error": f"'{market_name}' is a top-level market with no parent."}

    parent = conn.execute(
        "SELECT * FROM taxonomy WHERE market_name = ? COLLATE NOCASE", (child['parent_market'],)
    ).fetchone()

    if not parent:
        conn.close()
        return {"error": f"Parent '{child['parent_market']}' not found in taxonomy."}

    result = dict(parent)
    size_row = conn.execute(
        "SELECT value, units FROM size_data WHERE market_name = ? COLLATE NOCASE AND country = ? AND year = ?",
        (result['market_name'], country, year)
    ).fetchone()
    if size_row:
        result['value'] = size_row['value']
        result['units'] = size_row['units']
    result['country'] = country
    result['year'] = year
    result = _attach_cagr(conn, [result], country)[0]
    conn.close()
    return result


# ── TOOL 5: get_hierarchy_path ────────────────────────────────────────────────

def get_hierarchy_path(market_name: str = None, market_id: str = None,
                       subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    conn = get_conn()
    if market_id and not market_name:
        row = conn.execute("SELECT market_name FROM taxonomy WHERE market_id = ?", (market_id,)).fetchone()
        market_name = row['market_name'] if row else None

    path = []
    current = market_name
    while current:
        row = conn.execute(
            "SELECT market_id, market_name, parent_market, level, level_num, industry FROM taxonomy "
            "WHERE market_name = ? COLLATE NOCASE", (current,)
        ).fetchone()
        if not row:
            break
        path.insert(0, dict(row))
        current = row['parent_market']

    conn.close()
    return path


# ── TOOL 6: filter_markets ────────────────────────────────────────────────────

def filter_markets(industry: str = None, level: int = None, l1_market: str = None,
                   ancestor_market: str = None,
                   min_value: float = None, max_value: float = None,
                   min_cagr_hist_pct: float = None, max_cagr_hist_pct: float = None,
                   min_cagr_fcast_pct: float = None, max_cagr_fcast_pct: float = None,
                   country: str = "Global",
                   sort_by: str = "value_desc",
                   year: int = 2025, limit: int = 20,
                   subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):

    # country="*" is the sentinel for "rank all individual countries"
    all_countries = (country == "*")
    entitlement_country = None if all_countries else country
    params = apply_entitlement({"industry": industry, "country": entitlement_country},
                               subscription_tier, licensed_industries, licensed_geographies)
    country_allowlist = params.get("_country_allowlist")

    sort_map = {
        "value_desc":      "s.value DESC",
        "value_asc":       "s.value ASC",
        "cagr_hist_desc":  "c.cagr_hist_5yr DESC",
        "cagr_hist_asc":   "c.cagr_hist_5yr ASC",
        "cagr_fcast_desc": "c.cagr_fcast_5yr DESC",
        "cagr_fcast_asc":  "c.cagr_fcast_5yr ASC",
        "name_asc":        "t.market_name ASC",
    }
    order = sort_map.get(sort_by, "s.value DESC")

    conn = get_conn()

    # ── Ancestor-market path (recursive CTE across all descendants) ───────────
    if ancestor_market:
        cte = """
            WITH RECURSIVE sub(market_name) AS (
                SELECT market_name FROM taxonomy
                WHERE market_name = ? COLLATE NOCASE
                UNION ALL
                SELECT t.market_name FROM taxonomy t
                JOIN sub ON t.parent_market = sub.market_name COLLATE NOCASE
            )
        """
        cols = """
            SELECT t.market_id, t.market_name, t.parent_market, t.level, t.level_num,
                   t.industry, t.short_definition,
                   s.value, s.units, s.country, s.year,
                   ROUND(c.cagr_hist_5yr * 100, 2)  AS cagr_hist_5yr_pct,
                   ROUND(c.cagr_fcast_5yr * 100, 2) AS cagr_fcast_5yr_pct
            FROM taxonomy t
            JOIN sub ON t.market_name = sub.market_name COLLATE NOCASE
        """

        if not all_countries:
            sql = cte + cols + """
                LEFT JOIN size_data s ON t.market_name = s.market_name COLLATE NOCASE
                                      AND s.country = ? AND s.year = ?
                LEFT JOIN cagr_cache c ON t.market_name = c.market_name COLLATE NOCASE
                                       AND c.country = ?
                WHERE t.market_name != ? COLLATE NOCASE
            """
            args = [ancestor_market, country, year, country, ancestor_market]
        else:
            # Return every individual country — no Global aggregate
            sql = cte + cols + """
                JOIN size_data s ON t.market_name = s.market_name COLLATE NOCASE
                                 AND s.year = ?
                LEFT JOIN cagr_cache c ON t.market_name = c.market_name COLLATE NOCASE
                                       AND c.country = s.country
                WHERE t.market_name != ? COLLATE NOCASE
                  AND s.country != 'Global'
            """
            args = [ancestor_market, year, ancestor_market]
            if country_allowlist:
                ph = ",".join("?" * len(country_allowlist))
                sql += f" AND s.country IN ({ph})"
                args.extend(country_allowlist)

        if level:
            sql += " AND t.level_num = ?"; args.append(level)
        if min_value is not None:
            sql += " AND s.value >= ?"; args.append(min_value)
        if max_value is not None:
            sql += " AND s.value <= ?"; args.append(max_value)
        if min_cagr_hist_pct is not None:
            sql += " AND c.cagr_hist_5yr >= ?"; args.append(min_cagr_hist_pct / 100)
        if max_cagr_hist_pct is not None:
            sql += " AND c.cagr_hist_5yr <= ?"; args.append(max_cagr_hist_pct / 100)
        if min_cagr_fcast_pct is not None:
            sql += " AND c.cagr_fcast_5yr >= ?"; args.append(min_cagr_fcast_pct / 100)
        if max_cagr_fcast_pct is not None:
            sql += " AND c.cagr_fcast_5yr <= ?"; args.append(max_cagr_fcast_pct / 100)

        sql += f" ORDER BY {order} LIMIT ?"; args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Standard flat filter (no ancestor traversal) ──────────────────────────
    cols = """
        SELECT t.market_id, t.market_name, t.parent_market, t.level, t.level_num,
               t.industry, t.short_definition,
               s.value, s.units, s.country, s.year,
               ROUND(c.cagr_hist_5yr * 100, 2)  AS cagr_hist_5yr_pct,
               ROUND(c.cagr_fcast_5yr * 100, 2) AS cagr_fcast_5yr_pct
        FROM taxonomy t
    """

    if not all_countries:
        sql = cols + """
            LEFT JOIN size_data s ON t.market_name = s.market_name COLLATE NOCASE
                                  AND s.country = ? AND s.year = ?
            LEFT JOIN cagr_cache c ON t.market_name = c.market_name COLLATE NOCASE
                                   AND c.country = ?
            WHERE 1=1
        """
        args = [country, year, country]
    else:
        sql = cols + """
            JOIN size_data s ON t.market_name = s.market_name COLLATE NOCASE
                             AND s.year = ?
            LEFT JOIN cagr_cache c ON t.market_name = c.market_name COLLATE NOCASE
                                   AND c.country = s.country
            WHERE s.country != 'Global'
        """
        args = [year]
        if country_allowlist:
            ph = ",".join("?" * len(country_allowlist))
            sql += f" AND s.country IN ({ph})"
            args.extend(country_allowlist)

    if industry:
        sql += " AND t.industry = ?"; args.append(industry)
    elif l1_market:
        sql += " AND s.l1_market = ?"; args.append(l1_market)
    elif params.get("_industry_allowlist"):
        ph = ",".join("?" * len(params["_industry_allowlist"]))
        sql += f" AND t.industry IN ({ph})"; args.extend(params["_industry_allowlist"])

    if level:
        sql += " AND t.level_num = ?"; args.append(level)
    if min_value is not None:
        sql += " AND s.value >= ?"; args.append(min_value)
    if max_value is not None:
        sql += " AND s.value <= ?"; args.append(max_value)
    if min_cagr_hist_pct is not None:
        sql += " AND c.cagr_hist_5yr >= ?"; args.append(min_cagr_hist_pct / 100)
    if max_cagr_hist_pct is not None:
        sql += " AND c.cagr_hist_5yr <= ?"; args.append(max_cagr_hist_pct / 100)
    if min_cagr_fcast_pct is not None:
        sql += " AND c.cagr_fcast_5yr >= ?"; args.append(min_cagr_fcast_pct / 100)
    if max_cagr_fcast_pct is not None:
        sql += " AND c.cagr_fcast_5yr <= ?"; args.append(max_cagr_fcast_pct / 100)

    sql += f" ORDER BY {order} LIMIT ?"; args.append(int(limit))
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── TOOL 7: compare_markets ───────────────────────────────────────────────────

def compare_markets(market_names: list, country: str = "Global", year: int = 2025,
                    subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    if len(market_names) < 2:
        return [{"error": "Provide at least 2 market names to compare."}]

    conn = get_conn()
    results = []
    for name in market_names:
        row = conn.execute("SELECT * FROM taxonomy WHERE market_name = ? COLLATE NOCASE", (name,)).fetchone()
        if not row:
            row = conn.execute("SELECT * FROM taxonomy WHERE market_name LIKE ? LIMIT 1",
                               (f"%{name}%",)).fetchone()
        if not row:
            results.append({"market_name": name, "error": "Not found"}); continue

        r = dict(row)
        size_row = conn.execute(
            "SELECT value, units FROM size_data WHERE market_name = ? COLLATE NOCASE AND country = ? AND year = ?",
            (r['market_name'], country, year)
        ).fetchone()
        r['country'] = country
        r['year'] = year
        if size_row and size_row['value'] is not None:
            r['value'] = size_row['value']
            r['units'] = size_row['units']
        else:
            r['value'] = None
            r['units'] = size_row['units'] if size_row else None
            r['_no_data_note'] = (
                f"No data for '{country}'. "
                f"Available geographies: {', '.join(_available_geographies(conn, r['market_name'], year))}"
            )
        results.append(r)

    results = _attach_cagr(conn, results, country)
    conn.close()

    results.sort(key=lambda x: x.get('value') or -1, reverse=True)

    units_set = {r.get('units') for r in results if r.get('units')}
    if len(units_set) > 1:
        warning = f"Warning: markets use different units ({', '.join(units_set)}). Direct value comparison may not be meaningful."
        for r in results:
            r['_unit_mismatch_warning'] = warning

    return results


# ── TOOL 8: get_market_timeseries ─────────────────────────────────────────────

def get_market_timeseries(market_name: str, country: str = "Global",
                          from_year: int = 2019, to_year: int = 2030,
                          subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    conn = get_conn()
    rows = conn.execute(
        "SELECT year, value, units FROM size_data "
        "WHERE market_name = ? COLLATE NOCASE AND country = ? AND year BETWEEN ? AND ? ORDER BY year ASC",
        (market_name, country, from_year, to_year)
    ).fetchall()
    conn.close()

    if not rows:
        return {"error": f"No data for '{market_name}' in {country}"}

    series = [{"year": r['year'], "value": r['value'], "units": r['units']} for r in rows]
    start_val = series[0]['value']
    end_val = series[-1]['value']
    years_span = series[-1]['year'] - series[0]['year']
    range_cagr = calculate_cagr(start_val, end_val, years_span) if years_span > 0 else None

    return {
        "market_name": market_name,
        "country": country,
        "from_year": from_year,
        "to_year": to_year,
        "units": series[0]['units'],
        "series": series,
        "cagr_over_range_pct": round(range_cagr * 100, 2) if range_cagr else None,
        "note": "Years ≤ 2024 are historical/estimated. Years ≥ 2025 are forecast."
    }


# ── TOOL 9: semantic_search_markets ──────────────────────────────────────────

def semantic_search_markets(query: str, limit: int = 10,
                            subscription_tier="enterprise", licensed_industries=None, licensed_geographies=None):
    _load_semantic_index()

    query_vec = _embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Hybrid scoring:
    #   60% name-only semantic similarity  — accurate for exact/near-exact market names
    #   25% name+definition semantic similarity — catches paraphrases and topic matches
    #   15% token-level string overlap     — ensures exact keyword matches always rank high
    name_scores = _name_matrix @ query_vec
    def_scores  = _def_matrix  @ query_vec
    str_scores  = np.array([_string_overlap_score(query, n) for n in _embed_names], dtype=np.float32)

    scores = 0.60 * name_scores + 0.25 * def_scores + 0.15 * str_scores

    top_indices = np.argsort(scores)[::-1][:limit]

    conn = get_conn()
    results = []
    for idx in top_indices:
        name = _embed_names[idx]
        row = conn.execute(
            "SELECT market_id, market_name, parent_market, level, level_num, industry, short_definition "
            "FROM taxonomy WHERE market_name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            r = dict(row)
            r["similarity_score"] = round(float(scores[idx]), 3)
            results.append(r)
    conn.close()
    return results