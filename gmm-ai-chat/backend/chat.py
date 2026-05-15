import json
import inspect
from openai import OpenAI
from .system_prompt import build_system_prompt
from .tools import (search_markets, get_market, get_children, get_parent,
                   get_hierarchy_path, filter_markets, compare_markets, get_market_timeseries,
                   semantic_search_markets)
from .models import ChatRequest, ChatResponse

client = OpenAI()

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_markets",
            "description": "Search GMM taxonomy by market name or keyword. Use to locate a market before calling get_market, or to explore what markets exist on a topic.",
            "parameters": {"type": "object", "properties": {
                "query":    {"type": "string",  "description": "Search term"},
                "industry": {"type": "string",  "description": "Optional: restrict to one industry"},
                "level":    {"type": "integer", "description": "Optional: restrict to level 1–6"},
                "limit":    {"type": "integer", "description": "Max results (default 20)"},
            }, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market",
            "description": "Full profile for one market: taxonomy info, value for a country+year, and both CAGRs.",
            "parameters": {"type": "object", "properties": {
                "market_name": {"type": "string",  "description": "Market name (preferred)"},
                "market_id":   {"type": "string",  "description": "Market ID if known"},
                "country":     {"type": "string",  "description": "Country or region (default: Global)"},
                "year":        {"type": "integer", "description": "Year (default: 2025). 2010–2025 are historical/estimated; 2026–2035 are forecast."},
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_children",
            "description": "Get all direct sub-markets (one level below) of a market. Use for 'what are the sub-markets of X?' questions.",
            "parameters": {"type": "object", "properties": {
                "market_name":  {"type": "string",  "description": "Parent market name"},
                "market_id":    {"type": "string",  "description": "Parent market ID if known"},
                "country":      {"type": "string",  "description": "Country for size data (default: Global)"},
                "year":         {"type": "integer", "description": "Year (default: 2025)"},
                "include_size": {"type": "boolean", "description": "Include value and CAGR (default: true)"},
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_parent",
            "description": "Get the parent market of a given market. Use to navigate upward in the hierarchy.",
            "parameters": {"type": "object", "properties": {
                "market_name": {"type": "string",  "description": "Market name"},
                "market_id":   {"type": "string",  "description": "Market ID if known"},
                "country":     {"type": "string",  "description": "Country (default: Global)"},
                "year":        {"type": "integer", "description": "Year (default: 2024)"},
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hierarchy_path",
            "description": "Full ancestry path from Level 1 root down to a market. Use for 'where does X sit in the taxonomy?'",
            "parameters": {"type": "object", "properties": {
                "market_name": {"type": "string", "description": "Market name"},
                "market_id":   {"type": "string", "description": "Market ID if known"},
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_markets",
            "description": "Primary analytical tool. Filter and rank markets by value, CAGR, industry, level, country. Use for ranking ('largest', 'fastest growing'), threshold ('above $100bn'), and filtered lists. For 'largest/fastest/smallest within X' queries always use ancestor_market=X to search ALL sub-markets at every level under X. sort_by options: value_desc, value_asc, cagr_hist_desc, cagr_hist_asc, cagr_fcast_desc, cagr_fcast_asc, name_asc.",
            "parameters": {"type": "object", "properties": {
                "ancestor_market":    {"type": "string",  "description": "Return ALL sub-markets at every level beneath this market (excludes the market itself). Use this for 'largest/fastest/smallest within X' queries."},
                "industry":           {"type": "string",  "description": "Filter to one industry (use only when no ancestor_market is set)"},
                "level":              {"type": "integer", "description": "Filter to hierarchy level 1–6"},
                "l1_market":          {"type": "string",  "description": "Filter by L1 Market name"},
                "min_value":          {"type": "number",  "description": "Minimum value (in market's native units)"},
                "max_value":          {"type": "number",  "description": "Maximum value"},
                "min_cagr_hist_pct":  {"type": "number",  "description": "Min historical CAGR % (e.g. 15 for 15%)"},
                "max_cagr_hist_pct":  {"type": "number",  "description": "Max historical CAGR %"},
                "min_cagr_fcast_pct": {"type": "number",  "description": "Min forecast CAGR %"},
                "max_cagr_fcast_pct": {"type": "number",  "description": "Max forecast CAGR %"},
                "country":            {"type": "string",  "description": "Country or region for data retrieval. Default: 'Global' (global aggregate). Pass an exact country name (e.g. 'USA', 'China', 'India') or region name ('Asia Pacific', 'North America', 'Western Europe', 'Eastern Europe', 'Middle East', 'South America') when the user mentions one. Pass '*' ONLY for 'which country is largest/fastest/smallest?' queries to rank all individual countries."},
                "sort_by":            {"type": "string",  "enum": ["value_desc","value_asc","cagr_hist_desc","cagr_hist_asc","cagr_fcast_desc","cagr_fcast_asc","name_asc"]},
                "year":               {"type": "integer", "description": "Year for values (default: 2025)"},
                "limit":              {"type": "integer", "description": "Max results (default: 20)"},
            }, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_markets",
            "description": "Side-by-side comparison of 2–5 markets. Returns value, units, and both CAGRs. Flags unit mismatches automatically. If a market has no data for the requested country, the result includes a _no_data_note listing available geographies.",
            "parameters": {"type": "object", "properties": {
                "market_names": {"type": "array", "items": {"type": "string"}, "description": "2–5 exact market names (use semantic_search_markets first to confirm names)"},
                "country":      {"type": "string",  "description": "Country or region (default: 'Global'). Use exact names from the geography rules — e.g. 'USA', 'China', 'North America'. Do not guess; check _no_data_note in results if data is missing."},
                "year":         {"type": "integer", "description": "Year (default: 2025)"},
            }, "required": ["market_names"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_timeseries",
            "description": "Year-by-year values for a market from 2010–2035. Use for trend questions: 'how has X grown since 2015?', 'what is the forecast to 2030?'. Computes CAGR over the requested range.",
            "parameters": {"type": "object", "properties": {
                "market_name": {"type": "string",  "description": "Market name"},
                "country":     {"type": "string",  "description": "Country (default: Global)"},
                "from_year":   {"type": "integer", "description": "Start year (min: 2010, default: 2019)"},
                "to_year":     {"type": "integer", "description": "End year (max: 2035, default: 2030)"},
            }, "required": ["market_name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_markets",
            "description": "Find markets by semantic similarity to a natural-language query. Use this FIRST whenever you need to identify the correct market name in the database — especially when the user's phrasing may differ from the exact database name. Returns ranked candidates with similarity scores.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Natural language description of the market to find"},
                "limit": {"type": "integer", "description": "Number of candidates to return (default 10)"},
            }, "required": ["query"]}
        }
    },
]

TOOL_REGISTRY = {
    "search_markets":          search_markets,
    "get_market":              get_market,
    "get_children":            get_children,
    "get_parent":              get_parent,
    "get_hierarchy_path":      get_hierarchy_path,
    "filter_markets":          filter_markets,
    "compare_markets":         compare_markets,
    "get_market_timeseries":   get_market_timeseries,
    "semantic_search_markets": semantic_search_markets,
}


def dispatch_tool(tool_name: str, tool_input: dict, request: ChatRequest) -> str:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    tool_input["subscription_tier"]    = request.subscription_tier
    tool_input["licensed_industries"]  = request.licensed_industries
    tool_input["licensed_geographies"] = request.licensed_geographies

    valid_params = inspect.signature(fn).parameters.keys()
    filtered = {k: v for k, v in tool_input.items() if k in valid_params}

    try:
        return json.dumps(fn(**filtered), default=str)
    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Tool error: {str(e)}"})


def chat(request: ChatRequest) -> ChatResponse:
    system = build_system_prompt(
        request.user_id, request.subscription_tier,
        request.licensed_industries, request.licensed_geographies
    )
    messages = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    messages.append({"role": "user", "content": request.message})

    tool_calls_made = []

    for _ in range(10):  # max 10 tool-call iterations per query
        all_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            tools=TOOL_DEFINITIONS,
            messages=all_messages,
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return ChatResponse(response=choice.message.content, tool_calls_made=tool_calls_made)

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_calls_made.append(tool_call.function.name)
                tool_input = json.loads(tool_call.function.arguments)
                result = dispatch_tool(tool_call.function.name, tool_input, request)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

    return ChatResponse(
        response="Unable to complete analysis. Please try a more specific question.",
        tool_calls_made=tool_calls_made
    )
