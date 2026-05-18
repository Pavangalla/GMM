import pandas as pd
import sqlite3
import os
import numpy as np
from .cagr import calculate_cagr_hist, calculate_cagr_fcast

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = "/data/gmm_data.db"
EMBEDDINGS_PATH = "/data/market_embeddings.npz"
YEAR_COLUMNS = [str(y) for y in range(2010, 2036)]


def build_database():
    """
    Loads taxonomy + size data into SQLite at startup.
    Melts wide year columns to long format.
    Pre-computes CAGR cache.
    """
    conn = sqlite3.connect(DB_PATH)

    # ── TAXONOMY ──────────────────────────────────────────────────────────────
    df = pd.read_excel(os.path.join(_BASE, "data", "Basic_Taxonomy.xlsx"))
    df.columns = ['market_name', 'parent_market', 'level', 'industry',
                  'long_definition', 'short_definition']
    df['level_num'] = df['level'].str.extract(r'(\d+)').astype(int)
    df['market_id'] = ['MKT' + str(i).zfill(5) for i in range(len(df))]

    # Resolve parent_id from parent_market name
    name_to_id = dict(zip(df['market_name'], df['market_id']))
    df['parent_id'] = df['parent_market'].map(name_to_id)

    df.to_sql('taxonomy', conn, if_exists='replace', index=False)

    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_tax_id          ON taxonomy(market_id)",
        "CREATE INDEX IF NOT EXISTS idx_tax_name        ON taxonomy(market_name COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_tax_parent_id   ON taxonomy(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_tax_parent_name ON taxonomy(parent_market COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_tax_industry    ON taxonomy(industry)",
        "CREATE INDEX IF NOT EXISTS idx_tax_level       ON taxonomy(level_num)",
    ]:
        conn.execute(idx_sql)

    # ── SIZE DATA ─────────────────────────────────────────────────────────────
    size_df = pd.read_csv(os.path.join(_BASE, "data", "size_data.csv"), low_memory=False)
    size_df.columns = [c.strip() for c in size_df.columns]

    present_years = [c for c in YEAR_COLUMNS if c in size_df.columns]
    for col in present_years:
        # Strip thousands-separator commas (e.g. "3,404.61" → "3404.61") before parsing
        size_df[col] = size_df[col].astype(str).str.replace(',', '', regex=False)
        size_df[col] = pd.to_numeric(size_df[col], errors='coerce')

    id_vars = [c for c in ['Country', 'Market Name', 'Level', 'Parent Market', 'L1 Market', 'Units']
               if c in size_df.columns]

    melted = size_df.melt(id_vars=id_vars, value_vars=present_years,
                          var_name='year', value_name='value')
    melted['year'] = melted['year'].astype(int)
    melted = melted.rename(columns={
        'Country': 'country', 'Market Name': 'market_name', 'Level': 'level',
        'Parent Market': 'parent_market', 'L1 Market': 'l1_market', 'Units': 'units',
    })

    melted.to_sql('size_data', conn, if_exists='replace', index=False)

    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_size_market  ON size_data(market_name COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_size_country ON size_data(country)",
        "CREATE INDEX IF NOT EXISTS idx_size_year    ON size_data(year)",
        "CREATE INDEX IF NOT EXISTS idx_size_l1      ON size_data(l1_market)",
        "CREATE INDEX IF NOT EXISTS idx_size_market_country_year ON size_data(market_name COLLATE NOCASE, country, year)",
    ]:
        conn.execute(idx_sql)

    # ── CAGR CACHE ────────────────────────────────────────────────────────────
    print("Pre-computing CAGR cache...")
    records = []
    for _, row in size_df.iterrows():
        row_dict = row.to_dict()
        records.append({
            'market_name':    row.get('Market Name'),
            'country':        row.get('Country'),
            'units':          row.get('Units'),
            'cagr_hist_5yr':  calculate_cagr_hist(row_dict, from_year=2019, to_year=2025),
            'cagr_fcast_5yr': calculate_cagr_fcast(row_dict, from_year=2025, to_year=2030),
        })

    cagr_df = pd.DataFrame(records)
    cagr_df.to_sql('cagr_cache', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cagr_market  ON cagr_cache(market_name COLLATE NOCASE)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cagr_country ON cagr_cache(country)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cagr_market_country ON cagr_cache(market_name COLLATE NOCASE, country)")

    conn.commit()
    conn.close()
    print(f"Database built: {DB_PATH}")

    # ── EMBEDDINGS ────────────────────────────────────────────────────────────
    build_embeddings()


def build_embeddings():
    from sentence_transformers import SentenceTransformer
    print("Building semantic embeddings (this runs once)...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT market_name, short_definition FROM taxonomy").fetchall()
    conn.close()

    names = [r["market_name"] for r in rows]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Name-only embeddings: captures the market concept without definition noise
    name_embeddings = model.encode(names, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    # Definition embeddings: captures full semantic context
    def_texts = [f"{r['market_name']}. {r['short_definition'] or ''}" for r in rows]
    def_embeddings = model.encode(def_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    np.savez(
        EMBEDDINGS_PATH,
        names=np.array(names),
        name_embeddings=name_embeddings.astype(np.float32),
        def_embeddings=def_embeddings.astype(np.float32),
    )
    print(f"Embeddings saved: {len(names)} markets")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

if __name__ == "__main__":
    build_database()
