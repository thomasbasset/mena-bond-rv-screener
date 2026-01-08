# em_mena_rv_dashboard.py
# Run: streamlit run em_mena_rv_dashboard.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="MENA Bond RV (Z-spreads)", layout="wide")

# -----------------------------
# Hide the left menu/sidebar completely
# -----------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {display: none;}
    div[data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / "data").is_dir():
            return p
        p = p.parent
    return start.resolve().parent


APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = find_project_root(APP_DIR)
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EXCEL = DATA_DIR / "RV_Model_Raw_Data.xlsx"

# -----------------------------
# Rating scope definition
# -----------------------------
SCOPE_MAP = {
    "ALL": None,
    "IG": {
        "ISRAEL", "ISR",
        "KSA",
        "KUWIB",
        "QATAR", "QAT",
        "UAE",
        "OMAN",
        "ARAMCO",
        "QPETRO",
        "RASGAS",
        "ADNOCM", "ADNOC",
    },
    "HY": {
        "EGYPT", "EGYP",
        "ESCWPC",
        "SWEHAN",
        "JORDAN", "JOR",
        "MOROC", "MAR",
        "BHRAIN", "BHR", "BAHRAIN",
    },
    "QUASI": {
        "DPWDU",
        "PIFKSA",
        "MTVD",
        "TAQAUH",
        "RPCUH",
        "QATARGRES", "QATARGRS", "QATGRES", "QGAS",
    },
}


def issuer_in_scope(issuer: str, scope_choice: str) -> bool:
    if scope_choice == "ALL":
        return True
    s = SCOPE_MAP.get(scope_choice, set())
    return str(issuer).upper().strip() in s


# -----------------------------
# Ticker parsing
# -----------------------------
ISSUER_TO_COUNTRY = {
    "KSA": "KSA",
    "UAE": "UAE",
    "OMAN": "OMAN",
    "QATAR": "QAT",
    "QAT": "QAT",
    "QPETRO": "QAT",
    "RASGAS": "QAT",
    "BHRAIN": "BHR",
    "BHR": "BHR",
    "BAHRAIN": "BHR",
    "EGYPT": "EGYP",
    "EGYP": "EGYP",
    "JORDAN": "JOR",
    "JOR": "JOR",
    "MOROC": "MAR",
    "MAR": "MAR",
    "ISRAEL": "ISR",
    "ISR": "ISR",
    "ARAMCO": "KSA",
    "PIFKSA": "KSA",
    "KUWIB": "KWT",
    "KWT": "KWT",
    "ADNOCM": "UAE",
    "ADNOC": "UAE",
    "DPWDU": "UAE",
    "TAQAUH": "UAE",
    "ESCWPC": "UAE",
    "SWEHAN": "UAE",
    "RPCUH": "UAE",
    "MTVD": "OTHER",
}


def norm_token(tok: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(tok).upper()) if tok else ""


def parse_maturity_from_ticker(ticker: str):
    m = re.search(r"(\d{2}/\d{2}/\d{2,4})", str(ticker))
    if not m:
        return None
    dt = pd.to_datetime(m.group(1), errors="coerce", dayfirst=False)
    return None if pd.isna(dt) else dt


def parse_ticker_meta(t: str):
    raw = str(t).strip()
    first = raw.split()[0] if raw.split() else raw
    issuer = norm_token(first)
    country = ISSUER_TO_COUNTRY.get(issuer, "OTHER")
    maturity = parse_maturity_from_ticker(raw)
    return {"ticker": raw, "issuer": issuer, "country": country, "maturity": maturity}


# -----------------------------
# Bloomberg-style coupon fraction formatting (DISPLAY ONLY)
# -----------------------------
_SUP = str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")
_SUB = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎")

_VULGAR = {
    (1, 2): "½",
    (1, 3): "⅓",
    (2, 3): "⅔",
    (1, 4): "¼",
    (3, 4): "¾",
    (1, 5): "⅕",
    (2, 5): "⅖",
    (3, 5): "⅗",
    (4, 5): "⅘",
    (1, 6): "⅙",
    (5, 6): "⅚",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (5, 8): "⅝",
    (7, 8): "⅞",
}

def _to_sup(s: str) -> str:
    return str(s).translate(_SUP)

def _to_sub(s: str) -> str:
    return str(s).translate(_SUB)

_COUPON_FRAC_RE = re.compile(r"(?<!\.)\b(\d+)\s+(\d+)\s*/\s*(\d+)\b(?!\s*/\s*\d)")

def format_coupon_superscript(ticker: str) -> str:
    txt = str(ticker)

    def _repl(m):
        whole = m.group(1)
        num = int(m.group(2))
        den = int(m.group(3))
        if (num, den) in _VULGAR:
            return f"{whole}{_VULGAR[(num, den)]}"
        return f"{whole}{_to_sup(num)}⁄{_to_sub(den)}"

    return _COUPON_FRAC_RE.sub(_repl, txt)

def display_label_map(tickers):
    return {t: format_coupon_superscript(t) for t in tickers}


# -----------------------------
# Excel parsing
# -----------------------------
def parse_dates_best(series: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return d1 if d1.isna().sum() < d0.isna().sum() else d0


def load_excel_wide(path: Path, sheet: str):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    pairs = []
    for i in range(0, raw.shape[1], 2):
        t = raw.iat[0, i]
        if pd.isna(t) or i + 1 >= raw.shape[1]:
            continue
        pairs.append((i, i + 1, str(t)))
    return raw, pairs


def to_long_df(raw: pd.DataFrame, pairs):
    out = []
    for c_date, c_z, t in pairs:
        dates = parse_dates_best(raw.iloc[1:, c_date])
        z = pd.to_numeric(raw.iloc[1:, c_z], errors="coerce")
        tmp = pd.DataFrame({"date": dates, "zspread": z})
        tmp["ticker"] = t
        tmp = tmp.dropna(subset=["date", "zspread"])
        out.append(tmp[["date", "ticker", "zspread"]])
    if not out:
        return pd.DataFrame(columns=["date", "ticker", "zspread"])
    return pd.concat(out, ignore_index=True).sort_values(["ticker", "date"])


# -----------------------------
# Helpers
# -----------------------------
def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    s2 = s.dropna()
    if s2.empty or p <= 0:
        return s
    lo = s2.quantile(p)
    hi = s2.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sig = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sig


def last_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby("ticker").tail(1)


def ytm_years(maturity: pd.Timestamp, asof: pd.Timestamp) -> float:
    return (pd.to_datetime(maturity) - pd.to_datetime(asof)).days / 365.25


# -----------------------------
# Maturity buckets
# -----------------------------
MAT_BUCKETS = {
    "2Y area": (0.5, 3.5),
    "5Y area": (3.5, 7.5),
    "10Y area": (7.5, 13.5),
    "20Y area": (13.5, 25.0),
    "30Y+ area": (25.0, 80.0),
}


def maturity_and_scope_filter(meta: pd.DataFrame, df: pd.DataFrame, asof: pd.Timestamp, bucket_name: str, scope_choice: str):
    lo, hi = MAT_BUCKETS[bucket_name]
    m = meta.dropna(subset=["maturity"]).copy()

    if scope_choice != "ALL":
        m = m[m["issuer"].apply(lambda x: issuer_in_scope(x, scope_choice))]

    m["ytm"] = m["maturity"].apply(lambda x: ytm_years(x, asof))
    m = m[(m["ytm"] >= lo) & (m["ytm"] <= hi)]

    tickers_sorted = sorted(m["ticker"].unique().tolist())
    return df[df["ticker"].isin(tickers_sorted)].copy(), m.copy(), tickers_sorted


# -----------------------------
# Z computations
# -----------------------------
def compute_raw_last(df: pd.DataFrame, window: int, min_periods: int, min_obs: int, asof: pd.Timestamp, winsor_p: float):
    d = df[df["date"] <= asof].copy()
    d["zspread_w"] = d.groupby("ticker")["zspread"].transform(lambda x: winsorize_series(x, p=winsor_p))
    d["raw_z"] = d.groupby("ticker")["zspread_w"].transform(lambda x: rolling_zscore(x, window, min_periods))
    last = last_per_ticker(d)
    obs = d.groupby("ticker")["zspread"].count().rename("n_obs")
    last = last.merge(obs, on="ticker", how="left")
    return last[last["n_obs"] >= min_obs]


def compute_resid_last(df: pd.DataFrame, window: int, min_periods: int, min_obs: int, asof: pd.Timestamp, winsor_p: float):
    d = df[df["date"] <= asof].copy()
    d["zspread_w"] = d.groupby("ticker")["zspread"].transform(lambda x: winsorize_series(x, p=winsor_p))

    country_factor = (
        d.groupby(["country", "date"])["zspread_w"]
        .median()
        .rename("country_med")
        .reset_index()
    )
    d = d.merge(country_factor, on=["country", "date"], how="left")

    d = d.sort_values(["ticker", "date"]).copy()
    d["resid"] = np.nan

    for tkr, g in d.groupby("ticker"):
        g = g.dropna(subset=["zspread_w", "country_med"]).copy()
        if g.shape[0] < max(min_obs, min_periods):
            continue

        y = g["zspread_w"].astype(float)
        X = sm.add_constant(g["country_med"].astype(float), has_constant="add")

        try:
            rols = RollingOLS(y, X, window=window, min_nobs=min_periods).fit()
            pred = (rols.params["const"] + rols.params["country_med"] * g["country_med"])
            resid = y - pred
            d.loc[g.index, "resid"] = resid
        except Exception:
            continue

    d["resid_z"] = d.groupby("ticker")["resid"].transform(lambda x: rolling_zscore(x, window, min_periods))
    last = last_per_ticker(d)
    obs = d.groupby("ticker")["zspread"].count().rename("n_obs")
    last = last.merge(obs, on="ticker", how="left")
    return last[last["n_obs"] >= min_obs]


# -----------------------------
# Pair matrix for directional coloring (FULL SYMMETRIC)
# - compute once (i<j), then mirror to (j,i) so full matrix shows
# - diagonal NaN (blank)
# Color depends on ROW action:
#   - row bond is BUY => +|Z| (green)
#   - row bond is SELL => -|Z| (red)
# -----------------------------
def pair_spread_z_matrix_directional_full(
    df_bucket: pd.DataFrame,
    tickers_sorted: list,
    asof: pd.Timestamp,
    window: int,
    min_periods: int,
):
    wide = (
        df_bucket[df_bucket["date"] <= asof]
        .pivot_table(index="date", columns="ticker", values="zspread", aggfunc="last")
        .sort_index()
    )

    n = len(tickers_sorted)
    if n < 2:
        return pd.DataFrame(), None

    M = np.full((n, n), np.nan, dtype=float)  # signed for color (+ green, - red)
    CD = np.empty((n, n), dtype=object)       # hover: [buy, sell, cur, mean, std, signed_z, abs_z]

    for i in range(n):
        CD[i, i] = ["", "", np.nan, np.nan, np.nan, np.nan, np.nan]

    for i in range(n):
        x = tickers_sorted[i]
        if x not in wide.columns:
            continue

        for j in range(i + 1, n):
            y = tickers_sorted[j]
            if y not in wide.columns:
                continue

            spread_ts = (wide[y] - wide[x]).dropna()
            if spread_ts.shape[0] < min_periods:
                continue

            mu = spread_ts.rolling(window, min_periods=min_periods).mean().iloc[-1]
            sd = spread_ts.rolling(window, min_periods=min_periods).std(ddof=0).iloc[-1]
            cur = spread_ts.iloc[-1]

            if pd.isna(mu) or pd.isna(sd) or sd == 0 or pd.isna(cur):
                continue

            z = float((cur - mu) / sd)
            z_abs = abs(z)

            # Recommendation (spread = y - x):
            # z >= 0 => BUY x, SELL y
            # z <  0 => BUY y, SELL x
            if z >= 0:
                buy, sell = x, y
                cur_sb = float(cur)     # (sell - buy) = (y - x)
                mu_sb = float(mu)
            else:
                buy, sell = y, x
                cur_sb = float(-cur)    # flip to (sell - buy)
                mu_sb = float(-mu)

            buy_fmt = format_coupon_superscript(buy)
            sell_fmt = format_coupon_superscript(sell)
            cd_common = [buy_fmt, sell_fmt, cur_sb, mu_sb, float(sd), float(z), float(z_abs)]

            CD[i, j] = cd_common
            CD[j, i] = cd_common

            # ✅ COLOR RULE: use COLUMN bond action (not row)
            # Cell (i,j): column bond is y
            col_j_is_buy = (y == buy)
            M[i, j] = z_abs if col_j_is_buy else -z_abs

            # Mirror cell (j,i): column bond is x
            col_i_is_buy = (x == buy)
            M[j, i] = z_abs if col_i_is_buy else -z_abs

    mat = pd.DataFrame(M, index=tickers_sorted, columns=tickers_sorted)
    return mat, CD



# -----------------------------
# Plotly heatmap for directional coloring
# -----------------------------
def plotly_z_heatmap_directional(mat: pd.DataFrame, customdata, title: str, vmax: float, key: str):
    if mat is None or mat.empty or mat.shape[0] < 2:
        st.info("Not enough bonds in this selection.")
        return

    st.markdown(f"#### {title}")

    labels = display_label_map(mat.index.tolist())
    x_disp = [labels.get(x, x) for x in mat.columns]
    y_disp = [labels.get(y, y) for y in mat.index]

    colors = px.colors.diverging.RdYlGn[::-1]  # + => green, - => red

    fig = go.Figure(
        data=go.Heatmap(
            z=np.round(mat.values.astype(float), 2),
            x=x_disp,
            y=y_disp,
            zmin=-vmax,
            zmax=vmax,
            colorscale=colors,
            showscale=False,
            customdata=customdata,
            hovertemplate=(
                "<b>Buy:</b> %{customdata[0]}<br>"
                "<b>Sell:</b> %{customdata[1]}<br>"
                "<b>Current spread (Sell-Buy):</b> %{customdata[2]:.2f}<br>"
                "<b>Avg spread:</b> %{customdata[3]:.2f}<br>"
                "<b>Std:</b> %{customdata[4]:.2f}<br>"
                "<b>|Z|:</b> %{customdata[6]:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=900,
        hovermode="closest",
    )

    fig.update_xaxes(tickangle=90, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))

    st.plotly_chart(fig, use_container_width=True, key=key)


# -----------------------------
# Top 10 RV trades (pair table)
# -----------------------------
def build_rv_pairs_table(
    df_bucket: pd.DataFrame,
    last: pd.DataFrame,
    z_col: str,
    tickers_sorted: list,
    asof: pd.Timestamp,
    window: int,
    min_periods: int,
    topn: int = 10,
):
    if last is None or last.empty or z_col not in last.columns:
        return pd.DataFrame()

    s = last[["ticker", "zspread", z_col]].copy().rename(columns={z_col: "z"})
    s = s[s["ticker"].isin(tickers_sorted)].dropna(subset=["z", "zspread"]).copy()
    if s.shape[0] < 2:
        return pd.DataFrame()

    buy = s.assign(_k=1)
    sell = s.assign(_k=1)
    pairs = buy.merge(sell, on="_k", suffixes=("_buy", "_sell")).drop(columns=["_k"])
    pairs = pairs[pairs["ticker_buy"] != pairs["ticker_sell"]].copy()

    pairs["z_score"] = pairs["z_sell"] - pairs["z_buy"]
    pairs = pairs[pairs["z_score"] > 0].copy()

    pairs["current_spread"] = pairs["zspread_sell"] - pairs["zspread_buy"]

    wide = (
        df_bucket[df_bucket["date"] <= asof]
        .pivot_table(index="date", columns="ticker", values="zspread", aggfunc="last")
        .sort_index()
    )

    avg_spreads = []
    for _, r in pairs.iterrows():
        b = r["ticker_buy"]
        s_ = r["ticker_sell"]
        if b not in wide.columns or s_ not in wide.columns:
            avg_spreads.append(np.nan)
            continue
        spread_ts = (wide[s_] - wide[b]).dropna()
        if spread_ts.empty:
            avg_spreads.append(np.nan)
            continue
        avg_spreads.append(float(spread_ts.tail(max(min_periods, window)).mean()))
    pairs["average_spread"] = avg_spreads

    pairs = pairs.dropna(subset=["average_spread", "current_spread", "z_score"]).copy()

    pairs["pair_key"] = pairs.apply(lambda r: "||".join(sorted([r["ticker_buy"], r["ticker_sell"]])), axis=1)
    pairs = pairs.sort_values("z_score", ascending=False).drop_duplicates("pair_key")

    top = pairs.head(topn).copy()
    if top.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "Buy": top["ticker_buy"].values,
            "Sell": top["ticker_sell"].values,
            "Current Spread": top["current_spread"].values,
            "Average Spread": top["average_spread"].values,
            "Z-score": top["z_score"].values,
        }
    )

    for c in ["Current Spread", "Average Spread", "Z-score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out = out.sort_values("Z-score", ascending=False).head(topn).reset_index(drop=True)

    out["Buy"] = out["Buy"].map(format_coupon_superscript)
    out["Sell"] = out["Sell"].map(format_coupon_superscript)

    return out


def style_rv_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    NOMURA_RED = "#d32f2f"

    sty = df.style
    sty = sty.set_table_styles(
        [
            {
                "selector": "th.blank",
                "props": [
                    ("background-color", NOMURA_RED),
                    ("color", "white"),
                    ("text-align", "center !important"),
                ],
            },
            {
                "selector": "th.col_heading",
                "props": [
                    ("background-color", NOMURA_RED),
                    ("color", "white"),
                    ("font-weight", "700"),
                    ("text-align", "center !important"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "th.row_heading",
                "props": [
                    ("background-color", "white"),
                    ("color", "black"),
                    ("font-weight", "500"),
                    ("text-align", "center !important"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center !important"),
                    ("font-size", "12px"),
                    ("padding", "6px 10px"),
                    ("white-space", "nowrap"),
                    ("vertical-align", "middle"),
                ],
            },
            {"selector": "table", "props": [("width", "100%")]},
        ],
        overwrite=True,
    )
    return sty


def show_rv_pairs_table(
    df_bucket: pd.DataFrame,
    last: pd.DataFrame,
    z_col: str,
    tickers_sorted: list,
    asof: pd.Timestamp,
    window: int,
    min_periods: int,
):
    st.markdown("### Top 10 RV trades")

    tbl = build_rv_pairs_table(df_bucket, last, z_col, tickers_sorted, asof, window, min_periods, topn=10)
    if tbl.empty:
        st.info("Not enough data to build RV pair trades in this selection.")
        return

    tbl = tbl.reset_index(drop=True)
    tbl.index = np.arange(1, len(tbl) + 1)

    sty = (
        style_rv_table(tbl)
        .format({"Current Spread": "{:.2f}", "Average Spread": "{:.2f}", "Z-score": "{:.2f}"})
    )

    st.table(sty)


# -----------------------------
# Load (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all(excel_path: str, sheet: str):
    p = Path(excel_path)
    if not p.is_file():
        raise FileNotFoundError(f"Excel not found: {p}")
    raw, pairs = load_excel_wide(p, sheet)
    df_long = to_long_df(raw, pairs)
    meta = pd.DataFrame([parse_ticker_meta(t) for t in df_long["ticker"].unique()])
    df = df_long.merge(meta, on="ticker", how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df, meta


# -----------------------------
# Header
# -----------------------------
st.title("MENA Bonds RV Screener")

# -----------------------------
# Parameters (collapsible) + Diagnostics inside
# -----------------------------
with st.expander("Parameters", expanded=False):
    excel_path = st.text_input("Excel file path", value=str(DEFAULT_EXCEL))
    sheet = st.text_input("Sheet name", value="Time_Series_Z_Spreads")

    st.markdown("#### Short-history params")
    window = st.slider("Rolling window (days)", 10, 120, 20, step=5)
    min_periods = st.slider("Min periods (rolling)", 5, 60, 10, step=1)
    min_obs = st.slider("Min obs per bond", 10, 120, 25, step=1)
    winsor_p = st.slider("Winsor tail p", 0.0, 0.05, 0.01, step=0.005)

    st.markdown("#### Heatmap scale")
    vmax = st.slider("Color scale (max |Z diff|)", 0.5, 6.0, 3.0, step=0.5)

    try:
        df, meta = load_all(excel_path, sheet)
    except Exception as e:
        st.error("Excel load failed.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.error("Parsed dataframe is empty.")
        st.stop()

    asof_default = df["date"].max()
    asof_in = st.date_input("As-of date", value=asof_default.date())
    asof = pd.to_datetime(asof_in)

    st.markdown("#### Diagnostics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total bonds", int(df["ticker"].nunique()))
    with c2:
        st.metric("As-of", str(asof.date()))
    with c3:
        st.metric("Data rows", int(df.shape[0]))

# Defaults if expander not opened this run
if "df" not in globals() or "meta" not in globals() or "asof" not in globals():
    excel_path = str(DEFAULT_EXCEL)
    sheet = "Time_Series_Z_Spreads"
    window = 20
    min_periods = 10
    min_obs = 25
    winsor_p = 0.01
    vmax = 3.0

    try:
        df, meta = load_all(excel_path, sheet)
    except Exception as e:
        st.error("Excel load failed.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.error("Parsed dataframe is empty.")
        st.stop()

    asof = pd.to_datetime(df["date"].max())

# -----------------------------
# Tabs
# -----------------------------
tab_raw, tab_resid, tab_coint = st.tabs(["Raw Z", "Residual Z", "Cointegration Z (proxy)"])


def header_row_with_filters(key_prefix: str, default_bucket: str = "5Y area", default_scope: str = "ALL"):
    left, r1, r2 = st.columns([3, 1, 1])
    with left:
        st.write("")
    with r1:
        scope_choice = st.selectbox(
            "Rating scope",
            options=["ALL", "IG", "HY", "QUASI"],
            index=["ALL", "IG", "HY", "QUASI"].index(default_scope),
            key=f"{key_prefix}_scope",
        )
    with r2:
        bucket_choice = st.selectbox(
            "Maturity bucket",
            options=list(MAT_BUCKETS.keys()),
            index=list(MAT_BUCKETS.keys()).index(default_bucket),
            key=f"{key_prefix}_bucket",
        )
    return scope_choice, bucket_choice


# -----------------------------
# Raw tab
# -----------------------------
with tab_raw:
    scope_choice, bucket_choice = header_row_with_filters("raw", default_bucket="5Y area", default_scope="ALL")
    st.markdown(f"### Raw Z-score — {bucket_choice}")

    df_bucket, meta_bucket, tickers_sorted = maturity_and_scope_filter(meta, df, asof, bucket_choice, scope_choice)

    last = compute_raw_last(df_bucket, window, min_periods, min_obs, asof, winsor_p)

    mat, cd = pair_spread_z_matrix_directional_full(df_bucket, tickers_sorted, asof, window, min_periods)
    plotly_z_heatmap_directional(
        mat, cd,
        title=f"Raw Z-score Matrix — {bucket_choice}",
        vmax=vmax,
        key=f"hm_raw_{scope_choice}_{bucket_choice}_{asof.date()}_{window}_{min_periods}_{min_obs}_{winsor_p}_{vmax}",
    )

    show_rv_pairs_table(df_bucket, last, "raw_z", tickers_sorted, asof, window, min_periods)

# -----------------------------
# Residual tab
# -----------------------------
with tab_resid:
    scope_choice, bucket_choice = header_row_with_filters("resid", default_bucket="5Y area", default_scope="ALL")
    st.markdown(f"### Residual Z-score — {bucket_choice}")

    df_bucket, meta_bucket, tickers_sorted = maturity_and_scope_filter(meta, df, asof, bucket_choice, scope_choice)

    last = compute_resid_last(df_bucket, window, min_periods, min_obs, asof, winsor_p)

    mat, cd = pair_spread_z_matrix_directional_full(df_bucket, tickers_sorted, asof, window, min_periods)
    plotly_z_heatmap_directional(
        mat, cd,
        title=f"Residual Z-score Matrix — {bucket_choice}",
        vmax=vmax,
        key=f"hm_residual_{scope_choice}_{bucket_choice}_{asof.date()}_{window}_{min_periods}_{min_obs}_{winsor_p}_{vmax}",
    )

    show_rv_pairs_table(df_bucket, last, "resid_z", tickers_sorted, asof, window, min_periods)

# -----------------------------
# Cointegration proxy tab
# -----------------------------
with tab_coint:
    scope_choice, bucket_choice = header_row_with_filters("coint", default_bucket="5Y area", default_scope="ALL")
    st.markdown(f"### Cointegration Z-score — {bucket_choice} (proxy)")

    st.info("Proxy mode: uses single-bond z-score differences (same format as Raw/Residual).")

    df_bucket, meta_bucket, tickers_sorted = maturity_and_scope_filter(meta, df, asof, bucket_choice, scope_choice)

    last = compute_raw_last(df_bucket, window, min_periods, min_obs, asof, winsor_p)

    mat, cd = pair_spread_z_matrix_directional_full(df_bucket, tickers_sorted, asof, window, min_periods)
    plotly_z_heatmap_directional(
        mat, cd,
        title=f"Cointegration Z-score Matrix — {bucket_choice}",
        vmax=vmax,
        key=f"hm_cointegration_{scope_choice}_{bucket_choice}_{asof.date()}_{window}_{min_periods}_{min_obs}_{winsor_p}_{vmax}",
    )

    show_rv_pairs_table(df_bucket, last, "raw_z", tickers_sorted, asof, window, min_periods)

