# em_mena_rv_dashboard.py
# IMPORTANT: This is a pure .py file. Do NOT paste any ```python fences into the file.

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
# Paths
# -----------------------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / "data").is_dir():
            return p
        p = p.parent
    return start.resolve().parent


APP_DIR = Path(__file__).resolve().parent
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
        # "Qatar gres (if applicable)" — common tokens:
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


def maturity_and_scope_filter(
    meta: pd.DataFrame,
    df: pd.DataFrame,
    asof: pd.Timestamp,
    bucket_name: str,
    scope_choice: str,
):
    lo, hi = MAT_BUCKETS[bucket_name]
    m = meta.dropna(subset=["maturity"]).copy()

    # scope filter
    if scope_choice != "ALL":
        m = m[m["issuer"].apply(lambda x: issuer_in_scope(x, scope_choice))]

    # maturity filter
    m["ytm"] = m["maturity"].apply(lambda x: ytm_years(x, asof))
    m = m[(m["ytm"] >= lo) & (m["ytm"] <= hi)]

    tickers_sorted = sorted(m["ticker"].unique().tolist())  # alphabetical
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
    """
    Residual z-score = z-score of residuals from rolling regression:
        zspread_w ~ const + country_median(date,country)
    """
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
            # predicted = b0 + b1*country_med, aligned on g.index
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
# Heatmap matrix (single-bond zscore differences)
# -----------------------------
def z_diff_matrix(z: pd.Series, tickers_sorted: list):
    z = z.dropna()
    keep = [t for t in tickers_sorted if t in z.index]
    z = z.loc[keep]
    if z.shape[0] < 2:
        return pd.DataFrame()
    arr = z.values.astype(float)
    M = arr.reshape(-1, 1) - arr.reshape(1, -1)  # row i - col j
    return pd.DataFrame(M, index=z.index, columns=z.index)


# -----------------------------
# Plotly interactive heatmap
# -----------------------------
def plotly_z_heatmap(mat: pd.DataFrame, title: str, vmax: float):
    if mat is None or mat.empty or mat.shape[0] < 2:
        st.info("Not enough bonds in this selection.")
        return

    colors = px.colors.diverging.RdYlGn[::-1]  # negative green, positive red

    # mat[y, x] = Z(y) - Z(x): positive means "Sell y / Buy x"
    fig = go.Figure(
        data=go.Heatmap(
            z=np.round(mat.values.astype(float), 2),  # 2 decimals in cells (requested)
            x=mat.columns,  # Buy
            y=mat.index,    # Sell
            zmin=-vmax,
            zmax=vmax,
            colorscale=colors,
            colorbar=dict(title="Z-score"),
            hovertemplate=(
                "<b>Buy:</b> %{x}<br>"
                "<b>Sell:</b> %{y}<br>"
                "Z-score: <b>%{z:.2f}</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=650,
        hovermode="closest",
    )

    fig.update_xaxes(
        tickangle=90,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikecolor="rgba(0,0,0,0.45)",
        spikethickness=1,
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikecolor="rgba(0,0,0,0.45)",
        spikethickness=1,
        tickfont=dict(size=9),
    )

    st.plotly_chart(fig, use_container_width=True)


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
    """
    Buy = cheaper (lower z), Sell = richer (higher z)

    Current spread = zspread(Sell) - zspread(Buy)
    Average spread = mean of historical (Sell-Buy) spread
    Z score = Z(Sell) - Z(Buy)

    OUTPUT is ranked by Z score (desc), as requested.
    """
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

    # average spread from history
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

    # de-dup unordered pairs
    pairs["pair_key"] = pairs.apply(lambda r: "||".join(sorted([r["ticker_buy"], r["ticker_sell"]])), axis=1)
    pairs = pairs.sort_values("z_score", ascending=False).drop_duplicates("pair_key")

    top = pairs.head(topn).copy()
    if top.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "Buy": top["ticker_buy"].values,
            "Sell": top["ticker_sell"].values,
            "Current spread": top["current_spread"].values,
            "Average spread": top["average_spread"].values,
            "Z score": top["z_score"].values,
        }
    )

    for c in ["Current spread", "Average spread", "Z score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    # final rank: biggest Z score first (requested)
    out = out.sort_values("Z score", ascending=False).head(topn).reset_index(drop=True)

    return out


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
    else:
        st.dataframe(tbl, use_container_width=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Paths")
    st.code(f"PROJECT_ROOT = {PROJECT_ROOT}")
    st.code(f"DATA_DIR     = {DATA_DIR}")
    st.code(f"OUTPUTS_DIR  = {OUTPUTS_DIR}")
    st.code(f"DEFAULT_EXCEL= {DEFAULT_EXCEL}")

    st.header("Data")
    excel_path = st.text_input("Excel file path", value=str(DEFAULT_EXCEL))
    sheet = st.text_input("Sheet name", value="Time_Series_Z_Spreads")

    st.header("Short-history params")
    window = st.slider("Rolling window (days)", 10, 120, 20, step=5)
    min_periods = st.slider("Min periods (rolling)", 5, 60, 10, step=1)
    min_obs = st.slider("Min obs per bond", 10, 120, 25, step=1)
    winsor_p = st.slider("Winsor tail p", 0.0, 0.05, 0.01, step=0.005)

    st.header("Heatmap scale")
    vmax = st.slider("Color scale (max |Z diff|)", 0.5, 6.0, 3.0, step=0.5)

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
# Load
# -----------------------------
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
asof = st.sidebar.date_input("As-of date", value=asof_default.date())
asof = pd.to_datetime(asof)

# -----------------------------
# Diagnostics
# -----------------------------
st.markdown("#### Diagnostics")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total bonds", int(df["ticker"].nunique()))
with c2:
    st.metric("As-of", str(asof.date()))
with c3:
    st.metric("Data rows", int(df.shape[0]))

# -----------------------------
# Tabs
# -----------------------------
tab_raw, tab_resid, tab_coint = st.tabs(["Raw Z", "Residual Z", "Cointegration Z (proxy)"])

def header_row_with_filters(key_prefix: str, default_bucket: str = "5Y area", default_scope: str = "ALL"):
    left, r1, r2 = st.columns([3, 1, 1])
    with left:
        # removed "All MENA Bonds RV Screener" (requested)
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
    z = last.set_index("ticker")["raw_z"] if not last.empty else pd.Series(dtype=float)
    mat = z_diff_matrix(z, tickers_sorted)

    plotly_z_heatmap(mat, title=f"Raw Z-score Matrix — {bucket_choice}", vmax=vmax)
    show_rv_pairs_table(df_bucket, last, "raw_z", tickers_sorted, asof, window, min_periods)

# -----------------------------
# Residual tab
# -----------------------------
with tab_resid:
    scope_choice, bucket_choice = header_row_with_filters("resid", default_bucket="5Y area", default_scope="ALL")
    st.markdown(f"### Residual Z-score — {bucket_choice}")

    df_bucket, meta_bucket, tickers_sorted = maturity_and_scope_filter(meta, df, asof, bucket_choice, scope_choice)

    last = compute_resid_last(df_bucket, window, min_periods, min_obs, asof, winsor_p)

    # IMPORTANT: use resid_z everywhere (table + heatmap). This fixes "no z score in table".
    z = last.set_index("ticker")["resid_z"] if not last.empty else pd.Series(dtype=float)
    mat = z_diff_matrix(z, tickers_sorted)

    plotly_z_heatmap(mat, title=f"Residual Z-score Matrix — {bucket_choice}", vmax=vmax)
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
    z = last.set_index("ticker")["raw_z"] if not last.empty else pd.Series(dtype=float)
    mat = z_diff_matrix(z, tickers_sorted)

    plotly_z_heatmap(mat, title=f"Cointegration (proxy) Z-score Matrix — {bucket_choice}", vmax=vmax)
    show_rv_pairs_table(df_bucket, last, "raw_z", tickers_sorted, asof, window, min_periods)
