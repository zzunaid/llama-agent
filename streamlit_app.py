# Streamlit Lead Insights — Two Windows • Time Series • A vs B Time Series
# Works with your schema:
# name, phone, lead_priority, identifier, utm_source, utm_medium, utm_campaign,
# disposition_status, sub_disposition_status, remarks, created_at(or your date col)

import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare, chi2_contingency

# ---------------- Page setup ----------------
st.set_page_config(page_title="Lead Insights (Cohort & Time Series)", layout="wide")
st.title("Lead Insights — Cohort & Attribute Comparison")
st.caption("Plug in your DB in Streamlit Secrets to go live. Falls back to demo data if DB is not configured.")

# ---------------- Defaults ----------------
today = date.today()
curr_end = today
curr_start = curr_end - timedelta(days=7)
base_end = curr_start
base_start = base_end - timedelta(days=7)

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Choose analysis mode",
        ["Two windows", "Time series", "Compare two time series (A vs B)"],
        index=0
    )

# ---------------- Attributes / schema ----------------
ATTRS = [
    "lead_priority",
    "identifier",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "disposition_status",
    "sub_disposition_status",
]
PII_COLS = ["name", "phone", "remarks"]  # kept for preview only

# ---------------- DB (optional) ----------------
ENGINE = None
TABLE = st.secrets.get("TABLE", "public.leads")             # you can set in Secrets
DATE_COLUMN = st.secrets.get("DATE_COLUMN", "created_at")   # you can set in Secrets
DB_URL = st.secrets.get("DATABASE_URL", None)
if DB_URL:
    try:
        from sqlalchemy import create_engine
        ENGINE = create_engine(DB_URL)
    except Exception as e:
        st.warning(f"Could not init DB engine. Falling back to demo data. Error: {e}")
        ENGINE = None

# ---------------- Helpers: normalization & privacy ----------------
def normalize_leads(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and categorical values; mask phones for preview."""
    df = df.copy()

    # Rename headers (handle title-cased exports)
    rename = {
        "Name": "name",
        "Phone": "phone",
        "Lead Priority": "lead_priority",
        "Identifier": "identifier",
        "UTM Source": "utm_source",
        "UTM Medium": "utm_medium",
        "UTM medium": "utm_medium",
        "UTM Campaign": "utm_campaign",
        "Disposition Status": "disposition_status",
        "Sub-Disposition Status": "sub_disposition_status",
        "Remarks": "remarks",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Coalesce / normalize attributes
    for col in ATTRS:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "unknown", "none": "unknown", "": "unknown"})
        )

    # Ensure date column exists
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    elif "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        # if missing entirely in demo data, we'll add later as needed
        pass

    # PII preview fields
    if "phone" in df.columns:
        df["phone_masked"] = (
            df["phone"]
            .astype(str)
            .str.replace(r"(\d{2})\d{6}(\d{2})", r"\1******\2", regex=True)
        )

    return df

def preview_leads(df: pd.DataFrame):
    cols = [c for c in ["name", "phone_masked", "lead_priority", "identifier",
                        "utm_source", "utm_medium", "utm_campaign",
                        "disposition_status", "sub_disposition_status", "remarks"] if c in df.columns]
    if cols:
        st.expander("Sample rows (PII masked)").dataframe(df.head(20)[cols], use_container_width=True)

# ---------------- DB fetch ----------------
def fetch_range(start_date, end_date) -> pd.DataFrame:
    """Fetch leads from DB for [start, end). Falls back to empty if DB not configured."""
    if ENGINE is None or DB_URL is None:
        return pd.DataFrame()
    from sqlalchemy import text
    sql = f"""
    select
      name, phone,
      lead_priority, identifier,
      utm_source, utm_medium, utm_campaign,
      disposition_status, sub_disposition_status,
      remarks,
      {DATE_COLUMN} as created_at
    from {TABLE}
    where {DATE_COLUMN} >= :start and {DATE_COLUMN} < :end
    """
    df = pd.read_sql(text(sql), ENGINE, params={"start": str(start_date), "end": str(end_date)})
    return normalize_leads(df)

# ---------------- Demo data generators (used if DB not set) ----------------
DEVICES = ["android", "ios", "desktop"]  # for biasing synthetic mix
def demo_generate_leads(n_users: int, seed_val=42):
    """Generate synthetic leads using your ATTRS (no PII)."""
    rng = random.Random(seed_val)
    def pick(vals): return rng.choice(vals)

    # Simple synthetic vocab for each attribute
    pri = ["hot", "warm", "cold"]
    ident = ["partner_a", "partner_b", "webform", "agent_x"]
    utm_s = ["google", "facebook", "instagram", "referral", "direct"]
    utm_m = ["cpc", "cpm", "organic", "email", "whatsapp"]
    utm_c = ["mba_brand", "bca_offer", "retarget", "remarketing", "summer_push"]
    disp = ["new", "contacted", "qualified", "in_progress", "closed_won", "closed_lost"]
    subd = ["callback", "not_reachable", "price_issue", "no_interest", "enrolled", "follow_up"]

    rows = []
    for i in range(max(int(n_users), 0)):
        rows.append({
            "name": f"user_{i+1}",
            "phone": f"98{rng.randint(10000000, 99999999)}",
            "lead_priority": pick(pri),
            "identifier": pick(ident),
            "utm_source": pick(utm_s),
            "utm_medium": pick(utm_m),
            "utm_campaign": pick(utm_c),
            "disposition_status": pick(disp),
            "sub_disposition_status": pick(subd),
            "remarks": rng.choice(["", "asked for brochure", "will decide next week", ""]),
        })
    return normalize_leads(pd.DataFrame(rows))

def demo_add_dates(df: pd.DataFrame, start_date, end_date, seed_val=42):
    rng = np.random.default_rng(seed_val)
    days = max((end_date - start_date).days, 1)
    df = df.copy()
    df["created_at"] = [start_date + timedelta(days=int(rng.integers(0, days))) for _ in range(len(df))]
    return df

# ---------------- Stats & comparison helpers ----------------
def cat_dist(df, col):
    if df.empty:
        return pd.DataFrame({col: [], "count": [], "pct": []})
    s = df[col].value_counts(dropna=False)
    total = int(s.sum())
    pct = (s / total) if total else (s * 0.0)
    return pd.concat([s.rename("count"), pct.rename("pct")], axis=1).reset_index().rename(columns={"index": col})

def merge_dist(df_b, df_c, col):
    b = cat_dist(df_b, col).rename(columns={"count": "base_cnt", "pct": "base_pct"})
    c = cat_dist(df_c, col).rename(columns={"count": "curr_cnt", "pct": "curr_pct"})
    m = b.merge(c, on=col, how="outer").fillna(0.0)
    m["delta_pp"] = (m["curr_pct"] - m["base_pct"]) * 100 if "curr_pct" in m and "base_pct" in m else 0.0
    return m

def chi_for_two_dists(m):
    """Safe Pearson chi-square for two categorical distributions."""
    if m is None or m.empty or m.shape[0] < 2:
        return 0.0, 1.0
    obs = m["curr_cnt"].to_numpy(dtype=float)
    exp = m["base_cnt"].to_numpy(dtype=float)
    if obs.sum() <= 0 or exp.sum() <= 0:
        return 0.0, 1.0
    eps = 1e-9
    obs = obs + eps
    exp = exp + eps
    exp = exp * (obs.sum() / exp.sum())  # scale expected to observed total
    try:
        stat, p = chisquare(f_obs=obs, f_exp=exp)
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0

def slice_periods(df, start_date, end_date, freq="W"):
    """Split df into [(label, slice_df)] buckets by D/W/M."""
    if df.empty:
        return []
    d = df.copy()
    if "created_at" not in d.columns:
        return []
    d["created_at"] = pd.to_datetime(d["created_at"])
    periods = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")
    if len(periods) == 0:
        periods = pd.DatetimeIndex([pd.to_datetime(start_date)])
    buckets = []
    for i, s in enumerate(periods):
        e = periods[i + 1] if i + 1 < len(periods) else pd.to_datetime(end_date)
        mask = (d["created_at"] >= s) & (d["created_at"] < e)
        label = f"{s.date()} → {pd.to_datetime(e).date()}"
        buckets.append((label, d.loc[mask]))
    return buckets

def cat_dist_table(dfs_by_period, col):
    frames = []
    for label, d in dfs_by_period:
        s = d[col].value_counts()
        total = int(s.sum())
        row = {("count", v): int(c) for v, c in s.items()}
        row.update({("pct", v): (int(c) / total) if total else 0.0 for v, c in s.items()})
        row[("meta", "period")] = label
        frames.append(pd.Series(row))
    wide = pd.DataFrame(frames).fillna(0.0)
    if not wide.empty:
        wide = wide.reindex(sorted(wide.columns, key=lambda x: (x[0], str(x[1]))), axis=1)
        wide = wide.sort_values(("meta", "period"))
    return wide

def chi_across_periods(dfs_by_period, col):
    values = sorted(set(v for _, d in dfs_by_period for v in d[col].dropna().unique()))
    if len(values) < 2 or len(dfs_by_period) < 2:
        return 0.0, 1.0
    mat = []
    for _, d in dfs_by_period:
        counts = d[col].value_counts()
        mat.append([int(counts.get(v, 0)) for v in values])
    mat = np.array(mat, dtype=float)
    if mat.sum() == 0:
        return 0.0, 1.0
    stat, p, _, _ = chi2_contingency(mat)
    return float(stat), float(p)

# ---- A vs B time series helpers ----
def align_by_index(seriesA, seriesB):
    n = min(len(seriesA), len(seriesB))
    return seriesA[:n], seriesB[:n], [f"P{i+1}" for i in range(n)]

def per_period_compare(seriesA, seriesB, col):
    rows = []
    for idx, ((labelA, dA), (labelB, dB)) in enumerate(zip(seriesA, seriesB), start=1):
        cats = sorted(set(dA[col].dropna().unique()).union(set(dB[col].dropna().unique())))
        cA = dA[col].value_counts()
        cB = dB[col].value_counts()
        totalA, totalB = int(cA.sum()), int(cB.sum())
        # per-period chi-square
        if len(cats) >= 2 and (totalA + totalB) > 0:
            mat = np.array([[int(cA.get(k, 0)) for k in cats],
                            [int(cB.get(k, 0)) for k in cats]], dtype=float)
            try:
                stat, p, _, _ = chi2_contingency(mat)
            except Exception:
                stat, p = 0.0, 1.0
        else:
            stat, p = 0.0, 1.0
        for k in cats:
            a_pct = (int(cA.get(k, 0)) / totalA) if totalA else 0.0
            b_pct = (int(cB.get(k, 0)) / totalB) if totalB else 0.0
            rows.append({
                "period_idx": idx,
                "period_A": labelA,
                "period_B": labelB,
                "attribute": col,
                "category": k,
                "A_pct": a_pct,
                "B_pct": b_pct,
                "delta_pp": (a_pct - b_pct) * 100,
                "chi2_p": p,
                "chi2_stat": stat
            })
    return pd.DataFrame(rows)

def pooled_chi_A_vs_B(seriesA, seriesB, col):
    allA = pd.concat([d for _, d in seriesA], ignore_index=True) if seriesA else pd.DataFrame(columns=[col])
    allB = pd.concat([d for _, d in seriesB], ignore_index=True) if seriesB else pd.DataFrame(columns=[col])
    cats = sorted(set(allA[col].dropna().unique()).union(set(allB[col].dropna().unique())))
    if len(cats) < 2:
        return 0.0, 1.0
    cA = allA[col].value_counts()
    cB = allB[col].value_counts()
    mat = np.array([[int(cA.get(k, 0)) for k in cats],
                    [int(cB.get(k, 0)) for k in cats]], dtype=float)
    if mat.sum() == 0:
        return 0.0, 1.0
    try:
        stat, p, _, _ = chi2_contingency(mat)
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0

# ---------------- UI: Two windows ----------------
def run_two_windows():
    with st.sidebar:
        st.header("Two windows inputs")
        b1 = st.date_input("Baseline start", value=base_start, key="b1")
        b2 = st.date_input("Baseline end (exclusive)", value=base_end, key="b2")
        c1 = st.date_input("Current start", value=curr_start, key="c1")
        c2 = st.date_input("Current end (exclusive)", value=curr_end, key="c2")
        seed = st.number_input("Random seed (demo fallback)", value=42, step=1, key="seed_tw")
        run = st.button("Run comparison", key="run_tw")

    if not run:
        st.info("Pick dates in the sidebar and click **Run comparison**.")
        return

    # Prefer DB; fallback to demo
    if ENGINE:
        df_base = fetch_range(b1, b2)
        df_curr = fetch_range(c1, c2)
        if df_base.empty and df_curr.empty:
            st.warning("DB returned no rows; showing demo data instead.")
    else:
        df_base = pd.DataFrame()
        df_curr = pd.DataFrame()

    if df_base.empty and df_curr.empty:
        random.seed(seed); np.random.seed(seed)
        base_days = max((b2 - b1).days, 1)
        df_base = demo_generate_leads(2000 + base_days * 50, seed_val=seed)
        df_base = demo_add_dates(df_base, b1, b2, seed_val=seed)
        df_curr = demo_generate_leads(len(df_base) + int(0.6 * len(df_base)), seed_val=seed + 1)
        df_curr = demo_add_dates(df_curr, c1, c2, seed_val=seed + 1)

    preview_leads(df_base)

    # Topline & overlap (by name+phone if present, else by index)
    base_ids = df_base["name"].astype(str) + "|" + df_base["phone"].astype(str) if {"name","phone"} <= set(df_base.columns) else pd.Series(df_base.index.astype(str))
    curr_ids = df_curr["name"].astype(str) + "|" + df_curr["phone"].astype(str) if {"name","phone"} <= set(df_curr.columns) else pd.Series(df_curr.index.astype(str))
    base_uu = base_ids.nunique()
    curr_uu = curr_ids.nunique()
    pct = None if base_uu == 0 else (curr_uu - base_uu) / base_uu * 100
    retained = len(set(base_ids) & set(curr_ids))
    new = max(curr_uu - retained, 0); lost = max(base_uu - retained, 0)

    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Baseline unique leads", f"{base_uu:,}")
    c2_.metric("Current unique leads", f"{curr_uu:,}", None if pct is None else f"{pct:+.1f}%")
    c3_.metric("Retained leads", f"{retained:,}")
    st.caption(f"New: {new:,} • Lost: {lost:,}")

    # Per-attribute
    insights = []
    for col in ATTRS:
        st.subheader(f"Attribute: {col}")
        m = merge_dist(df_base, df_curr, col)
        if m.empty or m.shape[0] < 2:
            st.write("Not enough buckets for chi-square.")
            st.dataframe(m, use_container_width=True); continue
        stat, p = chi_for_two_dists(m)
        st.write(f"Chi-square p-value: **{p:.4g}**  (stat={stat:.2f})")
        st.dataframe(m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False), use_container_width=True)
        top = m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False).head(5)[[col,"delta_pp"]]
        if not top.empty: st.bar_chart(top.set_index(col))
        idx = int(np.nanargmax(np.abs(m["delta_pp"].to_numpy())))
        row = m.iloc[idx]; insights.append((col, row[col], float(row["delta_pp"]), p))

    st.subheader("Auto-summary")
    if insights:
        bullets = []
        for col, bucket, dpp, p in sorted(insights, key=lambda x: abs(x[2]), reverse=True):
            conf = "significant" if p < 0.01 else ("likely" if p < 0.05 else "weak")
            bullets.append(f"- **{col}**: '{bucket}' shifted by **{dpp:+.2f} pp** ({conf}, p={p:.3g}).")
        st.markdown("\n".join(bullets))
    else:
        st.write("No shifts detected.")

# ---------------- UI: Time series ----------------
def run_time_series():
    with st.sidebar:
        st.header("Time series inputs")
        overall_start = st.date_input("Overall start", value=base_start, key="ts_start")
        overall_end   = st.date_input("Overall end (exclusive)", value=curr_end, key="ts_end")
        freq = st.selectbox("Granularity", ["D","W","M"], index=1, help="D=day, W=week, M=month", key="ts_freq")
        seed = st.number_input("Random seed (demo fallback)", value=42, step=1, key="seed_ts")
        run_ts = st.button("Run time-series", key="run_ts")

    if not run_ts:
        st.info("Pick an overall range and click **Run time-series**.")
        return

    if ENGINE:
        df_all = fetch_range(overall_start, overall_end)
        if df_all.empty:
            st.warning("DB returned no rows; showing demo data instead.")
    else:
        df_all = pd.DataFrame()

    if df_all.empty:
        random.seed(seed); np.random.seed(seed)
        n_rows = 4000 + max((overall_end - overall_start).days, 1) * 80
        df_all = demo_generate_leads(n_rows, seed_val=seed)
        df_all = demo_add_dates(df_all, overall_start, overall_end, seed_val=seed)

    preview_leads(df_all)

    dfs_by_period = slice_periods(df_all, overall_start, overall_end, freq=freq)

    vol = pd.DataFrame({
        "period": [lbl for lbl, _ in dfs_by_period],
        "leads":  [d.shape[0] for _, d in dfs_by_period]
    })
    st.subheader("Topline leads by period")
    st.dataframe(vol, use_container_width=True)
    if not vol.empty:
        st.line_chart(vol.set_index("period"))

    for col in ATTRS:
        st.subheader(f"Attribute over time: {col}")
        wide = cat_dist_table(dfs_by_period, col)
        stat, p = chi_across_periods(dfs_by_period, col)
        st.write(f"Chi-square across periods p-value: **{p:.4g}**  (stat={stat:.2f})")
        pct_cols = [c for c in wide.columns if c[0] == "pct"]
        pct_view = wide[[("meta","period")] + pct_cols] if not wide.empty and pct_cols else pd.DataFrame()
        st.dataframe(pct_view, use_container_width=True)
        if not pct_view.empty:
            avg_share = pct_view[pct_cols].mean().sort_values(ascending=False)
            top_k = list(avg_share.head(5).index)
            chart_df = pct_view.set_index(("meta","period"))[top_k]
            chart_df.columns = [c[1] for c in chart_df.columns]
            st.line_chart(chart_df)

        if len(dfs_by_period) >= 2:
            last_label, last_df = dfs_by_period[-1]
            prev_label, prev_df = dfs_by_period[-2]
            last = cat_dist(last_df, col).rename(columns={"count":"last_cnt","pct":"last_pct"})
            prev = cat_dist(prev_df, col).rename(columns={"count":"prev_cnt","pct":"prev_pct"})
            delta = prev.merge(last, on=col, how="outer").fillna(0.0)
            delta["delta_pp"] = (delta["last_pct"] - delta["prev_pct"]) * 100
            st.markdown(f"**Last vs Previous:** {prev_label} → {last_label}")
            st.dataframe(delta.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False),
                         use_container_width=True)

# ---------------- UI: Compare two time series (A vs B) ----------------
def run_compare_two_ts():
    with st.sidebar:
        st.header("A vs B time series")
        a_start = st.date_input("A: start", value=base_start, key="a_start")
        a_end   = st.date_input("A: end (exclusive)", value=base_end, key="a_end")
        b_start = st.date_input("B: start", value=curr_start, key="b_start")
        b_end   = st.date_input("B: end (exclusive)", value=curr_end, key="b_end")
        freq = st.selectbox("Granularity", ["D","W","M"], index=1, key="ab_freq")
        align = st.selectbox("Align periods by", ["Index (P1..Pn)","Calendar"], index=0,
                             help="Index: A-P1 vs B-P1, …  Calendar: match only overlapping calendar buckets")
        seed = st.number_input("Random seed (demo fallback)", value=42, step=1, key="seed_ab")
        run_ab = st.button("Run A vs B", key="run_ab")

    if not run_ab:
        st.info("Set two ranges and click **Run A vs B**.")
        return

    # Prefer DB; fallback to demo
    if ENGINE:
        dfA = fetch_range(a_start, a_end)
        dfB = fetch_range(b_start, b_end)
        if dfA.empty and dfB.empty:
            st.warning("DB returned no rows; showing demo data instead.")
    else:
        dfA = pd.DataFrame(); dfB = pd.DataFrame()

    if dfA.empty and dfB.empty:
        random.seed(seed); np.random.seed(seed)
        nA = 3000 + max((a_end - a_start).days, 1) * 60
        nB = 3000 + max((b_end - b_start).days, 1) * 60
        dfA = demo_generate_leads(nA, seed_val=seed)
        dfB = demo_generate_leads(nB, seed_val=seed + 7)
        dfA = demo_add_dates(dfA, a_start, a_end, seed_val=seed)
        dfB = demo_add_dates(dfB, b_start, b_end, seed_val=seed + 7)

    preview_leads(dfA)

    seriesA = slice_periods(dfA, a_start, a_end, freq=freq)
    seriesB = slice_periods(dfB, b_start, b_end, freq=freq)

    if align.startswith("Index"):
        seriesA, seriesB, labels = align_by_index(seriesA, seriesB)
        st.caption(f"Aligned by index. A periods: {len(seriesA)}, B periods: {len(seriesB)}.")
    else:
        # calendar label match
        mapA = {la: d for la, d in seriesA}
        mapB = {lb: d for lb, d in seriesB}
        labels = [lab for lab in mapA.keys() if lab in mapB]
        seriesA = [(lab, mapA[lab]) for lab in labels]
        seriesB = [(lab, mapB[lab]) for lab in labels]
        st.caption(f"Aligned by calendar labels. Matched periods: {len(labels)}.")

    if not seriesA or not seriesB:
        st.warning("No overlapping/alignable periods. Try a different alignment or frequency.")
        return

    vol = pd.DataFrame({
        "period": labels,
        "A_leads": [d.shape[0] for _, d in seriesA],
        "B_leads": [d.shape[0] for _, d in seriesB],
    })
    st.subheader("Topline leads per period (A vs B)")
    st.dataframe(vol, use_container_width=True)
    st.line_chart(vol.set_index("period"))

    for col in ATTRS:
        st.subheader(f"A vs B over time: {col}")
        comp = per_period_compare(seriesA, seriesB, col)
        if comp.empty:
            st.write("No data for this attribute."); continue

        # Show last period’s biggest differences
        last_idx = comp["period_idx"].max()
        last_delta = comp.loc[comp["period_idx"] == last_idx].copy()
        last_delta = last_delta.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False)

        # Trend chart for top categories by average |gap|
        avg_gap = comp.groupby("category")["delta_pp"].apply(lambda s: np.mean(np.abs(s))).sort_values(ascending=False)
        top_cats = list(avg_gap.head(5).index)

        # Plot each top category’s A vs B trend
        if top_cats:
            wide_rows = []
            for i, _ in enumerate(labels, start=1):
                row = {"period": labels[i-1]}
                for k in top_cats:
                    a_pct = comp[(comp.period_idx==i) & (comp.category==k)]["A_pct"]
                    b_pct = comp[(comp.period_idx==i) & (comp.category==k)]["B_pct"]
                    row[f"{k}_A"] = float(a_pct.iloc[0]) if len(a_pct) else 0.0
                    row[f"{k}_B"] = float(b_pct.iloc[0]) if len(b_pct) else 0.0
                wide_rows.append(row)
            wide = pd.DataFrame(wide_rows)
            for k in top_cats:
                sub = wide[["period", f"{k}_A", f"{k}_B"]].set_index("period").rename(
                    columns={f"{k}_A": f"{k} (A)", f"{k}_B": f"{k} (B)"}
                )
                st.line_chart(sub)

        # Per-period significance + pooled
        st.markdown("**Per-period chi-square (A vs B)**")
        per_p = comp[["period_idx","period_A","period_B","chi2_p","chi2_stat"]].drop_duplicates().reset_index(drop=True)
        per_p["period"] = labels[:len(per_p)]
        per_p = per_p[["period","chi2_stat","chi2_p","period_A","period_B"]]
        st.dataframe(per_p, use_container_width=True)

        st.markdown("**Last period — top category gaps (A − B)**")
        st.dataframe(last_delta[["category","A_pct","B_pct","delta_pp"]], use_container_width=True)

        stat_all, p_all = pooled_chi_A_vs_B(seriesA, seriesB, col)
        st.write(f"Pooled chi-square across full range: **p={p_all:.4g}** (stat={stat_all:.2f})")

# ---------------- Router ----------------
if mode == "Two windows":
    run_two_windows()
elif mode == "Time series":
    run_time_series()
else:
    run_compare_two_ts()

st.caption("Setup: add DATABASE_URL, TABLE, DATE_COLUMN in Streamlit Secrets to use your real DB. Otherwise, the app runs on demo data.")