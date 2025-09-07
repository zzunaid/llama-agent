import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare, chi2_contingency

# ---------------- Page setup ----------------
st.set_page_config(page_title="Cohort Compare (Demo + Time Series)", layout="wide")
st.title("Baseline vs Current — Attribute Comparison")
st.caption("Demo uses synthetic data so you can verify charts & insights. Swap generators with SQL when ready.")

# ---------------- Defaults ----------------
today = date.today()
curr_end = today
curr_start = curr_end - timedelta(days=7)
base_end = curr_start
base_start = base_end - timedelta(days=7)

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose analysis mode", ["Two windows", "Time series"])

# ---------------- Dummy universe ----------------
DEVICES = ["android", "ios", "desktop"]
COUNTRIES = ["IN", "US", "AE", "SG", "UK"]
CHANNELS = ["ads", "seo", "social", "referral", "other"]

# ---------------- Data generators (dummy) ----------------
def generate_users(n_users: int, bias=None, id_prefix="B", start_id=1, seed_val=42):
    """Generate synthetic users with categorical attributes."""
    rng = random.Random(seed_val)
    bias = bias or {}

    def pick(values, weights=None):
        if weights is None:
            return rng.choice(values)
        w = [weights.get(v, 1.0) for v in values]
        s = sum(w) if sum(w) else 1.0
        w = [x / s for x in w]
        return rng.choices(values, weights=w, k=1)[0]

    rows = []
    for i in range(start_id, start_id + max(int(n_users), 0)):
        rows.append({
            "user_id": f"{id_prefix}{i}",
            "device":  pick(DEVICES,  bias.get("device")),
            "country": pick(COUNTRIES, bias.get("country")),
            "channel": pick(CHANNELS,  bias.get("channel")),
        })
    return pd.DataFrame(rows, columns=["user_id","device","country","channel"])

def make_overlap(df_base, overlap_rate=0.55, id_prefix="C", seed_val=42):
    """Create a current df with some overlap vs base + shifted distributions."""
    rng = np.random.default_rng(seed_val)
    base_ids = df_base["user_id"].to_numpy() if not df_base.empty else np.array([])
    n_overlap = int(len(base_ids) * overlap_rate)
    n_overlap = min(max(n_overlap, 0), len(base_ids))
    overlap_ids = rng.choice(base_ids, size=n_overlap, replace=False) if len(base_ids) else np.array([])

    bias = {
        "device":  {"ios": 1.6, "android": 0.9, "desktop": 0.8},
        "country": {"US": 1.5, "IN": 0.9, "AE": 1.1, "SG": 1.0, "UK": 1.0},
        "channel": {"ads": 1.7, "seo": 0.9, "social": 0.9, "referral": 1.0, "other": 0.8},
    }

    retained = generate_users(n_overlap, bias=bias, id_prefix="", start_id=0, seed_val=seed_val)
    if n_overlap:
        retained["user_id"] = overlap_ids
    n_new = int(len(df_base) * 0.6)
    new_users = generate_users(n_new, bias=bias, id_prefix=id_prefix, start_id=1, seed_val=seed_val)
    return pd.concat([retained, new_users], ignore_index=True) if (n_overlap + n_new) else pd.DataFrame(
        columns=["user_id","device","country","channel"]
    )

def make_dated_dummy(n, start_date, end_date, seed_val=42):
    """Generate users with created_at between [start_date, end_date)."""
    df = generate_users(n, seed_val=seed_val)
    rng = np.random.default_rng(seed_val)
    days = max((end_date - start_date).days, 1)
    df["created_at"] = [start_date + timedelta(days=int(rng.integers(0, days))) for _ in range(len(df))]
    return df

# ---------------- Stats helpers ----------------
def cat_dist(df, col):
    if df.empty:
        return pd.DataFrame({col: [], "count": [], "pct": []})
    s = df[col].value_counts(dropna=False)
    total = s.sum()
    pct = (s / total).rename("pct") if total else pd.Series([0]*len(s), index=s.index, name="pct")
    return pd.concat([s.rename("count"), pct], axis=1).reset_index().rename(columns={"index": col})

def merge_dist(df_b, df_c, col):
    b = cat_dist(df_b, col).rename(columns={"count":"base_cnt","pct":"base_pct"})
    c = cat_dist(df_c, col).rename(columns={"count":"curr_cnt","pct":"curr_pct"})
    m = b.merge(c, on=col, how="outer").fillna(0.0)
    m["delta_pp"] = (m["curr_pct"] - m["base_pct"]) * 100 if "curr_pct" in m and "base_pct" in m else 0.0
    return m

def chi_for(m):
    """Safe chi-square (Pearson) for two distributions."""
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
    """Split df into [(label, slice_df)] buckets by freq."""
    if df.empty:
        return []
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"])
    periods = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")
    if len(periods) == 0:
        periods = pd.DatetimeIndex([pd.to_datetime(start_date)])
    buckets = []
    for i, s in enumerate(periods):
        e = periods[i+1] if i+1 < len(periods) else pd.to_datetime(end_date)
        mask = (d["created_at"] >= s) & (d["created_at"] < e)
        label = f"{s.date()} → {pd.to_datetime(e).date()}"
        buckets.append((label, d.loc[mask]))
    return buckets

def cat_dist_table(dfs_by_period, col):
    """Wide table with counts & pcts per period for a categorical attribute."""
    frames = []
    for label, d in dfs_by_period:
        s = d[col].value_counts()
        total = int(s.sum())
        row = {("count", v): int(c) for v, c in s.items()}
        row.update({("pct", v): (int(c) / total) if total else 0.0 for v, c in s.items()})
        row[("meta","period")] = label
        frames.append(pd.Series(row))
    wide = pd.DataFrame(frames).fillna(0.0)
    wide = wide.reindex(sorted(wide.columns, key=lambda x:(x[0], str(x[1]))), axis=1)
    return wide.sort_values(("meta","period"))

def chi_across_periods(dfs_by_period, col):
    """Chi-square contingency across multiple periods."""
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

# ---------------- UI: Two windows ----------------
def run_two_windows():
    with st.sidebar:
        st.header("Two windows inputs")
        b1 = st.date_input("Baseline start", value=base_start, key="b1")
        b2 = st.date_input("Baseline end (exclusive)", value=base_end, key="b2")
        c1 = st.date_input("Current start", value=curr_start, key="c1")
        c2 = st.date_input("Current end (exclusive)", value=curr_end, key="c2")
        seed = st.number_input("Random seed", value=42, step=1, key="seed_tw")
        run = st.button("Run comparison", key="run_tw")

    if not run:
        st.info("Pick dates in the sidebar and click **Run comparison**.")
        return

    random.seed(seed); np.random.seed(seed)
    base_days = max((b2 - b1).days, 1)
    base_size = 2000 + base_days * 50
    df_base = generate_users(base_size, id_prefix="B", start_id=1, seed_val=seed)
    df_curr = make_overlap(df_base, overlap_rate=0.55, id_prefix="C", seed_val=seed)

    # Topline & overlap
    base_uu = df_base.user_id.nunique()
    curr_uu = df_curr.user_id.nunique()
    pct = None if base_uu == 0 else (curr_uu - base_uu) / base_uu * 100
    retained = len(set(df_base.user_id) & set(df_curr.user_id))
    new = max(curr_uu - retained, 0); lost = max(base_uu - retained, 0)

    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Baseline unique users", f"{base_uu:,}")
    c2_.metric("Current unique users", f"{curr_uu:,}", None if pct is None else f"{pct:+.1f}%")
    c3_.metric("Retained users", f"{retained:,}")
    st.caption(f"New: {new:,} • Lost: {lost:,}")

    # Per-attribute
    ATTRS = ["device","country","channel"]
    insights = []
    for col in ATTRS:
        st.subheader(f"Attribute: {col}")
        m = merge_dist(df_base, df_curr, col)
        if m.empty or m.shape[0] < 2:
            st.write("Not enough buckets for chi-square.")
            st.dataframe(m, use_container_width=True)
            continue
        stat, p = chi_for(m)
        st.write(f"Chi-square p-value: **{p:.4g}**  (stat={stat:.2f})")
        st.dataframe(m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False), use_container_width=True)
        top = m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False).head(5)[[col,"delta_pp"]]
        if not top.empty: st.bar_chart(top.set_index(col))
        idx = int(np.nanargmax(np.abs(m["delta_pp"].to_numpy())))
        row = m.iloc[idx]
        insights.append((col, row[col], float(row["delta_pp"]), p))

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
        seed = st.number_input("Random seed", value=42, step=1, key="seed_ts")
        run_ts = st.button("Run time-series", key="run_ts")

    if not run_ts:
        st.info("Pick an overall range and click **Run time-series**.")
        return

    random.seed(seed); np.random.seed(seed)
    span_days = max((overall_end - overall_start).days, 1)
    n_rows = 4000 + span_days * 80
    df_all = make_dated_dummy(n_rows, overall_start, overall_end, seed_val=seed)

    dfs_by_period = slice_periods(df_all, overall_start, overall_end, freq=freq)

    # volume trend
    vol = pd.DataFrame({
        "period": [lbl for lbl, _ in dfs_by_period],
        "users":  [d.user_id.nunique() for _, d in dfs_by_period]
    })
    st.subheader("Topline users by period")
    st.dataframe(vol, use_container_width=True)
    if not vol.empty:
        st.line_chart(vol.set_index("period"))

    ATTRS = ["device","country","channel"]
    for col in ATTRS:
        st.subheader(f"Attribute over time: {col}")
        wide = cat_dist_table(dfs_by_period, col)
        stat, p = chi_across_periods(dfs_by_period, col)
        st.write(f"Chi-square across periods p-value: **{p:.4g}**  (stat={stat:.2f})")

        pct_cols = [c for c in wide.columns if c[0] == "pct"]
        pct_view = wide[[("meta","period")] + pct_cols] if pct_cols else pd.DataFrame()
        st.dataframe(pct_view, use_container_width=True)

        if not pct_view.empty:
            avg_share = pct_view[pct_cols].mean().sort_values(ascending=False)
            top_k = list(avg_share.head(5).index)
            chart_df = pct_view.set_index(("meta","period"))[top_k]
            chart_df.columns = [c[1] for c in chart_df.columns]  # flatten
            st.line_chart(chart_df)

        # last vs previous period deltas
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

# ---------------- Router ----------------
if mode == "Two windows":
    run_two_windows()
else:
    run_time_series()

st.caption("Tip: Replace the dummy generators with SQL queries to go live.")