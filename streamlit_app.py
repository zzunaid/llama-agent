import math, random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# ------------ Page setup ------------
st.set_page_config(page_title="Cohort Compare (Dummy Data)", layout="wide")
st.title("Baseline vs Current — Attribute Comparison (Demo)")
st.caption("This demo generates synthetic data so you can verify charts & insights. Replace the generator with your SQL later.")

# ------------ Defaults (last 7d vs previous 7d) ------------
today = date.today()
curr_end = today
curr_start = curr_end - timedelta(days=7)
base_end = curr_start
base_start = base_end - timedelta(days=7)

with st.sidebar:
    st.header("Pick your date ranges")
    b1 = st.date_input("Baseline start", value=base_start)
    b2 = st.date_input("Baseline end (exclusive)", value=base_end)
    c1 = st.date_input("Current start", value=curr_start)
    c2 = st.date_input("Current end (exclusive)", value=curr_end)
    seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)
    run = st.button("Run comparison")

# ------------ Dummy data universe ------------
DEVICES = ["android", "ios", "desktop"]
COUNTRIES = ["IN", "US", "AE", "SG", "UK"]
CHANNELS = ["ads", "seo", "social", "referral", "other"]

# ------------ Dummy data generator ------------
def generate_users(n_users: int, bias=None, id_prefix="B", start_id=1, seed_val=42):
    """
    Generate a synthetic user table with categorical attrs.
    bias: dict(attr -> dict(value -> weight)) to shift distributions.
    """
    rng = random.Random(seed_val)
    bias = bias or {}

    def sample_from(values, weights=None):
        if weights is None:
            return rng.choice(values)
        w = [weights.get(v, 1.0) for v in values]
        s = sum(w)
        w = [x / s for x in w]
        return rng.choices(values, weights=w, k=1)[0]

    rows = []
    for i in range(start_id, start_id + n_users):
        device = sample_from(DEVICES, bias.get("device"))
        country = sample_from(COUNTRIES, bias.get("country"))
        channel = sample_from(CHANNELS, bias.get("channel"))
        rows.append(
            {"user_id": f"{id_prefix}{i}", "device": device, "country": country, "channel": channel}
        )
    return pd.DataFrame(rows)

def make_overlap(df_base, overlap_rate=0.5, id_prefix="C", seed_val=42):
    """
    Create a 'current' df with some overlap and drifted distribution.
    """
    rng = np.random.default_rng(seed_val)
    base_ids = df_base["user_id"].to_numpy()
    n_overlap = int(len(base_ids) * overlap_rate)
    if n_overlap < 0:
        n_overlap = 0
    if n_overlap > len(base_ids):
        n_overlap = len(base_ids)
    overlap_ids = rng.choice(base_ids, size=n_overlap, replace=False) if len(base_ids) else np.array([])

    # Bias: e.g., more iOS, more US + ads in current window
    bias = {
        "device": {"ios": 1.6, "android": 0.9, "desktop": 0.8},
        "country": {"US": 1.5, "IN": 0.9, "AE": 1.1, "SG": 1.0, "UK": 1.0},
        "channel": {"ads": 1.7, "seo": 0.9, "social": 0.9, "referral": 1.0, "other": 0.8},
    }

    # Retained users keep their IDs but can change attributes
    retained = generate_users(n_overlap, bias=bias, id_prefix="", start_id=0, seed_val=seed_val)
    if n_overlap:
        retained["user_id"] = overlap_ids

    # New users
    n_new = int(max(len(df_base) * 0.6, 0))
    new_users = generate_users(n_new, bias=bias, id_prefix=id_prefix, start_id=1, seed_val=seed_val)

    return pd.concat([retained, new_users], ignore_index=True) if (n_overlap + n_new) else pd.DataFrame(
        columns=["user_id","device","country","channel"]
    )

# ------------ Comparison helpers ------------
def cat_dist(df, col):
    if df.empty:
        return pd.DataFrame({col: [], "count": [], "pct": []})
    s = df[col].value_counts(dropna=False)
    total = s.sum()
    if total == 0:
        return pd.DataFrame({col: s.index, "count": s.values, "pct": [0]*len(s)})
    pct = (s / total).rename("pct")
    return pd.concat([s.rename("count"), pct], axis=1).reset_index().rename(columns={"index": col})

def merge_dist(df_b, df_c, col):
    b = cat_dist(df_b, col).rename(columns={"count":"base_cnt","pct":"base_pct"})
    c = cat_dist(df_c, col).rename(columns={"count":"curr_cnt","pct":"curr_pct"})
    m = b.merge(c, on=col, how="outer").fillna(0.0)
    if "curr_pct" not in m or "base_pct" not in m:
        m["delta_pp"] = 0.0
    else:
        m["delta_pp"] = (m["curr_pct"] - m["base_pct"]) * 100
    return m

def chi_for(m):
    """
    Safe Chi-square for categorical drift.
    - Requires ≥ 2 buckets.
    - Scales expected to observed total.
    - Handles zero totals gracefully.
    """
    if m is None or m.empty or m.shape[0] < 2:
        return 0.0, 1.0

    obs = m["curr_cnt"].to_numpy(dtype=float)
    exp = m["base_cnt"].to_numpy(dtype=float)

    # guard: no data
    if obs.sum() <= 0 or exp.sum() <= 0:
        return 0.0, 1.0

    eps = 1e-9
    obs = obs + eps
    exp = exp + eps

    # scale expected to observed total (required by scipy.stats.chisquare)
    exp = exp * (obs.sum() / exp.sum())

    try:
        from scipy.stats import chisquare
        stat, p = chisquare(f_obs=obs, f_exp=exp)
        return float(stat), float(p)
    except Exception:
        # any numerical issues -> neutral result
        return 0.0, 1.0

# ------------ UI run ------------
if not run:
    st.info("Click **Run comparison** in the sidebar to generate demo data and see results.")
    st.stop()

# Generate baseline & current (dummy)
random.seed(seed)
np.random.seed(seed)

# Size scales with date span just for realism
base_days = max((b2 - b1).days, 1)
curr_days = max((c2 - c1).days, 1)
base_size = 2000 + base_days * 50
# Start with same baseline size; 'make_overlap' will add new users
df_base = generate_users(base_size, id_prefix="B", start_id=1, seed_val=seed)

df_curr = make_overlap(df_base, overlap_rate=0.55, id_prefix="C", seed_val=seed)

# ------------ Topline & overlap ------------
base_uu = df_base.user_id.nunique() if not df_base.empty else 0
curr_uu = df_curr.user_id.nunique() if not df_curr.empty else 0
pct = None if base_uu == 0 else (curr_uu - base_uu) / base_uu * 100

retained = len(set(df_base.user_id).intersection(set(df_curr.user_id))) if base_uu and curr_uu else 0
new = max(curr_uu - retained, 0)
lost = max(base_uu - retained, 0)

c1_, c2_, c3_ = st.columns(3)
c1_.metric("Baseline unique users", f"{base_uu:,}")
c2_.metric("Current unique users", f"{curr_uu:,}", None if pct is None else f"{pct:+.1f}%")
c3_.metric("Retained users", f"{retained:,}")
st.caption(f"New: {new:,} • Lost: {lost:,}")

# ------------ Per-attribute comparisons ------------
ATTRS = ["device", "country", "channel"]
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
    st.dataframe(
        m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False),
        use_container_width=True
    )

    # chart
    try:
        top = m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False).head(5)[[col, "delta_pp"]]
        if not top.empty:
            st.bar_chart(top.set_index(col))
    except Exception:
        pass

    # for narrative
    if len(m):
        idx = int(np.nanargmax(np.abs(m["delta_pp"].to_numpy())))
        row = m.iloc[idx]
        insights.append((col, row[col], float(row["delta_pp"]), p))

# ------------ Narrative summary ------------
st.subheader("Auto-summary")
if insights:
    bullets = []
    for col, bucket, dpp, p in sorted(insights, key=lambda x: abs(x[2]), reverse=True):
        conf = "significant" if p < 0.01 else ("likely" if p < 0.05 else "weak")
        bullets.append(f"- **{col}**: '{bucket}' shifted by **{dpp:+.2f} pp** ({conf}, p={p:.3g}).")
    st.markdown("\n".join(bullets))
else:
    st.write("No shifts detected.")

st.caption("Tip: Replace the dummy generator with your SQL pulls to go live.")