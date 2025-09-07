import math, random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare

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

# ------------ Dummy data generator ------------
DEVICES = ["android", "ios", "desktop"]
COUNTRIES = ["IN", "US", "AE", "SG", "UK"]
CHANNELS = ["ads", "seo", "social", "referral", "other"]

def generate_users(n_users: int, bias=None, id_prefix="B", start_id=1):
    """
    Generate a synthetic user table with categorical attrs.
    bias: dict(attr -> dict(value -> weight)) to shift distributions.
    """
    rng = random.Random(seed)
    bias = bias or {}

    def sample_from(values, weights=None):
        if weights is None:
            return rng.choice(values)
        # weights dict aligned to values
        w = [weights.get(v, 1.0) for v in values]
        # normalize
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

def make_overlap(df_base, overlap_rate=0.5, id_prefix="C"):
    """
    Create a 'current' df with some overlap and drifted distribution.
    """
    rng = np.random.default_rng(seed)
    base_ids = df_base["user_id"].to_numpy()
    n_overlap = int(len(base_ids) * overlap_rate)
    overlap_ids = rng.choice(base_ids, size=n_overlap, replace=False)

    # Bias: e.g., more iOS, more US + ads in current window
    bias = {
        "device": {"ios": 1.6, "android": 0.9, "desktop": 0.8},
        "country": {"US": 1.5, "IN": 0.9, "AE": 1.1, "SG": 1.0, "UK": 1.0},
        "channel": {"ads": 1.7, "seo": 0.9, "social": 0.9, "referral": 1.0, "other": 0.8},
    }

    # Retained users keep their IDs but can change attributes
    retained = generate_users(n_overlap, bias=bias, id_prefix="", start_id=0)
    retained["user_id"] = overlap_ids

    # New users
    n_new = int(len(df_base) * 0.6)  # tweak size for effect
    new_users = generate_users(n_new, bias=bias, id_prefix=id_prefix, start_id=1)

    return pd.concat([retained, new_users], ignore_index=True)

# ------------ Comparison helpers ------------
def cat_dist(df, col):
    s = df[col].value_counts(dropna=False)
    pct = (s / s.sum()).rename("pct")
    return pd.concat([s.rename("count"), pct], axis=1).reset_index().rename(columns={"index": col})

def merge_dist(df_b, df_c, col):
    b = cat_dist(df_b, col).rename(columns={"count":"base_cnt","pct":"base_pct"})
    c = cat_dist(df_c, col).rename(columns={"count":"curr_cnt","pct":"curr_pct"})
    m = b.merge(c, on=col, how="outer").fillna(0.0)
    m["delta_pp"] = (m["curr_pct"] - m["base_pct"]) * 100
    return m

def chi_for(m):
    # add epsilon to avoid zero in expected
    obs = m["curr_cnt"].values + 1e-9
    exp = m["base_cnt"].values + 1e-9
    stat, p = chisquare(obs, f_exp=exp)
    return float(stat), float(p)

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
curr_size = 2000 + curr_days * 50

df_base = generate_users(base_size, id_prefix="B", start_id=1)
df_curr = make_overlap(df_base, overlap_rate=0.55, id_prefix="C")

# ------------ Topline & overlap ------------
base_uu = df_base.user_id.nunique()
curr_uu = df_curr.user_id.nunique()
pct = None if base_uu == 0 else (curr_uu - base_uu) / base_uu * 100

retained = len(set(df_base.user_id).intersection(set(df_curr.user_id)))
new = curr_uu - retained
lost = base_uu - retained

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
    stat, p = chi_for(m)

    # table
    st.write(f"Chi-square p-value: **{p:.4g}**  (stat={stat:.2f})")
    st.dataframe(
        m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False),
        use_container_width=True
    )

    # chart
    top = m.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False).head(5)[[col, "delta_pp"]]
    st.bar_chart(top.set_index(col))

    if len(m):
        row = m.iloc[m["delta_pp"].abs().argmax()]
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