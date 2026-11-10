"""
STEP 2 - DATA CLEANING & TRANSFORMATION
Every action is logged to data/cleaning_log.csv (issue, rows affected, action, rationale)
so the report can show exactly what was fixed and why.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent / "data"
RAW, CLEAN = BASE / "raw", BASE / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)
log = []

def note(ds, issue, n, action, why):
    log.append({"dataset": ds, "issue": issue, "rows_affected": int(n), "action": action, "rationale": why})

# ============================================================== ORDERS
o = pd.read_csv(RAW / "orders_raw.csv")
start = len(o)

# 1. exact duplicates
d = o.duplicated().sum(); o = o.drop_duplicates()
note("orders", "Exact duplicate rows", d, "Dropped", "Double-booked orders inflate revenue and volume.")
d = o.duplicated("order_id").sum(); o = o.drop_duplicates("order_id")
note("orders", "Duplicate order_id", d, "Kept first", "order_id is the primary key and must be unique.")

# 2. text standardisation
cmap = {"usa": "United States", "u.s.a.": "United States", "us": "United States", "united states": "United States",
        "uk": "United Kingdom", "u.k.": "United Kingdom", "england": "United Kingdom", "united kingdom": "United Kingdom",
        "united arab emirates": "UAE", "u.a.e": "UAE", "uae": "UAE", "deutschland": "Germany", "germany": "Germany",
        "india": "India"}
before = o["country"].copy()
key = o["country"].str.strip().str.lower()
o["country"] = key.map(cmap).fillna(o["country"].str.strip())
note("orders", "Inconsistent country names (USA, U.K., Deutschland...)", (before != o["country"]).sum(),
     "Mapped to one standard name", "Otherwise the same market is split into several groups in country analysis.")

before = o["tile_category"].copy()
cat_std = {c.upper(): c for c in o["tile_category"].dropna().str.strip().unique() if not c.strip().isupper()}
o["tile_category"] = o["tile_category"].str.strip().apply(lambda x: cat_std.get(x.upper(), x))
note("orders", "Tile category in UPPER CASE with trailing spaces", (before != o["tile_category"]).sum(),
     "Trimmed and mapped to proper case", "Ensures 6 categories, not 12.")
before = o["customer_type"].copy()
ct_std = {c.lower(): c for c in o["customer_type"].unique() if c != c.lower()}
o["customer_type"] = o["customer_type"].map(lambda x: ct_std.get(x.lower(), x))
note("orders", "customer_type in lower case", (before != o["customer_type"]).sum(), "Mapped to proper case", "Consistent segment labels.")

# 3. dates: two formats (YYYY-MM-DD and DD/MM/YYYY)
iso = pd.to_datetime(o["order_date"], format="%Y-%m-%d", errors="coerce")
dmy = pd.to_datetime(o["order_date"], format="%d/%m/%Y", errors="coerce")
note("orders", "Mixed date formats (DD/MM/YYYY)", iso.isna().sum(), "Parsed both formats explicitly",
     "Auto-parsing would swap day and month for dates like 03/04/2025.")
o["order_date"] = iso.fillna(dmy)

# 4. numeric columns stored as text ("USD 8.01")
bad = o["unit_price_usd"].astype(str).str.contains("USD", na=False).sum()
o["unit_price_usd"] = pd.to_numeric(o["unit_price_usd"].astype(str).str.replace("USD", "").str.strip(), errors="coerce")
note("orders", "unit_price_usd contains text prefix 'USD'", bad, "Stripped prefix, cast to float", "Price must be numeric to compute revenue.")
o["quantity_sqm"] = pd.to_numeric(o["quantity_sqm"], errors="coerce")
o["customer_rating"] = pd.to_numeric(o["customer_rating"], errors="coerce")

# 5. impossible values
n = (o["quantity_sqm"] <= 0).sum(); o.loc[o["quantity_sqm"] <= 0, "quantity_sqm"] = o["quantity_sqm"].abs()
note("orders", "Negative quantity", n, "Converted to absolute value", "Sign-entry error; order exists (has price/delivery).")
n = (o["customer_rating"] > 5).sum(); o.loc[o["customer_rating"] > 5, "customer_rating"] = np.nan
note("orders", "Rating outside 1-5 scale", n, "Set to missing, then imputed", "A 7 on a 5-point scale is a data entry error.")
n = (o["discount_pct"] > 40).sum(); o.loc[o["discount_pct"] > 40, "discount_pct"] = np.nan
note("orders", "Discount > 40% (policy cap)", n, "Set to missing, then imputed", "Company policy caps discounts; 95% is a typo.")

# 6. extreme outliers in quantity: cap at segment-wise 99.5th percentile (keep the order, limit leverage)
cap = o.groupby("customer_type")["quantity_sqm"].transform(lambda s: s.quantile(0.995))
n = (o["quantity_sqm"] > cap).sum(); o["quantity_sqm"] = np.minimum(o["quantity_sqm"], cap)
note("orders", "Extreme quantity outliers (x40 entry error)", n, "Capped at 99.5th percentile within customer type",
     "Caps per segment because a distributor legitimately buys far more than a homeowner.")

# 7. missing values
miss = o.isna().sum()
o["unit_price_usd"] = o["unit_price_usd"].fillna(o.groupby(["tile_category", "country"])["unit_price_usd"].transform("median"))
o["unit_price_usd"] = o["unit_price_usd"].fillna(o.groupby("tile_category")["unit_price_usd"].transform("median"))
note("orders", "Missing unit price", miss["unit_price_usd"], "Median by category x country", "Price depends mostly on product and market.")
o["discount_pct"] = o["discount_pct"].fillna(o.groupby("customer_type")["discount_pct"].transform("median"))
note("orders", "Missing discount", miss["discount_pct"], "Median by customer type", "Discount policy is set per customer segment.")
o["customer_rating"] = o["customer_rating"].fillna(o["customer_rating"].median())
note("orders", "Missing rating", miss["customer_rating"], "Overall median", "Median is robust to the skewed rating distribution.")
o["shipping_mode"] = np.where(o["shipping_mode"].isna() & (o["country"] == "India"), "Road", o["shipping_mode"])
o["shipping_mode"] = o["shipping_mode"].fillna(o["shipping_mode"].mode()[0])
note("orders", "Missing shipping mode", miss["shipping_mode"], "Road if India, else mode (Sea Freight)", "Domestic orders go by road; exports mostly by sea.")
o["dashboard_time_min"] = np.where(o["dashboard_time_min"].isna() & (o["used_7d_view"] + o["used_room_visualizer"] == 0), 0, o["dashboard_time_min"])
o["dashboard_time_min"] = o["dashboard_time_min"].fillna(o.groupby("used_room_visualizer")["dashboard_time_min"].transform("median"))
note("orders", "Missing dashboard time", miss["dashboard_time_min"], "Median by visualizer usage", "Visualizer users spend much longer on the dashboard.")
n = o["actual_lead_days"].isna().sum(); o = o.dropna(subset=["actual_lead_days"])
note("orders", "Missing actual lead time", n, "Rows dropped", "Delivery is the target of the logistics analysis; imputing it would invent the outcome.")

# 8. FEATURE ENGINEERING (transformation)
o["year"] = o["order_date"].dt.year
o["month"] = o["order_date"].dt.to_period("M").astype(str)
o["quarter"] = o["order_date"].dt.to_period("Q").astype(str)
o["gross_revenue_usd"] = (o["quantity_sqm"] * o["unit_price_usd"]).round(2)
o["net_revenue_usd"] = (o["gross_revenue_usd"] * (1 - o["discount_pct"] / 100)).round(2)
o["cogs_usd"] = (o["quantity_sqm"] * o["unit_cost_usd"]).round(2)
o["gross_margin_usd"] = (o["net_revenue_usd"] - o["cogs_usd"]).round(2)
o["margin_pct"] = (o["gross_margin_usd"] / o["net_revenue_usd"] * 100).round(2)
o["delay_days"] = (o["actual_lead_days"] - o["promised_lead_days"]).round(1)
o["on_time"] = (o["delay_days"] <= 0).astype(int)
o["is_export"] = (o["country"] != "India").astype(int)
o["digital_engagement"] = np.select(
    [o["used_room_visualizer"] == 1, o["used_7d_view"] == 1], ["7D + Room Visualizer", "7D View only"], "No digital tool")
o["is_repeat_order"] = (o["order_seq"] > 1).astype(int)
o["order_size_band"] = pd.cut(o["net_revenue_usd"], [0, 2000, 10000, 30000, np.inf], labels=["Small <2k", "Medium 2-10k", "Large 10-30k", "Bulk >30k"])
o["satisfied"] = (o["customer_rating"] >= 4).astype(int)
# customer lifetime aggregates (RFM-lite)
snap = o["order_date"].max() + pd.Timedelta(days=1)
cust = o.groupby("customer_id").agg(cust_orders=("order_id", "count"), cust_revenue=("net_revenue_usd", "sum"),
                                     last_order=("order_date", "max"))
cust["recency_days"] = (snap - cust["last_order"]).dt.days
o = o.merge(cust[["cust_orders", "cust_revenue", "recency_days"]], left_on="customer_id", right_index=True)
note("orders", "Feature engineering", len(o), "Added revenue, margin, delay, on_time, digital_engagement, repeat, size band, RFM",
     "Raw columns don't directly answer business questions; derived KPIs do.")
o = o.sort_values("order_date").reset_index(drop=True)
o.to_csv(CLEAN / "orders_clean.csv", index=False)

# ============================================================== PRODUCTION
p = pd.read_csv(RAW / "production_raw.csv")
d = p.duplicated().sum(); p = p.drop_duplicates()
note("production", "Duplicate batch rows", d, "Dropped", "Double counts output.")
before = p["automation_level"].copy()
p["automation_level"] = p["automation_level"].str.strip().str.lower().str.replace("-", " ").map(
    {"fully automated": "Fully Automated", "semi automated": "Semi-Automated", "manual": "Manual"})
note("production", "automation_level spelling variants", (before != p["automation_level"]).sum(), "Standardised to 3 levels", "Core grouping variable.")
n = (p["kiln_temp_c"] > 2000).sum(); p.loc[p["kiln_temp_c"] > 2000, "kiln_temp_c"] = np.nan
note("production", "Kiln temp 11800 C (sensor glitch, extra zero)", n, "Set to missing then imputed", "Physically impossible for a ceramic kiln.")
n = (p["defect_rate_pct"] < 0).sum(); p.loc[p["defect_rate_pct"] < 0, "defect_rate_pct"] = np.nan
note("production", "Negative defect rate", n, "Set to missing then imputed", "A rate cannot be negative.")
for c in ["kiln_temp_c", "defect_rate_pct", "energy_kwh"]:
    m = p[c].isna().sum()
    p[c] = p[c].fillna(p.groupby(["automation_level", "tile_category"])[c].transform("median"))
    note("production", f"Missing {c}", m, "Median by automation level x category", "Values differ strongly by line type and product.")
p["production_date"] = pd.to_datetime(p["production_date"])
p["month"] = p["production_date"].dt.to_period("M").astype(str)
p["good_output_sqm"] = (p["output_sqm"] * (1 - p["defect_rate_pct"] / 100)).round(0)
p["energy_per_sqm"] = (p["energy_kwh"] / p["output_sqm"]).round(3)
p["sqm_per_operator"] = (p["output_sqm"] / p["operators_on_shift"]).round(1)
p["oee_proxy_pct"] = ((1 - p["downtime_hours"] / 24) * (1 - p["defect_rate_pct"] / 100) * 100).round(2)
note("production", "Feature engineering", len(p), "Added good output, energy/sqm, sqm/operator, OEE proxy", "Standard manufacturing KPIs.")
p.to_csv(CLEAN / "production_clean.csv", index=False)

# ============================================================== SESSIONS
s = pd.read_csv(RAW / "sessions_raw.csv")
d = s.duplicated().sum(); s = s.drop_duplicates()
note("sessions", "Duplicate sessions", d, "Dropped", "Tracking double-fire.")
n = (s["session_duration_min"] == 0).sum(); s = s[s["session_duration_min"] != 0]
note("sessions", "Zero-duration sessions (bots/bounces)", n, "Dropped", "Would distort conversion rates.")
for c in ["device", "traffic_source"]:
    m = s[c].isna().sum(); s[c] = s[c].fillna("Unknown")
    note("sessions", f"Missing {c}", m, "Labelled 'Unknown'", "Keeps the session for funnel totals without guessing.")
m = s["session_duration_min"].isna().sum()
s["session_duration_min"] = s["session_duration_min"].fillna(s.groupby(["viewed_7d", "uploaded_room_photo"])["session_duration_min"].transform("median"))
note("sessions", "Missing duration", m, "Median by tool usage", "Tool users have longer sessions.")
s["session_datetime"] = pd.to_datetime(s["session_datetime"])
s["month"] = s["session_datetime"].dt.to_period("M").astype(str)
s["journey"] = np.select([s["uploaded_room_photo"] == 1, s["viewed_7d"] == 1], ["7D + Room upload", "7D only"], "Browse only")
s.to_csv(CLEAN / "sessions_clean.csv", index=False)

pd.DataFrame(log).to_csv(BASE / "cleaning_log.csv", index=False)
print(f"orders {start} -> {len(o)} | production -> {len(p)} | sessions -> {len(s)}")
print(pd.DataFrame(log)[["dataset", "issue", "rows_affected"]].to_string())
