"""
STEP 1 - SYNTHETIC DATA GENERATION
Business: "TerraLuxe Tiles" - an automated tile manufacturing unit (India) that exports worldwide.
Differentiators to validate with data:
  (1) 7D product viewer on the customer dashboard
  (2) AI Room Visualizer (customer uploads a photo of home/office and sees tiles applied)
  (3) Automated production lines (less defects, less downtime)
  (4) Timely, tracked delivery to export customers

Three linked datasets are created:
  orders_raw.csv      - one row per customer order (sales + delivery + satisfaction)
  production_raw.csv  - one row per production batch (operations / automation)
  sessions_raw.csv    - one row per dashboard visit (digital funnel: 7D view, room upload, quote, order)

Realistic "dirt" is injected on purpose (missing values, duplicates, typos, mixed formats,
impossible values, outliers) so that Step 2 (cleaning) has real work to do.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- master lists
countries = {  # country: (region, share, transit_days_sea, price_index)
    "India": ("South Asia", 0.14, 4, 0.85), "United States": ("North America", 0.13, 32, 1.30),
    "United Kingdom": ("Europe", 0.08, 26, 1.20), "Germany": ("Europe", 0.07, 27, 1.18),
    "UAE": ("Middle East", 0.10, 7, 1.05), "Saudi Arabia": ("Middle East", 0.07, 9, 1.05),
    "Australia": ("Oceania", 0.06, 22, 1.22), "Canada": ("North America", 0.05, 35, 1.25),
    "Kenya": ("Africa", 0.05, 14, 0.90), "Nigeria": ("Africa", 0.05, 18, 0.90),
    "Vietnam": ("South-East Asia", 0.05, 12, 0.92), "Indonesia": ("South-East Asia", 0.05, 13, 0.92),
    "Brazil": ("South America", 0.05, 34, 1.00), "Mexico": ("North America", 0.05, 30, 1.02),
}
categories = {  # category: (base USD per sqm, cost ratio, base defect %)
    "Ceramic Wall": (7.5, 0.58, 2.0), "Ceramic Floor": (8.5, 0.57, 2.3),
    "Vitrified (GVT)": (13.0, 0.55, 2.8), "Porcelain Slab": (24.0, 0.52, 3.5),
    "Outdoor / Anti-skid": (11.0, 0.56, 2.5), "Designer Mosaic": (32.0, 0.48, 4.0),
}
finishes = ["Glossy", "Matte", "Satin", "Rustic", "Carving"]
sizes = ["300x300", "600x600", "600x1200", "800x1600", "1200x2400"]
cust_types = {"Homeowner": 0.30, "Builder/Contractor": 0.22, "Architect/Interior Designer": 0.18,
              "Distributor/Dealer": 0.18, "Hotel/Commercial": 0.12}
space_types = ["Living Room", "Kitchen", "Bathroom", "Bedroom", "Office", "Hotel Lobby", "Outdoor/Patio", "Retail Store"]
channels = ["Website Dashboard", "Mobile App", "Distributor", "Trade Fair", "Sales Rep / Email"]

# ---------------------------------------------------------------- production batches
N_BATCH = 1800
dates_b = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N_BATCH), unit="D")
lines = rng.choice(["L1", "L2", "L3", "L4", "L5", "L6"], N_BATCH)
# Automation is rolled out over time: L1-L2 fully automated from start, L3-L4 upgraded mid 2025
auto = []
for d, l in zip(dates_b, lines):
    if l in ("L1", "L2"):
        auto.append("Fully Automated")
    elif l in ("L3", "L4"):
        auto.append("Fully Automated" if d >= pd.Timestamp("2025-07-01") else "Semi-Automated")
    elif l == "L5":
        auto.append("Semi-Automated")
    else:
        auto.append("Manual")
auto = np.array(auto)
cat_b = rng.choice(list(categories), N_BATCH, p=[.2, .2, .25, .13, .12, .10])
auto_eff = {"Fully Automated": 0.45, "Semi-Automated": 0.80, "Manual": 1.25}
kiln = rng.normal(1180, 12, N_BATCH)
defect = np.array([categories[c][2] for c in cat_b]) * np.array([auto_eff[a] for a in auto]) \
    + 0.04 * np.abs(kiln - 1180) + rng.normal(0, 0.4, N_BATCH)
defect = np.clip(defect, 0.2, None)
output = np.array([{"Fully Automated": 5200, "Semi-Automated": 4100, "Manual": 3000}[a] for a in auto]) \
    * rng.normal(1, 0.08, N_BATCH)
downtime = np.clip(np.array([{"Fully Automated": 0.8, "Semi-Automated": 1.9, "Manual": 3.4}[a] for a in auto])
                   + rng.exponential(0.6, N_BATCH), 0, None)
output = output * (1 - downtime / 30)
energy = output * rng.normal(1.9, 0.12, N_BATCH) * np.where(auto == "Fully Automated", 0.88, 1.0)
cost = np.array([categories[c][0] * categories[c][1] for c in cat_b]) \
    * np.where(auto == "Fully Automated", 0.90, np.where(auto == "Manual", 1.10, 1.0)) * rng.normal(1, 0.04, N_BATCH)

prod = pd.DataFrame({
    "batch_id": [f"B{100000+i}" for i in range(N_BATCH)],
    "production_date": dates_b.strftime("%Y-%m-%d"),
    "line_id": lines, "automation_level": auto, "tile_category": cat_b,
    "kiln_temp_c": kiln.round(1), "output_sqm": output.round(0), "defect_rate_pct": defect.round(2),
    "downtime_hours": downtime.round(2), "energy_kwh": energy.round(0), "cost_per_sqm_usd": cost.round(2),
    "operators_on_shift": np.array([{"Fully Automated": 4, "Semi-Automated": 9, "Manual": 16}[a] for a in auto])
                          + rng.integers(-1, 3, N_BATCH),
})
prod = prod.sort_values("production_date").reset_index(drop=True)
# line-level defect by month for linking to orders
prod["_month"] = prod["production_date"].str[:7]

# ---------------------------------------------------------------- orders
N_ORD = 6000
n_cust = 2200
cust_ids = [f"C{10000+i}" for i in range(n_cust)]
cust_country = rng.choice(list(countries), n_cust, p=np.array([v[1] for v in countries.values()]) / sum(v[1] for v in countries.values()))
cust_type = rng.choice(list(cust_types), n_cust, p=list(cust_types.values()))
cust_digital = rng.random(n_cust)  # propensity to use digital tools
cmap = dict(zip(cust_ids, zip(cust_country, cust_type, cust_digital)))

o_cust = rng.choice(cust_ids, N_ORD, p=(w := rng.pareto(2.5, n_cust) + 1) / w.sum())
o_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N_ORD), unit="D")
# seasonality: more orders Oct-Mar (construction season in many markets) -> resample some
season_w = np.where(o_date.month.isin([10, 11, 12, 1, 2, 3]), 1.25, 0.85)
keep = rng.random(N_ORD) < season_w / season_w.max()
o_date = pd.DatetimeIndex(np.where(keep, o_date, o_date + pd.to_timedelta(rng.integers(90, 150, N_ORD), unit="D")))
o_date = pd.DatetimeIndex([d if d <= pd.Timestamp("2025-12-31") else d - pd.Timedelta(days=365) for d in o_date])

rows = []
for i in range(N_ORD):
    cid = o_cust[i]; country, ctype, dig = cmap[cid]
    region, _, sea_days, pidx = countries[country]
    d = o_date[i]
    # digital adoption grows over time (dashboard launched Jan-2024, AR visualizer Jul-2024)
    t = (d - pd.Timestamp("2024-01-01")).days / 730
    p7d = np.clip(0.25 + 0.45 * t + 0.35 * (dig - 0.5) + (0.1 if ctype in ("Homeowner", "Architect/Interior Designer") else -0.05), 0.02, 0.95)
    used7d = rng.random() < p7d
    room_ok = d >= pd.Timestamp("2024-07-01")
    proom = np.clip(p7d * 0.75 + (0.15 if ctype == "Homeowner" else 0), 0, 0.95) if room_ok else 0
    usedroom = rng.random() < proom
    channel = rng.choice(channels, p=[.38, .17, .2, .1, .15]) if (used7d or usedroom) else rng.choice(channels, p=[.15, .07, .4, .16, .22])
    cat = rng.choice(list(categories), p=[.18, .2, .25, .13, .12, .12] if ctype != "Hotel/Commercial" else [.08, .12, .25, .3, .1, .15])
    base_price, cost_ratio, _ = categories[cat]
    qty_scale = {"Homeowner": 90, "Builder/Contractor": 1400, "Architect/Interior Designer": 380,
                 "Distributor/Dealer": 2600, "Hotel/Commercial": 1800}[ctype]
    qty = rng.lognormal(np.log(qty_scale), 0.55)
    # digital tools -> bigger baskets (upsell to premium) and more confidence
    qty *= (1.12 if used7d else 1) * (1.18 if usedroom else 1)
    unit_price = base_price * pidx * rng.normal(1, 0.07) * (1.05 if usedroom else 1)
    disc = np.clip(rng.normal({"Distributor/Dealer": 12, "Builder/Contractor": 9, "Hotel/Commercial": 8}.get(ctype, 4), 2.5), 0, 25)
    ship = "Road" if country == "India" else rng.choice(["Sea Freight", "Air Freight"], p=[.9, .1])
    if ship == "Air Freight": transit = max(2, sea_days // 6)
    elif ship == "Road": transit = 4
    else: transit = sea_days
    automated_line = rng.random() < (0.45 + 0.35 * t)  # share of orders produced on fully automated lines
    prod_days = rng.integers(5, 9) if automated_line else rng.integers(8, 15)
    promised = prod_days + transit + 5
    delay = rng.gamma(1.4, 2.0) * (0.55 if automated_line else 1.0) \
        + (rng.gamma(2, 3) if (ship == "Sea Freight" and rng.random() < 0.12) else 0) \
        + (3 if d.month in (12, 1) else 0) - 2.5
    actual = max(promised - 3, promised + delay)
    damage = np.clip(rng.normal(1.6 if ship == "Sea Freight" else 0.9, 0.6) * (0.75 if automated_line else 1), 0, None)
    late = actual - promised
    expect_mismatch = rng.random() < (0.05 if usedroom else 0.09 if used7d else 0.16)
    returned = expect_mismatch or (damage > 3.2)
    rating = 4.4 - 0.11 * max(late, 0) - 0.9 * expect_mismatch + 0.25 * usedroom + 0.12 * used7d + rng.normal(0, 0.45)
    rating = float(np.clip(round(rating * 2) / 2, 1, 5))
    rows.append([f"ORD{200000+i}", d, cid, country, region, ctype, channel, cat,
                 rng.choice(finishes), rng.choice(sizes, p=[.1, .35, .3, .15, .1]), rng.choice(space_types),
                 round(qty, 1), round(unit_price, 2), round(disc, 1), round(cost_ratio * base_price * rng.normal(1, .03), 2),
                 int(used7d), int(usedroom), int(rng.poisson(3 + 6 * usedroom)) if (used7d or usedroom) else 0,
                 round(rng.gamma(2, 4) * (1.8 if usedroom else 1.3 if used7d else 0.6), 1),
                 "Fully Automated" if automated_line else "Semi/Manual", ship,
                 int(promised), round(actual, 1), round(damage, 2), int(returned), rating])

orders = pd.DataFrame(rows, columns=[
    "order_id", "order_date", "customer_id", "country", "region", "customer_type", "sales_channel", "tile_category",
    "finish", "size_mm", "space_type", "quantity_sqm", "unit_price_usd", "discount_pct", "unit_cost_usd",
    "used_7d_view", "used_room_visualizer", "visualizer_renders", "dashboard_time_min",
    "production_line_type", "shipping_mode", "promised_lead_days", "actual_lead_days", "damage_pct", "returned",
    "customer_rating"])
orders = orders.sort_values("order_date").reset_index(drop=True)
# churn mechanism: after a bad experience (>5 days late, return, or rating <= 3) a customer
# keeps ordering with only 45% probability per later order -> experience drives retention
keep = np.ones(len(orders), bool)
for cid, idx in orders.groupby("customer_id").groups.items():
    hurt = False
    for k in idx:
        if hurt and rng.random() > 0.45:
            keep[k] = False; continue
        r = orders.loc[k]
        if (r.actual_lead_days - r.promised_lead_days > 5) or r.returned == 1 or r.customer_rating <= 3:
            hurt = True
orders = orders[keep].reset_index(drop=True)
# repeat purchase flag: customer ordered again later
orders["order_seq"] = orders.groupby("customer_id").cumcount() + 1
orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")

# ---------------------------------------------------------------- dashboard sessions (digital funnel)
N_S = 12000
s_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N_S), unit="D")
s_country = rng.choice(list(countries), N_S, p=np.array([v[1] for v in countries.values()]) / sum(v[1] for v in countries.values()))
device = rng.choice(["Desktop", "Mobile", "Tablet"], N_S, p=[.45, .45, .10])
source = rng.choice(["Google Search", "Instagram", "LinkedIn", "Referral", "Direct", "Trade Fair QR"], N_S, p=[.3, .22, .12, .12, .14, .10])
t = (s_date - pd.Timestamp("2024-01-01")).days / 730
v7d = rng.random(N_S) < np.clip(0.35 + 0.3 * t, 0, 1)
upload = (s_date >= pd.Timestamp("2024-07-01")) & (rng.random(N_S) < np.where(v7d, 0.42, 0.12))
pages = rng.poisson(4 + 3 * v7d + 4 * upload)
dur = np.round(rng.gamma(2, 2.2, N_S) * (1 + 0.8 * v7d + 1.2 * upload), 1)
p_quote = 0.06 + 0.09 * v7d + 0.16 * upload + np.where(device == "Mobile", -0.02, 0.01) + np.where(source == "Trade Fair QR", 0.06, 0)
quote = rng.random(N_S) < p_quote
order = quote & (rng.random(N_S) < (0.30 + 0.15 * upload + 0.05 * v7d))
sessions = pd.DataFrame({
    "session_id": [f"S{500000+i}" for i in range(N_S)], "session_datetime": s_date.strftime("%Y-%m-%d"),
    "visitor_country": s_country, "device": device, "traffic_source": source, "pages_viewed": pages,
    "session_duration_min": dur, "viewed_7d": v7d.astype(int), "uploaded_room_photo": upload.astype(int),
    "requested_quote": quote.astype(int), "placed_order": order.astype(int)})

# ================================================================ INJECT DIRT
def dirty(df, cols_missing, frac=0.03):
    for c in cols_missing:
        idx = rng.choice(df.index, int(len(df) * frac), replace=False)
        df.loc[idx, c] = np.nan
    return df

orders = orders.astype({"quantity_sqm": object, "unit_price_usd": object, "customer_rating": object, "order_date": object})
orders = dirty(orders, ["unit_price_usd", "discount_pct", "customer_rating", "shipping_mode", "dashboard_time_min", "actual_lead_days"], 0.025)
# inconsistent country spellings
alias = {"United States": ["USA", "U.S.A.", "united states", "US "], "United Kingdom": ["UK", "U.K.", "england"],
         "UAE": ["United Arab Emirates", "U.A.E", "uae"], "Germany": ["germany", "Deutschland"], "India": ["india ", "INDIA"]}
for k, v in alias.items():
    idx = orders.index[orders["country"] == k]
    pick = rng.choice(idx, int(len(idx) * 0.15), replace=False)
    orders.loc[pick, "country"] = rng.choice(v, len(pick))
# messy category text
idx = rng.choice(orders.index, 250, replace=False)
orders.loc[idx, "tile_category"] = orders.loc[idx, "tile_category"].str.upper().str.strip() + "  "
idx = rng.choice(orders.index, 200, replace=False)
orders.loc[idx, "customer_type"] = orders.loc[idx, "customer_type"].str.lower()
# mixed date formats
idx = rng.choice(orders.index, 400, replace=False)
orders.loc[idx, "order_date"] = pd.to_datetime(orders.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
# impossible / outlier values
idx = rng.choice(orders.index, 25, replace=False); orders.loc[idx, "quantity_sqm"] = -orders.loc[idx, "quantity_sqm"].astype(float)
idx = rng.choice(orders.index, 15, replace=False); orders.loc[idx, "quantity_sqm"] = orders.loc[idx, "quantity_sqm"].astype(float) * 40
idx = rng.choice(orders.index, 20, replace=False); orders.loc[idx, "unit_price_usd"] = "USD " + orders.loc[idx, "unit_price_usd"].astype(str)
idx = rng.choice(orders.index, 12, replace=False); orders.loc[idx, "customer_rating"] = 7
idx = rng.choice(orders.index, 18, replace=False); orders.loc[idx, "discount_pct"] = 95
# duplicates
orders = pd.concat([orders, orders.sample(120, random_state=1)], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)

prod = prod.drop(columns="_month")
prod = dirty(prod, ["kiln_temp_c", "defect_rate_pct", "energy_kwh"], 0.02)
idx = rng.choice(prod.index, 30, replace=False); prod.loc[idx, "automation_level"] = rng.choice(["fully automated", "Semi Automated", "MANUAL"], 30)
idx = rng.choice(prod.index, 10, replace=False); prod.loc[idx, "kiln_temp_c"] = 11800  # sensor glitch (extra zero)
idx = rng.choice(prod.index, 8, replace=False); prod.loc[idx, "defect_rate_pct"] = -1
prod = pd.concat([prod, prod.sample(40, random_state=3)], ignore_index=True)

sessions = dirty(sessions, ["session_duration_min", "device", "traffic_source"], 0.02)
idx = rng.choice(sessions.index, 40, replace=False); sessions.loc[idx, "session_duration_min"] = 0
sessions = pd.concat([sessions, sessions.sample(80, random_state=4)], ignore_index=True)

orders.to_csv(OUT / "orders_raw.csv", index=False)
prod.to_csv(OUT / "production_raw.csv", index=False)
sessions.to_csv(OUT / "sessions_raw.csv", index=False)
print("orders", orders.shape, "| production", prod.shape, "| sessions", sessions.shape)
