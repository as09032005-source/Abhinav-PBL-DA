"""
TerraLuxe Tiles - Export Intelligence Dashboard (Streamlit)
Run:  pip install -r requirements.txt   then   streamlit run app.py
Pages: Overview | Sales | Digital Experience | Delivery | Factory Automation | Correlation Lab | Data Quality | Strategy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="TerraLuxe Tiles - Export Intelligence", layout="wide")
BASE = Path(__file__).parent / "data"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENG = ["No digital tool", "7D View only", "7D + Room Visualizer"]
ENG_COL = dict(zip(ENG, PAL[:3]))
px.defaults.color_discrete_sequence = PAL
px.defaults.template = "plotly_white"


def ensure_data():
    """On Streamlit Cloud the data folder may not be in the repo: build it from the scripts."""
    import subprocess, sys
    here = Path(__file__).parent
    needed = [BASE / "clean/orders_clean.csv", BASE / "clean/production_clean.csv",
              BASE / "clean/sessions_clean.csv", BASE / "cleaning_log.csv", BASE / "raw/orders_raw.csv"]
    if all(f.exists() for f in needed):
        return
    with st.spinner("First run: generating and cleaning the data (about 30 seconds)..."):
        for script in ["01_generate_synthetic_data.py", "02_clean_transform.py"]:
            path = here / script
            if not path.exists():
                st.error(f"Data files are missing and {script} is not in the repository. "
                         "Upload 01_generate_synthetic_data.py and 02_clean_transform.py next to app.py, "
                         "or upload the whole data/ folder.")
                st.stop()
            res = subprocess.run([sys.executable, str(path)], cwd=str(here), capture_output=True, text=True)
            if res.returncode != 0:
                st.error(f"{script} failed:\n\n{res.stderr[-1500:]}")
                st.stop()


ensure_data()


@st.cache_data
def load():
    o = pd.read_csv(BASE / "clean/orders_clean.csv", parse_dates=["order_date"])
    p = pd.read_csv(BASE / "clean/production_clean.csv", parse_dates=["production_date"])
    s = pd.read_csv(BASE / "clean/sessions_clean.csv", parse_dates=["session_datetime"])
    log = pd.read_csv(BASE / "cleaning_log.csv")
    raw = pd.read_csv(BASE / "raw/orders_raw.csv")
    return o, p, s, log, raw


o_all, p_all, s_all, clog, raw = load()


def explain(what, why, insight):
    st.markdown(f"**What it shows:** {what}  \n**Why this chart:** {why}  \n**Insight:** {insight}")
    st.divider()


# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("TerraLuxe Tiles")
st.sidebar.caption("Automated tile manufacturing for the world")
page = st.sidebar.radio("Page", ["Overview", "Sales (descriptive)", "Digital Experience (diagnostic)", "Delivery & Logistics",
                                 "Factory Automation", "Correlation Lab", "Data Quality", "Strategy"])
years = st.sidebar.multiselect("Year", sorted(o_all.year.unique()), default=sorted(o_all.year.unique()))
regions = st.sidebar.multiselect("Region", sorted(o_all.region.unique()), default=sorted(o_all.region.unique()))
ctypes = st.sidebar.multiselect("Customer type", sorted(o_all.customer_type.unique()), default=sorted(o_all.customer_type.unique()))
o = o_all[o_all.year.isin(years) & o_all.region.isin(regions) & o_all.customer_type.isin(ctypes)]
p = p_all[p_all.production_date.dt.year.isin(years)]
s = s_all[s_all.session_datetime.dt.year.isin(years)]
if o.empty:
    st.warning("No orders match the filters."); st.stop()

# ================================================================ OVERVIEW
if page == "Overview":
    st.title("Export Intelligence Dashboard")
    st.caption("Synthetic data, Jan-2024 to Dec-2025. Business: automated tile factory exporting to 14 countries, "
               "with a 7D product viewer and an AI Room Visualizer on the customer dashboard.")
    k = st.columns(6)
    k[0].metric("Net revenue", f"${o.net_revenue_usd.sum()/1e6:,.1f}M")
    k[1].metric("Gross margin", f"{o.gross_margin_usd.sum()/o.net_revenue_usd.sum()*100:.1f}%")
    k[2].metric("Orders", f"{len(o):,}")
    k[3].metric("On-time delivery", f"{o.on_time.mean()*100:.1f}%")
    k[4].metric("Return rate", f"{o.returned.mean()*100:.1f}%")
    k[5].metric("Avg rating", f"{o.customer_rating.mean():.2f} / 5")
    m = o.groupby(o.order_date.dt.to_period("M").dt.to_timestamp())["net_revenue_usd"].sum().reset_index()
    st.plotly_chart(px.line(m, x="order_date", y="net_revenue_usd", markers=True, title="Monthly net revenue (USD)",
                            labels={"order_date": "", "net_revenue_usd": "USD"}), use_container_width=True)
    explain("Total net revenue per month.", "A line chart is the clearest way to show change over time and seasonality.",
            "Peaks from October to March match the construction season in our main markets. 2025 is flat or slightly down "
            "against 2024 even though digital adoption grew. The Delivery page shows why: late deliveries are losing repeat customers.")
    c1, c2 = st.columns(2)
    geo = o.groupby("country", as_index=False)["net_revenue_usd"].sum()
    c1.plotly_chart(px.choropleth(geo, locations="country", locationmode="country names", color="net_revenue_usd",
                                  color_continuous_scale="Blues", title="Revenue by export market"), use_container_width=True)
    rg = o.groupby("region", as_index=False)["net_revenue_usd"].sum().sort_values("net_revenue_usd")
    c2.plotly_chart(px.bar(rg, x="net_revenue_usd", y="region", orientation="h", title="Revenue by region"), use_container_width=True)
    explain("Where our revenue comes from geographically.", "The map gives a global view for an export business; the bar chart ranks the regions exactly.",
            "North America, the Middle East and Europe bring in most of the revenue. India (domestic) is about 11%, so this is mainly an export business.")

# ================================================================ SALES
elif page == "Sales (descriptive)":
    st.title("Sales: what happened?")
    c1, c2 = st.columns(2)
    cat = o.groupby("tile_category").agg(revenue=("net_revenue_usd", "sum"), margin=("gross_margin_usd", "sum")).reset_index()
    cat["margin_pct"] = cat.margin / cat.revenue * 100
    c1.plotly_chart(px.bar(cat.sort_values("revenue"), x="revenue", y="tile_category", orientation="h", title="Revenue by tile category"), use_container_width=True)
    c2.plotly_chart(px.bar(cat.sort_values("margin_pct"), x="margin_pct", y="tile_category", orientation="h", title="Gross margin % by category",
                           color_discrete_sequence=[PAL[2]]), use_container_width=True)
    explain("Revenue and margin for each product line.", "Comparing revenue with margin separates volume drivers from profit drivers.",
            "Designer Mosaic and Porcelain Slab earn the most revenue and the highest margins (about 46-52%). These premium lines are what the 7D viewer and Room Visualizer should promote.")
    seg = o.groupby("customer_type").agg(revenue=("net_revenue_usd", "sum"), aov=("net_revenue_usd", "mean"), orders=("order_id", "count")).reset_index()
    st.plotly_chart(px.scatter(seg, x="orders", y="aov", size="revenue", color="customer_type", text="customer_type",
                               title="Segments: number of orders vs average order value (bubble = revenue)"), use_container_width=True)
    explain("Each customer segment by frequency, ticket size and total revenue.", "A bubble chart shows three measures at once without needing a second axis.",
            "Distributors and builders bring in most of the revenue (B2B bulk). Homeowners place many small orders but are the heaviest Visualizer users, so they are the brand-building segment.")
    st.plotly_chart(px.histogram(o, x="net_revenue_usd", nbins=60, log_x=True, title="Order value distribution (log scale)"), use_container_width=True)
    explain("How order values are spread.", "A histogram shows skew; a log scale is used because orders range from $100 to over $100k.",
            "Order values are strongly right-skewed: a few bulk orders carry the revenue, so medians and log-transforms are used in the tests.")
    st.subheader("Descriptive statistics")
    st.dataframe(o[["quantity_sqm", "unit_price_usd", "discount_pct", "net_revenue_usd", "margin_pct", "delay_days", "customer_rating"]].describe().round(2))

# ================================================================ DIGITAL
elif page == "Digital Experience (diagnostic)":
    st.title("Do the 7D viewer and Room Visualizer work?")
    eng = o.groupby("digital_engagement").agg(aov=("net_revenue_usd", "mean"), return_rate=("returned", "mean"),
                                             rating=("customer_rating", "mean"), orders=("order_id", "count")).reindex(ENG).reset_index()
    c = st.columns(3)
    c[0].plotly_chart(px.bar(eng, x="digital_engagement", y="return_rate", color="digital_engagement", color_discrete_map=ENG_COL, title="Return rate"), use_container_width=True)
    c[1].plotly_chart(px.bar(eng, x="digital_engagement", y="rating", color="digital_engagement", color_discrete_map=ENG_COL, title="Average rating", range_y=[3, 5]), use_container_width=True)
    c[2].plotly_chart(px.bar(eng, x="digital_engagement", y="aov", color="digital_engagement", color_discrete_map=ENG_COL, title="Average order value (overall)"), use_container_width=True)
    ct = pd.crosstab(o.digital_engagement, o.returned); chi = stats.chi2_contingency(ct)
    st.info(f"Chi-square test (returns vs tool use): chi2 = {chi[0]:.1f}, p = {chi[1]:.1e}. "
            + ("Significant: tool use and returns are related." if chi[1] < 0.05 else "Not significant."))
    explain("Returns, rating and order value for each level of digital engagement.",
            "Grouped comparisons with a hypothesis test are the core diagnostic tool: they show whether a difference is real or just noise.",
            "Customers who saw their own room before buying return about 3x less often (15% vs 5%) and rate us higher, because there is less 'it looked different at home' regret. "
            "Overall order value looks LOWER for tool users, which is checked below.")
    st.subheader("Simpson's paradox check: compare like with like")
    w = o.pivot_table(index="customer_type", columns="digital_engagement", values="net_revenue_usd", aggfunc="mean").reindex(columns=ENG).reset_index()
    wl = w.melt(id_vars="customer_type", var_name="digital_engagement", value_name="aov")
    st.plotly_chart(px.bar(wl, x="customer_type", y="aov", color="digital_engagement", barmode="group", color_discrete_map=ENG_COL,
                           title="Average order value within each segment"), use_container_width=True)
    explain("Order value by tool use, split by customer segment.",
            "The overall comparison mixes segments. Homeowners (small orders) use the Visualizer most, which drags the overall average down.",
            "Within every segment, Visualizer users spend about 11-40% MORE. The tool raises basket size; the overall drop was a segment-mix effect (Simpson's paradox).")
    fn = s.groupby("journey").agg(sessions=("session_id", "count"), quote=("requested_quote", "mean"), order=("placed_order", "mean")).reindex(["Browse only", "7D only", "7D + Room upload"]).reset_index()
    fnl = fn.melt(id_vars="journey", value_vars=["quote", "order"], var_name="stage", value_name="rate")
    st.plotly_chart(px.bar(fnl, x="journey", y="rate", color="stage", barmode="group", title="Website/app funnel: conversion by visitor journey"), use_container_width=True)
    explain("Share of dashboard sessions that request a quote or place an order.", "A funnel comparison measures the tool's effect on conversion, not just satisfaction.",
            f"Visitors who upload a room photo convert to orders about {fn.order.iloc[2]/fn.order.iloc[0]:.0f}x more often than those who only browse.")
    q = o.groupby("quarter")[["used_7d_view", "used_room_visualizer"]].mean().reset_index().melt(id_vars="quarter", var_name="tool", value_name="share")
    st.plotly_chart(px.line(q, x="quarter", y="share", color="tool", markers=True, title="Adoption of digital tools by quarter"), use_container_width=True)
    explain("Share of orders where each tool was used.", "Tracks whether customers actually take up the innovation over time.",
            "Adoption keeps rising (about 70% for 7D by late 2025). The Visualizer jumped as soon as it launched in Q3-2024.")

# ================================================================ DELIVERY
elif page == "Delivery & Logistics":
    st.title("Timely delivery: where do we fail and what does it cost?")
    st.plotly_chart(px.box(o, x="shipping_mode", y="delay_days", color="shipping_mode", points=False,
                           title="Delay vs promised date (days) by shipping mode"), use_container_width=True)
    explain("Spread of delivery delays for each shipping mode.", "A box plot shows median, spread and outliers, and the outliers are the problem here.",
            "Sea freight has a long tail of delays of up to about 30 days (port congestion, missed vessels). That tail is what hurts the brand.")
    ot = o.groupby(["shipping_mode", "production_line_type"])["on_time"].mean().reset_index()
    st.plotly_chart(px.bar(ot, x="shipping_mode", y="on_time", color="production_line_type", barmode="group",
                           title="On-time rate: automated vs semi/manual production"), use_container_width=True)
    explain("On-time delivery split by production line type and shipping mode.", "Two-way grouping separates factory-caused from logistics-caused delays.",
            "Orders made on fully automated lines arrive on time 16-21 points more often in every shipping mode. Factory automation is also a delivery lever.")
    o2 = o.assign(delay_band=pd.cut(o.delay_days, [-10, 0, 3, 7, 14, 100], labels=["On time", "1-3", "4-7", "8-14", ">14"]))
    dr = o2.groupby("delay_band", observed=True)["customer_rating"].mean().reset_index()
    dr["delay_band"] = dr["delay_band"].astype(str)
    st.plotly_chart(px.bar(dr, x="delay_band", y="customer_rating", title="Average rating by days late", range_y=[2, 5]), use_container_width=True)
    rho = stats.spearmanr(o.delay_days, o.customer_rating)
    st.info(f"Spearman correlation between delay and rating: rho = {rho.statistic:.2f} (p = {rho.pvalue:.1e})")
    explain("Customer rating for each band of lateness.", "Bins turn a noisy scatter into a clear dose-response pattern.",
            "Ratings fall steadily, from about 4.4 when on time to about 2.5 when more than two weeks late.")
    first = o[(o.order_seq == 1) & (o.order_date < "2025-06-30")].merge(
        o[o.order_seq == 2][["customer_id", "order_date"]].rename(columns={"order_date": "d2"}), on="customer_id", how="left")
    first["came_back"] = ((first.d2 - first.order_date).dt.days <= 180)
    first["experience"] = np.where((first.delay_days > 5) | (first.returned == 1), "Bad first order", "Good first order")
    rt = first.groupby("experience")["came_back"].mean().reset_index()
    st.plotly_chart(px.bar(rt, x="experience", y="came_back", color="experience", color_discrete_sequence=[PAL[7], PAL[2]],
                           title="% of customers who re-order within 180 days"), use_container_width=True)
    explain("Retention after a good vs bad first order.", "Links operations (delay/returns) to customer lifetime value, the root cause of revenue stalling.",
            "A bad first experience cuts the chance of a second order by more than a third (about 27% vs 42%). This is why 2025 revenue stalled despite more digital users.")

# ================================================================ FACTORY
elif page == "Factory Automation":
    st.title("Operations automation: quality, productivity, cost")
    lv = ["Manual", "Semi-Automated", "Fully Automated"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(p, x="automation_level", y="defect_rate_pct", category_orders={"automation_level": lv}, color="automation_level",
                           title="Defect rate % by automation level"), use_container_width=True)
    agg = p.groupby("automation_level")[["sqm_per_operator", "downtime_hours", "cost_per_sqm_usd", "energy_per_sqm", "oee_proxy_pct"]].mean().reindex(lv).reset_index()
    c2.plotly_chart(px.bar(agg, x="automation_level", y="sqm_per_operator", title="Output per operator (sqm/batch)"), use_container_width=True)
    an = stats.f_oneway(*[p.loc[p.automation_level == l, "defect_rate_pct"] for l in lv])
    st.info(f"One-way ANOVA (defect rate across automation levels): F = {an.statistic:.0f}, p = {an.pvalue:.1e}")
    explain("Quality and labour productivity by line type.", "ANOVA tests whether the mean defect rate truly differs across more than two groups.",
            "Fully automated lines have less than half the defects of manual lines and about 7x the output per operator.")
    st.dataframe(agg.round(2), use_container_width=True)
    u = p[p.line_id.isin(["L3", "L4"])].assign(group="L3-L4 (upgraded Jul-2025)")
    c = p[p.line_id == "L5"].assign(group="L5 (control)")
    t = pd.concat([u, c]).groupby(["month", "group"])["defect_rate_pct"].mean().reset_index()
    fig = px.line(t, x="month", y="defect_rate_pct", color="group", title="Before/after upgrade vs control line")
    fig.add_vline(x="2025-07", line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)
    explain("Monthly defect rate of the upgraded lines against a line that was not upgraded.",
            "Difference-in-differences: the control line removes seasonal or raw-material effects, isolating the impact of the upgrade.",
            "Defects on L3-L4 fell by about 0.9 percentage points right after the upgrade while L5 stayed flat. The upgrade caused the improvement.")
    st.plotly_chart(px.scatter(p, x="kiln_temp_c", y="defect_rate_pct", color="automation_level", opacity=.6,
                               title="Kiln temperature vs defect rate"), use_container_width=True)
    explain("Process parameter against quality.", "A scatter plot finds non-linear (U-shaped) relationships that a correlation number would miss.",
            "Defects rise when firing drifts away from about 1180 C, which makes the case for IoT temperature control with automatic alerts.")

# ================================================================ CORRELATION
elif page == "Correlation Lab":
    st.title("Correlation Lab")
    num = ["quantity_sqm", "unit_price_usd", "discount_pct", "net_revenue_usd", "margin_pct", "used_7d_view", "used_room_visualizer",
           "visualizer_renders", "dashboard_time_min", "delay_days", "damage_pct", "returned", "customer_rating", "cust_orders"]
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    cm = o[num].corr(method=method).round(2)
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale=["#2a78d6", "#f0efec", "#eb6834"], zmin=-1, zmax=1, aspect="auto",
                              title=f"{method.title()} correlation matrix"), use_container_width=True)
    explain("Pairwise linear (Pearson) or rank (Spearman) correlation between numeric features.",
            "A quick screen for relationships worth testing. Spearman is offered because revenue and delay are skewed.",
            "Strongest business links: delay vs rating (negative), returned vs rating (negative), Visualizer use vs renders and dashboard time (positive). Quantity drives revenue and discount.")
    c1, c2 = st.columns(2)
    xv = c1.selectbox("X", num, index=num.index("delay_days")); yv = c2.selectbox("Y", num, index=num.index("customer_rating"))
    samp = o.sample(min(2000, len(o)), random_state=1)
    st.plotly_chart(px.scatter(samp, x=xv, y=yv, color="digital_engagement", color_discrete_map=ENG_COL, opacity=.5,
                               title=f"{yv} vs {xv} (sample of 2,000 orders)"), use_container_width=True)
    r = stats.pearsonr(o[xv], o[yv]); st.caption(f"Pearson r = {r.statistic:.3f}, p = {r.pvalue:.1e}. Correlation is not causation: check the diagnostic pages.")

# ================================================================ DATA QUALITY
elif page == "Data Quality":
    st.title("Data preparation: what was cleaned and why")
    c = st.columns(3)
    c[0].metric("Raw order rows", f"{len(raw):,}"); c[1].metric("Clean order rows", f"{len(o_all):,}"); c[2].metric("Issues fixed", len(clog))
    st.dataframe(clog, use_container_width=True, hide_index=True)
    st.subheader("Missing values in the raw orders file")
    mv = raw.isna().sum().loc[lambda x: x > 0].reset_index(); mv.columns = ["column", "missing"]
    st.plotly_chart(px.bar(mv, x="column", y="missing", title="Missing values per column (raw)"), use_container_width=True)
    st.subheader("Raw sample (before cleaning)"); st.dataframe(raw.head(15), use_container_width=True)

# ================================================================ STRATEGY
elif page == "Strategy":
    st.title("Data-driven strategy")
    st.markdown("""
| # | Finding (evidence) | Strategy | KPI target |
|---|---|---|---|
| 1 | Room Visualizer users: about 3x fewer returns, +0.38 rating, +11-40% basket within segment, 7x funnel conversion | Make the room upload the **default step** in every quote; give distributors a white-label version for their showrooms | Visualizer use above 75% of orders; returns below 6% |
| 2 | Late delivery is the #1 driver of low ratings; a bad first order cuts the 180-day re-order rate from 42% to 27% | **Delivery promise engine**: quote dates by lane and line type, hold buffer stock in UAE, UK and US warehouses, send proactive delay alerts | On-time above 85%; 180-day re-order above 45% |
| 3 | Automated lines: -57% defects, about 7x sqm/operator, 19% lower cost/sqm, and better on-time delivery | Upgrade L5 and L6 next (proven by the L3-L4 diff-in-diff); route export orders to automated lines first | Defects below 1.8% factory-wide |
| 4 | Kiln deviation drives defects | IoT kiln sensors with automatic alerts at +/-8 C | 0 batches outside +/-8 C |
| 5 | Mosaic and Porcelain have the highest margin; USA, UAE and UK are the largest markets | Promote premium lines in the 7D viewer for those markets; run Oct-Mar campaigns | Premium share of revenue above 60% |
| 6 | Trade-fair QR traffic converts best (8.8% of sessions) | Put 7D/AR QR codes on every sample box and at fair booths | Session to order conversion above 7% |
""")
