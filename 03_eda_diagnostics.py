"""
STEP 3 - DESCRIPTIVE & DIAGNOSTIC ANALYTICS (EDA)
Descriptive  = WHAT happened (trends, distributions, mix, KPIs)
Diagnostic   = WHY it happened (correlations, group comparisons, hypothesis tests, driver model)
Outputs: charts/*.png and outputs/kpis_and_tests.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE = Path(__file__).parent
CH = BASE / "charts"; CH.mkdir(exist_ok=True)
OUTD = BASE / "outputs"; OUTD.mkdir(exist_ok=True)
o = pd.read_csv(BASE / "data/clean/orders_clean.csv", parse_dates=["order_date"])
p = pd.read_csv(BASE / "data/clean/production_clean.csv", parse_dates=["production_date"])
s = pd.read_csv(BASE / "data/clean/sessions_clean.csv", parse_dates=["session_datetime"])

# ---- style (validated categorical palette, fixed order; muted ink; recessive grid)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "lines.linewidth": 2, "figure.dpi": 130})
ENG_ORDER = ["No digital tool", "7D View only", "7D + Room Visualizer"]
R = {}

def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, bbox_inches="tight"); plt.close(fig)

def barlabels(ax, fmt="{:,.0f}", horizontal=False):
    for b in ax.patches:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK2)

# ================================================================ KPIs
R["kpis"] = {
    "orders": int(len(o)), "customers": int(o.customer_id.nunique()), "countries": int(o.country.nunique()),
    "net_revenue_usd": round(o.net_revenue_usd.sum()), "gross_margin_pct": round(o.gross_margin_usd.sum() / o.net_revenue_usd.sum() * 100, 1),
    "avg_order_value_usd": round(o.net_revenue_usd.mean()), "export_share_revenue_pct": round(o.loc[o.is_export == 1, "net_revenue_usd"].sum() / o.net_revenue_usd.sum() * 100, 1),
    "on_time_pct": round(o.on_time.mean() * 100, 1), "return_rate_pct": round(o.returned.mean() * 100, 1),
    "avg_rating": round(o.customer_rating.mean(), 2), "repeat_order_share_pct": round(o.is_repeat_order.mean() * 100, 1),
    "avg_defect_pct": round(p.defect_rate_pct.mean(), 2),
    "revenue_2024": round(o.loc[o.year == 2024, "net_revenue_usd"].sum()), "revenue_2025": round(o.loc[o.year == 2025, "net_revenue_usd"].sum()),
}
R["kpis"]["yoy_growth_pct"] = round((R["kpis"]["revenue_2025"] / R["kpis"]["revenue_2024"] - 1) * 100, 1)
R["descriptive_stats"] = o[["quantity_sqm", "unit_price_usd", "discount_pct", "net_revenue_usd", "margin_pct",
                            "delay_days", "damage_pct", "customer_rating", "dashboard_time_min"]].describe().round(2).to_dict()

# ================================================================ 1. Monthly revenue trend
m = o.groupby(o.order_date.dt.to_period("M"))["net_revenue_usd"].sum() / 1e6
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(m.index.to_timestamp(), m.values, color=C[0], marker="o", markersize=4)
ax.set_title("Fig 1. Monthly net revenue (USD million)"); ax.set_ylabel("USD mn")
pk = m.idxmax(); ax.annotate(f"Peak {pk}: ${m.max():.2f}M", (pk.to_timestamp(), m.max()), xytext=(10, -5), textcoords="offset points", fontsize=8.5, color=INK2)
save(fig, "01_monthly_revenue.png")
R["monthly_peak"] = str(pk)
season = o.assign(season=np.where(o.order_date.dt.month.isin([10, 11, 12, 1, 2, 3]), "Oct-Mar", "Apr-Sep")).groupby("season")["net_revenue_usd"].sum()
R["season_share_oct_mar_pct"] = round(season["Oct-Mar"] / season.sum() * 100, 1)

# ================================================================ 2. Revenue by country
cr = o.groupby("country")["net_revenue_usd"].sum().sort_values() / 1e6
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(cr.index, cr.values, color=[C[1] if c == "India" else C[0] for c in cr.index], height=0.7)
barlabels(ax, "{:.2f}M", horizontal=True)
ax.set_title("Fig 2. Net revenue by country (orange = domestic)"); ax.set_xlabel("USD mn"); ax.grid(axis="y", visible=False)
save(fig, "02_revenue_by_country.png")
R["top_countries"] = cr.sort_values(ascending=False).head(5).round(2).to_dict()

# ================================================================ 3. Category revenue and margin
cat = o.groupby("tile_category").agg(rev=("net_revenue_usd", "sum"), gm=("gross_margin_usd", "sum")).sort_values("rev")
cat["margin_pct"] = cat.gm / cat.rev * 100
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
axs[0].barh(cat.index, cat.rev / 1e6, color=C[0], height=0.65); barlabels(axs[0], "{:.1f}M", True); axs[0].set_title("Fig 3a. Revenue by category (USD mn)")
axs[1].barh(cat.index, cat.margin_pct, color=C[2], height=0.65); barlabels(axs[1], "{:.1f}%", True); axs[1].set_title("Fig 3b. Gross margin %")
for a in axs: a.grid(axis="y", visible=False)
save(fig, "03_category_revenue_margin.png")
R["category"] = cat[["rev", "margin_pct"]].round(1).to_dict()

# ================================================================ 4. Customer segment
seg = o.groupby("customer_type").agg(aov=("net_revenue_usd", "mean"), orders=("order_id", "count"), rev=("net_revenue_usd", "sum")).sort_values("rev")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
axs[0].barh(seg.index, seg.rev / 1e6, color=C[0], height=0.65); barlabels(axs[0], "{:.1f}M", True); axs[0].set_title("Fig 4a. Revenue by segment (USD mn)")
axs[1].barh(seg.index, seg.aov, color=C[6], height=0.65); barlabels(axs[1], "${:,.0f}", True); axs[1].set_title("Fig 4b. Average order value")
for a in axs: a.grid(axis="y", visible=False)
save(fig, "04_segments.png")
R["segments"] = seg.round(0).to_dict()

# ================================================================ 5. Order value distribution
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(np.log10(o.net_revenue_usd), bins=45, color=C[0], edgecolor="#fcfcfb", linewidth=1)
ax.set_xticks([2, 3, 4, 5]); ax.set_xticklabels(["$100", "$1k", "$10k", "$100k"])
ax.axvline(np.log10(o.net_revenue_usd.median()), color=INK2, ls="--", lw=1)
ax.text(np.log10(o.net_revenue_usd.median()), ax.get_ylim()[1] * 0.92, f"  median ${o.net_revenue_usd.median():,.0f}", fontsize=8.5, color=INK2)
ax.set_title("Fig 5. Distribution of order value (log scale)"); ax.set_ylabel("orders")
save(fig, "05_order_value_distribution.png")
R["order_value_skew"] = round(float(stats.skew(o.net_revenue_usd)), 2)

# ================================================================ 6. Correlation heatmap
num = ["quantity_sqm", "unit_price_usd", "discount_pct", "net_revenue_usd", "margin_pct", "used_7d_view", "used_room_visualizer",
       "visualizer_renders", "dashboard_time_min", "delay_days", "damage_pct", "returned", "customer_rating", "cust_orders"]
corr = o[num].corr().round(2)
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=60, ha="right", fontsize=8.5)
ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=8.5); ax.grid(False)
for i in range(len(num)):
    for j in range(len(num)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
fig.colorbar(im, fraction=0.04); ax.set_title("Fig 6. Pearson correlation matrix - order features")
save(fig, "06_correlation_heatmap.png")
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False)
R["top_correlations"] = {f"{a} ~ {b}": v for (a, b), v in pairs.head(10).items()}

# ================================================================ 7. Digital engagement impact (diagnostic)
eng = o.groupby("digital_engagement").agg(aov=("net_revenue_usd", "mean"), returns=("returned", "mean"),
                                         rating=("customer_rating", "mean"), n=("order_id", "count")).reindex(ENG_ORDER)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for a, col, ttl, fmt, mult in [(axs[0], "aov", "Avg order value (USD)", "${:,.0f}", 1), (axs[1], "returns", "Return rate (%)", "{:.1f}%", 100),
                               (axs[2], "rating", "Avg rating (1-5)", "{:.2f}", 1)]:
    a.bar(range(3), eng[col] * mult, color=C[:3], width=0.6); barlabels(a, fmt); a.set_title(ttl, fontsize=10.5)
    a.set_xticks(range(3)); a.set_xticklabels(["None", "7D only", "7D + Room"], fontsize=9); a.grid(axis="x", visible=False)
axs[2].set_ylim(3, 5)
fig.suptitle("Fig 7. Do the 7D viewer and Room Visualizer change customer behaviour?", x=0.01, ha="left", fontweight="bold")
save(fig, "07_digital_tools_impact.png")
g0 = o.loc[o.digital_engagement == "No digital tool", "net_revenue_usd"]; g2 = o.loc[o.digital_engagement == "7D + Room Visualizer", "net_revenue_usd"]
t = stats.ttest_ind(np.log(g2), np.log(g0), equal_var=False)
ct = pd.crosstab(o.digital_engagement, o.returned); chi = stats.chi2_contingency(ct)
kw = stats.kruskal(*[o.loc[o.digital_engagement == e, "customer_rating"] for e in ENG_ORDER])
R["digital_impact"] = {"table": eng.round(3).to_dict(),
                       "aov_uplift_room_vs_none_pct": round((g2.mean() / g0.mean() - 1) * 100, 1),
                       "welch_t_log_aov": {"t": round(t.statistic, 2), "p": float(f"{t.pvalue:.2e}")},
                       "chi2_returns": {"chi2": round(chi[0], 2), "p": float(f"{chi[1]:.2e}")},
                       "kruskal_rating": {"H": round(kw.statistic, 2), "p": float(f"{kw.pvalue:.2e}")}}

# ================================================================ 8. Digital adoption over time
q = o.groupby("quarter")[["used_7d_view", "used_room_visualizer"]].mean() * 100
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.plot(q.index, q.used_7d_view, color=C[0], marker="o", label="Used 7D view")
ax.plot(q.index, q.used_room_visualizer, color=C[1], marker="o", label="Used Room Visualizer")
for col, c in [("used_7d_view", C[0]), ("used_room_visualizer", C[1])]:
    ax.text(len(q) - 1 + 0.1, q[col].iloc[-1], f"{q[col].iloc[-1]:.0f}%", va="center", fontsize=8.5, color=INK2)
ax.axvline("2024Q3", color=INK2, ls=":", lw=1); ax.text("2024Q3", 3, " Visualizer launched", fontsize=8, color=INK2)
ax.set_ylabel("% of orders"); ax.legend(frameon=False, loc="upper left"); ax.set_title("Fig 8. Digital tool adoption by quarter")
save(fig, "08_digital_adoption.png")
R["adoption"] = q.round(1).to_dict()

# ================================================================ 9. Session funnel
fn = s.groupby("journey").agg(sessions=("session_id", "count"), quote=("requested_quote", "mean"), order=("placed_order", "mean")).reindex(["Browse only", "7D only", "7D + Room upload"])
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, fn.quote * 100, w, color=C[0], label="Requested quote")
ax.bar(x + w / 2, fn.order * 100, w, color=C[1], label="Placed order")
barlabels(ax, "{:.1f}%"); ax.set_xticks(x); ax.set_xticklabels(fn.index); ax.legend(frameon=False)
ax.set_ylabel("% of sessions"); ax.set_title("Fig 9. Dashboard funnel: conversion by visitor journey"); ax.grid(axis="x", visible=False)
save(fig, "09_funnel_conversion.png")
chi_f = stats.chi2_contingency(pd.crosstab(s.journey, s.placed_order))
R["funnel"] = {"table": fn.round(4).to_dict(), "chi2_order": {"chi2": round(chi_f[0], 1), "p": float(f"{chi_f[1]:.2e}")},
               "lift_room_vs_browse_x": round(fn.loc["7D + Room upload", "order"] / fn.loc["Browse only", "order"], 1)}
src = s.groupby("traffic_source")["placed_order"].mean().sort_values() * 100
R["source_conversion_pct"] = src.round(2).to_dict()

# ================================================================ 10. Delivery delay by shipping mode
modes = ["Road", "Air Freight", "Sea Freight"]
fig, ax = plt.subplots(figsize=(8, 3.6))
bp = ax.boxplot([o.loc[o.shipping_mode == mo, "delay_days"] for mo in modes], tick_labels=modes, patch_artist=True, widths=0.5,
                medianprops=dict(color=INK), flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2, alpha=.4, markeredgewidth=0))
for b, c in zip(bp["boxes"], C): b.set_facecolor(c); b.set_alpha(.8); b.set_edgecolor("#fcfcfb")
ax.axhline(0, color=C[7], lw=1, ls="--"); ax.text(3.3, 0.3, "promised date", fontsize=8, color=INK2)
ax.set_ylabel("days late (+) / early (-)"); ax.set_title("Fig 10. Delivery delay vs promise, by shipping mode")
save(fig, "10_delay_by_shipping.png")
R["delay_by_mode"] = o.groupby("shipping_mode")["delay_days"].describe()[["mean", "50%", "75%", "max"]].round(2).to_dict()

# ================================================================ 11. On-time % by line type x shipping
ot = o.pivot_table(index="shipping_mode", columns="production_line_type", values="on_time", aggfunc="mean").reindex(modes) * 100
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x - w / 2, ot["Semi/Manual"], w, color=C[1], label="Semi/Manual line")
ax.bar(x + w / 2, ot["Fully Automated"], w, color=C[0], label="Fully automated line")
barlabels(ax, "{:.0f}%"); ax.set_xticks(x); ax.set_xticklabels(modes); ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_ylabel("% delivered on time"); ax.set_title("Fig 11. On-time delivery: automation x shipping mode"); ax.set_ylim(0, 115); ax.grid(axis="x", visible=False)
save(fig, "11_ontime_automation_shipping.png")
R["on_time_matrix"] = ot.round(1).to_dict()
ct2 = stats.chi2_contingency(pd.crosstab(o.production_line_type, o.on_time))
R["chi2_ontime_automation"] = {"chi2": round(ct2[0], 1), "p": float(f"{ct2[1]:.2e}")}

# ================================================================ 12. Delay vs rating & repeat
o["delay_band"] = pd.cut(o.delay_days, [-10, 0, 3, 7, 14, 100], labels=["On time/early", "1-3 late", "4-7 late", "8-14 late", ">14 late"])
dr = o.groupby("delay_band", observed=True).agg(rating=("customer_rating", "mean"), n=("order_id", "count"))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(dr.index.astype(str), dr.rating, color=C[0], width=0.6); barlabels(ax, "{:.2f}")
ax.set_ylim(2.5, 5); ax.set_ylabel("avg rating"); ax.set_title("Fig 12. Customer rating falls as delivery delay grows"); ax.grid(axis="x", visible=False)
save(fig, "12_delay_vs_rating.png")
r = stats.spearmanr(o.delay_days, o.customer_rating)
R["delay_rating"] = {"table": dr.round(2).to_dict(), "spearman_rho": round(r.statistic, 3), "p": float(f"{r.pvalue:.2e}")}

# ================================================================ 13. Automation vs defects & productivity
lv = ["Manual", "Semi-Automated", "Fully Automated"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
bp = axs[0].boxplot([p.loc[p.automation_level == l, "defect_rate_pct"] for l in lv], tick_labels=["Manual", "Semi", "Full"], patch_artist=True, widths=.5,
                    medianprops=dict(color=INK), flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2, alpha=.4, markeredgewidth=0))
for b, c in zip(bp["boxes"], [C[1], C[3], C[0]]): b.set_facecolor(c); b.set_edgecolor("#fcfcfb")
axs[0].set_title("Defect rate %", fontsize=10.5)
pa = p.groupby("automation_level")[["sqm_per_operator", "downtime_hours"]].mean().reindex(lv)
axs[1].bar(["Manual", "Semi", "Full"], pa.sqm_per_operator, color=[C[1], C[3], C[0]], width=.6); barlabels(axs[1]); axs[1].set_title("Output per operator (sqm)", fontsize=10.5)
axs[2].bar(["Manual", "Semi", "Full"], pa.downtime_hours, color=[C[1], C[3], C[0]], width=.6); barlabels(axs[2], "{:.1f}h"); axs[2].set_title("Avg downtime per batch", fontsize=10.5)
for a in axs[1:]: a.grid(axis="x", visible=False)
fig.suptitle("Fig 13. Factory automation vs quality and productivity", x=0.01, ha="left", fontweight="bold")
save(fig, "13_automation_quality.png")
an = stats.f_oneway(*[p.loc[p.automation_level == l, "defect_rate_pct"] for l in lv])
R["automation"] = {"table": p.groupby("automation_level")[["defect_rate_pct", "sqm_per_operator", "downtime_hours", "energy_per_sqm", "cost_per_sqm_usd", "oee_proxy_pct"]].mean().reindex(lv).round(2).to_dict(),
                   "anova_defect": {"F": round(an.statistic, 1), "p": float(f"{an.pvalue:.2e}")}}

# ================================================================ 14. Before/after: lines L3-L4 upgrade
u = p[p.line_id.isin(["L3", "L4"])].groupby("month")["defect_rate_pct"].mean()
ctrl = p[p.line_id.isin(["L5"])].groupby("month")["defect_rate_pct"].mean()
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(pd.PeriodIndex(u.index, freq="M").to_timestamp(), u.values, color=C[0], label="Lines L3-L4 (upgraded Jul-2025)")
ax.plot(pd.PeriodIndex(ctrl.index, freq="M").to_timestamp(), ctrl.values, color=C[1], label="Line L5 (not upgraded - control)")
ax.axvline(pd.Timestamp("2025-07-01"), color=INK2, ls=":", lw=1)
ax.set_ylabel("avg defect %"); ax.legend(frameon=False); ax.set_title("Fig 14. Before/after automation upgrade (difference-in-differences view)")
save(fig, "14_automation_before_after.png")
pre = p[(p.line_id.isin(["L3", "L4"])) & (p.production_date < "2025-07-01")].defect_rate_pct
post = p[(p.line_id.isin(["L3", "L4"])) & (p.production_date >= "2025-07-01")].defect_rate_pct
cpre = p[(p.line_id == "L5") & (p.production_date < "2025-07-01")].defect_rate_pct
cpost = p[(p.line_id == "L5") & (p.production_date >= "2025-07-01")].defect_rate_pct
R["did"] = {"upgraded_pre": round(pre.mean(), 2), "upgraded_post": round(post.mean(), 2), "control_pre": round(cpre.mean(), 2),
            "control_post": round(cpost.mean(), 2), "did_effect_pp": round((post.mean() - pre.mean()) - (cpost.mean() - cpre.mean()), 2)}

# ================================================================ 15. Kiln temperature vs defect
fig, ax = plt.subplots(figsize=(8, 3.6))
for l, c in zip(lv, [C[1], C[3], C[0]]):
    d = p[p.automation_level == l]; ax.scatter(d.kiln_temp_c, d.defect_rate_pct, s=10, color=c, alpha=.55, label=l, edgecolors="none")
ax.axvline(1180, color=INK2, ls=":", lw=1); ax.text(1181, ax.get_ylim()[1] * 0.92, "target 1180 C", fontsize=8, color=INK2)
ax.set_xlabel("kiln temperature (C)"); ax.set_ylabel("defect %"); ax.legend(frameon=False, markerscale=2)
ax.set_title("Fig 15. Kiln temperature deviation drives defects (U-shape)")
save(fig, "15_kiln_vs_defect.png")
R["kiln_dev_defect_corr"] = round(float(np.corrcoef(np.abs(p.kiln_temp_c - 1180), p.defect_rate_pct)[0, 1]), 3)

# ================================================================ 16. Driver model for rating (diagnostic regression)
X = o[["delay_days", "damage_pct", "used_7d_view", "used_room_visualizer", "discount_pct", "unit_price_usd"]].copy()
X["is_sea"] = (o.shipping_mode == "Sea Freight").astype(int)
Xs = StandardScaler().fit_transform(X); lr = LinearRegression().fit(Xs, o.customer_rating)
coef = pd.Series(lr.coef_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(coef.index, coef.values, color=[C[7] if v < 0 else C[2] for v in coef.values], height=.6); ax.axvline(0, color=INK2, lw=.8)
for b, v in zip(ax.patches, coef.values):
    ax.text(v, b.get_y() + b.get_height() / 2, f" {v:+.3f} " , va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
ax.set_xlim(coef.min() * 1.35, coef.max() * 1.5); ax.grid(axis="y", visible=False)
ax.set_title(f"Fig 16. What drives customer rating? (standardised coefficients, R2={lr.score(Xs, o.customer_rating):.2f})")
save(fig, "16_rating_drivers.png")
R["rating_model"] = {"r2": round(lr.score(Xs, o.customer_rating), 3), "std_coefs": coef.round(3).to_dict()}

# ================================================================ 7b. Simpson's paradox: AOV uplift WITHIN segment
w_seg = o.pivot_table(index="customer_type", columns="digital_engagement", values="net_revenue_usd", aggfunc="mean")[ENG_ORDER]
upl = ((w_seg["7D + Room Visualizer"] / w_seg["No digital tool"] - 1) * 100).sort_values()
mix = pd.crosstab(o.digital_engagement, o.customer_type, normalize="index").loc[ENG_ORDER, "Homeowner"] * 100
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
axs[0].barh(upl.index, upl.values, color=C[2], height=.6); barlabels(axs[0], "+{:.0f}%", True); axs[0].grid(axis="y", visible=False)
axs[0].set_title("Fig 7b. AOV uplift, Room Visualizer vs no tool, within each segment", fontsize=10.5)
axs[1].bar(["None", "7D only", "7D + Room"], mix.values, color=C[:3], width=.6); barlabels(axs[1], "{:.0f}%"); axs[1].grid(axis="x", visible=False)
axs[1].set_title("Share of orders from Homeowners", fontsize=10.5)
save(fig, "07b_simpsons_paradox_aov.png")
R["within_segment_aov_uplift_pct"] = upl.round(1).to_dict()
R["homeowner_share_by_engagement_pct"] = mix.round(1).to_dict()

# ================================================================ 17. Retention: does the first experience bring customers back?
first = o[o.order_seq == 1].copy()
first = first[first.order_date < "2025-06-30"]  # cohort with at least 6 months to come back
nxt = o[o.order_seq == 2][["customer_id", "order_date"]].rename(columns={"order_date": "second_date"})
first = first.merge(nxt, on="customer_id", how="left")
first["came_back_180d"] = ((first.second_date - first.order_date).dt.days <= 180).astype(int)
first["first_exp"] = np.where((first.delay_days > 5) | (first.returned == 1), "Bad (late >5d or return)", "Good")
ret = first.groupby("first_exp")["came_back_180d"].mean() * 100
ret2 = first.groupby("digital_engagement")["came_back_180d"].mean().reindex(ENG_ORDER) * 100
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
axs[0].bar(ret.index, ret.values, color=[C[7], C[2]], width=.55); barlabels(axs[0], "{:.0f}%"); axs[0].set_title("By first-order experience", fontsize=10.5)
axs[1].bar(["None", "7D only", "7D + Room"], ret2.values, color=C[:3], width=.55); barlabels(axs[1], "{:.0f}%"); axs[1].set_title("By digital tool used on first order", fontsize=10.5)
axs[0].set_ylabel("% re-ordered within 180 days")
for a in axs: a.grid(axis="x", visible=False)
fig.suptitle("Fig 17. Customer retention after the first order", x=0.01, ha="left", fontweight="bold")
save(fig, "17_retention_first_experience.png")
chi_r = stats.chi2_contingency(pd.crosstab(first.first_exp, first.came_back_180d))
R["retention"] = {"by_experience_pct": ret.round(1).to_dict(), "by_tool_pct": ret2.round(1).to_dict(),
                  "chi2": round(chi_r[0], 1), "p": float(f"{chi_r[1]:.2e}"), "cohort_size": int(len(first))}

json.dump(R, open(OUTD / "kpis_and_tests.json", "w"), indent=2, default=str)
print(json.dumps({k: R[k] for k in ["kpis", "digital_impact", "funnel", "on_time_matrix", "automation", "did", "delay_rating", "rating_model", "retention", "within_segment_aov_uplift_pct", "homeowner_share_by_engagement_pct", "top_correlations", "season_share_oct_mar_pct", "top_countries", "kiln_dev_defect_corr", "source_conversion_pct"]}, indent=1, default=str))
