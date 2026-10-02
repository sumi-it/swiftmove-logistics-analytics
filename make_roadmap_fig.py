import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

steps = [("1  Data\ncollection", "Orders, GPS, stock,\ncost, weather,\ncalendar"),
         ("2  Data\ncleaning", "Duplicates, types,\nmissing values,\noutliers"),
         ("3  Exploratory\nanalysis", "Baseline KPIs,\ntrends, delay and\ncost patterns"),
         ("4  Feature\nengineering", "Lags, calendar,\nutilisation,\nSKU clusters"),
         ("5  Predictive\nmodelling", "Demand forecast,\ndelivery time,\nlate-risk"),
         ("6  Optimisation", "Safety stock,\nroutes, vehicle\nmix"),
         ("7  Reporting and\ndecisions", "Dashboard,\nrecommendations,\nmonitoring")]
cols = ["#2E75B6", "#2E75B6", "#2E75B6", "#1F3864", "#1F3864", "#ED7D31", "#70AD47"]
fig, ax = plt.subplots(figsize=(11.5, 3.6)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.6); ax.axis("off")
bw, gp, x0, y0 = 1.34, 0.22, 0.2, 1.3
for i, ((title, sub), c) in enumerate(zip(steps, cols)):
    x = x0 + i * (bw + gp)
    ax.add_patch(FancyBboxPatch((x, y0), bw, 1.5, boxstyle="round,pad=0.03", fc=c, ec="none"))
    ax.text(x + bw / 2, y0 + 1.1, title, ha="center", va="center", color="white", fontsize=8.6, fontweight="bold")
    ax.text(x + bw / 2, y0 + 0.4, sub, ha="center", va="center", color="white", fontsize=6.4)
    if i < 6:
        ax.add_patch(FancyArrowPatch((x + bw + 0.03, y0 + 0.75), (x + bw + gp - 0.03, y0 + 0.75), arrowstyle="-|>", mutation_scale=10, color="#555"))
def span(a, b, label, col):
    xa, xb = x0 + a * (bw + gp), x0 + b * (bw + gp) + bw
    ax.plot([xa, xb], [1.1, 1.1], color=col, lw=2.5)
    ax.text((xa + xb) / 2, 0.8, label, ha="center", fontsize=8.2, color=col, fontweight="bold")
span(0, 1, "Week 2: collect and clean", "#2E75B6"); span(2, 3, "Week 3: explore and engineer", "#2E75B6")
span(4, 5, "Week 4: model and optimise", "#1F3864"); span(6, 6, "Report", "#70AD47")
ax.text(0.2 + 3.0 * (bw + gp) + bw / 2, 3.3, "Week 1 (this report): plan and define", fontsize=8.2, color="#595959", ha="center", style="italic")
ax.text(5.75, 0.25, "Feedback loop: monitor KPIs, retrain models and refresh data regularly", fontsize=8, color="#595959", ha="center", style="italic")
plt.tight_layout(); plt.savefig("figures/w1_roadmap.png", dpi=170); plt.close()
