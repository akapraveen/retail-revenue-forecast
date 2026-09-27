"""Render README charts from results.json into figures/."""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
C1, C2, INK, GRID = "#2a78d6", "#eb6834", "#2b2f2b", "#e1e0d9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": "#666",
                     "ytick.color": "#666", "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})
gbp_k = plt.FuncFormatter(lambda v, _: f"£{v / 1000:.0f}k")


def main():
    r = json.loads((HERE / "results.json").read_text())
    OUT.mkdir(exist_ok=True)

    h = r["weekly_history"][1:-1]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = pd.to_datetime([d for d, _ in h]); y = [v for _, v in h]
    ax.plot(x, y, color=C1, lw=2)
    ax.fill_between(x, y, color=C1, alpha=0.1)
    ax.yaxis.set_major_formatter(gbp_k)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set(title="Weekly product revenue, Dec 2009 to Dec 2011", ylim=(0, None))
    fig.savefig(OUT / "weekly_revenue.png"); plt.close(fig)

    d = r["detail"]["2011-11-01"]
    dates = pd.to_datetime(d["dates"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(dates, d["p10"], d["p90"], color=C2, alpha=0.15, lw=0, label="Boosting 10–90% range")
    ax.plot(dates, d["Blend: boosting + last year"], color=C2, lw=2, label="Blended forecast")
    ax.plot(dates, d["actual"], color=C1, lw=2, marker="o", ms=4, label="Actual")
    ax.yaxis.set_major_formatter(gbp_k)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set(title="November 2011 reforecast (made 1 Nov with data to 31 Oct)", ylim=(0, None))
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "november_forecast.png"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
