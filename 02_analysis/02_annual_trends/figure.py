import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import style as S

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "09_dataset_hardening" / "results" / "apparel_recalls_v2.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.parse_args()
    d = pd.read_csv(V2)
    counts = d[d.year.between(1974, 2025)].groupby("year").size().reindex(range(1974, 2026), fill_value=0)
    roll = counts.rolling(5, center=True).mean()

    def build(fig):
        ax = fig.subplots(1, 1)
        ax.fill_between(counts.index, counts.values, color=S.BLUE, alpha=0.16, zorder=1)
        ax.plot(counts.index, counts.values, color=S.BLUE, linewidth=1.0, alpha=0.8, label="Annual recalls", zorder=2)
        ax.plot(roll.index, roll.values, color=S.PINK, linewidth=2.6, label="5-year rolling mean", zorder=3)
        for y in (2007, 2023):
            v = counts[y]
            ax.scatter([y], [v], s=26, color=S.BLUE, zorder=4)
            ax.annotate(f"{v} in {y}", (y, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=S.INK)
        ax.set_xlim(1973, 2027)
        ax.set_ylim(0, 47)
        ax.set_ylabel("Apparel and home-textile recalls")
        ax.set_xlabel("Year")
        S.tidy(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        S.freeze_free_text(fig)

    out = S.save_figure(build, HERE / "figure_annual_trends", height_in=4.2)
    print("[02] wrote", ", ".join(Path(p).name for p in out))

if __name__ == "__main__":
    main()
