"""Score the team-written diary sanity set with the app's text model and plot it.

    backend/.venv/bin/python ml/text/eval_diary.py

Reads ml/text/diary_sanity.csv (id, label, variant, text). Low day N mirrors good day N, so the
plot groups entries by day. Writes ml/reports/text/diary.png and prints a table.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import text as tp  # noqa: E402

THRESHOLD = tp.THRESHOLD
DATA = ROOT / "ml" / "text" / "diary_sanity.csv"
RESULTS = ROOT / "ml" / "reports" / "text"

# Reference palette (dataviz skill), light mode. Categorical slots 1 to 3 in order.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [
    ("good", "as written", "Good day, as written", "#2a78d6"),
    ("low", "as written", "Low day, as written", "#eb6834"),
    ("good", "I added (Claude rewrite)", 'Good day, "I" and "my" added', "#1baf7a"),
]


def score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    res = [tp.analyze_text(t) for t in out["text"]]
    out["score"] = [r["p_depressive"] for r in res]
    out["logit"] = [r["features"]["text_logit"] for r in res]
    out["first_person_per100"] = [r["features"]["first_person"] for r in res]
    out["day"] = out["id"].str.extract(r"_(\d)").astype(int)
    out["correct"] = (out["score"] >= THRESHOLD) == (out["label"] == "low")
    return out


def plot(scored: pd.DataFrame, title: str, path: Path) -> None:
    present = [s for s in SERIES if ((scored["label"] == s[0]) & (scored["variant"] == s[1])).any()]
    days = sorted(scored["day"].unique())
    width = 0.8 / len(present)
    group_x = list(range(len(days))) + [len(days) + 0.6]  # gap before Average

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for k, (label, variant, name, color) in enumerate(present):
        d = scored[(scored["label"] == label) & (scored["variant"] == variant)].set_index("day")
        values = [d.loc[day, "score"] for day in days] + [d["score"].mean()]
        xs = [x + (k - (len(present) - 1) / 2) * width for x in group_x]
        # 2px surface edge keeps a gap between adjacent bars.
        ax.bar(xs, values, width=width, color=color, edgecolor=SURFACE, linewidth=2,
               label=name, zorder=3)
        # Only the averages get value labels.
        ax.text(xs[-1], values[-1] + 0.02, f"{values[-1]:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK, zorder=4)

    ax.axhline(THRESHOLD, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(group_x[-1] + 1.25, THRESHOLD + 0.015, f"cut-off {THRESHOLD}", va="bottom", ha="right",
            fontsize=9, color=INK_2)

    ax.set_xticks(group_x, [f"Day {d}" for d in days] + ["Average"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.6, group_x[-1] + 1.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Score  (0 = everyday language, 1 = depressive language)", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0)
    ax.axvline(len(days) - 0.2, color=GRID, linewidth=1, zorder=1)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=28)
    ax.text(0, 1.035, "Each low day mirrors the good day with the same number. "
            "Higher score = reads more like r/depression.",
            transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=len(present), frameon=False,
              fontsize=9.5, labelcolor=INK_2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    scored = score(pd.read_csv(DATA))
    pd.set_option("display.width", 200)
    print(scored[["id", "variant", "score", "logit", "first_person_per100", "correct"]]
          .round(3).to_string(index=False))
    print()
    summary = scored.groupby(["label", "variant"]).agg(
        n=("score", "size"), mean_score=("score", "mean"), mean_logit=("logit", "mean"),
        first_person_per100=("first_person_per100", "mean"), correct=("correct", "sum"))
    print(summary.round(3).to_string())

    low = scored[(scored["label"] == "low") & (scored["variant"] == "as written")]["logit"]
    for variant in scored.loc[scored["label"] == "good", "variant"].unique():
        good = scored[(scored["label"] == "good") & (scored["variant"] == variant)]["logit"]
        wins = sum(lo > g for lo in low for g in good)
        print(f"low day scored above good day ({variant}): {wins} of {len(low) * len(good)} pairs")

    main_set = scored[scored["variant"] == "as written"]
    correct = int(main_set["correct"].sum())
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "diary.png"
    plot(scored, f"Text model: {correct} of {len(main_set)} diary entries correct", path)
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
