"""Render ``results/workflow_summary.json`` as a grouped bar chart.

Shows the relax-then-MD throughput per framework, with the compile /
no-compile pairs adjacent for direct visual comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY_JSON = Path("results") / "workflow_summary.json"
CHART_PNG = Path("results") / "workflow_throughput.png"


def main() -> None:
    summary_path = SUMMARY_JSON.resolve()
    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} not found — run `modal run modal_workflow.py` first."
        )
    summary = json.loads(summary_path.read_text())
    data = summary["workflow"]

    order = ["ase", "torchsim", "alchemi"]
    frameworks = [f for f in order if f in data]
    values = [data[f]["atom_steps_per_second"] for f in frameworks]

    colors = {
        "ase": "gray",
        "torchsim": "steelblue",
        "alchemi": "darkorange",
    }
    bar_colors = [colors[f] for f in frameworks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frameworks, values, width=0.55, color=bar_colors)
    for x, v in zip(frameworks, values, strict=True):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    n_systems = data[frameworks[0]]["n_systems"]
    n_atoms = data[frameworks[0]]["n_atoms_per_system"]
    ax.set_ylabel("atom-steps / second (budgeted)")
    ax.set_title(
        f"Workflow throughput (relax → MD) — N={n_systems} × {n_atoms} atoms, "
        f"MACE-MPA-0 (A100, fp32, no cueq)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    out = CHART_PNG.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
