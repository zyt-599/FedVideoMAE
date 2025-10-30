#!/usr/bin/env python3
"""
Dataflow schematic for this project starting from pretraining.

Draws a high‑level diagram (no code execution), and saves PNG/SVG/PDF to
Report/architecture (creating the folder if needed).

Usage
  python3 plot_dataflow.py

Dependencies
  - matplotlib

Install (if needed)
  python3 -m pip install matplotlib
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def box(ax, xy, w, h, text, fc="#F3F6FA", ec="#333", fontsize=10, weight="semibold"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=6",
        linewidth=1.2, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(ax, xy_from, xy_to, text=None):
    ax.annotate(
        "",
        xy=xy_to, xycoords='data',
        xytext=xy_from, textcoords='data',
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
    )
    if text:
        xm = (xy_from[0] + xy_to[0]) / 2
        ym = (xy_from[1] + xy_to[1]) / 2
        ax.text(xm, ym, text, ha="center", va="bottom", fontsize=9)


def draw_diagram(out_dir: Path):
    fig_w, fig_h = 12, 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Row Y coordinates (top -> bottom)
    y_top = 78
    y_mid1 = 58
    y_mid2 = 38
    y_mid3 = 18

    w = 20; h = 10

    # Top row: Data -> Transforms -> Partitions
    b_data = box(ax, (5, y_top), w, h, "Datasets\n(RWF2000 / RLVS)")
    b_tf = box(ax, (30, y_top), w, h, "Transforms\n(build_video_transform)")
    b_part = box(ax, (55, y_top), w, h, "Partitions\n(clients shards)")
    arrow(ax, (5+w, y_top+h/2), (30, y_top+h/2))
    arrow(ax, (30+w, y_top+h/2), (55, y_top+h/2))

    # Row 2: DataLoader -> Clients
    b_loader = box(ax, (5, y_mid1), w, h, "DataLoader\n(batch, num_workers)")
    b_clients = box(ax, (30, y_mid1), w, h, "Federated Clients\n(K shards)")
    arrow(ax, (5+w, y_mid1+h/2), (30, y_mid1+h/2))
    arrow(ax, (55, y_top), (15, y_mid1+h))  # down diagonal grouping

    # Row 3 left: Local training
    b_local = box(ax, (5, y_mid2), w+10, h+2, "Local Train\nVideoMAE (encoder) +\nPretrainLoss (MSE) + AdamW")
    arrow(ax, (30, y_mid1), (10, y_mid2+h+1))

    # DP on client (optional)
    b_dp = box(ax, (35, y_mid2), w, h, "(Optional) DP\nclip + noise")
    arrow(ax, (15+w+10, y_mid2+h/2+1), (35, y_mid2+h/2))

    # Row 3 right: Client updates -> Server aggregation
    b_update = box(ax, (60, y_mid2), w, h, "Client Update\n(trainable params)")
    arrow(ax, (35+w, y_mid2+h/2), (60, y_mid2+h/2))

    b_server = box(ax, (60, y_mid1), w, h, "Server Aggregate\nFedAvg (+SA/Noise)")
    arrow(ax, (70, y_mid2+h), (70, y_mid1))

    # Row 4: Global model, validation, logs
    b_global = box(ax, (60, y_mid3), w, h, "Global Model\n(updated per round)")
    arrow(ax, (70, y_mid1), (70, y_mid3+h))

    b_val = box(ax, (35, y_mid3), w, h, "Validation\n(optional per round)")
    arrow(ax, (60, y_mid3+h/2), (35+w, y_mid3+h/2), text="metrics")

    b_logs = box(ax, (10, y_mid3), w, h, "Logs & CKPTs\n(logs/, runs/)")
    arrow(ax, (35, y_mid3+h/2), (10+w, y_mid3+h/2))

    # Loop annotation
    ax.text(70, y_mid2-6, "Repeat for R rounds", ha="center", va="center", fontsize=10)

    # Title
    ax.set_title("Pretraining Dataflow (with DP & SA)", fontsize=13, weight="bold")

    out_png = out_dir / "dataflow_pretrain.png"
    out_svg = out_dir / "dataflow_pretrain.svg"
    out_pdf = out_dir / "dataflow_pretrain.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=220)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_png}\nSaved: {out_svg}\nSaved: {out_pdf}")


def main():
    project = Path(__file__).resolve().parent
    out_dir = _ensure_dir(project / "Report" / "architecture")
    draw_diagram(out_dir)


if __name__ == "__main__":
    main()

