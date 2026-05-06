"""Generate hero figures for the Gradient Decoupled DPO release.

Outputs:
    fig_memory_vs_sequence.png  - peak GPU memory across 4 sequence configs
    fig_memory_vs_layers.png    - peak GPU memory across 3 model depths
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

C_STANDARD = "#E07A5F"
C_DECOUPLED = "#2A9D8F"
C_OOM = "#B23A48"
C_CAP = "#6B6B6B"
CAPACITY_GB = 80.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})


def _annotate_bar(ax, x, height, text, color="black", weight="normal", dy=1.2):
    ax.text(x, height + dy, text, ha="center", va="bottom",
            fontsize=10, color=color, fontweight=weight)


def fig_memory_vs_sequence():
    configs = ["49f / 480p", "81f / 720p", "81f / 1080p", "121f / 1080p"]
    tokens = [20280, 75600, 171360, 252960]

    standard_mem = [79.0, 80.0, 80.0, 80.0]
    standard_oom = [True, True, True, True]
    decoupled_mem = [42.91, 48.72, 67.10, 80.0]
    decoupled_oom = [False, False, False, True]

    x = np.arange(len(configs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)

    bars_s = ax.bar(x - width / 2, standard_mem, width,
                    color=C_STANDARD, label="Standard DPO", zorder=3)
    bars_d = ax.bar(x + width / 2, decoupled_mem, width,
                    color=C_DECOUPLED, label="Gradient Decoupled DPO", zorder=3)

    for i, (b, oom) in enumerate(zip(bars_s, standard_oom)):
        b.set_hatch("///") if oom else None
        b.set_edgecolor(C_OOM if oom else "none")
        b.set_linewidth(1.4 if oom else 0)
        if oom:
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          "OOM", color=C_OOM, weight="bold")
        else:
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          f"{standard_mem[i]:.1f} GB")

    for i, (b, oom) in enumerate(zip(bars_d, decoupled_oom)):
        if oom:
            b.set_hatch("///")
            b.set_edgecolor(C_OOM)
            b.set_linewidth(1.4)
            b.set_color("#E07A5F")
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          "OOM", color=C_OOM, weight="bold")
        else:
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          f"{decoupled_mem[i]:.2f} GB",
                          color="#1F6F65", weight="bold")

    ax.axhline(CAPACITY_GB, ls="--", color=C_CAP, lw=1.2, zorder=2)

    xt_labels = [f"{c}\n~{t/1000:.1f}k tokens" for c, t in zip(configs, tokens)]
    ax.set_xticks(x)
    ax.set_xticklabels(xt_labels)
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_ylim(0, 92)
    ax.set_title("Wan 14B I2V DPO — peak memory across sequence configs\n"
                 "(40 layers, CP=8 DP=4, bf16 + ZeRO-2 + recompute=full, 32×H800)",
                 loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=1)

    fig.text(0.5, -0.02,
             "Dashed line = H800 80 GB capacity. "
             "Decoupled DPO unlocks production-config training (49 f/480p) "
             "that standard DPO cannot fit, and scales to ~8.5× the visual-token length.",
             ha="center", fontsize=10, color="#444")

    fig.tight_layout()
    out = OUT_DIR / "fig_memory_vs_sequence.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_memory_vs_layers():
    configs = [
        "Wan 14B / 40 layers\n8×H800, fixture seq (≈504 tokens)",
        "Wan 14B / 40 layers\n32×H800, production seq (49 f / 480p, ≈20k tokens)",
    ]
    standard = [80.0, 79.0]
    decoupled = [65.17, 42.91]
    standard_oom = [True, True]
    standard_oom_label = ["OOM (>80 GB)", "OOM 79 GB"]
    deltas = ["qualitative ✓", "−46% / qualitative ✓"]

    x = np.arange(len(configs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    bars_s = ax.bar(x - width / 2, standard, width,
                    color=C_STANDARD, label="Standard DPO", zorder=3)
    bars_d = ax.bar(x + width / 2, decoupled, width,
                    color=C_DECOUPLED, label="Gradient Decoupled DPO", zorder=3)

    for i, b in enumerate(bars_s):
        if standard_oom[i]:
            b.set_hatch("///")
            b.set_edgecolor(C_OOM)
            b.set_linewidth(1.4)
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          standard_oom_label[i], color=C_OOM, weight="bold")
        else:
            _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                          f"{standard[i]:.2f} GB")

    for i, b in enumerate(bars_d):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, b.get_height(),
                      f"{decoupled[i]:.2f} GB",
                      color="#1F6F65", weight="bold")

    for i, d in enumerate(deltas):
        top = max(standard[i], decoupled[i])
        ax.annotate(d, xy=(x[i], top + 6),
                    ha="center", va="center", fontsize=10.5,
                    color="#1F6F65" if "−" in d else "#444",
                    fontweight="bold" if "−" in d else "normal",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#EAF6F3" if "−" in d else "#F0F0F0",
                              ec="#2A9D8F" if "−" in d else "#999",
                              lw=0.8))

    ax.axhline(CAPACITY_GB, ls="--", color=C_CAP, lw=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_ylim(0, 102)
    ax.set_title("Wan 14B I2V DPO — peak GPU memory at production depth\n"
                 "bf16 + ZeRO-2 + recompute=full + flash-attn 3",
                 loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=1)

    fig.text(0.5, -0.02,
             "Dashed line = H800 80 GB capacity. "
             "At full production scale (32×H800, 49 f / 480p, ~20k visual tokens), "
             "Gradient Decoupled DPO cuts peak memory by ~46% — from a standard-DPO OOM down to 42.91 GB.",
             ha="center", fontsize=10, color="#444")

    fig.tight_layout()
    out = OUT_DIR / "fig_memory_vs_layers.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    # Hero figure shipped with the release.
    fig_memory_vs_layers()
    # Sequence-scaling figure kept for internal reference; uncomment to render.
    # fig_memory_vs_sequence()
