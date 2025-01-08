import json
import os
import sys

import matplotlib.pyplot as plt


def plot_bw_uvp_vs_epsilon_combined(data, d, json_file):
    """
    Plots BW-UVP vs epsilon with dot size and color representing hidden dimensions.
    """
    lambdas = sorted(
        set(float(key.split("_")[0].split("=")[1]) for key in data[d].keys())
    )
    hidden_dims = sorted(
        set(int(key.split("_")[1].split("=")[1]) for key in data[d].keys())
    )
    filename = os.path.basename(json_file)[: -len(".json")] + "bw-uvp-vs-epsilon"

    bproj_results = {hd: [] for hd in hidden_dims}
    scones_results = {hd: [] for hd in hidden_dims}

    for hd in hidden_dims:
        for lb in lambdas:
            key = f"lambda={lb}_hdim={hd}"
            if key in data[d]:
                bproj_results[hd].append(data[d][key]["bproj-mean-bw-uvp"])
                scones_results[hd].append(data[d][key]["scones-mean-bw-uvp"])

    fig, ax = plt.subplots(figsize=(12, 6))
    norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
    colors = plt.cm.viridis(norm(hidden_dims))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    for i, hd in enumerate(hidden_dims):
        ax.plot(
            lambdas,
            bproj_results[hd],
            "-o",
            label="BProj" if i == 0 else None,
            color=colors[i],
            alpha=0.8,
        )
        ax.plot(
            lambdas,
            scones_results[hd],
            "--o",
            label="SCONES" if i == 0 else None,
            color=colors[i],
            alpha=0.8,
        )

    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
    cbar.set_label("Hidden Dimension")

    ax.legend(loc="upper right")

    ax.set_title(f"Dimension d={d}: BW-UVP vs. Epsilon (Color = Hidden Dimensions)")
    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("BW-UVP")
    plt.savefig(filename + ".png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with open(json_file, "r") as f:
        data = json.load(f)

    for d in data.keys():
        plot_bw_uvp_vs_epsilon_combined(data, d, json_file)
