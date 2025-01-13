import json
import os
import sys

import matplotlib.pyplot as plt


def plot_bw_uvp_vs_epsilon_combined(data, d, json_file):
    """
    Plots BW-UVP vs epsilon with dot size and color representing hidden dimensions.
    """
    epsilons = sorted(
        set(float(key.split("_")[0].split("=")[1]) for key in data[d].keys())
    )
    hidden_dims = sorted(
        set(int(key.split("_")[1].split("=")[1]) for key in data[d].keys())
    )
    for i, eps in enumerate(epsilons):
        if eps == 1.0:
            epsilons[i] = 1
    filename = os.path.basename(json_file)[: -len(".json")] + "bw-uvp-vs-epsilon"

    bproj_results = {hd: [] for hd in hidden_dims}
    scones_results = {hd: [] for hd in hidden_dims}

    for hd in hidden_dims:
        for eps in epsilons:
            key = f"eps={eps}_hdim={hd}"
            if key in data[d]:
                bproj_results[hd].append(data[d][key]["bproj-mean-bw-uvp"])
                scones_results[hd].append(data[d][key]["scones-mean-bw-uvp"])

    fig, ax = plt.subplots(figsize=(12, 6))
    norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
    colors = plt.cm.viridis(norm(hidden_dims))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    for i, hd in enumerate(hidden_dims):
        ax.plot(
            epsilons,
            bproj_results[hd],
            # "-o",
            label="BProj" if i == 0 else None,
            color=colors[i],
            alpha=0.8,
        )
        ax.plot(
            epsilons,
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

    ax.set_title(f"Dimension d={d}: BW-UVP vs. Epsilon")
    ax.set_xlabel("ϵ")
    ax.set_ylabel("BW-UVP")
    plt.savefig(filename + ".png")


def plot_wasserstein_vs_epsilon_combined(data, json_file):
    """
    Expects `data` to have the structure:
      {
        "32": {
          "2.0": {
            "1e-5": {
              "runs": [...],
              "bproj_mean_wasserstein": ...,
              "scones_mean_wasserstein": ...
            },
            "1e-4": { ... },
            ...
          },
          "1.0": { ... },
          ...
        },
        "64": {
          "2.0": {
            "1e-5": { ... },
            ...
          },
          ...
        }
      }

    This function creates one figure per lambda (inside each hidden_dim sub-dict).
    The x-axis is treated as 'epsilon' = SCONES LR. Colors indicate different hidden_dims.
    Each figure plots both BProj and SCONES lines for all hidden_dims.
    The file is saved as "jsonfilename_lam={lam}.png".
    """

    hidden_dims = sorted(map(int, data.keys()))  # e.g. [32, 64, 128, ...]

    lambda_values = set()
    for hd in data:
        lambda_values.update(map(float, data[hd].keys()))
    lambda_values = sorted(lambda_values)  # e.g. [0.5, 1.0, 2.0, ...]

    for lam in lambda_values:
        fig, ax = plt.subplots(figsize=(10, 6))

        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
        colors = plt.cm.viridis(norm(hidden_dims))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

        for i, hd in enumerate(hidden_dims):
            hd_str = str(hd)
            lam_str = str(lam)

            if lam_str not in data[hd_str]:
                continue

            lr_dict = data[hd_str][lam_str]

            lrs = sorted(map(float, lr_dict.keys()))

            bproj_vals = []
            scones_vals = []
            for lr in lrs:
                lr_str = str(lr)
                entry = lr_dict[lr_str]
                bproj_vals.append(entry["bproj_mean_wasserstein"])
                scones_vals.append(entry["scones_mean_wasserstein"])

            ax.plot(
                lrs,
                bproj_vals,
                # "-o",
                color=colors[i],
                label=f"BProj" if i == 0 else None,
                alpha=0.8,
            )
            ax.plot(
                lrs,
                scones_vals,
                "--o",
                color=colors[i],
                label=f"SCONES" if i == 0 else None,
                alpha=0.8,
            )

        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
        cbar.set_label("Hidden Dimension")

        ax.set_title(f"Lambda = {lam}: Wasserstein vs. SCONES LR")
        ax.set_xlabel("ε")
        ax.set_ylabel("Wasserstein Distance")
        ax.set_xscale("log")  # TODO

        handles, labels = ax.get_legend_handles_labels()
        new_handles, new_labels = [], []
        for h, lbl in zip(handles, labels):
            if lbl not in new_labels:
                new_labels.append(lbl)
                new_handles.append(h)
        ax.legend(new_handles, new_labels, loc="best")

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        out_filename = f"{base_name}_lam={lam}.png"
        plt.savefig(out_filename)
        print(f"Saved figure: {out_filename}")

        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <json_file> <exp>")
        sys.exit(1)

    json_file = sys.argv[1]
    exp = sys.argv[2]

    with open(json_file, "r") as f:
        data = json.load(f)

    if exp == "g":
        for d in data.keys():
            plot_bw_uvp_vs_epsilon_combined(data, d, json_file)
    else:
        plot_wasserstein_vs_epsilon_combined(data, json_file)
