import json
import sys
import matplotlib.pyplot as plt


def plot_bw_uvp_vs_lambda(data, d):
    """
    Plots BW-UVP vs lambda for each hidden dimension at a given dimension d.
    """
    # data[d] has structure: { 'lambda=2.0_hdim=1024': {...}, 'lambda=2.0_hdim=2048': {...}, ... }
    # We'll parse out lambda, hdim, and mean bw-uvp
    lambda_hdim_pairs = list(data[d].keys())

    # Group by hidden dimension
    results_by_hdim = {}
    for pair_key in lambda_hdim_pairs:
        # e.g. pair_key = 'lambda=2.0_hdim=1024'
        splitted = pair_key.split("_")
        # splitted = ['lambda=2.0', 'hdim=1024']
        lambda_val_str = splitted[0].split("=")[1]  # '2.0'
        hdim_val_str = splitted[1].split("=")[1]  # '1024'

        lambda_val = float(lambda_val_str)
        hdim_val = int(hdim_val_str)

        # We'll store this so we can group them
        if hdim_val not in results_by_hdim:
            results_by_hdim[hdim_val] = []
        entry = data[d][pair_key]
        results_by_hdim[hdim_val].append(
            (lambda_val, entry["bproj-mean-bw-uvp"], entry["scones-mean-bw-uvp"])
        )

    # Now let's plot for each hdim
    for hdim_val, arr in results_by_hdim.items():
        # Sort by lambda
        arr.sort(key=lambda x: x[0])
        lambdas = [x[0] for x in arr]
        bproj_vals = [x[1] for x in arr]
        scones_vals = [x[2] for x in arr]

        plt.plot(lambdas, bproj_vals, "-o", label=f"BProj hdim={hdim_val}")
        plt.plot(lambdas, scones_vals, "-o", label=f"SCONES hdim={hdim_val}")

    plt.title(f"Dimension d={d}: BW-UVP vs. lambda")
    plt.xlabel("lambda")
    plt.ylabel("BW-UVP")
    plt.legend()
    plt.show()


def plot_bw_uvp_vs_hdim(data, d):
    """
    Plots BW-UVP vs hidden dimension for each lambda at a given dimension d.
    """
    lambda_hdim_pairs = list(data[d].keys())

    # Group by lambda
    results_by_lambda = {}
    for pair_key in lambda_hdim_pairs:
        splitted = pair_key.split("_")
        lambda_val_str = splitted[0].split("=")[1]
        hdim_val_str = splitted[1].split("=")[1]

        lambda_val = float(lambda_val_str)
        hdim_val = int(hdim_val_str)

        if lambda_val not in results_by_lambda:
            results_by_lambda[lambda_val] = []
        entry = data[d][pair_key]
        results_by_lambda[lambda_val].append(
            (hdim_val, entry["bproj-mean-bw-uvp"], entry["scones-mean-bw-uvp"])
        )

    # Now let's plot for each lambda
    for lambda_val, arr in results_by_lambda.items():
        arr.sort(key=lambda x: x[0])
        hdims = [x[0] for x in arr]
        bproj_vals = [x[1] for x in arr]
        scones_vals = [x[2] for x in arr]

        plt.plot(hdims, bproj_vals, "-o", label=f"BProj λ={lambda_val}")
        plt.plot(hdims, scones_vals, "-o", label=f"SCONES λ={lambda_val}")

    plt.title(f"Dimension d={d}: BW-UVP vs. hidden dimension")
    plt.xlabel("Hidden dimension")
    plt.ylabel("BW-UVP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with open(json_file, "r") as f:
        data = json.load(f)

    # data structure: { "2": { "lambda=2.0_hdim=1024": {...}, ... }, "4": {...}, ... }

    # Let’s just loop over available dimensions in the data
    for d in data.keys():
        # 1) Plot BW-UVP vs. lambda for each hidden dim
        plot_bw_uvp_vs_lambda(data, d)

        # 2) Plot BW-UVP vs. hidden dim for each lambda
        plot_bw_uvp_vs_hdim(data, d)
