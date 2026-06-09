import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "output" / "q_learning_model.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
DEFAULT_CSV_DIR = SCRIPT_DIR / "csv_tables"
FIXED_RHOS = (0.25, 0.50, 0.75)

sys.path.insert(0, str(SCRIPT_DIR))

import QLearning_Relay as ql  # noqa: E402


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)


def load_q_table(model_path):
    model_data = np.load(model_path)
    if "q_table" not in model_data.files:
        raise KeyError(f"No q_table found in saved model: {model_path}")
    return model_data["q_table"], model_data


def make_reference_state(snr_db):
    return ql.make_reference_state(
        snr_db=snr_db,
        delta=ql.BASELINE_DELTA.copy(),
        sigma_n2=ql.BASELINE_SIGMA_N2,
        eta=ql.BASELINE_ETA,
    )


def evaluate_snr_baselines(q_table):
    snr_grid = np.arange(0.0, 41.0, 2.0)
    results = {
        "snr_grid": snr_grid,
        "q_rho": [],
        "exhaustive_rho": [],
        "q_avg_pep": [],
        "exhaustive_avg_pep": [],
        "random_avg_pep": [],
        "fixed_avg_pep": {rho: [] for rho in FIXED_RHOS},
    }

    for snr_db in snr_grid:
        state = make_reference_state(snr_db)
        all_peps = np.vstack([ql.compute_pep_pair(state, rho) for rho in ql.RHO_ACTIONS])
        all_avg_peps = all_peps.mean(axis=1)

        q_action_index = ql.greedy_action(q_table[ql.state_to_index(state)])
        exhaustive_action_index = int(np.argmin(all_avg_peps))

        results["q_rho"].append(ql.RHO_ACTIONS[q_action_index])
        results["exhaustive_rho"].append(ql.RHO_ACTIONS[exhaustive_action_index])
        results["q_avg_pep"].append(all_avg_peps[q_action_index])
        results["exhaustive_avg_pep"].append(all_avg_peps[exhaustive_action_index])
        # Exact expectation of a uniformly random action over the same 19 rho values.
        results["random_avg_pep"].append(all_avg_peps.mean())

        for fixed_rho in FIXED_RHOS:
            results["fixed_avg_pep"][fixed_rho].append(
                ql.compute_pep_pair(state, fixed_rho).mean()
            )

    for key in ("q_rho", "exhaustive_rho", "q_avg_pep", "exhaustive_avg_pep", "random_avg_pep"):
        results[key] = np.asarray(results[key], dtype=float)
    for fixed_rho in FIXED_RHOS:
        results["fixed_avg_pep"][fixed_rho] = np.asarray(
            results["fixed_avg_pep"][fixed_rho],
            dtype=float,
        )
    return results


def plot_snr_baseline_comparison(results, output_path):
    snr_grid = results["snr_grid"]
    reference_state = make_reference_state(snr_grid[0])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    pep_series = [
        results["q_avg_pep"],
        results["exhaustive_avg_pep"],
        results["random_avg_pep"],
        *[results["fixed_avg_pep"][rho] for rho in FIXED_RHOS],
    ]
    all_pep_values = np.concatenate(pep_series)
    y_min = max(np.min(all_pep_values) * 0.75, 1e-4)
    y_max = min(np.max(all_pep_values) * 1.20, 1.0)
    outline = [path_effects.Stroke(linewidth=4.5, foreground="white"), path_effects.Normal()]

    q_rho_line = axes[0].plot(
        snr_grid,
        results["q_rho"],
        "-o",
        color="midnightblue",
        linewidth=2.6,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markevery=(0, 2),
        zorder=4,
        label=r"Q-learning $\rho$",
    )[0]
    q_rho_line.set_path_effects(outline)
    exhaustive_rho_line = axes[0].plot(
        snr_grid,
        results["exhaustive_rho"],
        "--s",
        color="crimson",
        linewidth=2.6,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markevery=(1, 2),
        zorder=5,
        label=r"Exhaustive-search $\rho^*$",
    )[0]
    exhaustive_rho_line.set_path_effects(outline)
    axes[0].set_title(r"Greedy $\rho$ vs SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    q_pep_line = axes[1].semilogy(
        snr_grid,
        results["q_avg_pep"],
        "-o",
        color="midnightblue",
        linewidth=2.8,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markevery=(0, 2),
        zorder=5,
        label="Q-learning",
    )[0]
    q_pep_line.set_path_effects(outline)
    fixed_styles = {
        0.25: {"color": "darkorange", "linestyle": (0, (1, 1)), "marker": "^", "markevery": (2, 4)},
        0.50: {"color": "gray", "linestyle": (0, (3, 2, 1, 2)), "marker": "D", "markevery": (0, 4)},
        0.75: {"color": "darkgreen", "linestyle": (0, (5, 2)), "marker": "v", "markevery": (1, 4)},
    }
    for fixed_rho in FIXED_RHOS:
        style = fixed_styles[fixed_rho]
        axes[1].semilogy(
            snr_grid,
            results["fixed_avg_pep"][fixed_rho],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markevery=style["markevery"],
            linewidth=2.5,
            zorder=2,
            label=rf"Fixed $\rho = {fixed_rho:.2f}$",
        )
    axes[1].semilogy(
        snr_grid,
        results["random_avg_pep"],
        "-.",
        color="purple",
        linewidth=2.6,
        marker="x",
        markersize=4,
        markevery=(3, 4),
        zorder=3,
        label=r"Random $\rho$, uniform over 19 actions",
    )
    exhaustive_pep_line = axes[1].semilogy(
        snr_grid,
        results["exhaustive_avg_pep"],
        "--",
        color="crimson",
        linewidth=2.8,
        marker="s",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markevery=(1, 2),
        zorder=6,
        label="Exhaustive-search best, 19 actions",
    )[0]
    exhaustive_pep_line.set_path_effects(outline)
    axes[1].set_title("Average PEP vs SNR")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Average PEP")
    axes[1].set_ylim(y_min, y_max)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=11)

    fig.suptitle(
        "Q-learning policy compared with fixed, random, and exhaustive-search baselines",
        fontsize=18,
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        rf"$[\delta_1,\delta_2] = [{ql.BASELINE_DELTA[0]:.2f}, {ql.BASELINE_DELTA[1]:.2f}],\ "
        rf"\sigma_n^2 = {ql.BASELINE_SIGMA_N2:.1f},\ \eta = {ql.BASELINE_ETA:.1f},\ "
        rf"\kappa = {reference_state.path_loss_exponent:.0f},\ "
        rf"|h_{{sr}}|^2 = {reference_state.selected_relay_gain:.2f}$",
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_snr_baseline_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        results["snr_grid"],
        results["q_rho"],
        results["exhaustive_rho"],
        results["q_avg_pep"],
        results["fixed_avg_pep"][0.25],
        results["fixed_avg_pep"][0.50],
        results["fixed_avg_pep"][0.75],
        results["random_avg_pep"],
        results["exhaustive_avg_pep"],
    ]
    table = np.column_stack(columns)
    header = ",".join(
        [
            "SNR_dB",
            "Q_learning_rho",
            "Exhaustive_search_rho",
            "Q_learning_avg_PEP",
            "Fixed_rho_0_25_avg_PEP",
            "Fixed_rho_0_50_avg_PEP",
            "Fixed_rho_0_75_avg_PEP",
            "Random_rho_avg_PEP",
            "Exhaustive_search_avg_PEP",
        ]
    )
    np.savetxt(output_path, table, delimiter=",", header=header, comments="", fmt="%.5f")


def save_snr_summary_csvs(results, pep_summary_path, rho_summary_path):
    pep_summary_path.parent.mkdir(parents=True, exist_ok=True)
    exhaustive_pep = results["exhaustive_avg_pep"]
    pep_series = [
        ("Q-learning", results["q_avg_pep"]),
        ("Fixed rho = 0.25", results["fixed_avg_pep"][0.25]),
        ("Fixed rho = 0.50", results["fixed_avg_pep"][0.50]),
        ("Fixed rho = 0.75", results["fixed_avg_pep"][0.75]),
        ("Random rho", results["random_avg_pep"]),
        ("Exhaustive-search", results["exhaustive_avg_pep"]),
    ]

    with pep_summary_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Method",
                "Mean_avg_PEP",
                "Median_avg_PEP",
                "Min_PEP",
                "Max_PEP",
                "Mean_gap_vs_exhaustive_percent",
            ]
        )
        for method, values in pep_series:
            mean_gap = np.mean(
                (values - exhaustive_pep) / np.maximum(exhaustive_pep, np.finfo(float).eps)
            )
            writer.writerow(
                [
                    method,
                    f"{np.mean(values):.5f}",
                    f"{np.median(values):.5f}",
                    f"{np.min(values):.5f}",
                    f"{np.max(values):.5f}",
                    f"{100.0 * mean_gap:.4f}",
                ]
            )

    with rho_summary_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Method", "Mean_rho", "Median_rho", "Min_rho", "Max_rho"])
        for method, values in (
            ("Q-learning", results["q_rho"]),
            ("Exhaustive-search", results["exhaustive_rho"]),
        ):
            writer.writerow(
                [
                    method,
                    f"{np.mean(values):.4f}",
                    f"{np.median(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Q-learning PEP-vs-SNR baseline comparison from a saved Q-table."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved Q-learning .npz model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where plots will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"Directory where CSV tables will be saved. Default: {DEFAULT_CSV_DIR}",
    )
    args = parser.parse_args()

    q_table, model_data = load_q_table(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    snr_results = evaluate_snr_baselines(q_table)

    policy_plot_path = args.output_dir / "q_learning_pep_vs_snr_baseline_comparison_600dpi_large_text.png"
    csv_path = args.csv_dir / "q_learning_pep_vs_snr_baseline_comparison.csv"
    pep_summary_csv_path = args.csv_dir / "q_learning_pep_vs_snr_summary.csv"
    rho_summary_csv_path = args.csv_dir / "q_learning_rho_summary.csv"

    plot_snr_baseline_comparison(snr_results, policy_plot_path)
    save_snr_baseline_csv(snr_results, csv_path)
    save_snr_summary_csvs(snr_results, pep_summary_csv_path, rho_summary_csv_path)

    print(f"Loaded model: {args.model}")
    print(f"Q-table shape: {q_table.shape}")
    if "training_episodes" in model_data.files:
        print(f"Training episodes in saved model: {int(model_data['training_episodes'][0])}")
    if "seed" in model_data.files:
        print(f"Seed in saved model: {int(model_data['seed'][0])}")
    print()
    print("Generated PEP-vs-SNR baseline comparison plot:")
    print(policy_plot_path)
    print("Generated CSV table:")
    print(csv_path)
    print("Generated summary CSV tables:")
    print(pep_summary_csv_path)
    print(rho_summary_csv_path)
    print()
    print("Note: QLearning_Relay.py also generates a training-progress plot during training.")
    print("That plot cannot be reconstructed from this saved model because episode history is not stored in the .npz file.")


if __name__ == "__main__":
    main()
