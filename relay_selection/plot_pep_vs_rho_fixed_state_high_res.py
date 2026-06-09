from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import QLearning_Relay as ql


OUTPUT_DIR = ql.PLOT_OUTPUT_DIR
OUTPUT_PATH = OUTPUT_DIR / "pep_vs_rho_fixed_state_600dpi_large_text.png"
CSV_DIR = SCRIPT_DIR / "csv_tables"
CSV_PATH = CSV_DIR / "pep_vs_rho_fixed_state.csv"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
)

SNR_VALUES_DB = [5.0, 10.0, 20.0]
# SNR_VALUES_DB = [0, 3.0103, 6.0206]
# SNR_VALUES_DB = [0, 9.0309, 40]

RHO_GRID = np.linspace(0.05, 0.95, 200)
FIXED_ALPHA = 2.0
REFERENCE_RELAY_SEED = 12345


def make_fixed_alpha_reference_state(snr_db, delta, sigma_n2, eta, alpha=FIXED_ALPHA, seed=REFERENCE_RELAY_SEED):
    rng = np.random.default_rng(seed)
    relay_distances_m = rng.uniform(ql.RELAY_DISTANCE_MIN_M, ql.RELAY_DISTANCE_MAX_M, size=ql.K)
    relay_distances = relay_distances_m / ql.REFERENCE_DISTANCE_M
    relay_omega = ql.PATH_LOSS_REFERENCE_GAIN * relay_distances ** (-alpha)
    x_values = rng.normal(0.0, 1.0, size=ql.K)
    y_values = rng.normal(0.0, 1.0, size=ql.K)
    relay_channels = np.sqrt(relay_omega / 2.0) * (x_values + 1j * y_values)
    relay_gains = np.abs(relay_channels) ** 2
    selected_relay_index = int(np.argmax(relay_gains))
    selected_relay_gain = float(relay_gains[selected_relay_index])

    return ql.State(
        snr_db=snr_db,
        delta=np.asarray(delta, dtype=float),
        sigma_n2=sigma_n2,
        eta=eta,
        path_loss_exponent=float(alpha),
        relay_distances_m=relay_distances_m,
        relay_distances=relay_distances,
        relay_omega=relay_omega,
        relay_channels=relay_channels,
        relay_gains=relay_gains,
        selected_relay_index=selected_relay_index,
        selected_relay_gain=selected_relay_gain,
    )


def evaluate_pep_curves(snr_db, delta, sigma_n2, eta):
    state = make_fixed_alpha_reference_state(
        snr_db=snr_db,
        delta=np.asarray(delta, dtype=float),
        sigma_n2=sigma_n2,
        eta=eta,
    )
    pep_curves = np.vstack([ql.compute_pep_pair(state, rho) for rho in RHO_GRID])
    return state, pep_curves


def plot_pep_vs_rho():
    delta = ql.BASELINE_DELTA.copy()
    sigma_n2 = ql.BASELINE_SIGMA_N2
    eta = ql.BASELINE_ETA

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=False)
    reference_state = None

    for axis, snr_db in zip(axes, SNR_VALUES_DB):
        state, pep_curves = evaluate_pep_curves(snr_db, delta, sigma_n2, eta)
        if reference_state is None:
            reference_state = state

        axis.semilogy(
            RHO_GRID,
            pep_curves[:, 0],
            color="darkorange",
            linewidth=2.8,
            label="Near user PEP",
        )
        axis.semilogy(
            RHO_GRID,
            pep_curves[:, 1],
            color="forestgreen",
            linewidth=2.8,
            label="Far user PEP",
        )
        axis.set_title(rf"Average PEP vs $\rho$ at SNR = {snr_db:.0f} dB")
        axis.set_xlabel(r"$\rho$ (power-splitting ratio)")
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Average PEP")
    axes[1].legend(loc="best")

    fig.suptitle(
        r"PEP vs $\rho$ for Fixed State Variables" "\n"
        rf"$[\delta_1, \delta_2] = [{delta[0]:.2f}, {delta[1]:.2f}], "
        rf"\sigma_n^2 = {sigma_n2:.1f}, \eta = {eta:.1f}, "
        rf"\kappa = {reference_state.path_loss_exponent:.0f}, "
        rf"|h_{{sr}}|^2 = {reference_state.selected_relay_gain:.2f}$",
        fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_pep_vs_rho_csv():
    delta = ql.BASELINE_DELTA.copy()
    sigma_n2 = ql.BASELINE_SIGMA_N2
    eta = ql.BASELINE_ETA

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for snr_db in SNR_VALUES_DB:
        state, pep_curves = evaluate_pep_curves(snr_db, delta, sigma_n2, eta)
        avg_pep = pep_curves.mean(axis=1)
        for rho, near_pep, far_pep, average_pep in zip(
            RHO_GRID,
            pep_curves[:, 0],
            pep_curves[:, 1],
            avg_pep,
        ):
            rows.append(
                [
                    snr_db,
                    rho,
                    near_pep,
                    far_pep,
                    average_pep,
                    state.path_loss_exponent,
                    state.selected_relay_index + 1,
                    state.selected_relay_gain,
                ]
            )

    table = np.asarray(rows, dtype=float)
    header = "SNR_dB,rho,Near_user_PEP,Far_user_PEP,Average_PEP,kappa,selected_relay,selected_relay_gain"
    np.savetxt(CSV_PATH, table, delimiter=",", header=header, comments="", fmt="%.5f")


def main():
    plot_pep_vs_rho()
    save_pep_vs_rho_csv()
    print(f"Saved PEP vs rho plot to: {OUTPUT_PATH}")
    print(f"Saved PEP vs rho CSV table to: {CSV_PATH}")


if __name__ == "__main__":
    main()
