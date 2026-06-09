import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import kve


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import QLearning_Relay as ql  # noqa: E402


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "dense_rho_sweep"
PERCENTILES = (10, 25, 40, 50, 60, 75, 90)


def vectorized_user_pep(snr_db, delta_u, sigma_n2, eta, rho_grid, relay_gain):
    snr_linear = 10.0 ** (snr_db / 10.0)
    pb = snr_linear * sigma_n2
    pr = eta * rho_grid * relay_gain * pb

    g_fixed = np.sqrt(
        pr / ((1.0 - rho_grid) * ql.LAMBDA * pb + (2.0 - rho_grid) * sigma_n2 + ql.EPS)
    )
    g_fixed_prime = g_fixed * np.sqrt(1.0 - rho_grid)
    g_relay = np.sqrt(g_fixed**2 + g_fixed_prime**2)
    signal_term = g_fixed**2 * (1.0 - rho_grid) * pb
    interference_term = g_fixed**2 * sigma_n2 + g_relay**2 * pb * (delta_u**2)

    sum_total = np.zeros_like(rho_grid, dtype=float)

    for ell in range(ql.SELECTED_RELAY_ORDER):
        for ii in range(ql.SELECTED_USER_ORDER):
            k_prime = ql.K - ql.SELECTED_RELAY_ORDER + ell + 1
            m_prime = ql.M - ql.SELECTED_USER_ORDER + ii + 1

            uk = np.sqrt(
                1.0
                + (4.0 * m_prime * interference_term) / (ql.LAMBDA * signal_term + ql.EPS)
            )
            ak = k_prime * (1.0 - 1.0 / (uk**2)) / (2.0 * ql.LAMBDA * signal_term + ql.EPS)
            valid = ak > 0.0

            term = np.zeros_like(rho_grid, dtype=float)
            term[valid] = (
                ql.math.comb(ql.SELECTED_RELAY_ORDER - 1, ell)
                * ((-1) ** ell)
                * ql.math.comb(ql.SELECTED_USER_ORDER - 1, ii)
                * ((-1) ** ii)
                * (ak[valid] / (k_prime * m_prime * uk[valid] + ql.EPS))
                * (kve(1, ak[valid]) - kve(0, ak[valid]))
            )

            finite = np.isfinite(term)
            sum_total[finite] += term[finite]

    pep_value = 0.5 - (ql.AK * ql.AM / 2.0) * sum_total
    return np.clip(np.real_if_close(pep_value), 1e-8, 1.0).astype(float)


def vectorized_avg_pep(state_vars, rho_grid, relay_gain):
    near = vectorized_user_pep(
        state_vars["snr_db"],
        state_vars["delta"][0],
        state_vars["sigma_n2"],
        state_vars["eta"],
        rho_grid,
        relay_gain,
    )
    far = vectorized_user_pep(
        state_vars["snr_db"],
        state_vars["delta"][1],
        state_vars["sigma_n2"],
        state_vars["eta"],
        rho_grid,
        relay_gain,
    )
    return (near + far) / 2.0


def sample_state_vars(rng):
    delta_near = ql.sample_from_edges(rng, ql.DELTA_EDGES, ql.DELTA_BIN_WEIGHTS)
    delta_far = ql.sample_delta_above(rng, delta_near)
    return {
        "snr_db": ql.sample_from_edges(rng, ql.SNR_EDGES),
        "delta": np.array([delta_near, delta_far], dtype=float),
        "sigma_n2": ql.sample_from_edges(rng, ql.SIGMA_EDGES),
        "eta": ql.sample_from_edges(rng, ql.ETA_EDGES),
    }


def sample_relay_components(rng):
    relay_distances_m = rng.uniform(ql.RELAY_DISTANCE_MIN_M, ql.RELAY_DISTANCE_MAX_M, size=ql.K)
    relay_distances = relay_distances_m / ql.REFERENCE_DISTANCE_M
    x_values = rng.normal(0.0, 1.0, size=ql.K)
    y_values = rng.normal(0.0, 1.0, size=ql.K)
    fading_power = (x_values**2 + y_values**2) / 2.0
    return relay_distances, fading_power


def selected_gain_for_alpha(relay_distances, fading_power, alpha):
    relay_omega = ql.PATH_LOSS_REFERENCE_GAIN * relay_distances ** (-alpha)
    relay_gains = relay_omega * fading_power
    selected_relay_index = int(np.argmax(relay_gains))
    return selected_relay_index, float(relay_gains[selected_relay_index])


def optimal_rho_for_gain(state_vars, rho_grid, relay_gain):
    avg_pep = vectorized_avg_pep(state_vars, rho_grid, relay_gain)
    best_index = int(np.argmin(avg_pep))
    return float(rho_grid[best_index]), float(avg_pep[best_index])


def summarize(values, min_peps, gains=None):
    values = np.asarray(values, dtype=float)
    min_peps = np.asarray(min_peps, dtype=float)
    summary = {
        "rho_min": float(np.min(values)),
        "rho_max": float(np.max(values)),
        "rho_mean": float(np.mean(values)),
        "rho_std": float(np.std(values)),
        "min_avg_pep_mean": float(np.mean(min_peps)),
        "min_avg_pep_median": float(np.median(min_peps)),
        "share_rho_le_0_005": float(np.mean(values <= 0.005)),
        "share_rho_le_0_010": float(np.mean(values <= 0.010)),
        "share_rho_le_0_020": float(np.mean(values <= 0.020)),
        "share_rho_le_0_050": float(np.mean(values <= 0.050)),
        "share_rho_le_0_100": float(np.mean(values <= 0.100)),
        "share_rho_le_0_200": float(np.mean(values <= 0.200)),
    }
    for percentile in PERCENTILES:
        summary[f"rho_p{percentile}"] = float(np.percentile(values, percentile))

    if gains is not None:
        gains = np.asarray(gains, dtype=float)
        summary.update(
            {
                "gain_min": float(np.min(gains)),
                "gain_max": float(np.max(gains)),
                "gain_mean": float(np.mean(gains)),
                "gain_median": float(np.median(gains)),
                "gain_p40": float(np.percentile(gains, 40)),
                "gain_p60": float(np.percentile(gains, 60)),
            }
        )
    else:
        summary.update(
            {
                "gain_min": 1.0,
                "gain_max": 1.0,
                "gain_mean": 1.0,
                "gain_median": 1.0,
                "gain_p40": 1.0,
                "gain_p60": 1.0,
            }
        )
    return summary


def write_summary_csv(summary_path, summaries):
    fieldnames = ["case"] + list(next(iter(summaries.values())).keys())
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, summary in summaries.items():
            writer.writerow({"case": case_name, **summary})


def write_report(report_path, args, summaries):
    with report_path.open("w") as report:
        report.write("Dense rho sweep report\n")
        report.write(f"states: {args.states}\n")
        report.write(f"seed: {args.seed}\n")
        report.write(f"rho grid: {args.rho_min} to {args.rho_max}, points={args.rho_points}\n")
        report.write("optimal rho minimizes average PEP = (PEP_near + PEP_far) / 2\n\n")
        for case_name, summary in summaries.items():
            report.write(f"{case_name}\n")
            report.write(
                "rho percentiles p10/p25/p40/p50/p60/p75/p90: "
                + " ".join(f"{summary[f'rho_p{p}']:.6f}" for p in PERCENTILES)
                + "\n"
            )
            report.write(
                f"rho min/max/mean/std: {summary['rho_min']:.6f} "
                f"{summary['rho_max']:.6f} {summary['rho_mean']:.6f} {summary['rho_std']:.6f}\n"
            )
            report.write(
                f"share rho <= 0.005/0.010/0.020/0.050/0.100/0.200: "
                f"{100.0 * summary['share_rho_le_0_005']:.2f}% "
                f"{100.0 * summary['share_rho_le_0_010']:.2f}% "
                f"{100.0 * summary['share_rho_le_0_020']:.2f}% "
                f"{100.0 * summary['share_rho_le_0_050']:.2f}% "
                f"{100.0 * summary['share_rho_le_0_100']:.2f}% "
                f"{100.0 * summary['share_rho_le_0_200']:.2f}%\n"
            )
            report.write(
                f"gain mean/median/p40/p60: {summary['gain_mean']:.6f} "
                f"{summary['gain_median']:.6f} {summary['gain_p40']:.6f} {summary['gain_p60']:.6f}\n"
            )
            report.write(
                f"min avg PEP mean/median: {summary['min_avg_pep_mean']:.8f} "
                f"{summary['min_avg_pep_median']:.8f}\n\n"
            )


def run_sweep(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_state_path = output_dir / "dense_rho_sweep_per_state.csv"
    summary_path = output_dir / "dense_rho_sweep_summary.csv"
    report_path = output_dir / "dense_rho_sweep_report.txt"
    log_path = output_dir / "dense_rho_sweep_progress.log"

    rng = np.random.default_rng(args.seed)
    rho_grid = np.linspace(args.rho_min, args.rho_max, args.rho_points)

    results = {
        "relay_alpha_2": {"rho": [], "pep": [], "gain": []},
        "relay_alpha_3": {"rho": [], "pep": [], "gain": []},
        "lambda_1_no_relay_selection": {"rho": [], "pep": [], "gain": []},
    }

    with per_state_path.open("w", newline="") as csv_file, log_path.open("w") as log_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "state_index",
                "snr_db",
                "delta_near",
                "delta_far",
                "sigma_n2",
                "eta",
                "relay_alpha_2_selected_relay",
                "relay_alpha_2_gain",
                "relay_alpha_2_optimal_rho",
                "relay_alpha_2_min_avg_pep",
                "relay_alpha_3_selected_relay",
                "relay_alpha_3_gain",
                "relay_alpha_3_optimal_rho",
                "relay_alpha_3_min_avg_pep",
                "lambda_1_optimal_rho",
                "lambda_1_min_avg_pep",
            ]
        )

        for state_index in range(args.states):
            state_vars = sample_state_vars(rng)
            relay_distances, fading_power = sample_relay_components(rng)

            row = [
                state_index,
                state_vars["snr_db"],
                state_vars["delta"][0],
                state_vars["delta"][1],
                state_vars["sigma_n2"],
                state_vars["eta"],
            ]

            for alpha, case_name in ((2.0, "relay_alpha_2"), (3.0, "relay_alpha_3")):
                selected_relay_index, gain = selected_gain_for_alpha(relay_distances, fading_power, alpha)
                optimal_rho, min_pep = optimal_rho_for_gain(state_vars, rho_grid, gain)
                results[case_name]["rho"].append(optimal_rho)
                results[case_name]["pep"].append(min_pep)
                results[case_name]["gain"].append(gain)
                row.extend([selected_relay_index + 1, gain, optimal_rho, min_pep])

            optimal_rho, min_pep = optimal_rho_for_gain(state_vars, rho_grid, ql.LAMBDA)
            results["lambda_1_no_relay_selection"]["rho"].append(optimal_rho)
            results["lambda_1_no_relay_selection"]["pep"].append(min_pep)
            results["lambda_1_no_relay_selection"]["gain"].append(ql.LAMBDA)
            row.extend([optimal_rho, min_pep])
            writer.writerow(row)

            if (state_index + 1) % args.progress_every == 0 or state_index + 1 == args.states:
                message = f"completed {state_index + 1}/{args.states} states"
                print(message, flush=True)
                log_file.write(message + "\n")
                log_file.flush()

    summaries = {
        case_name: summarize(
            case_values["rho"],
            case_values["pep"],
            case_values["gain"],
        )
        for case_name, case_values in results.items()
    }
    write_summary_csv(summary_path, summaries)
    write_report(report_path, args, summaries)
    return per_state_path, summary_path, report_path, log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a file-backed dense rho sweep.")
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rho-min", type=float, default=0.001)
    parser.add_argument("--rho-max", type=float, default=0.999)
    parser.add_argument("--rho-points", type=int, default=999)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = run_sweep(args)
    print("Saved dense sweep outputs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
