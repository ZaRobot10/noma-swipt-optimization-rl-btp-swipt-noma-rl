import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kve


# ===================== System Parameters =====================
K = 8 # relay
M = 3 # User
LAMBDA = 1.0
REFERENCE_DISTANCE_M = 100.0
RELAY_DISTANCE_MIN_M = 10.0
RELAY_DISTANCE_MAX_M = 90.0
PATH_LOSS_REFERENCE_GAIN = 0.02
PATH_LOSS_EXPONENTS = np.array([2.0, 3.0], dtype=float)
SELECTED_RELAY_ORDER = 8
SELECTED_USER_ORDER = 2
BASELINE_RHO = 0.5
BASELINE_SNR_DB = 20.0
BASELINE_DELTA = np.array([0.01, 0.10], dtype=float)
BASELINE_SIGMA_N2 = 1.0
BASELINE_ETA = 0.8

EPS = np.finfo(float).eps
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = OUTPUT_DIR / "output"
MODEL_PATH = MODEL_OUTPUT_DIR / "q_learning_model.npz"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "plots"

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


# ===================== RL Design =====================
# Single-step Q-learning with gamma = 0 is equivalent to a contextual bandit,
# which is the right fit here because rho only affects the current episode.
TRAINING_EPISODES = 100000
EPSILON_START = 1.0
EPSILON_END = 0.05
# Tabular Q-learning needs a finite action set, so rho is discretized here.
RHO_ACTIONS = np.linspace(0.05, 0.95, 19)
DELTA_EVAL_GRID = np.linspace(0.01, 0.20, 10)
ETA_EVAL_GRID = np.linspace(0.6, 1.0, 9)

# Continuous channel conditions are aggregated into bins so nearby samples
# share Q-values and the agent can learn from repeated visits.
SNR_EDGES = np.linspace(0.0, 40.0, 7)  # 6 bins
DELTA_EDGES = np.array([0.0, 0.03, 0.08, 0.20, 1.0])  # 4 bins
SIGMA_EDGES = np.linspace(0.5, 1.0, 4)  # 3 bins
ETA_EDGES = np.linspace(0.6, 1.0, 4)  # 3 bins
# Previous relay-gain bins after C0 scaling:
# RELAY_GAIN_EDGES = np.array([0.0, 0.25, 0.75, 1.5, 3.0, 6.0, 15.0, 60.0])
RELAY_GAIN_EDGES = np.array([0.0, 0.35, 0.75, 1.25, 2.25, 4.0, 8.0, 30.0])

DELTA_BIN_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10], dtype=float)
DELTA_BIN_WEIGHTS /= DELTA_BIN_WEIGHTS.sum()

AK = math.factorial(K) / (
    math.factorial(SELECTED_RELAY_ORDER - 1) * math.factorial(K - SELECTED_RELAY_ORDER)
)
AM = math.factorial(M) / (
    math.factorial(SELECTED_USER_ORDER - 1) * math.factorial(M - SELECTED_USER_ORDER)
)


@dataclass(frozen=True)
class State:
    snr_db: float
    delta: np.ndarray
    sigma_n2: float
    eta: float
    path_loss_exponent: float = 2.0
    relay_distances_m: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    # Normalized distance d = distance_m / REFERENCE_DISTANCE_M, used in Omega(d).
    relay_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    relay_omega: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    relay_channels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=complex))
    relay_gains: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    selected_relay_index: int = -1
    selected_relay_gain: float = LAMBDA


def clip_open_interval(value, low, high):
    low_open = np.nextafter(low, high)
    high_open = np.nextafter(high, low)
    if not low_open < high_open:
        return float((low + high) / 2.0)
    return float(np.clip(value, low_open, high_open))


def sample_from_edges(rng, edges, weights=None):
    bin_index = int(rng.choice(len(edges) - 1, p=weights))
    low = edges[bin_index]
    high = edges[bin_index + 1]
    return clip_open_interval(rng.uniform(low, high), low, high)


def sample_delta_above(rng, lower_bound):
    conditional_weights = np.zeros(len(DELTA_EDGES) - 1, dtype=float)

    for bin_index, base_weight in enumerate(DELTA_BIN_WEIGHTS):
        bin_low = DELTA_EDGES[bin_index]
        bin_high = DELTA_EDGES[bin_index + 1]
        valid_low = max(bin_low, lower_bound)
        valid_length = max(bin_high - valid_low, 0.0)
        bin_width = bin_high - bin_low
        conditional_weights[bin_index] = base_weight * (valid_length / bin_width)

    weight_sum = conditional_weights.sum()
    if weight_sum <= 0.0:
        raise ValueError(f"No feasible Delta_far value exists above Delta_near={lower_bound}.")

    conditional_weights /= weight_sum
    bin_index = int(rng.choice(len(DELTA_EDGES) - 1, p=conditional_weights))
    low = max(DELTA_EDGES[bin_index], lower_bound)
    high = DELTA_EDGES[bin_index + 1]
    return clip_open_interval(rng.uniform(low, high), low, high)


def sample_relay_block_fading(rng):
    path_loss_exponent = float(rng.choice(PATH_LOSS_EXPONENTS))
    relay_distances_m = rng.uniform(RELAY_DISTANCE_MIN_M, RELAY_DISTANCE_MAX_M, size=K)
    relay_distances = relay_distances_m / REFERENCE_DISTANCE_M

    relay_omega = PATH_LOSS_REFERENCE_GAIN * relay_distances ** (-path_loss_exponent)
    x_values = rng.normal(0.0, 1.0, size=K)
    y_values = rng.normal(0.0, 1.0, size=K)
    relay_channels = np.sqrt(relay_omega / 2.0) * (x_values + 1j * y_values)
    relay_gains = np.abs(relay_channels) ** 2

    selected_relay_index = int(np.argmax(relay_gains))
    selected_relay_gain = float(relay_gains[selected_relay_index])

    return (
        relay_distances_m,
        relay_distances,
        relay_omega,
        path_loss_exponent,
        relay_channels,
        relay_gains,
        selected_relay_index,
        selected_relay_gain,
    )


def sample_state(rng):
    # Each episode still draws continuous values; only the Q-table index is binned.
    delta_near = sample_from_edges(rng, DELTA_EDGES, DELTA_BIN_WEIGHTS)
    # Enforce the system assumption that the far user experiences larger
    # residual SIC error than the near user.
    delta_far = sample_delta_above(rng, delta_near)
    (
        relay_distances_m,
        relay_distances,
        relay_omega,
        path_loss_exponent,
        relay_channels,
        relay_gains,
        selected_relay_index,
        selected_relay_gain,
    ) = sample_relay_block_fading(rng)
    return State(
        snr_db=sample_from_edges(rng, SNR_EDGES),
        delta=np.array(
            [delta_near, delta_far],
            dtype=float,
        ),
        sigma_n2=sample_from_edges(rng, SIGMA_EDGES),
        eta=sample_from_edges(rng, ETA_EDGES),
        path_loss_exponent=path_loss_exponent,
        relay_distances_m=relay_distances_m,
        relay_distances=relay_distances,
        relay_omega=relay_omega,
        relay_channels=relay_channels,
        relay_gains=relay_gains,
        selected_relay_index=selected_relay_index,
        selected_relay_gain=selected_relay_gain,
    )


def make_reference_state(snr_db, delta, sigma_n2, eta, seed=12345):
    (
        relay_distances_m,
        relay_distances,
        relay_omega,
        path_loss_exponent,
        relay_channels,
        relay_gains,
        selected_relay_index,
        selected_relay_gain,
    ) = sample_relay_block_fading(np.random.default_rng(seed))
    return State(
        snr_db=snr_db,
        delta=np.asarray(delta, dtype=float),
        sigma_n2=sigma_n2,
        eta=eta,
        path_loss_exponent=path_loss_exponent,
        relay_distances_m=relay_distances_m,
        relay_distances=relay_distances,
        relay_omega=relay_omega,
        relay_channels=relay_channels,
        relay_gains=relay_gains,
        selected_relay_index=selected_relay_index,
        selected_relay_gain=selected_relay_gain,
    )


def discretize(value, edges):
    return int(np.digitize(value, edges[1:-1], right=False))


def state_to_index(state):
    # Map a continuous state to a discrete bucket for tabular learning.
    return (
        discretize(state.snr_db, SNR_EDGES),
        discretize(state.delta[0], DELTA_EDGES),
        discretize(state.delta[1], DELTA_EDGES),
        discretize(state.sigma_n2, SIGMA_EDGES),
        discretize(state.eta, ETA_EDGES),
        discretize(state.selected_relay_gain, RELAY_GAIN_EDGES),
    )


def compute_user_pep(snr_db, delta_u, sigma_n2, eta, rho, user_index, relay_gain=LAMBDA):
    snr_linear = 10.0 ** (snr_db / 10.0)
    pb = snr_linear * sigma_n2
    pr = eta * rho * relay_gain * pb

    # The updated closed form uses the same combinatorial orders for both users;
    # the user-specific dependence enters through delta_u only.
    _ = user_index

    g_fixed = np.sqrt(pr / ((1.0 - rho) * LAMBDA * pb + (2.0 - rho) * sigma_n2 + EPS))
    g_fixed_prime = g_fixed * np.sqrt(1.0 - rho)
    g_relay = np.sqrt(g_fixed**2 + g_fixed_prime**2)
    signal_term = g_fixed**2 * (1.0 - rho) * pb
    interference_term = g_fixed**2 * sigma_n2 + g_relay**2 * pb * (delta_u**2)

    sum_total = 0.0

    for ell in range(SELECTED_RELAY_ORDER):
        for ii in range(SELECTED_USER_ORDER):
            k_prime = K - SELECTED_RELAY_ORDER + ell + 1
            m_prime = M - SELECTED_USER_ORDER + ii + 1

            uk = np.sqrt(
                1.0
                + (4.0 * m_prime * interference_term) / (LAMBDA * signal_term + EPS)
            )
            ak = k_prime * (1.0 - 1.0 / (uk**2)) / (2.0 * LAMBDA * signal_term + EPS)

            if ak <= 0.0:
                continue

            term = (
                math.comb(SELECTED_RELAY_ORDER - 1, ell)
                * ((-1) ** ell)
                * math.comb(SELECTED_USER_ORDER - 1, ii)
                * ((-1) ** ii)
                * (ak / (k_prime * m_prime * uk + EPS))
                * (kve(1, ak) - kve(0, ak))
            )

            if np.isfinite(term):
                sum_total += term

    pep_value = 0.5 - (AK * AM / 2.0) * sum_total
    return float(np.clip(np.real_if_close(pep_value), 1e-8, 1.0))


def compute_pep_pair(state, rho):
    return np.array(
        [
            compute_user_pep(
                state.snr_db,
                state.delta[0],
                state.sigma_n2,
                state.eta,
                rho,
                0,
                state.selected_relay_gain,
            ),
            compute_user_pep(
                state.snr_db,
                state.delta[1],
                state.sigma_n2,
                state.eta,
                rho,
                1,
                state.selected_relay_gain,
            ),
        ],
        dtype=float,
    )


def pareto_front_metrics(costs):
    # A lower PEP vector is better; non-dominated actions form the Pareto front.
    num_actions = costs.shape[0]
    domination_counts = np.zeros(num_actions, dtype=int)
    frontier_mask = np.ones(num_actions, dtype=bool)

    for i in range(num_actions):
        for j in range(num_actions):
            if i == j:
                continue
            dominates = np.all(costs[j] <= costs[i] + 1e-12) and np.any(costs[j] < costs[i] - 1e-12)
            if dominates:
                domination_counts[i] += 1
                frontier_mask[i] = False

    return frontier_mask, domination_counts


def pareto_reward(costs, action_index):
    # Reward combines three signals:
    # 1) lower normalized PEPs for both users,
    # 2) a penalty if other rho values dominate this action,
    # 3) a bonus if this action lies on the Pareto front.
    normalized_costs = (costs - costs.min(axis=0)) / np.maximum(np.ptp(costs, axis=0), EPS)
    hypervolume_scores = np.prod(1.1 - normalized_costs, axis=1)
    frontier_mask, domination_counts = pareto_front_metrics(costs)

    reward = hypervolume_scores[action_index] - domination_counts[action_index] / max(len(costs) - 1, 1)
    if frontier_mask[action_index]:
        reward += 0.5

    return float(reward), bool(frontier_mask[action_index]), float(hypervolume_scores[action_index])


def epsilon_for_episode(episode):
    decay_fraction = episode / max(TRAINING_EPISODES - 1, 1)
    return float(EPSILON_END + (EPSILON_START - EPSILON_END) * (1.0 - decay_fraction))


def select_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))

    best_indices = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(rng.choice(best_indices))


def greedy_action(q_values):
    best_indices = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(best_indices[0])


def rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / window
    averaged = np.convolve(values, kernel, mode="valid")
    padded = np.full(values.size, np.nan, dtype=float)
    padded[window - 1 :] = averaged
    return padded


def train_agent(seed=7):
    rng = np.random.default_rng(seed)
    q_shape = (
        len(SNR_EDGES) - 1,
        len(DELTA_EDGES) - 1,
        len(DELTA_EDGES) - 1,
        len(SIGMA_EDGES) - 1,
        len(ETA_EDGES) - 1,
        len(RELAY_GAIN_EDGES) - 1,
        len(RHO_ACTIONS),
    )
    q_table = np.zeros(q_shape, dtype=float)
    visit_counts = np.zeros(q_shape, dtype=int)

    history = {
        "reward": [],
        "pep_near": [],
        "pep_far": [],
        "rho": [],
        "epsilon": [],
        "pareto_hit": [],
        "hypervolume": [],
    }

    for episode in range(TRAINING_EPISODES):
        epsilon = epsilon_for_episode(episode)
        state = sample_state(rng)
        state_index = state_to_index(state)
        q_values = q_table[state_index]
        action_index = select_action(q_values, epsilon, rng)

        # Evaluate every rho for the current state so the reward can compare the
        # chosen action against the state-wise Pareto frontier.
        all_peps = np.vstack([compute_pep_pair(state, rho) for rho in RHO_ACTIONS])
        chosen_peps = all_peps[action_index]
        reward, pareto_hit, hypervolume = pareto_reward(all_peps, action_index)

        state_action_index = state_index + (action_index,)
        visit_counts[state_action_index] += 1
        # With gamma = 0 there is no bootstrap term; the update tracks expected
        # immediate reward for this state-action pair.
        alpha = 1.0 / visit_counts[state_action_index]
        q_table[state_action_index] += alpha * (reward - q_table[state_action_index])

        history["reward"].append(reward)
        history["pep_near"].append(chosen_peps[0])
        history["pep_far"].append(chosen_peps[1])
        history["rho"].append(RHO_ACTIONS[action_index])
        history["epsilon"].append(epsilon)
        history["pareto_hit"].append(float(pareto_hit))
        history["hypervolume"].append(hypervolume)

    return q_table, history


def save_q_learning_model(q_table, seed=7, output_path=MODEL_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        q_table=q_table,
        rho_actions=RHO_ACTIONS,
        snr_edges=SNR_EDGES,
        delta_edges=DELTA_EDGES,
        sigma_edges=SIGMA_EDGES,
        eta_edges=ETA_EDGES,
        relay_gain_edges=RELAY_GAIN_EDGES,
        path_loss_reference_gain=np.array([PATH_LOSS_REFERENCE_GAIN], dtype=float),
        path_loss_exponents=PATH_LOSS_EXPONENTS,
        reference_distance_m=np.array([REFERENCE_DISTANCE_M], dtype=float),
        relay_distance_min_m=np.array([RELAY_DISTANCE_MIN_M], dtype=float),
        relay_distance_max_m=np.array([RELAY_DISTANCE_MAX_M], dtype=float),
        baseline_snr_db=np.array([BASELINE_SNR_DB], dtype=float),
        baseline_delta=BASELINE_DELTA,
        baseline_sigma_n2=np.array([BASELINE_SIGMA_N2], dtype=float),
        baseline_eta=np.array([BASELINE_ETA], dtype=float),
        training_episodes=np.array([TRAINING_EPISODES], dtype=int),
        epsilon_start=np.array([EPSILON_START], dtype=float),
        epsilon_end=np.array([EPSILON_END], dtype=float),
        seed=np.array([seed], dtype=int),
        selected_relay_order=np.array([SELECTED_RELAY_ORDER], dtype=int),
        selected_user_order=np.array([SELECTED_USER_ORDER], dtype=int),
        k_relays=np.array([K], dtype=int),
        m_users=np.array([M], dtype=int),
    )
    return output_path


def evaluate_policy(q_table):
    snr_grid = np.arange(0.0, 41.0, 2.0)
    learned_rho = []
    learned_pep = []
    baseline_pep = []

    # Sweep SNR while keeping the other parameters fixed to show how the learned
    # rho policy behaves against the original fixed-rho baseline.
    for snr_db in snr_grid:
        state = make_reference_state(
            snr_db=snr_db,
            delta=BASELINE_DELTA.copy(),
            sigma_n2=BASELINE_SIGMA_N2,
            eta=BASELINE_ETA,
        )
        action_index = greedy_action(q_table[state_to_index(state)])
        rho_star = RHO_ACTIONS[action_index]

        learned_rho.append(rho_star)
        learned_pep.append(compute_pep_pair(state, rho_star))
        baseline_pep.append(compute_pep_pair(state, BASELINE_RHO))

    learned_pep = np.vstack(learned_pep)
    baseline_pep = np.vstack(baseline_pep)

    rng = np.random.default_rng(99)
    test_states = [sample_state(rng) for _ in range(400)]
    learned_test_peps = []
    baseline_test_peps = []
    pareto_hits = []

    for state in test_states:
        action_index = greedy_action(q_table[state_to_index(state)])
        learned_test_peps.append(compute_pep_pair(state, RHO_ACTIONS[action_index]))
        baseline_test_peps.append(compute_pep_pair(state, BASELINE_RHO))
        state_costs = np.vstack([compute_pep_pair(state, rho) for rho in RHO_ACTIONS])
        frontier_mask, _ = pareto_front_metrics(state_costs)
        pareto_hits.append(float(frontier_mask[action_index]))

    return {
        "snr_grid": snr_grid,
        "learned_rho": np.asarray(learned_rho, dtype=float),
        "learned_pep": learned_pep,
        "baseline_pep": baseline_pep,
        "learned_test_peps": np.vstack(learned_test_peps),
        "baseline_test_peps": np.vstack(baseline_test_peps),
        "test_pareto_hit_rate": float(np.mean(pareto_hits)),
    }


def evaluate_delta_sensitivity(q_table):
    scenarios = {
        "delta1_only": {
            "title": rf"Varying $\delta_1$ with $\delta_2$ = {BASELINE_DELTA[1]:.2f}",
            "x_label": r"$\delta_1$",
            "states": [
                make_reference_state(
                    snr_db=BASELINE_SNR_DB,
                    delta=np.array([delta_value, BASELINE_DELTA[1]], dtype=float),
                    sigma_n2=BASELINE_SIGMA_N2,
                    eta=BASELINE_ETA,
                )
                for delta_value in DELTA_EVAL_GRID
            ],
        },
        "delta2_only": {
            "title": rf"Varying $\delta_2$ with $\delta_1$ = {BASELINE_DELTA[0]:.2f}",
            "x_label": r"$\delta_2$",
            "states": [
                make_reference_state(
                    snr_db=BASELINE_SNR_DB,
                    delta=np.array([BASELINE_DELTA[0], delta_value], dtype=float),
                    sigma_n2=BASELINE_SIGMA_N2,
                    eta=BASELINE_ETA,
                )
                for delta_value in DELTA_EVAL_GRID
            ],
        },
        "delta_both": {
            "title": r"Joint variation: $\delta_1 = \delta_2$",
            "x_label": "Common SIC imperfection value",
            "states": [
                make_reference_state(
                    snr_db=BASELINE_SNR_DB,
                    delta=np.array([delta_value, delta_value], dtype=float),
                    sigma_n2=BASELINE_SIGMA_N2,
                    eta=BASELINE_ETA,
                )
                for delta_value in DELTA_EVAL_GRID
            ],
        },
    }

    results = {}
    for scenario_name, config in scenarios.items():
        learned_rho = []
        learned_pep = []
        baseline_pep = []

        for state in config["states"]:
            action_index = greedy_action(q_table[state_to_index(state)])
            rho_star = RHO_ACTIONS[action_index]

            learned_rho.append(rho_star)
            learned_pep.append(compute_pep_pair(state, rho_star))
            baseline_pep.append(compute_pep_pair(state, BASELINE_RHO))

        results[scenario_name] = {
            "title": config["title"],
            "x_label": config["x_label"],
            "x_values": DELTA_EVAL_GRID.copy(),
            "learned_rho": np.asarray(learned_rho, dtype=float),
            "learned_pep": np.vstack(learned_pep),
            "baseline_pep": np.vstack(baseline_pep),
        }

    return results


def evaluate_eta_sensitivity(q_table):
    learned_rho = []
    learned_pep = []
    baseline_pep = []

    for eta_value in ETA_EVAL_GRID:
        state = make_reference_state(
            snr_db=BASELINE_SNR_DB,
            delta=BASELINE_DELTA.copy(),
            sigma_n2=BASELINE_SIGMA_N2,
            eta=eta_value,
        )
        action_index = greedy_action(q_table[state_to_index(state)])
        rho_star = RHO_ACTIONS[action_index]

        learned_rho.append(rho_star)
        learned_pep.append(compute_pep_pair(state, rho_star))
        baseline_pep.append(compute_pep_pair(state, BASELINE_RHO))

    return {
        "eta_grid": ETA_EVAL_GRID.copy(),
        "learned_rho": np.asarray(learned_rho, dtype=float),
        "learned_pep": np.vstack(learned_pep),
        "baseline_pep": np.vstack(baseline_pep),
    }


def plot_training_history(history, output_path):
    episodes = np.arange(1, len(history["reward"]) + 1)
    window = 250

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(episodes, history["reward"], color="steelblue", alpha=0.08, linewidth=0.8)
    axes[0, 0].plot(
        episodes,
        rolling_mean(history["reward"], window),
        color="midnightblue",
        linewidth=3,
        label=f"{window}-episode mean",
    )
    axes[0, 0].set_title("Pareto Reward During Training")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        episodes,
        rolling_mean(history["pep_near"], window),
        linewidth=3,
        color="darkorange",
        label="Near-user PEP",
    )
    axes[0, 1].plot(
        episodes,
        rolling_mean(history["pep_far"], window),
        linewidth=3,
        color="forestgreen",
        label="Far-user PEP",
    )
    axes[0, 1].set_title(r"Rolling Average PEP of Selected $\rho$")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average PEP")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        episodes,
        100.0 * rolling_mean(history["pareto_hit"], window),
        linewidth=3,
        color="firebrick",
        label="Pareto-front hit rate",
    )
    axes[1, 0].set_title(r"How Often the Agent Picks a Pareto-Optimal $\rho$")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Hit Rate (%)")
    axes[1, 0].set_ylim(0.0, 105.0)
    axes[1, 0].grid(True, alpha=0.3)

    epsilon_axis = axes[1, 0].twinx()
    epsilon_axis.plot(
        episodes,
        history["epsilon"],
        linewidth=2.4,
        color="slategray",
        linestyle="--",
        label="Epsilon",
    )
    epsilon_axis.set_ylabel("Epsilon")
    epsilon_axis.set_ylim(0.0, 1.05)

    axes[1, 1].plot(episodes, history["rho"], color="purple", alpha=0.08, linewidth=0.8)
    axes[1, 1].plot(
        episodes,
        rolling_mean(history["rho"], window),
        color="black",
        linewidth=3,
        label=rf"{window}-episode mean $\rho$",
    )
    axes[1, 1].set_title(r"$\rho$ Selected by the Agent")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel(r"$\rho$")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(r"Q-Learning on $\rho$ with $\gamma = 0$ (Contextual Bandit Formulation)", fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_policy_evaluation(results, output_path):
    snr_grid = results["snr_grid"]
    learned_pep = results["learned_pep"]
    baseline_pep = results["baseline_pep"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(snr_grid, results["learned_rho"], marker="o", linewidth=3, markersize=7, color="teal")
    axes[0].set_title(r"Greedy $\rho$ Chosen by the Learned Policy")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(
        snr_grid,
        learned_pep[:, 0],
        "-o",
        linewidth=3,
        markersize=7,
        color="darkorange",
        label=r"Near user, learned $\rho$",
    )
    axes[1].semilogy(
        snr_grid,
        learned_pep[:, 1],
        "-s",
        linewidth=3,
        markersize=7,
        color="forestgreen",
        label=r"Far user, learned $\rho$",
    )
    axes[1].semilogy(
        snr_grid,
        baseline_pep[:, 0],
        "--o",
        linewidth=2.4,
        markersize=7,
        color="peru",
        label=r"Near user, fixed $\rho = 0.5$",
    )
    axes[1].semilogy(
        snr_grid,
        baseline_pep[:, 1],
        "--s",
        linewidth=2.4,
        markersize=7,
        color="limegreen",
        label=r"Far user, fixed $\rho = 0.5$",
    )
    axes[1].set_title(r"PEP vs SNR: Learned Policy Compared with Fixed $\rho$")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Average PEP")
    axes[1].set_ylim(1e-5, 1.0)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_delta_sensitivity(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_state = make_reference_state(
        snr_db=BASELINE_SNR_DB,
        delta=BASELINE_DELTA,
        sigma_n2=BASELINE_SIGMA_N2,
        eta=BASELINE_ETA,
    )
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=False)

    for axis, scenario_key in zip(axes, ("delta1_only", "delta2_only", "delta_both")):
        scenario = results[scenario_key]
        x_values = scenario["x_values"]
        learned_pep = scenario["learned_pep"]
        baseline_pep = scenario["baseline_pep"]

        axis.semilogy(
            x_values,
            learned_pep[:, 0],
            "-o",
            linewidth=3,
            markersize=7,
            color="darkorange",
            label=r"Near user, learned $\rho$",
        )
        axis.semilogy(
            x_values,
            learned_pep[:, 1],
            "-s",
            linewidth=3,
            markersize=7,
            color="forestgreen",
            label=r"Far user, learned $\rho$",
        )
        axis.semilogy(
            x_values,
            baseline_pep[:, 0],
            "--o",
            linewidth=2.4,
            markersize=7,
            color="peru",
            label=r"Near user, fixed $\rho = 0.5$",
        )
        axis.semilogy(
            x_values,
            baseline_pep[:, 1],
            "--s",
            linewidth=2.4,
            markersize=7,
            color="limegreen",
            label=r"Far user, fixed $\rho = 0.5$",
        )
        axis.set_title(scenario["title"])
        axis.set_xlabel(scenario["x_label"])
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Average PEP")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", ncol=1, framealpha=0.92, fontsize=10)
    fig.suptitle(
        "PEP vs Delta Sensitivity\n"
        rf"$\mathrm{{SNR}} = {BASELINE_SNR_DB:.0f}\,\mathrm{{dB}},\ "
        rf"\sigma_n^2 = {BASELINE_SIGMA_N2:.1f},\ \eta = {BASELINE_ETA:.1f},\ "
        rf"\kappa = {reference_state.path_loss_exponent:.0f},\ "
        rf"|h_{{sr}}|^2 = {reference_state.selected_relay_gain:.2f}$",
        fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_eta_sensitivity(results, output_path):
    eta_grid = results["eta_grid"]
    learned_pep = results["learned_pep"]
    baseline_pep = results["baseline_pep"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(eta_grid, results["learned_rho"], marker="o", linewidth=3, markersize=7, color="teal")
    axes[0].set_title(r"Greedy $\rho$ Chosen by the Learned Policy")
    axes[0].set_xlabel(r"$\eta$ (Energy Conversion Efficiency)")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(
        eta_grid,
        learned_pep[:, 0],
        "-o",
        linewidth=3,
        markersize=7,
        color="darkorange",
        label=r"Near user, learned $\rho$",
    )
    axes[1].semilogy(
        eta_grid,
        learned_pep[:, 1],
        "-s",
        linewidth=3,
        markersize=7,
        color="forestgreen",
        label=r"Far user, learned $\rho$",
    )
    axes[1].semilogy(
        eta_grid,
        baseline_pep[:, 0],
        "--o",
        linewidth=2.4,
        markersize=7,
        color="peru",
        label=r"Near user, fixed $\rho = 0.5$",
    )
    axes[1].semilogy(
        eta_grid,
        baseline_pep[:, 1],
        "--s",
        linewidth=2.4,
        markersize=7,
        color="limegreen",
        label=r"Far user, fixed $\rho = 0.5$",
    )
    axes[1].set_title(r"PEP vs $\eta$: Learned Policy Compared with Fixed $\rho$")
    axes[1].set_xlabel(r"$\eta$ (Energy Conversion Efficiency)")
    axes[1].set_ylabel("Average PEP")
    axes[1].set_ylim(1e-5, 1.0)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "PEP vs " r"$\eta$" "\n"
        rf"$\mathrm{{SNR}} = {BASELINE_SNR_DB:.0f}\,\mathrm{{dB}},\ "
        rf"[\delta_1, \delta_2] = [{BASELINE_DELTA[0]:.2f}, {BASELINE_DELTA[1]:.2f}],\ "
        rf"\sigma_n^2 = {BASELINE_SIGMA_N2:.1f}$",
        fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    training_seed = 7
    q_table, history = train_agent(seed=training_seed)
    model_path = save_q_learning_model(q_table, seed=training_seed)
    evaluation_results = evaluate_policy(q_table)
    delta_sensitivity_results = evaluate_delta_sensitivity(q_table)
    eta_sensitivity_results = evaluate_eta_sensitivity(q_table)

    training_plot_path = PLOT_OUTPUT_DIR / "q_learning_training_progress.png"
    evaluation_plot_path = PLOT_OUTPUT_DIR / "q_learning_policy_evaluation.png"
    delta_plot_path = PLOT_OUTPUT_DIR / "q_learning_delta_sensitivity.png"
    eta_plot_path = PLOT_OUTPUT_DIR / "q_learning_eta_sensitivity.png"

    plot_training_history(history, training_plot_path)
    plot_policy_evaluation(evaluation_results, evaluation_plot_path)
    plot_delta_sensitivity(delta_sensitivity_results, delta_plot_path)
    plot_eta_sensitivity(eta_sensitivity_results, eta_plot_path)

    learned_test_mean = evaluation_results["learned_test_peps"].mean(axis=0)
    baseline_test_mean = evaluation_results["baseline_test_peps"].mean(axis=0)

    print("Training complete.")
    print(f"Episodes: {TRAINING_EPISODES}")
    print("Formulation: single-step Q-learning with gamma = 0")
    print(f"Reward mean over last 250 episodes: {np.mean(history['reward'][-250:]):.4f}")
    print(f"Pareto-front hit rate over last 250 episodes: {100.0 * np.mean(history['pareto_hit'][-250:]):.2f}%")
    print(f"Average learned rho over last 250 episodes: {np.mean(history['rho'][-250:]):.4f}")
    print(
        "Average test PEP (near user): "
        f"learned = {learned_test_mean[0]:.6f}, fixed rho=0.5 = {baseline_test_mean[0]:.6f}"
    )
    print(
        "Average test PEP (far user): "
        f"learned = {learned_test_mean[1]:.6f}, fixed rho=0.5 = {baseline_test_mean[1]:.6f}"
    )
    print(f"Test Pareto-front hit rate: {100.0 * evaluation_results['test_pareto_hit_rate']:.2f}%")
    print(f"Saved Q-learning model: {model_path}")
    print(f"Saved training plot: {training_plot_path}")
    print(f"Saved evaluation plot: {evaluation_plot_path}")
    print(f"Saved delta sensitivity plot: {delta_plot_path}")
    print(f"Saved eta sensitivity plot: {eta_plot_path}")


if __name__ == "__main__":
    main()
