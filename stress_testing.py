# Stress-testing harness

from copy import deepcopy
import pandas as pd
import numpy as np

TERMINAL_FAIL_MIDPOINT_DAYS = 730.0
TERMINAL_FAIL_LOG_SCALE = 0.50
TERMINAL_FAIL_WEIGHT = 0.15

def fixed_terminal_fail_prob(delta_days: float, last_obs_idx: int, obs_vocab: list[str]) -> float:
    if obs_vocab[last_obs_idx] == "EXITLIKE":
        return 0.0
    if not np.isfinite(delta_days):
        return 0.0

    z = (np.log1p(float(delta_days)) - np.log1p(TERMINAL_FAIL_MIDPOINT_DAYS)) / TERMINAL_FAIL_LOG_SCALE
    z = float(np.clip(z, -30, 30))
    return 1.0 / (1.0 + np.exp(-z))

def build_terminal_pseudo_xi(gamma_last: np.ndarray, q_fail: float, allowed_next: dict[int, list[int]]) -> np.ndarray:
    n_states = len(STATES)
    xi_terminal = np.zeros((n_states, n_states), dtype=float)

    exit_idx = STATE_INDEX["Exit"]
    fail_idx = STATE_INDEX["Fail"]

    for i in range(n_states):
        mass = float(gamma_last[i])
        if mass <= 0:
            continue

        if i == exit_idx:
            xi_terminal[i, exit_idx] = mass
        elif i == fail_idx:
            xi_terminal[i, fail_idx] = mass
        else:
            xi_terminal[i, i] = mass * (1.0 - q_fail)
            if fail_idx in allowed_next[i]:
                xi_terminal[i, fail_idx] = mass * q_fail

    return xi_terminal

def make_cfg(**overrides):
    cfg = {
        "name": "baseline",
        "max_iter": MAX_ITER,
        "tol": TOL,
        "ridge_l2": RIDGE_L2,
        "emission_alpha": EMISSION_DIRICHLET,
        "learn_emissions": True,            # toggle hidden layer
        "fixed_emission_init": "heuristic", # heuristic or uniform
        "use_covariates": True,             # toggle covariates
        "covariate_mode": "log1p",          # log1p or winsor_log1p
        "winsor_q": (0.01, 0.99),
        "include_fail_state": True,         # toggle failure state
        "verbose": False,
    }
    cfg.update(overrides)
    return cfg

def build_allowed_next_states_cfg(include_fail_state=True):
    allowed = {}
    for s in PROGRESSION_STATES:
        i = STATE_INDEX[s]
        nxt = min(i + 1, STATE_INDEX["Series E"])
        choices = [i, nxt, STATE_INDEX["Exit"]]
        if include_fail_state:
            choices.append(STATE_INDEX["Fail"])
        allowed[i] = sorted(set(choices))
    allowed[STATE_INDEX["Exit"]] = [STATE_INDEX["Exit"]]
    allowed[STATE_INDEX["Fail"]] = [STATE_INDEX["Fail"]]
    return allowed

def apply_covariate_transform(df_in: pd.DataFrame, cfg: dict):
    x = df_in["log_raise"].to_numpy(dtype=float).copy()
    meta = {"mode": cfg["covariate_mode"], "clip_lo": None, "clip_hi": None}
    mask = np.isfinite(x)

    if cfg["covariate_mode"] == "winsor_log1p" and mask.sum() > 0:
        q_lo, q_hi = cfg["winsor_q"]
        lo, hi = np.quantile(x[mask], [q_lo, q_hi])
        x[mask] = np.clip(x[mask], lo, hi)
        meta["clip_lo"] = float(lo)
        meta["clip_hi"] = float(hi)
    elif cfg["covariate_mode"] != "log1p":
        raise ValueError(f"Unknown covariate_mode: {cfg['covariate_mode']}")

    return x, meta

def transform_size_point(size_usd_mn: float, cfg: dict, meta: dict):
    r = float(np.log1p(size_usd_mn))
    if cfg["covariate_mode"] == "winsor_log1p":
        lo, hi = meta["clip_lo"], meta["clip_hi"]
        if lo is not None:
            r = max(r, lo)
        if hi is not None:
            r = min(r, hi)
    return r

def initialize_emissions_experiment(obs_vocab: list[str], cfg: dict) -> np.ndarray:
    if cfg["fixed_emission_init"] == "uniform":
        return np.full((len(STATES), len(obs_vocab)), 1.0 / len(obs_vocab), dtype=float)
    return initialize_emissions(obs_vocab)

def transition_probs_for_time_cfg(i: int, x: np.ndarray, params_i: np.ndarray, choices: list[int]) -> np.ndarray:
    m = len(choices)
    if m == 1:
        return np.array([1.0], dtype=float)
    logits = np.zeros(m, dtype=float)
    for k in range(1, m):
        logits[k] = float(np.dot(params_i[k - 1], x))
    return softmax(logits)

def build_transition_matrices_for_sequence_cfg(r_seq_std, miss_seq, params, allowed_next, cfg):
    n_states = len(STATES)
    T = len(r_seq_std)
    A_list = []
    for t in range(1, T):
        if cfg["use_covariates"]:
            x = np.array([1.0, float(r_seq_std[t]), float(miss_seq[t])], dtype=float)
        else:
            x = np.array([1.0, 0.0, 0.0], dtype=float)

        A = np.zeros((n_states, n_states), dtype=float)
        for i in range(n_states):
            choices = allowed_next[i]
            if len(choices) == 1:
                A[i, choices[0]] = 1.0
                continue
            probs = transition_probs_for_time_cfg(i, x, params[i], choices)
            for idx, j in enumerate(choices):
                A[i, j] = probs[idx]
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-300)
        A_list.append(A)
    return A_list

def transition_probs_at_size_table_cfg(params, r_mu, r_sd, allowed_next, cfg, transform_meta):
    rows = []
    for size in SIZE_POINTS_USD_MN:
        r = transform_size_point(size, cfg, transform_meta)
        r_std = 0.0 if r_sd < 1e-8 else (r - r_mu) / r_sd
        if cfg["use_covariates"]:
            x = np.array([1.0, float(r_std), 0.0], dtype=float)
        else:
            x = np.array([1.0, 0.0, 0.0], dtype=float)

        for from_state in PROGRESSION_STATES:
            i = STATE_INDEX[from_state]
            choices = allowed_next[i]
            probs = transition_probs_for_time_cfg(i, x, params[i], choices)
            for idx, j in enumerate(choices):
                rows.append({
                    "from_state": from_state,
                    "deal_size_usd_mn": size,
                    "to_state": STATES[j],
                    "prob": float(probs[idx]),
                })
    return pd.DataFrame(rows)

def final_posterior_summaries(sequences, B, pi, params, allowed_next, cfg):
    n_states = len(STATES)
    gamma_totals = np.zeros(n_states, dtype=float)
    entropy_list = []

    for _, obs_seq, r_seq, m_seq, gap_seq in sequences:
        T = len(obs_seq)
        if T == 0:
            continue

        A_list = build_transition_matrices_for_sequence_cfg(r_seq, m_seq, params, allowed_next, cfg)
        logB = np.log(np.maximum(B[:, obs_seq], 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))

        log_alpha = np.full((T, n_states), -np.inf)
        log_alpha[0, :] = log_pi + logB[:, 0]
        for t in range(1, T):
            logA = np.log(np.maximum(A_list[t - 1], 1e-300))
            tmp = log_alpha[t - 1, :][:, None] + logA
            log_alpha[t, :] = logB[:, t] + logsumexp(tmp, axis=0)

        loglik = float(logsumexp(log_alpha[T - 1, :], axis=0))

        log_beta = np.full((T, n_states), -np.inf)
        log_beta[T - 1, :] = 0.0
        for t in range(T - 2, -1, -1):
            logA = np.log(np.maximum(A_list[t], 1e-300))
            tmp = logA + (logB[:, t + 1] + log_beta[t + 1, :])[None, :]
            log_beta[t, :] = logsumexp(tmp, axis=1)

        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)
        gamma_totals += gamma.sum(axis=0)

        ent = -np.sum(gamma * np.log(np.maximum(gamma, 1e-300)), axis=1)
        entropy_list.extend(ent.tolist())

    gamma_df = pd.DataFrame({"state": STATES, "posterior_mass": gamma_totals})
    gamma_df["posterior_mass"] = gamma_df["posterior_mass"] / gamma_df["posterior_mass"].sum()
    entropy_mean = float(np.mean(entropy_list)) if entropy_list else np.nan
    return gamma_df, entropy_mean

def fit_model_experiment(df_in: pd.DataFrame, cfg: dict):
    np.random.seed(RANDOM_SEED)
    allowed_next = build_allowed_next_states_cfg(cfg["include_fail_state"])

    df_local = df_in.copy()
    transformed_log_raise, transform_meta = apply_covariate_transform(df_local, cfg)
    df_local["log_raise_used"] = transformed_log_raise
    df_local["delta_days"] = pd.to_numeric(df_local["delta_days"], errors="coerce")

    obs_vocab = sorted(df_local["obs_token"].astype(str).unique().tolist())
    obs_index = {o: k for k, o in enumerate(obs_vocab)}
    df_local["obs_idx"] = df_local["obs_token"].map(obs_index).astype(int)

    tmp = df_local[["log_raise_used"]].rename(columns={"log_raise_used": "log_raise"})
    r_std, r_mu, r_sd = standardize_feature(tmp)
    miss = (~np.isfinite(df_local["log_raise"].to_numpy())).astype(float)
    df_local["r_std"] = r_std
    df_local["r_miss"] = miss

    sequences = []
    for cid, g in df_local.groupby("company_id", sort=False):
        sequences.append((
            int(cid),
            g["obs_idx"].to_numpy(dtype=int),
            g["r_std"].to_numpy(dtype=float),
            g["r_miss"].to_numpy(dtype=float),
            g["delta_days"].to_numpy(dtype=float),
            ))

    n_states = len(STATES)
    n_obs = len(obs_vocab)

    B = initialize_emissions_experiment(obs_vocab, cfg)
    pi = initialize_pi()

    p = 3
    params = {}
    for i in range(n_states):
        m_i = len(allowed_next[i])
        params[i] = np.zeros((max(m_i - 1, 0), p), dtype=float)

    prev_ll = None
    ll_total = np.nan
    xi_total = np.zeros((n_states, n_states), dtype=float)

    for it in range(1, cfg["max_iter"] + 1):
        pi_acc = np.zeros(n_states, dtype=float)
        emiss_counts = np.zeros((n_states, n_obs), dtype=float)
        xi_total = np.zeros((n_states, n_states), dtype=float)
        ll_total = 0.0

        trans_X = {i: [] for i in range(n_states)}
        trans_W = {i: [] for i in range(n_states)}

        for _, obs_seq, r_seq, m_seq, gap_seq in sequences:
            T = len(obs_seq)
            if T == 0:
                continue

            A_list = build_transition_matrices_for_sequence_cfg(r_seq, m_seq, params, allowed_next, cfg)
            logB = np.log(np.maximum(B[:, obs_seq], 1e-300))
            log_pi = np.log(np.maximum(pi, 1e-300))

            log_alpha = np.full((T, n_states), -np.inf)
            log_alpha[0, :] = log_pi + logB[:, 0]

            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                tmp = log_alpha[t - 1, :][:, None] + logA
                log_alpha[t, :] = logB[:, t] + logsumexp(tmp, axis=0)

            loglik = float(logsumexp(log_alpha[T - 1, :], axis=0))
            ll_total += loglik

            log_beta = np.full((T, n_states), -np.inf)
            log_beta[T - 1, :] = 0.0
            for t in range(T - 2, -1, -1):
                logA = np.log(np.maximum(A_list[t], 1e-300))
                tmp = logA + (logB[:, t + 1] + log_beta[t + 1, :])[None, :]
                log_beta[t, :] = logsumexp(tmp, axis=1)

            log_gamma = log_alpha + log_beta - loglik
            gamma = np.exp(log_gamma)

            pi_acc += gamma[0, :]
            if cfg["learn_emissions"]:
                for t in range(T):
                    emiss_counts[:, obs_seq[t]] += gamma[t, :]

            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                log_xi = (
                    log_alpha[t - 1, :][:, None]
                    + logA
                    + logB[:, t][None, :]
                    + log_beta[t, :][None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                xi_total += xi

                if cfg["use_covariates"]:
                    x_t = np.array([1.0, float(r_seq[t]), float(m_seq[t])], dtype=float)
                else:
                    x_t = np.array([1.0, 0.0, 0.0], dtype=float)

                for i in range(n_states):
                    choices = allowed_next[i]
                    if len(choices) <= 1:
                        continue
                    w = np.array([xi[i, j] for j in choices], dtype=float)
                    if w.sum() > 1e-12:
                        trans_X[i].append(x_t)
                        trans_W[i].append(w)

            # Terminal pseudo-transition based on delta_days
            gap_last = float(gap_seq[-1]) if T > 0 else np.nan
            q_fail = fixed_terminal_fail_prob(
                delta_days=gap_last,
                last_obs_idx=int(obs_seq[-1]),
                obs_vocab=obs_vocab
            )

            gamma_last = gamma[-1, :]
            xi_terminal = build_terminal_pseudo_xi(gamma_last, q_fail, allowed_next)
            xi_terminal *= TERMINAL_FAIL_WEIGHT
            xi_total += xi_terminal

            if cfg["use_covariates"]:
                x_terminal = np.array([1.0, float(r_seq[-1]), float(m_seq[-1])], dtype=float)
            else:
                x_terminal = np.array([1.0, 0.0, 0.0], dtype=float)

            for i in range(n_states):
                choices = allowed_next[i]
                if len(choices) <= 1:
                    continue
                w = np.array([xi_terminal[i, j] for j in choices], dtype=float)
                if w.sum() > 1e-12:
                    trans_X[i].append(x_terminal)
                    trans_W[i].append(w)

        pi = np.maximum(pi_acc, 1e-300)
        pi = pi / pi.sum()

        if cfg["learn_emissions"]:
            B = emiss_counts + cfg["emission_alpha"]
            B = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-300)

        for i in range(n_states):
            if len(allowed_next[i]) <= 1:
                continue
            X_i = np.array(trans_X[i], dtype=float)
            W_i = np.array(trans_W[i], dtype=float)
            if X_i.shape[0] == 0:
                continue
            params[i] = fit_weighted_multinomial_logit(X_i, W_i, ridge_l2=cfg["ridge_l2"])

        if cfg["verbose"]:
            print(f"[{cfg['name']}] EM iter {it:02d} | total log-likelihood: {ll_total:,.3f}")

        if prev_ll is not None and abs(ll_total - prev_ll) < cfg["tol"] * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll_total

    counts = xi_total
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-300)

    df_counts = pd.DataFrame(counts, index=STATES, columns=STATES)
    df_probs  = pd.DataFrame(probs,  index=STATES, columns=STATES)
    df_emiss  = pd.DataFrame(B, index=STATES, columns=obs_vocab)
    df_pi     = pd.DataFrame({"state": STATES, "pi": pi})
    df_abs    = compute_absorption_probabilities(probs, STATE_INDEX["Exit"], STATE_INDEX["Fail"])
    df_size   = transition_probs_at_size_table_cfg(params, r_mu, r_sd, allowed_next, cfg, transform_meta)
    gamma_df, post_entropy = final_posterior_summaries(sequences, B, pi, params, allowed_next, cfg)

    return {
        "cfg": deepcopy(cfg),
        "allowed_next": allowed_next,
        "transform_meta": transform_meta,
        "loglik": float(ll_total),
        "counts": df_counts,
        "probs": df_probs,
        "emiss": df_emiss,
        "pi": df_pi,
        "absorb": df_abs,
        "size_table": df_size,
        "params": params,
        "r_mu": r_mu,
        "r_sd": r_sd,
        "posterior_mass": gamma_df,
        "posterior_entropy_mean": post_entropy,
    }

def compare_results(base: dict, alt: dict) -> pd.DataFrame:
    trans_mad = float(np.mean(np.abs(base["probs"].values - alt["probs"].reindex(index=STATES, columns=STATES).values)))
    abs_base = base["absorb"].set_index("state").reindex(STATES)
    abs_alt  = alt["absorb"].set_index("state").reindex(STATES)
    absorb_mad = float(np.mean(np.abs(
        abs_base[["prob_absorb_exit", "prob_absorb_fail"]].values -
        abs_alt[["prob_absorb_exit", "prob_absorb_fail"]].values
    )))
    return pd.DataFrame([{
        "baseline": base["cfg"]["name"],
        "alternative": alt["cfg"]["name"],
        "loglik_diff": float(alt["loglik"] - base["loglik"]),
        "transition_MAD": trans_mad,
        "absorption_MAD": absorb_mad,
        "posterior_entropy_diff": float(alt["posterior_entropy_mean"] - base["posterior_entropy_mean"]),
    }])

def competition_decomposition(result: dict) -> pd.DataFrame:
    rows = []
    params = result["params"]
    cfg = result["cfg"]
    allowed_next = result["allowed_next"]

    for size in SIZE_POINTS_USD_MN:
        r = transform_size_point(size, cfg, result["transform_meta"])
        r_std = 0.0 if result["r_sd"] < 1e-8 else (r - result["r_mu"]) / result["r_sd"]
        x = np.array([1.0, float(r_std), 0.0], dtype=float) if cfg["use_covariates"] else np.array([1.0, 0.0, 0.0])

        for from_state in PROGRESSION_STATES:
            i = STATE_INDEX[from_state]
            choices = allowed_next[i]
            m = len(choices)

            logits = np.zeros(m, dtype=float)
            for k in range(1, m):
                logits[k] = float(np.dot(params[i][k - 1], x))
            probs = softmax(logits)

            row = {"from_state": from_state, "deal_size_usd_mn": size}
            for idx, j in enumerate(choices):
                row[f"logit_{STATES[j]}"] = logits[idx]
                row[f"prob_{STATES[j]}"] = probs[idx]

            if STATE_INDEX["Exit"] in choices:
                stay_idx = choices.index(i)
                next_idx = choices.index(min(i + 1, STATE_INDEX["Series E"]))
                exit_idx = choices.index(STATE_INDEX["Exit"])
                row["gap_next_minus_stay"] = logits[next_idx] - logits[stay_idx]
                row["gap_next_minus_exit"] = logits[next_idx] - logits[exit_idx]

            rows.append(row)

    return pd.DataFrame(rows)

def emission_identifiability_table(result: dict) -> pd.DataFrame:
    emiss = result["emiss"].copy()
    rows = []

    canonical = {
        "Seed": "Seed",
        "Series A": "Series A",
        "Series B": "Series B",
        "Series C": "Series C",
        "Series D": "Series D",
        "Series E": "Series E",
        "Exit": "EXITLIKE",
        "Fail": "UNSPEC" if "UNSPEC" in emiss.columns else emiss.columns[np.argmax(emiss.loc["Fail"].values)],
    }

    for s in STATES:
        p = emiss.loc[s].to_numpy(dtype=float)
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)))
        top_tok = emiss.loc[s].idxmax()
        top_prob = float(emiss.loc[s].max())
        canon_tok = canonical.get(s, top_tok)
        canon_prob = float(emiss.loc[s, canon_tok]) if canon_tok in emiss.columns else np.nan

        rows.append({
            "state": s,
            "top_token": top_tok,
            "top_prob": top_prob,
            "canonical_token": canon_tok,
            "canonical_prob": canon_prob,
            "row_entropy": float(entropy),
        })

    return pd.DataFrame(rows)

def run_experiment_suite(df_in: pd.DataFrame, experiments: list[dict]):
    results = {}
    comparison_rows = []

    for cfg in experiments:
        print(f"Running: {cfg['name']}")
        res = fit_model_experiment(df_in, cfg)
        results[cfg["name"]] = res

    base = results[experiments[0]["name"]]
    for cfg in experiments[1:]:
        comparison_rows.append(compare_results(base, results[cfg["name"]]))

    summary = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    return results, summary

# Stress-testing and component analysis

import os
import pandas as pd

# Output directory

STRESS_TEST_DIR = "stress_test_outputs"
os.makedirs(STRESS_TEST_DIR, exist_ok=True)

# Reuse the loaded model input

df_model = df.copy()

# Display helper

def show_table(title: str, df_to_show: pd.DataFrame):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    display(df_to_show)

# Summary labeling helper

def label_summary(summary_df: pd.DataFrame, aspect_name: str) -> pd.DataFrame:
    out = summary_df.copy()
    out.insert(0, "aspect_tested", aspect_name)
    return out

# 1. Hidden layer stress test

hidden_layer_experiments = [
    make_cfg(name="baseline"),
    make_cfg(name="fixed_emissions", learn_emissions=False, fixed_emission_init="heuristic"),
]

hidden_results, hidden_summary = run_experiment_suite(df_model, hidden_layer_experiments)
hidden_summary = label_summary(hidden_summary, "hidden_layer")
show_table("1. HIDDEN LAYER STRESS TEST", hidden_summary)

# 2. Covariate-dependent transition stress test

covariate_experiments = [
    make_cfg(name="baseline"),
    make_cfg(name="no_covariates", use_covariates=False),
]

cov_results, cov_summary = run_experiment_suite(df_model, covariate_experiments)
cov_summary = label_summary(cov_summary, "covariate_dependence")
show_table("2. COVARIATE-DEPENDENT TRANSITIONS STRESS TEST", cov_summary)

# 3. Covariate specification robustness test

covspec_experiments = [
    make_cfg(name="baseline"),
    make_cfg(name="winsorized_size", covariate_mode="winsor_log1p", winsor_q=(0.01, 0.99)),
]

covspec_results, covspec_summary = run_experiment_suite(df_model, covspec_experiments)
covspec_summary = label_summary(covspec_summary, "covariate_specification")
show_table("3. COVARIATE SPECIFICATION ROBUSTNESS TEST", covspec_summary)

# 4. Failure-state stress test

fail_experiments = [
    make_cfg(name="baseline"),
    make_cfg(name="no_fail_state", include_fail_state=False),
]

fail_results, fail_summary = run_experiment_suite(df_model, fail_experiments)
fail_summary = label_summary(fail_summary, "failure_state")
show_table("4. FAILURE-STATE STRESS TEST", fail_summary)

# 5. Regularization sensitivity test

ridge_experiments = [
    make_cfg(name="baseline"),
    make_cfg(name="ridge_0.01", ridge_l2=0.01),
    make_cfg(name="ridge_1.0", ridge_l2=1.0),
]

ridge_results, ridge_summary = run_experiment_suite(df_model, ridge_experiments)
ridge_summary = label_summary(ridge_summary, "regularization")
show_table("5. REGULARIZATION SENSITIVITY TEST", ridge_summary)

# Master summary table

# Key summary for the first five component tests.

master_summary = pd.concat(
    [
        hidden_summary,
        cov_summary,
        covspec_summary,
        fail_summary,
        ridge_summary,
    ],
    ignore_index=True
)

show_table("MASTER STRESS-TEST SUMMARY (FIRST FIVE MODEL COMPONENTS)", master_summary)

master_summary.to_csv(
    os.path.join(STRESS_TEST_DIR, "stress_test_master_summary.csv"),
    index=False
)

# Baseline model for diagnostic tables

# Reuse the baseline fit.

baseline_result = hidden_results["baseline"]

# 6. Multinomial competition decomposition

# Diagnose probability mass reallocation across outcomes.

competition_df = competition_decomposition(baseline_result)

show_table("6. MULTINOMIAL COMPETITION DECOMPOSITION", competition_df)

competition_df.to_csv(
    os.path.join(STRESS_TEST_DIR, "competition_decomposition.csv"),
    index=False
)

# 7. Emission sharpness and identifiability diagnostics

# Summarize emission concentration and state distinctness.

ident_df = emission_identifiability_table(baseline_result)

show_table("7. EMISSION SHARPNESS / IDENTIFIABILITY DIAGNOSTICS", ident_df)

ident_df.to_csv(
    os.path.join(STRESS_TEST_DIR, "emission_identifiability.csv"),
    index=False
)

# Posterior entropy summary

# Summarize baseline posterior uncertainty.

entropy_df = pd.DataFrame([{
    "metric": "mean_posterior_state_entropy",
    "value": baseline_result["posterior_entropy_mean"]
}])

show_table("POSTERIOR ENTROPY SUMMARY", entropy_df)

entropy_df.to_csv(
    os.path.join(STRESS_TEST_DIR, "posterior_entropy_summary.csv"),
    index=False
)

# Final status message

print("\n" + "=" * 90)
print("STRESS TEST EXPORT COMPLETE")
print("=" * 90)
print(f"All results were saved to: {STRESS_TEST_DIR}")

print("\nExported files:")
print(" - stress_test_master_summary.csv")
print(" - competition_decomposition.csv")
print(" - emission_identifiability.csv")
print(" - posterior_entropy_summary.csv")
