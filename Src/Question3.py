# ML Assignment 2 - Q3 (patched): bias and variance as measured quantities
# Requirements: numpy, pandas, matplotlib, scikit-learn
#
# NO re-tuning happens here. The script loads the frozen choices from
#   outputs/q1_frozen_config.json    {"max_depth": 8|null, "min_samples_leaf": 1, "bag_B": 50, "bag_seed": 33005}
#   outputs/q2b_selected_setting.json {"depth": 2, "requested_rounds": 50, ...}
# and stops with a clear error if either file is missing (run Q1 and Q2 first).
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL_NUMBER = "M26ECC005"
SEED = int("".join(ch for ch in ROLL_NUMBER if ch.isdigit()))  # 26005
N_CLASSES = 40
N_REFITS = 30
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2))


# ------------------------------------------------------------------ data
def load_split():
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data.astype(np.float32), data.target.astype(int)
    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == cls))
        train_idx.extend(ids[:6])
        val_idx.extend(ids[6:8])
        test_idx.extend(ids[8:10])
    train_idx, val_idx, test_idx = map(np.asarray, (train_idx, val_idx, test_idx))
    np.savez(OUT / "split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    save_json("split.json", {"registration_number": ROLL_NUMBER, "seed": SEED,
                             "train_idx": train_idx.tolist(), "validation_idx": val_idx.tolist(),
                             "test_idx": test_idx.tolist()})
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx])


def mask_faces(X):
    images = X.reshape(-1, 64, 64).copy()
    images[:, 24:40, 24:40] = 0.0
    return images.reshape(len(X), -1)


def acc(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ------------------------------------------------------------------ trees / bagging
def fit_tree(X, y, depth, leaf=1, seed=SEED):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                  random_state=seed).fit(X, y)


def majority_vote(preds):
    return np.array([np.bincount(preds[:, j], minlength=N_CLASSES).argmax()
                     for j in range(preds.shape[1])])


def bag_predict(trees, X):
    return np.asarray([t.predict(X).astype(int) for t in trees])


def train_bag(X, y, B, depth, leaf, seed=SEED):
    """Trees are generated sequentially from one rng, so trees[:k] is exactly the k-tree bag."""
    rng = np.random.default_rng(seed)
    trees = []
    for b in range(B):
        sample = rng.integers(0, len(y), size=len(y))
        trees.append(fit_tree(X[sample], y[sample], depth, leaf, seed + b + 1))
    return trees


def prob_from_preds(preds):
    out = np.zeros((preds.shape[1], N_CLASSES))
    for j in range(preds.shape[1]):
        out[j] = np.bincount(preds[:, j], minlength=N_CLASSES) / len(preds)
    return out


def class_probability_tree(model, X):
    out = np.zeros((len(X), N_CLASSES))
    probs = model.predict_proba(X)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = probs[:, col]
    return out


def pairwise_disagreement(preds):
    vals = [np.mean(preds[i] != preds[j])
            for i in range(len(preds)) for j in range(i + 1, len(preds))]
    return float(np.mean(vals)) if vals else 0.0


def vote_margin(preds):
    out = []
    for j in range(preds.shape[1]):
        c = np.sort(np.bincount(preds[:, j], minlength=N_CLASSES) / len(preds))
        out.append(c[-1] - c[-2])
    return float(np.mean(out))


# ------------------------------------------------------------------ SAMME (same patched class as Q2)
class SAMME:
    """Multiclass SAMME. A learner with error >= 1-1/K is discarded, weights reset to
    uniform, the event logged in self.events; boosting continues (stops if it fails
    again immediately after a reset)."""
    def __init__(self, depth=1, rounds=10, seed=SEED):
        self.depth, self.rounds, self.seed = depth, rounds, seed
        self.estimators, self.alphas, self.events = [], [], []

    def fit(self, X, y):
        n = len(y)
        self.classes = np.unique(y)
        K = len(self.classes)
        uniform = np.full(n, 1.0 / n)
        weights, at_uniform = uniform.copy(), True
        self.estimators, self.alphas, self.events = [], [], []
        attempt, max_attempts = 0, self.rounds * 3
        while len(self.estimators) < self.rounds and attempt < max_attempts:
            attempt += 1
            model = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed + attempt - 1)
            model.fit(X, y, sample_weight=weights)
            pred = model.predict(X).astype(int)
            error = float(weights[pred != y].sum())
            if error >= 1.0 - 1.0 / K:
                ev = {"attempt": attempt, "accepted_rounds_so_far": len(self.estimators),
                      "weighted_error": error}
                if at_uniform:
                    ev["action"] = "stopped: not better than random even with uniform weights"
                    self.events.append(ev)
                    break
                ev["action"] = "learner discarded, weights reset to uniform"
                self.events.append(ev)
                weights, at_uniform = uniform.copy(), True
                continue
            alpha = 20.0 if error <= 0 else float(np.log((1 - error) / error) + np.log(K - 1))
            self.estimators.append(model)
            self.alphas.append(alpha)
            weights = weights * np.exp(alpha * (pred != y))
            weights /= weights.sum()
            at_uniform = False
            if error <= 0:
                break
        self.alphas = np.asarray(self.alphas, dtype=float)
        return self

    def scores(self, X):
        s = np.zeros((len(X), len(self.classes)))
        for alpha, model in zip(self.alphas, self.estimators):
            pred = model.predict(X).astype(int)
            for col, cls in enumerate(self.classes):
                s[:, col] += alpha * (pred == cls)
        return s

    def predict(self, X):
        return self.classes[np.argmax(self.scores(X), axis=1)]

    def predict_proba_full(self, X):
        """Normalised weighted vote q_k, placed in fixed 0..39 class columns."""
        out = np.zeros((len(X), N_CLASSES))
        if len(self.alphas) == 0 or self.alphas.sum() == 0:
            return np.full((len(X), N_CLASSES), 1.0 / N_CLASSES)
        q = self.scores(X) / self.alphas.sum()
        for col, cls in enumerate(self.classes):
            out[:, int(cls)] = q[:, col]
        return out


# ------------------------------------------------------------------ measurement helpers
def bias_variance(prob_runs, y):
    """prob_runs: R x N x 40. Returns Bias^2, Variance, and mean squared error
    (which must equal Bias^2 + Variance exactly)."""
    mean_p = prob_runs.mean(axis=0)
    onehot = np.eye(N_CLASSES)[y]
    bias2 = float(np.mean(np.sum((mean_p - onehot) ** 2, axis=1)))
    variance = float(np.mean(np.sum((prob_runs - mean_p[None]) ** 2, axis=2)))
    mse = float(np.mean(np.sum((prob_runs - onehot[None]) ** 2, axis=2)))
    return bias2, variance, mse


def resample_indices(y, rng):
    """Within each identity: 6 draws with replacement from that identity's 6 training faces."""
    idx = []
    for cls in np.unique(y):
        idx.extend(rng.choice(np.flatnonzero(y == cls), size=6, replace=True))
    return np.asarray(idx)


def leaf_stats(tree, idx, Xtr):
    """Leaf occupancy of a tree fitted on the resample Xtr[idx] (rows include duplicates)."""
    leaves = tree.apply(Xtr[idx])
    df = pd.DataFrame({"leaf": leaves, "face": idx})
    rows = df.groupby("leaf").size()
    distinct = df.groupby("leaf")["face"].nunique()
    return {"n_leaves": int(tree.get_n_leaves()),
            "mean_rows_per_leaf": float(rows.mean()),
            "mean_distinct_faces_per_leaf": float(distinct.mean()),
            "frac_leaves_single_row": float((rows == 1).mean()),
            "frac_leaves_single_distinct_face": float((distinct == 1).mean())}


def load_frozen():
    p1, p2 = OUT / "q1_frozen_config.json", OUT / "q2b_selected_setting.json"
    missing = [p.name for p in (p1, p2) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in outputs/. Q3 must not re-tune: run the Q1 and Q2 scripts first.")
    q1, q2 = json.loads(p1.read_text()), json.loads(p2.read_text())
    return {"tree_depth": q1["max_depth"], "tree_leaf": int(q1["min_samples_leaf"]),
            "bag_B": int(q1["bag_B"]), "bag_seed": int(q1.get("bag_seed", SEED + 7000)),
            "boost_depth": int(q2["depth"]), "boost_rounds": int(q2["requested_rounds"])}


def choose_depth_levels(Xtr, ytr, d_sel, leaf):
    """Three DISTINCT effective depths: shallow, selected, much deeper (unrestricted).
    If the selected tree is already (effectively) unrestricted, an intermediate depth
    is added so the comparison still has three rows; 'selected' then tags the deepest row."""
    full = fit_tree(Xtr, ytr, None, leaf).get_depth()
    eff = lambda d: full if d is None else min(d, full)
    eff_sel = eff(d_sel)
    shallow = 2 if eff_sel > 2 else 1
    levels = [{"depth": shallow, "roles": ["shallow"]}]
    if eff_sel < full:
        levels.append({"depth": d_sel, "roles": ["selected"]})
        levels.append({"depth": None, "roles": ["deep"]})
    else:
        mid = max(shallow + 1, full // 2)
        if mid < full:
            levels.append({"depth": mid, "roles": ["intermediate"]})
        levels.append({"depth": None, "roles": ["selected", "deep"]})
    for L in levels:
        L["effective_depth"] = eff(L["depth"])
        L["label"] = ("None(full=%d)" % full) if L["depth"] is None else str(L["depth"])
    return levels, full


# ------------------------------------------------------------------ main
def main():
    log_file = open(OUT / "q3_run_log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


def run():
    start_all = time.perf_counter()
    Xtr, ytr, Xv, yv, Xte, yte = load_split()
    masked = mask_faces(Xte)
    cfg = load_frozen()
    print("Frozen config (loaded, not re-tuned):", cfg)

    # 30 resamples generated ONCE and reused by every experiment
    rng = np.random.default_rng(SEED + 9000)
    resamples = [resample_indices(ytr, rng) for _ in range(N_REFITS)]
    save_json("q3_resample_indices.json", {"seed": SEED + 9000, "n_refits": N_REFITS,
                                           "indices_into_train": [r.tolist() for r in resamples]})
    distinct_per_set = float(np.mean([len(np.unique(r)) for r in resamples]))
    print(f"Mean distinct training faces per resampled set: {distinct_per_set:.2f} of 240")

    levels, full_depth = choose_depth_levels(Xtr, ytr, cfg["tree_depth"], cfg["tree_leaf"])
    print("Depth levels:", [(L["label"], L["roles"], L["effective_depth"]) for L in levels])
    if all("deep" not in L["roles"] for L in levels):
        print("WARNING: no deep level")

    probs, accs, accs_m = {}, {}, {}   # keyed by model name
    lvl = {}                           # per-level storage
    boost_log = []

    for L in levels:
        is_sel = "selected" in L["roles"]
        B_needed = max(100, cfg["bag_B"]) if is_sel else 100
        S = {k: [] for k in ["s_p", "s_a", "s_am", "s_leaf", "b100_p", "b100_a", "b100_am",
                             "b100_dis", "b100_margin", "b100_leaves",
                             "bB_p", "bB_a", "bB_am", "bo_p", "bo_a", "bo_am"]}
        for r, idx in enumerate(resamples):
            Xr, yr = Xtr[idx], ytr[idx]
            tree = fit_tree(Xr, yr, L["depth"], cfg["tree_leaf"], SEED + 10000 + r)
            S["s_p"].append(class_probability_tree(tree, Xte))
            S["s_a"].append(acc(yte, tree.predict(Xte)))
            S["s_am"].append(acc(yte, tree.predict(masked)))
            S["s_leaf"].append(leaf_stats(tree, idx, Xtr))

            bag = train_bag(Xr, yr, B_needed, L["depth"], cfg["tree_leaf"], SEED + 11000 + r)
            pc, pm = bag_predict(bag[:100], Xte), bag_predict(bag[:100], masked)
            S["b100_p"].append(prob_from_preds(pc))
            S["b100_a"].append(acc(yte, majority_vote(pc)))
            S["b100_am"].append(acc(yte, majority_vote(pm)))
            S["b100_dis"].append(pairwise_disagreement(pc))
            S["b100_margin"].append(vote_margin(pc))
            S["b100_leaves"].append(float(np.mean([t.get_n_leaves() for t in bag[:100]])))
            if is_sel:
                bB = bag[:cfg["bag_B"]]
                pcB, pmB = bag_predict(bB, Xte), bag_predict(bB, masked)
                S["bB_p"].append(prob_from_preds(pcB))
                S["bB_a"].append(acc(yte, majority_vote(pcB)))
                S["bB_am"].append(acc(yte, majority_vote(pmB)))
                boost = SAMME(cfg["boost_depth"], cfg["boost_rounds"], SEED + 12000 + r).fit(Xr, yr)
                S["bo_p"].append(boost.predict_proba_full(Xte))
                S["bo_a"].append(acc(yte, boost.predict(Xte)))
                S["bo_am"].append(acc(yte, boost.predict(masked)))
                boost_log.append({"repetition": r, "actual_rounds": len(boost.estimators),
                                  "requested_rounds": cfg["boost_rounds"],
                                  "n_rejection_events": len(boost.events)})
            if (r + 1) % 10 == 0:
                print(f"depth {L['label']}: refit {r+1}/{N_REFITS} done")
        lvl[L["label"]] = S
        if is_sel:
            probs["single_tree_selected"] = np.asarray(S["s_p"])
            accs["single_tree_selected"], accs_m["single_tree_selected"] = S["s_a"], S["s_am"]
            probs["bagging_selected"] = np.asarray(S["bB_p"])
            accs["bagging_selected"], accs_m["bagging_selected"] = S["bB_a"], S["bB_am"]
            probs["boosting_selected"] = np.asarray(S["bo_p"])
            accs["boosting_selected"], accs_m["boosting_selected"] = S["bo_a"], S["bo_am"]

    pd.DataFrame(boost_log).to_csv(OUT / "q3_boosting_refit_log.csv", index=False)
    np.savez_compressed(OUT / "q3_probability_runs.npz",
                        **{k: v.astype(np.float32) for k, v in probs.items()},
                        **{f"depth_{lab}_single": np.asarray(S["s_p"], dtype=np.float32)
                           for lab, S in lvl.items()},
                        **{f"depth_{lab}_bag100": np.asarray(S["b100_p"], dtype=np.float32)
                           for lab, S in lvl.items()},
                        y_test=yte)

    # ---------- Q3(a)
    rows, per_refit = [], []
    for name in ["single_tree_selected", "bagging_selected", "boosting_selected"]:
        b2, var, mse = bias_variance(probs[name], yte)
        assert abs(mse - (b2 + var)) < 1e-9, "bias-variance identity violated"
        a, am = np.asarray(accs[name]), np.asarray(accs_m[name])
        rows.append({"model": name, "bias_squared": b2, "variance": var,
                     "mse_check_bias2_plus_var": b2 + var,
                     "mean_test_accuracy": float(a.mean()), "std_test_accuracy": float(a.std(ddof=1)),
                     "mean_masked_accuracy": float(am.mean()),
                     "std_masked_accuracy": float(am.std(ddof=1))})
        for r in range(N_REFITS):
            per_refit.append({"model": name, "repetition": r,
                              "clean_accuracy": float(a[r]), "masked_accuracy": float(am[r])})
    q3a = pd.DataFrame(rows)
    q3a.to_csv(OUT / "q3a_bias_variance.csv", index=False)
    pd.DataFrame(per_refit).to_csv(OUT / "q3a_per_refit_accuracy.csv", index=False)
    print("\nQ3(a):\n", q3a.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([accs[n] for n in ["single_tree_selected", "bagging_selected", "boosting_selected"]],
               labels=["tree", "bagging", "boosting"])
    ax.set_ylabel("Clean test accuracy over 30 refits"); ax.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q3a_accuracy_spread.png", dpi=160); plt.close()

    # ---------- Q3(b) + Q3(c) measurements
    brows, crows = [], []
    for L in levels:
        S = lvl[L["label"]]
        sb, sv, _ = bias_variance(np.asarray(S["s_p"]), yte)
        bb, bv, _ = bias_variance(np.asarray(S["b100_p"]), yte)
        sa, ba = np.asarray(S["s_a"]), np.asarray(S["b100_a"])
        gain = ba - sa                      # paired: same resamples
        leaf = pd.DataFrame(S["s_leaf"]).mean()
        brows.append({
            "depth": L["label"], "roles": "+".join(L["roles"]), "effective_depth": L["effective_depth"],
            "single_bias_squared": sb, "single_variance": sv,
            "single_accuracy_mean": float(sa.mean()), "single_accuracy_std": float(sa.std(ddof=1)),
            "bag100_bias_squared": bb, "bag100_variance": bv,
            "bag100_accuracy_mean": float(ba.mean()), "bag100_accuracy_std": float(ba.std(ddof=1)),
            "variance_removed_by_bagging": sv - bv,
            "relative_variance_reduction": (sv - bv) / sv if sv > 0 else float("nan"),
            "bias_change_bag_minus_single": bb - sb,
            "accuracy_gain_bag_minus_single": float(gain.mean()),
            "accuracy_gain_std_paired": float(gain.std(ddof=1))})
        crows.append({
            "depth": L["label"], "roles": "+".join(L["roles"]),
            "single_tree_mean_leaves": float(leaf["n_leaves"]),
            "single_tree_mean_rows_per_leaf": float(leaf["mean_rows_per_leaf"]),
            "single_tree_mean_distinct_faces_per_leaf": float(leaf["mean_distinct_faces_per_leaf"]),
            "single_tree_frac_leaves_single_row": float(leaf["frac_leaves_single_row"]),
            "single_tree_frac_leaves_single_distinct_face": float(leaf["frac_leaves_single_distinct_face"]),
            "bag100_mean_leaves_per_tree": float(np.mean(S["b100_leaves"])),
            "bag100_pairwise_disagreement": float(np.mean(S["b100_dis"])),
            "bag100_vote_margin": float(np.mean(S["b100_margin"]))})
    q3b, q3c = pd.DataFrame(brows), pd.DataFrame(crows)
    q3b.to_csv(OUT / "q3b_depth_comparison.csv", index=False)
    q3c.to_csv(OUT / "q3c_leaf_disagreement.csv", index=False)
    print("\nQ3(b):\n", q3b.to_string(index=False))
    print("\nQ3(c) leaf occupancy and disagreement:\n", q3c.to_string(index=False))

    best_gain = q3b.loc[q3b["accuracy_gain_bag_minus_single"].idxmax()]
    deepest_label = [L["label"] for L in levels if "deep" in L["roles"]][0]
    save_json("q3c_summary.json", {
        "depth_with_largest_accuracy_gain": best_gain["depth"],
        "deepest_depth": deepest_label,
        "deepest_tree_is_depth_of_largest_gain": bool(best_gain["depth"] == deepest_label),
        "depth_with_largest_variance_removed": q3b.loc[q3b["variance_removed_by_bagging"].idxmax(), "depth"],
        "depth_with_largest_relative_variance_reduction":
            q3b.loc[q3b["relative_variance_reduction"].idxmax(), "depth"],
        "gains_by_depth": {r["depth"]: r["accuracy_gain_bag_minus_single"] for r in brows},
        "note": "Gains are paired differences over the same 30 resamples; compare with the "
                "paired std before claiming an ordering."})

    x = np.arange(len(levels)); lab = [L["label"] for L in levels]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a_, col, ttl in [(ax[0], "bias_squared", "Bias squared"), (ax[1], "variance", "Variance")]:
        a_.plot(x, q3b[f"single_{col}"], "o-", label="single tree")
        a_.plot(x, q3b[f"bag100_{col}"], "s-", label="bagging (100)")
        a_.set_xticks(x); a_.set_xticklabels(lab); a_.set_xlabel("Tree depth"); a_.set_ylabel(ttl)
        a_.grid(True, alpha=.3); a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "q3b_bias_variance_vs_depth.png", dpi=160); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(x, q3b["accuracy_gain_bag_minus_single"], yerr=q3b["accuracy_gain_std_paired"], capsize=3)
    ax[0].set_ylabel("Accuracy gain from bagging"); ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
    ax[1].bar(x, q3b["relative_variance_reduction"])
    ax[1].set_ylabel("Relative variance reduction"); ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
    for a_ in ax:
        a_.set_xlabel("Tree depth"); a_.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q3b_bagging_gain_vs_depth.png", dpi=160); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(x, q3c["single_tree_frac_leaves_single_distinct_face"], "o-", label="single distinct face")
    ax[0].plot(x, q3c["single_tree_frac_leaves_single_row"], "s--", label="single row")
    ax[0].set_ylabel("Fraction of leaves"); ax[0].legend(fontsize=8)
    ax[1].plot(x, q3c["bag100_pairwise_disagreement"], "o-", label="pairwise disagreement")
    ax[1].plot(x, q3c["bag100_vote_margin"], "s--", label="vote margin")
    ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(lab); a_.set_xlabel("Tree depth"); a_.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q3c_leaf_disagreement.png", dpi=160); plt.close()

    # ---------- Q3(d): final table. Frozen models are re-fitted here on the ORIGINAL
    # training set (no tuning), so clean/masked accuracy never depends on other CSV names.
    tree = fit_tree(Xtr, ytr, cfg["tree_depth"], cfg["tree_leaf"])
    bagF = train_bag(Xtr, ytr, cfg["bag_B"], cfg["tree_depth"], cfg["tree_leaf"], cfg["bag_seed"])
    boostF = SAMME(cfg["boost_depth"], cfg["boost_rounds"], SEED + 500).fit(Xtr, ytr)
    frozen = {
        "single_tree_selected": (acc(yte, tree.predict(Xte)), acc(yte, tree.predict(masked))),
        "bagging_selected": (acc(yte, majority_vote(bag_predict(bagF, Xte))),
                             acc(yte, majority_vote(bag_predict(bagF, masked)))),
        "boosting_selected": (acc(yte, boostF.predict(Xte)), acc(yte, boostF.predict(masked)))}
    summary = []
    for _, row in q3a.iterrows():
        n = row["model"]
        c, m = frozen[n]
        summary.append({"model": n, "clean_test_accuracy": c, "masked_test_accuracy": m,
                        "accuracy_drop_under_mask": c - m,
                        "bias_squared": row["bias_squared"], "variance": row["variance"],
                        "mean_clean_accuracy_over_30_refits": row["mean_test_accuracy"],
                        "mean_masked_accuracy_over_30_refits": row["mean_masked_accuracy"],
                        "std_clean_accuracy_over_30_refits": row["std_test_accuracy"]})
    q3d = pd.DataFrame(summary)
    q3d.to_csv(OUT / "q3d_final_summary.csv", index=False)
    print("\nQ3(d) final table:\n", q3d.to_string(index=False))

    # cross-check against the Q2 robustness file (warn, never silently NaN)
    mismatches = []
    p = OUT / "q2d_robustness.csv"
    if p.exists():
        r2 = pd.read_csv(p)
        key = {"single_tree_selected": "single_tree", "bagging_selected": "bagging",
               "boosting_selected": "boosting"}
        for n, (c, m) in frozen.items():
            for ts, val in [("clean", c), ("masked", m)]:
                hit = r2[(r2["model"] == key[n]) & (r2["test_set"] == ts)]
                if hit.empty or abs(float(hit["accuracy"].iloc[0]) - val) > 1e-9:
                    mismatches.append({"model": n, "test_set": ts, "q3_value": val,
                                       "q2_value": None if hit.empty else float(hit["accuracy"].iloc[0])})
        print("WARNING: Q3 frozen accuracies differ from q2d_robustness.csv:" if mismatches
              else "Cross-check OK: frozen accuracies match q2d_robustness.csv", mismatches or "")
    else:
        print("NOTE: q2d_robustness.csv not found; cross-check skipped.")

    lowest_var = q3d.loc[q3d["variance"].idxmin(), "model"]
    best_clean = q3d.loc[q3d["clean_test_accuracy"].idxmax(), "model"]
    save_json("q3d_facts.json", {
        "lowest_variance_model": lowest_var, "best_clean_accuracy_model": best_clean,
        "lowest_variance_is_best_accuracy": bool(lowest_var == best_clean),
        "most_stable_by_refit_accuracy_std": q3d.loc[q3d["std_clean_accuracy_over_30_refits"].idxmin(), "model"],
        "largest_accuracy_drop_under_mask": q3d.loc[q3d["accuracy_drop_under_mask"].idxmax(), "model"],
        "q2_crosscheck_mismatches": mismatches})

    fig, ax = plt.subplots(2, 2, figsize=(8, 5.6))
    names = ["tree", "bagging", "boosting"]
    for a_, col, ttl in [(ax[0, 0], "bias_squared", "Bias squared"), (ax[0, 1], "variance", "Variance"),
                         (ax[1, 0], "clean_test_accuracy", "Clean test accuracy"),
                         (ax[1, 1], "masked_test_accuracy", "Masked test accuracy")]:
        a_.bar(names, q3d[col]); a_.set_title(ttl, fontsize=10); a_.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q3d_final_table.png", dpi=160); plt.close()

    save_json("q3_config.json", {
        "roll_number": ROLL_NUMBER, "seed": SEED, "n_refits": N_REFITS,
        "resample_seed": SEED + 9000, "frozen_config": cfg,
        "seeds": {"tree": "SEED+10000+r", "bagging": "SEED+11000+r", "boosting": "SEED+12000+r",
                  "frozen_boosting": SEED + 500},
        "depth_levels": levels, "unrestricted_tree_depth_on_original_train": int(full_depth),
        "mean_distinct_faces_per_resample": distinct_per_set,
        "boosting_refits_with_rejections": int(sum(b["n_rejection_events"] > 0 for b in boost_log)),
        "boosting_refits_stopped_early": int(sum(b["actual_rounds"] < b["requested_rounds"] for b in boost_log)),
        "runtime_seconds": time.perf_counter() - start_all,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print("Outputs saved in outputs/")


if __name__ == "__main__":
    main()
