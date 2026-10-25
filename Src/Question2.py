# ML Assignment 2 - Q2 (patched): multiclass SAMME boosting from scratch
# Requirements: numpy, pandas, matplotlib, scikit-learn
#
# Reads the Q1 results from outputs/q1_frozen_config.json, e.g.
#   {"max_depth": 8, "min_samples_leaf": 1, "bag_B": 50, "bag_seed": 33005}
# (max_depth: null for an unlimited tree; bag_seed is optional.)
# If that file is missing, the script re-tunes on validation only and saves it.
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
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


class Tee:
    """Mirror stdout into a log file so the final run log is saved."""
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
    """One reproducible split: 6 train, 2 validation, 2 test per identity."""
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
    save_json("split.json", {
        "registration_number": ROLL_NUMBER, "seed": SEED,
        "train_idx": train_idx.tolist(), "validation_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist()})
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X[test_idx], y[test_idx], train_idx, test_idx)


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
    rng = np.random.default_rng(seed)
    trees, oob = [], []
    for b in range(B):
        sample = rng.integers(0, len(y), size=len(y))
        trees.append(fit_tree(X[sample], y[sample], depth, leaf, seed + b + 1))
        oob.append(np.setdiff1d(np.arange(len(y)), np.unique(sample)))
    return trees, oob


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


def vote_confidence(preds):
    return float(np.mean([np.bincount(preds[:, j], minlength=N_CLASSES).max() / len(preds)
                          for j in range(preds.shape[1])]))


# ------------------------------------------------------------------ Q1 config (frozen tree + final B)
def load_q1_config(Xtr, ytr, Xv, yv):
    path = OUT / "q1_frozen_config.json"
    if path.exists():
        cfg = json.loads(path.read_text())
        cfg.setdefault("bag_seed", SEED + 7000)
        cfg["source"] = "q1_frozen_config.json"
        print("Loaded Q1 config:", cfg)
        return cfg
    print("WARNING: outputs/q1_frozen_config.json not found; re-tuning on validation only.")
    best = (-1.0, None, None)
    for d in [2, 4, 8, 16, None]:
        for leaf in [1, 2, 4]:
            score = acc(yv, fit_tree(Xtr, ytr, d, leaf).predict(Xv))
            if score > best[0]:
                best = (score, d, leaf)
    _, d, leaf = best
    bag_seed = SEED + 7000
    best_b = (-1.0, None)
    for B in [1, 5, 10, 25, 50, 100, 200]:
        trees, _ = train_bag(Xtr, ytr, B, d, leaf, bag_seed)
        score = acc(yv, majority_vote(bag_predict(trees, Xv)))
        if score > best_b[0]:
            best_b = (score, B)
    cfg = {"max_depth": d, "min_samples_leaf": leaf, "bag_B": best_b[1],
           "bag_seed": bag_seed, "source": "fallback re-tuned inside q2 script"}
    save_json("q1_frozen_config.json", cfg)
    print("Fallback Q1 config:", cfg)
    return cfg


# ------------------------------------------------------------------ SAMME
class SAMME:
    """Multiclass SAMME AdaBoost, implemented directly (not sklearn AdaBoost).

    A weak learner that is not better than random multiclass guessing
    (weighted error >= 1 - 1/K) is DISCARDED, sample weights are reset to
    uniform, the event is logged in self.events, and boosting continues.
    If a learner still fails right after a reset, boosting stops (logged).
    """
    def __init__(self, depth=1, rounds=10, seed=SEED):
        self.depth, self.rounds, self.seed = depth, rounds, seed
        self.estimators, self.alphas, self.history, self.events = [], [], [], []

    def fit(self, X, y, verbose=False):
        n = len(y)
        self.classes = np.unique(y)
        K = len(self.classes)
        uniform = np.full(n, 1.0 / n)
        weights = uniform.copy()
        at_uniform = True
        self.estimators, self.alphas, self.history, self.events = [], [], [], []
        attempt, max_attempts = 0, self.rounds * 3
        while len(self.estimators) < self.rounds and attempt < max_attempts:
            attempt += 1
            model = DecisionTreeClassifier(max_depth=self.depth,
                                           random_state=self.seed + attempt - 1)
            model.fit(X, y, sample_weight=weights)
            pred = model.predict(X).astype(int)
            error = float(weights[pred != y].sum())
            if error >= 1.0 - 1.0 / K:
                ev = {"attempt": attempt, "accepted_rounds_so_far": len(self.estimators),
                      "weighted_error": error, "threshold": 1.0 - 1.0 / K}
                if at_uniform:
                    ev["action"] = "stopped: not better than random even with uniform weights"
                    self.events.append(ev)
                    if verbose:
                        print(ev)
                    break
                ev["action"] = "learner discarded, weights reset to uniform"
                self.events.append(ev)
                if verbose:
                    print(ev)
                weights, at_uniform = uniform.copy(), True
                continue
            alpha = 20.0 if error <= 0 else float(np.log((1 - error) / error) + np.log(K - 1))
            self.estimators.append(model)
            self.alphas.append(alpha)
            weights = weights * np.exp(alpha * (pred != y))
            weights /= weights.sum()
            at_uniform = False
            self.history.append({
                "round": len(self.estimators), "weighted_error": error, "alpha": alpha,
                "max_weight": float(weights.max()),
                "top10_weight_sum": float(np.sort(weights)[-10:].sum()),
                "weights": weights.copy(), "predictions": pred.copy()})
            if verbose:
                print(f"round={len(self.estimators)}: error={error:.6f}, alpha={alpha:.5f}, "
                      f"max_weight={weights.max():.6f}")
            if error <= 0:
                break
        self.alphas = np.asarray(self.alphas, dtype=float)
        return self

    def learner_predictions(self, X):
        return np.asarray([m.predict(X).astype(int) for m in self.estimators])

    def scores(self, X):
        s = np.zeros((len(X), len(self.classes)))
        for alpha, model in zip(self.alphas, self.estimators):
            pred = model.predict(X).astype(int)
            for col, cls in enumerate(self.classes):
                s[:, col] += alpha * (pred == cls)
        return s

    def predict(self, X):
        return self.classes[np.argmax(self.scores(X), axis=1)]

    def predict_proba(self, X):
        if len(self.alphas) == 0 or self.alphas.sum() == 0:
            return np.full((len(X), len(self.classes)), 1 / len(self.classes))
        return self.scores(X) / self.alphas.sum()


# ------------------------------------------------------------------ Q2(a) toy
def samme_round(X, y, w, forced_pred=None, label=""):
    """One complete SAMME round. forced_pred lets us feed a hand-made bad learner."""
    K = len(np.unique(y))
    if forced_pred is None:
        model = DecisionTreeClassifier(max_depth=1, random_state=SEED)
        model.fit(X, y, sample_weight=w)
        pred = model.predict(X).astype(int)
    else:
        pred = np.asarray(forced_pred, dtype=int)
    error = float(w[pred != y].sum())
    if error >= 1 - 1 / K:
        alpha, updated = 0.0, np.full(len(y), 1.0 / len(y))
        status = "REJECTED: not better than random; learner discarded, weights reset to uniform"
    else:
        alpha = 20.0 if error <= 0 else float(np.log((1 - error) / error) + np.log(K - 1))
        updated = w * np.exp(alpha * (pred != y))
        updated = updated / updated.sum()
        status = "accepted"
    return {"case": label, "features": X.ravel().tolist(), "labels": y.tolist(),
            "initial_weights": w.tolist(), "weak_predictions": pred.tolist(),
            "weighted_error": error, "random_threshold_1_minus_1_over_K": 1 - 1 / K,
            "alpha": alpha, "updated_normalized_weights": updated.tolist(), "status": status}


def toy_round():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    w = np.full(len(y), 1 / len(y))
    accepted = samme_round(X, y, w, label="stump on uniform weights")
    # Hand-made learner that is wrong on 6 of 8 samples -> error 0.75 >= 2/3
    bad = samme_round(X, y, w, forced_pred=[1, 1, 1, 2, 2, 2, 0, 1],
                      label="forced bad learner (demonstrates rejection branch)")
    save_json("q2a_toy_round.json", accepted)
    save_json("q2a_toy_rejection_demo.json", bad)
    print("Q2(a) complete SAMME round:\n", json.dumps(accepted, indent=2))
    print("Q2(a) rejection branch demo:\n", json.dumps(bad, indent=2))


# ------------------------------------------------------------------ helpers for plots / evidence
def show_face(ax, x, title):
    ax.imshow(np.asarray(x).reshape(64, 64), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def top_k(vec, k=5):
    idx = np.argsort(vec)[::-1][:k]
    return [{"class": int(c), "value": float(vec[c])} for c in idx]


def face_evidence(i, Xte, masked, yte, bag, boost):
    """Everything needed to explain a bagging-vs-boosting difference on test face i."""
    pc, pm = bag_predict(bag, Xte[[i]]), bag_predict(bag, masked[[i]])
    cnt_c = np.bincount(pc[:, 0], minlength=N_CLASSES)
    cnt_m = np.bincount(pm[:, 0], minlength=N_CLASSES)
    lc, lm = boost.learner_predictions(Xte[[i]])[:, 0], boost.learner_predictions(masked[[i]])[:, 0]
    flipped = lc != lm
    a = boost.alphas
    qc, qm = boost.predict_proba(Xte[[i]])[0], boost.predict_proba(masked[[i]])[0]
    true = int(yte[i])
    top_learners = np.argsort(a)[::-1][:5]
    return {
        "test_index_in_test_set": int(i), "true_label": true,
        "bagging": {
            "clean_prediction": int(cnt_c.argmax()), "masked_prediction": int(cnt_m.argmax()),
            "top5_vote_counts_clean": [{"class": int(c), "votes": int(cnt_c[c])} for c in np.argsort(cnt_c)[::-1][:5]],
            "top5_vote_counts_masked": [{"class": int(c), "votes": int(cnt_m[c])} for c in np.argsort(cnt_m)[::-1][:5]],
            "true_class_votes_clean": int(cnt_c[true]), "true_class_votes_masked": int(cnt_m[true]),
            "fraction_of_trees_whose_vote_changed": float(np.mean(pc[:, 0] != pm[:, 0])),
            "tree_pairwise_disagreement_clean": pairwise_disagreement(pc),
            "tree_pairwise_disagreement_masked": pairwise_disagreement(pm),
            "vote_margin_clean": vote_margin(pc), "vote_margin_masked": vote_margin(pm)},
        "boosting": {
            "clean_prediction": int(qc.argmax()), "masked_prediction": int(qm.argmax()),
            "top5_q_clean": top_k(qc), "top5_q_masked": top_k(qm),
            "true_class_q_clean": float(qc[true]), "true_class_q_masked": float(qm[true]),
            "fraction_of_learners_whose_vote_changed": float(np.mean(flipped)),
            "fraction_of_alpha_carried_by_changed_learners": float(a[flipped].sum() / a.sum()),
            "top5_alpha_learners": [{"round": int(j + 1), "alpha": float(a[j]),
                                     "clean_vote": int(lc[j]), "masked_vote": int(lm[j])}
                                    for j in top_learners],
            "final_round_sample_weights": {
                "max_weight": float(boost.history[-1]["max_weight"]) if boost.history else None,
                "top10_weight_sum": float(boost.history[-1]["top10_weight_sum"]) if boost.history else None}}}


# ------------------------------------------------------------------ main
def main():
    log_file = open(OUT / "q2_run_log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


def run():
    toy_round()
    Xtr, ytr, Xv, yv, Xte, yte, train_idx, test_idx = load_split()
    q1 = load_q1_config(Xtr, ytr, Xv, yv)
    q1_depth, q1_leaf, q1_B = q1["max_depth"], int(q1["min_samples_leaf"]), int(q1["bag_B"])

    # ---------- Q2(b): grid, validation-only selection
    rows, models, stop_rows = [], {}, []
    for depth in [1, 2, 3, 4]:
        for rounds in [10, 25, 50, 100]:
            start = time.perf_counter()
            model = SAMME(depth, rounds, SEED + 500).fit(Xtr, ytr)
            elapsed = time.perf_counter() - start
            models[(depth, rounds)] = model
            rows.append({"depth": depth, "requested_rounds": rounds,
                         "actual_rounds": len(model.estimators),
                         "n_rejection_events": len(model.events),
                         "train_accuracy": acc(ytr, model.predict(Xtr)),
                         "validation_accuracy": acc(yv, model.predict(Xv)),
                         "training_time_seconds": elapsed})
            for ev in model.events:
                stop_rows.append({"depth": depth, "requested_rounds": rounds, **ev})
    grid_unsorted = pd.DataFrame(rows)
    grid_unsorted.to_csv(OUT / "q2_boosting_grid.csv", index=False)
    pd.DataFrame(stop_rows, columns=["depth", "requested_rounds", "attempt",
                                     "accepted_rounds_so_far", "weighted_error",
                                     "threshold", "action"]).to_csv(OUT / "q2_stop_events.csv", index=False)
    grid = grid_unsorted.sort_values(["validation_accuracy", "requested_rounds", "depth"],
                                     ascending=[False, True, True])
    grid.to_csv(OUT / "q2_boosting_grid_sorted.csv", index=False)
    best = grid.iloc[0]
    setting = (int(best["depth"]), int(best["requested_rounds"]))
    boost = models[setting]
    save_json("q2b_selected_setting.json", {
        "depth": setting[0], "requested_rounds": setting[1],
        "actual_rounds": len(boost.estimators),
        "validation_accuracy": float(best["validation_accuracy"]),
        "rule": "max validation accuracy; ties -> fewer rounds, then smaller depth"})
    print("\nQ2(b) boosting grid (sorted):\n", grid.to_string(index=False))
    print("Selected setting:", setting, "| rejection events in grid:", len(stop_rows))

    history = pd.DataFrame([{
        "round": h["round"], "weighted_training_error": h["weighted_error"],
        "learner_weight_alpha": h["alpha"], "maximum_sample_weight": h["max_weight"],
        "top10_sample_weight_sum": h["top10_weight_sum"]} for h in boost.history])
    history.to_csv(OUT / "q2_selected_boosting_history.csv", index=False)
    for col, label in [("weighted_training_error", "Weighted training error"),
                       ("learner_weight_alpha", "Learner weight alpha"),
                       ("maximum_sample_weight", "Maximum sample weight"),
                       ("top10_sample_weight_sum", "Weight on 10 heaviest images")]:
        plt.figure()
        plt.plot(history["round"], history[col], marker="o", markersize=3)
        plt.xlabel("Boosting round"); plt.ylabel(label); plt.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(OUT / f"q2_{col}.png", dpi=160); plt.close()

    # ---------- Q2(c): bagging (frozen Q1 tree) vs boosting at 25/50/100 learners
    compare, bag_models = [], {}
    bag_seed = int(q1["bag_seed"])
    for B in sorted({25, 50, 100, q1_B}):
        start = time.perf_counter()
        trees, _ = train_bag(Xtr, ytr, B, q1_depth, q1_leaf, bag_seed)
        elapsed = time.perf_counter() - start
        bag_models[B] = trees
        if B in (25, 50, 100):
            compare.append({"model": f"bagging_{B}", "learners": B,
                            "train_accuracy": acc(ytr, majority_vote(bag_predict(trees, Xtr))),
                            "validation_accuracy": acc(yv, majority_vote(bag_predict(trees, Xv))),
                            "clean_test_accuracy": acc(yte, majority_vote(bag_predict(trees, Xte))),
                            "training_time_seconds": elapsed})
    d_sel = setting[0]
    for n in [25, 50, 100]:
        m = models[(d_sel, n)]
        t = float(grid_unsorted.query("depth==@d_sel and requested_rounds==@n")["training_time_seconds"].iloc[0])
        compare.append({"model": f"boosting_{n}_depth{d_sel}", "learners": len(m.estimators),
                        "train_accuracy": acc(ytr, m.predict(Xtr)),
                        "validation_accuracy": acc(yv, m.predict(Xv)),
                        "clean_test_accuracy": acc(yte, m.predict(Xte)),
                        "training_time_seconds": t})
    compare.append({"model": f"selected_boosting_depth{setting[0]}_rounds{setting[1]}",
                    "learners": len(boost.estimators),
                    "train_accuracy": acc(ytr, boost.predict(Xtr)),
                    "validation_accuracy": acc(yv, boost.predict(Xv)),
                    "clean_test_accuracy": acc(yte, boost.predict(Xte)),
                    "training_time_seconds": float(best["training_time_seconds"])})
    compare_df = pd.DataFrame(compare)
    compare_df.to_csv(OUT / "q2c_comparison.csv", index=False)
    print("\nQ2(c) comparison:\n", compare_df.to_string(index=False))

    # Heaviest face: sample with the largest weight at ANY round (robust to weight resets)
    if boost.history:
        W = np.array([h["weights"] for h in boost.history])
        i = int(np.argmax(W.max(axis=0)))
        peak_round = int(np.argmax(W[:, i])) + 1
        hist_i = [{"round": h["round"], "true_label": int(ytr[i]),
                   "prediction": int(h["predictions"][i]),
                   "correct": bool(h["predictions"][i] == ytr[i]),
                   "weight_after_update": float(h["weights"][i])} for h in boost.history]
        pd.DataFrame(hist_i).to_csv(OUT / "q2c_heaviest_sample_history.csv", index=False)
        save_json("q2c_heaviest_sample.json", {
            "training_index": i, "original_olivetti_index": int(train_idx[i]),
            "true_label": int(ytr[i]), "peak_weight": float(W[:, i].max()),
            "peak_round": peak_round, "final_weight": float(W[-1, i]),
            "fraction_of_rounds_misclassified": float(np.mean([not r["correct"] for r in hist_i])),
            "history": hist_i})
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        show_face(ax[0], Xtr[i], f"train idx {i} (Olivetti {int(train_idx[i])}), class {int(ytr[i])}")
        rounds_ = [r["round"] for r in hist_i]
        ws = [r["weight_after_update"] for r in hist_i]
        ax[1].plot(rounds_, ws, color="tab:blue", lw=1)
        wrong = [r for r in hist_i if not r["correct"]]
        ax[1].scatter([r["round"] for r in wrong], [r["weight_after_update"] for r in wrong],
                      color="red", s=14, label="misclassified in round", zorder=3)
        ax[1].set_xlabel("Boosting round"); ax[1].set_ylabel("Sample weight")
        ax[1].legend(fontsize=8); ax[1].grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(OUT / "q2c_heaviest_face.png", dpi=160); plt.close()

    # ---------- Q2(d): frozen tree, final bagging, boosting; clean vs masked
    tree = fit_tree(Xtr, ytr, q1_depth, q1_leaf)          # fitted ONCE, frozen Q1 setting
    bag = bag_models[q1_B]                                 # final Q1 ensemble size
    masked = mask_faces(Xte)
    robustness = []
    for name in ["single_tree", "bagging", "boosting"]:
        for set_name, X in [("clean", Xte), ("masked", masked)]:
            if name == "single_tree":
                p = tree.predict(X).astype(int)
                conf = float(tree.predict_proba(X).max(axis=1).mean())
            elif name == "bagging":
                pp = bag_predict(bag, X)
                p, conf = majority_vote(pp), vote_confidence(pp)
            else:
                p, conf = boost.predict(X), float(boost.predict_proba(X).max(axis=1).mean())
            robustness.append({"model": name, "test_set": set_name, "accuracy": acc(yte, p),
                               "mean_winning_vote_or_probability": conf})
    rob = pd.DataFrame(robustness)
    rob.to_csv(OUT / "q2d_robustness.csv", index=False)
    print("\nQ2(d) robustness:\n", rob.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    piv = rob.pivot(index="model", columns="test_set", values="accuracy").loc[["single_tree", "bagging", "boosting"]]
    piv[["clean", "masked"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Test accuracy"); ax.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q2d_robustness.png", dpi=160); plt.close()

    # One face where bagging and boosting respond differently to the mask
    bc, bm = majority_vote(bag_predict(bag, Xte)), majority_vote(bag_predict(bag, masked))
    sc, sm = boost.predict(Xte), boost.predict(masked)
    n_te = len(yte)
    def to_wrong(c, m, k):
        return c[k] == yte[k] and m[k] != yte[k]
    strong = [k for k in range(n_te) if to_wrong(bc, bm, k) != to_wrong(sc, sm, k)]
    weak = [k for k in range(n_te) if (bc[k] != bm[k]) != (sc[k] != sm[k])]
    if strong:
        k, rule = strong[0], "one model goes correct->wrong under the mask, the other does not"
    elif weak:
        k, rule = weak[0], "one model changes its prediction under the mask, the other does not"
    else:
        k, rule = None, None
    if k is None:
        save_json("q2d_different_mask_response.json", {
            "note": "No test face where bagging and boosting respond differently to the mask."})
        print("No differing-response face found.")
    else:
        ev = face_evidence(k, Xte, masked, yte, bag, boost)
        ev.update({"selection_rule": rule, "n_candidates_strict": len(strong),
                   "n_candidates_loose": len(weak), "original_olivetti_index": int(test_idx[k])})
        save_json("q2d_different_mask_response.json", ev)
        fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
        show_face(ax[0], Xte[k], f"clean | bag {int(bc[k])} boost {int(sc[k])} | true {int(yte[k])}")
        show_face(ax[1], masked[k], f"masked | bag {int(bm[k])} boost {int(sm[k])}")
        plt.tight_layout(); plt.savefig(OUT / "q2d_different_mask_response.png", dpi=160); plt.close()

    save_json("q2_config.json", {
        "roll_number": ROLL_NUMBER, "seed": SEED, "boost_seed": SEED + 500,
        "boost_grid": {"depth": [1, 2, 3, 4], "rounds": [10, 25, 50, 100]},
        "selected_boosting": {"depth": setting[0], "rounds": setting[1]},
        "q1_config_used": q1, "bagging_seed": bag_seed,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__}})
    print("Outputs saved in outputs/")


if __name__ == "__main__":
    main()