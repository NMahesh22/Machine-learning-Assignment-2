
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL_NUMBER = "M26ECC005"
SEED = int("".join(ch for ch in ROLL_NUMBER if ch.isdigit()))  # 26005
K = 40                                  # identities / classes
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
LOG = []


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    LOG.append(text)


# ----------------------------------------------------------------- data ----
def load_split():
    """6 train / 2 validation / 2 test images per identity; saved and reused."""
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data.astype(np.float32), data.target.astype(int)
    split_file = OUT / "split.npz"
    if split_file.exists():
        s = np.load(split_file)
        tr, va, te = s["train_idx"], s["val_idx"], s["test_idx"]
    else:
        rng = np.random.default_rng(SEED)
        tr, va, te = [], [], []
        for cls in np.unique(y):
            ids = rng.permutation(np.flatnonzero(y == cls))
            tr.extend(ids[:6]); va.extend(ids[6:8]); te.extend(ids[8:10])
        tr, va, te = map(np.asarray, (tr, va, te))
        np.savez(split_file, train_idx=tr, val_idx=va, test_idx=te)
        (OUT / "split.json").write_text(json.dumps({
            "roll_number": ROLL_NUMBER, "seed": SEED, "train_idx": tr.tolist(),
            "validation_idx": va.tolist(), "test_idx": te.tolist()}, indent=2))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def mask_faces(X):
    """Zero the centred 16x16 block (rows/cols 24..39)."""
    imgs = X.reshape(-1, 64, 64).copy()
    imgs[:, 24:40, 24:40] = 0.0
    return imgs.reshape(len(X), -1)


# -------------------------------------------------------------- helpers ----
def acc(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def fit_tree(X, y, depth, leaf, seed):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                  random_state=seed).fit(X, y)


def leaf_stats(model):
    t = model.tree_
    counts = t.n_node_samples[t.children_left == -1]
    return int(len(counts)), float(counts.mean()), float(np.mean(counts == 1))


def vote_counts(preds):
    """preds: (n_models, n_samples) integer labels -> (n_samples, K) vote counts."""
    counts = np.zeros((preds.shape[1], K))
    cols = np.arange(preds.shape[1])
    for p in preds:
        counts[cols, p] += 1
    return counts


def vote_margin(counts, B):
    s = np.sort(counts, axis=1)
    return float(np.mean((s[:, -1] - s[:, -2]) / B))


def vote_confidence(counts, B):
    return float(np.mean(counts.max(axis=1) / B))


def pairwise_disagreement(counts, B):
    """Exact mean pairwise disagreement from vote counts (0 pairs for B=1 -> NaN)."""
    if B < 2:
        return float("nan")
    agree = (counts * (counts - 1)).sum(axis=1) / (B * (B - 1))
    return float(1.0 - agree.mean())


def tree_confidence(model, X):
    return float(model.predict_proba(X).max(axis=1).mean())


def show_face(ax, x, title=""):
    ax.imshow(x.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ------------------------------------------------------------------ main ----
def main():
    Xtr, ytr, Xv, yv, Xte, yte = load_split()
    Xte_m = mask_faces(Xte)
    n = len(ytr)
    log(f"Roll number {ROLL_NUMBER}, seed {SEED}; train/val/test = {len(ytr)}/{len(yv)}/{len(yte)}")

    # ===== Q1(a) single tree baseline ====================================
    DEPTHS, LEAVES = [2, 4, 8, 16, None], [1, 2, 4]
    rows, models = [], {}
    for depth in DEPTHS:
        for leaf in LEAVES:
            m = fit_tree(Xtr, ytr, depth, leaf, SEED)
            models[(depth, leaf)] = m
            n_leaves, mean_leaf, frac_one = leaf_stats(m)
            rows.append({
                "max_depth": "None" if depth is None else depth,
                "min_samples_leaf": leaf, "actual_depth": int(m.get_depth()),
                "train_accuracy": acc(ytr, m.predict(Xtr)),
                "validation_accuracy": acc(yv, m.predict(Xv)),
                "test_accuracy": acc(yte, m.predict(Xte)),
                "n_leaves": n_leaves, "mean_samples_per_leaf": mean_leaf,
                "fraction_single_example_leaves": frac_one})
    grid_raw = pd.DataFrame(rows)

    # Selection uses VALIDATION accuracy only. Ties: shallower depth, then larger leaf.
    def sel_key(r):
        d = np.inf if r["max_depth"] == "None" else r["max_depth"]
        return (-r["validation_accuracy"], d, -r["min_samples_leaf"])
    best = min(rows, key=sel_key)
    best_depth = None if best["max_depth"] == "None" else int(best["max_depth"])
    best_leaf = int(best["min_samples_leaf"])
    tree = models[(best_depth, best_leaf)]
    grid_raw.to_csv(OUT / "q1a_tree_grid.csv", index=False)
    log("\nQ1(a) tree grid:\n" + grid_raw.to_string(index=False))
    log(f"FROZEN tree setting: max_depth={best_depth}, min_samples_leaf={best_leaf} "
        f"(selected on validation accuracy; ties -> shallower depth, larger leaf)")

    xs, xl = np.arange(len(DEPTHS)), [str(d) for d in DEPTHS]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    sub = grid_raw[grid_raw.min_samples_leaf == best_leaf]
    for col, lab in [("train_accuracy", "train"), ("validation_accuracy", "validation"),
                     ("test_accuracy", "test")]:
        ax[0, 0].plot(xs, sub[col].values, marker="o", label=lab)
    ax[0, 0].set_title(f"Accuracy vs depth (min_samples_leaf={best_leaf})"); ax[0, 0].legend()
    for panel, col, title in [(ax[0, 1], "n_leaves", "Number of leaves"),
                              (ax[1, 0], "mean_samples_per_leaf", "Mean training samples per leaf"),
                              (ax[1, 1], "fraction_single_example_leaves",
                               "Fraction of leaves with 1 training example")]:
        for leaf in LEAVES:
            panel.plot(xs, grid_raw[grid_raw.min_samples_leaf == leaf][col].values,
                       marker="o", label=f"min_samples_leaf={leaf}")
        panel.set_title(title); panel.legend(fontsize=8)
    for a in ax.ravel():
        a.set_xticks(xs); a.set_xticklabels(xl); a.set_xlabel("max_depth"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "q1a_depth_trends.png", dpi=160); plt.close()

    # ===== Q1(b) bagging from scratch ====================================
    B_LIST, B_MAX = [1, 5, 10, 25, 50, 100, 200], 200
    rng = np.random.default_rng(SEED + 1000)
    trees, distinct = [], []
    oob_mask = np.zeros((B_MAX, n), dtype=bool)   # True where example is out-of-bag
    for b in range(B_MAX):
        sample = rng.integers(0, n, size=n)        # bootstrap of original training size
        uniq = np.unique(sample)
        distinct.append(len(uniq))
        oob_mask[b] = True
        oob_mask[b, uniq] = False
        trees.append(fit_tree(Xtr[sample], ytr[sample], best_depth, best_leaf, SEED + b + 1))
    # Ensembles for smaller B are the first B trees of this single sequence (nested).
    P_tr = np.asarray([t.predict(Xtr).astype(int) for t in trees])
    P_va = np.asarray([t.predict(Xv).astype(int) for t in trees])
    P_te = np.asarray([t.predict(Xte).astype(int) for t in trees])
    P_te_m = np.asarray([t.predict(Xte_m).astype(int) for t in trees])
    theory = n * (1 - (1 - 1 / n) ** n)

    oob_counts, oob_res = np.zeros((n, K)), {}
    for b in range(B_MAX):
        idx = np.flatnonzero(oob_mask[b])
        oob_counts[idx, P_tr[b][idx]] += 1
        if b + 1 in B_LIST:
            cov = oob_counts.sum(axis=1) > 0
            oob_res[b + 1] = (acc(ytr[cov], oob_counts[cov].argmax(axis=1)), float(cov.mean()))

    results = []
    for B in B_LIST:
        c_tr, c_va, c_te = (vote_counts(P[:B]) for P in (P_tr, P_va, P_te))
        md = float(np.mean(distinct[:B]))
        results.append({
            "B": B,
            "train_accuracy": acc(ytr, c_tr.argmax(1)),
            "validation_accuracy": acc(yv, c_va.argmax(1)),
            "test_accuracy": acc(yte, c_te.argmax(1)),
            "mean_distinct_examples": md, "theoretical_expectation": theory,
            "difference_vs_theory": md - theory,
            "oob_accuracy": oob_res[B][0], "oob_coverage": oob_res[B][1],
            "pairwise_disagreement_val": pairwise_disagreement(c_va, B),
            "mean_vote_margin_val": vote_margin(c_va, B)})
    bag_df = pd.DataFrame(results)
    bag_df.to_csv(OUT / "q1_bagging_results.csv", index=False)
    log("\nQ1(b,c) bagging results:\n" + bag_df.to_string(index=False))
    log(f"Theoretical expected distinct examples per bootstrap: n(1-(1-1/n)^n) = {theory:.3f} "
        f"({theory / n:.4f} n)")

    # Final B: highest validation accuracy; ties -> higher OOB accuracy, then smaller B.
    best_B = int(sorted(results, key=lambda r: (-r["validation_accuracy"],
                                                -r["oob_accuracy"], r["B"]))[0]["B"])
    log(f"FINAL ensemble size selected on validation: B={best_B} "
        f"(ties -> higher OOB accuracy, then smaller B)")

    # --- figures for (b) and (c)
    plt.figure(figsize=(6, 4))
    plt.bar([str(b) for b in B_LIST], bag_df["mean_distinct_examples"], color="steelblue",
            label="measured mean distinct")
    plt.axhline(theory, color="red", ls="--", label=f"theory = {theory:.1f}")
    plt.xlabel("B"); plt.ylabel("distinct examples per bootstrap sample")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "q1b_distinct_vs_theory.png", dpi=160); plt.close()

    for metric, label in [("oob_accuracy", "OOB accuracy"),
                          ("pairwise_disagreement_val", "Pairwise disagreement (validation)"),
                          ("mean_vote_margin_val", "Mean vote margin (validation)")]:
        plt.figure(figsize=(6, 4))
        plt.plot(bag_df["B"], bag_df[metric], marker="o")
        plt.xscale("log"); plt.xlabel("Number of trees B"); plt.ylabel(label)
        plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(OUT / f"q1c_{metric}.png", dpi=160); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for col, lab in [("train_accuracy", "train"), ("validation_accuracy", "validation"),
                     ("test_accuracy", "test"), ("oob_accuracy", "OOB")]:
        ax[0].plot(bag_df["B"], bag_df[col], marker="o", label=lab)
    ax[0].set_title("Accuracy vs B"); ax[0].legend()
    ax[1].plot(bag_df["B"], bag_df["pairwise_disagreement_val"], marker="o", color="tab:red")
    ax[1].set_title("Pairwise disagreement (validation)")
    ax[2].plot(bag_df["B"], bag_df["mean_vote_margin_val"], marker="o", color="tab:green")
    ax[2].set_title("Mean vote margin (validation)")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("B"); a.grid(alpha=.3)
        a.axvline(best_B, color="k", ls=":", lw=1)
    plt.tight_layout(); plt.savefig(OUT / "q1c_combined.png", dpi=160); plt.close()

    # ===== Q1(d) tree vs bagging, clean and masked =======================
    Bn = best_B
    c_clean, c_mask = vote_counts(P_te[:Bn]), vote_counts(P_te_m[:Bn])
    tp_clean, tp_mask = tree.predict(Xte).astype(int), tree.predict(Xte_m).astype(int)
    bp_clean, bp_mask = c_clean.argmax(1), c_mask.argmax(1)
    rob = [
        {"model": "single_tree", "test_set": "clean", "accuracy": acc(yte, tp_clean),
         "mean_confidence": tree_confidence(tree, Xte)},
        {"model": "single_tree", "test_set": "masked", "accuracy": acc(yte, tp_mask),
         "mean_confidence": tree_confidence(tree, Xte_m)},
        {"model": f"bagging_B{Bn}", "test_set": "clean", "accuracy": acc(yte, bp_clean),
         "mean_confidence": vote_confidence(c_clean, Bn)},
        {"model": f"bagging_B{Bn}", "test_set": "masked", "accuracy": acc(yte, bp_mask),
         "mean_confidence": vote_confidence(c_mask, Bn)}]
    rob_df = pd.DataFrame(rob)
    rob_df.to_csv(OUT / "q1d_robustness.csv", index=False)
    log("\nQ1(d) clean vs masked:\n" + rob_df.to_string(index=False))

    # robustness bar chart
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, col, title in [(ax[0], "accuracy", "Test accuracy"),
                          (ax[1], "mean_confidence", "Mean prediction confidence")]:
        piv = rob_df.pivot(index="model", columns="test_set", values=col)[["clean", "masked"]]
        piv.plot.bar(ax=a, rot=0); a.set_title(title); a.set_ylim(0, 1.05); a.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "q1d_clean_vs_masked.png", dpi=160); plt.close()

    # masked-face illustration
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
    for j in range(6):
        show_face(ax[0, j], Xte[j * 13], f"clean, id {yte[j * 13]}")
        show_face(ax[1, j], Xte_m[j * 13], "masked")
    plt.tight_layout(); plt.savefig(OUT / "q1d_masked_examples.png", dpi=160); plt.close()

    # two faces bagging corrects + two faces where both are wrong
    true_frac = c_clean[np.arange(len(yte)), yte] / Bn
    corrected = [i for i in range(len(yte)) if tp_clean[i] != yte[i] and bp_clean[i] == yte[i]]
    both_wrong = [i for i in range(len(yte)) if tp_clean[i] != yte[i] and bp_clean[i] != yte[i]]
    corrected = sorted(corrected, key=lambda i: -true_frac[i])[:2]      # clearest corrections
    bw = sorted(both_wrong, key=lambda i: true_frac[i])                  # true-class votes ascending
    both_wrong = bw if len(bw) < 2 else [bw[0], bw[-1]]                  # 0 votes vs. closest miss
    log(f"\nBagging corrects the tree on {sum(1 for i in range(len(yte)) if tp_clean[i] != yte[i] and bp_clean[i] == yte[i])} "
        f"test faces; both wrong on {len(bw)} test faces.")
    cases = []
    for kind, ids in [("bagging_corrects_tree", corrected), ("both_wrong", both_wrong)]:
        for i in ids:
            order = np.argsort(-c_clean[i])[:5]
            cases.append({
                "case": kind, "test_index": int(i), "true_label": int(yte[i]),
                "tree_prediction": int(tp_clean[i]), "bagging_prediction": int(bp_clean[i]),
                "true_class_votes": int(c_clean[i, yte[i]]),
                "votes_for_tree_prediction": int(c_clean[i, tp_clean[i]]),
                "winning_votes": int(c_clean[i].max()), "total_trees": Bn,
                "top5_bagging_votes": json.dumps({int(k): int(c_clean[i, k]) for k in order}),
                "tree_pred_masked": int(tp_mask[i]), "bagging_pred_masked": int(bp_mask[i])})
    cases_df = pd.DataFrame(cases)
    cases_df.to_csv(OUT / "q1d_example_faces.csv", index=False)
    log("Q1(d) example faces:\n" + cases_df.to_string(index=False))

    if cases:
        fig, ax = plt.subplots(len(cases), 5, figsize=(14, 2.9 * len(cases)),
                               gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.7]}, squeeze=False)
        for r, cs in enumerate(cases):
            i, t_lab, tp_lab, bp_lab = (cs["test_index"], cs["true_label"],
                                        cs["tree_prediction"], cs["bagging_prediction"])
            first = lambda c: Xtr[np.flatnonzero(ytr == c)[0]]
            show_face(ax[r, 0], Xte[i], f"{cs['case']}\ntest face (true id {t_lab})")
            show_face(ax[r, 1], first(t_lab), f"train face, true id {t_lab}")
            show_face(ax[r, 2], first(tp_lab), f"tree predicts id {tp_lab}")
            show_face(ax[r, 3], first(bp_lab), f"bagging predicts id {bp_lab}")
            order = np.argsort(-c_clean[i])[:5]
            ax[r, 4].bar([str(k) for k in order], c_clean[i, order],
                         color=["tab:green" if k == t_lab else "tab:gray" for k in order])
            ax[r, 4].set_title(f"bagging votes (green = true id), B={Bn}", fontsize=8)
            ax[r, 4].set_xlabel("class id", fontsize=8)
        plt.tight_layout(); plt.savefig(OUT / "q1d_example_faces.png", dpi=160); plt.close()

    # ===== freeze for Q2 / Q3 ============================================
    (OUT / "frozen_config.json").write_text(json.dumps({
        "roll_number": ROLL_NUMBER, "seed": SEED,
        "single_tree": {"max_depth": best_depth, "min_samples_leaf": best_leaf},
        "bagging": {"B": best_B, "bootstrap_seed": SEED + 1000}}, indent=2))
    joblib.dump(trees[:best_B], OUT / "q1_final_bagging_trees.joblib")
    joblib.dump(tree, OUT / "q1_frozen_tree.joblib")
    (OUT / "q1_log.txt").write_text("\n".join(LOG))
    log("\nAll Q1 outputs saved in", OUT.resolve())


if __name__ == "__main__":
    main()