#!/usr/bin/env python3
"""Session 8 detector of AI-written Municibid descriptions.

Ground truth available:
  human  = descriptions of listings STARTED before 2022-11-30 (before ChatGPT existed);
  AI     = synthetic rewrites of 600 of those human descriptions (code/sample_for_synthetic.py; six
           generators, two instructions: V1 'enhance/appealing', V2 'concise/factual').
Three instruments, all applied to the base description text `desc` (what the seller submitted):
  C1  supervised: TF-IDF word 1-2 grams + format counts -> logistic regression; positives = synthetic,
      negatives = pre-ChatGPT human. 5 folds grouped by the original listing (a rewrite and its own
      original are always in the same fold). Threshold tau = 99th percentile of cross-fitted scores on
      human text (FPR 1%); checked out of time on listings started 2022-12-01..2025-01-28.
  L   Liang et al. (2024) distributional estimator: Bernoulli word-occurrence models P (human) and Q (AI);
      alpha = argmax sum log((1-a) P(doc) + a Q(doc)); validated on held-out mixtures of known alpha.
  NU  real pre-tool (2022-12-01..2025-01-28) vs real tool-era (2025-02-01..2025-09-30) classifier,
      grouped by agency; needs no synthetic text; robustness only.
Usage: python3 detector.py PANEL.parquet SYNTH.jsonl OUTDIR
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

LABEL_LINE = re.compile(r"(?m)^\s*(?:[-*•]\s*)?[A-Z][A-Za-z0-9 /&()'\-]{1,40}:\s+\S")


def fmt_feats(texts):
    rows = []
    for t in texts:
        w = len(t.split())
        rows.append([np.log1p(w), len(LABEL_LINE.findall(t)) / (1 + w / 50.0), t.count("—") > 0,
                     t.count("\n") / (1 + w / 50.0),
                     sum(1 for x in t.split() if x.isupper() and len(x) > 1) / (1 + w)])
    return csr_matrix(np.array(rows, dtype=float))


def norm(t):
    """Whitespace and quote normalisation. Curly quotes are a generator artefact (the synthetic
    generators write straight quotes; clerks pasting from word processors write curly ones), not
    evidence of authorship, so they are mapped to straight quotes before any feature is built."""
    t = (t or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]+", " ", t).strip()


class C1(object):
    def __init__(self, C=2.0):
        # min_df=10 drops item-unique words (e.g. 'polaroid', 'plotter') that the 600 rewrites share
        # only with their own originals
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=60000, sublinear_tf=True,
                                   token_pattern=r"(?u)\b[A-Za-z][A-Za-z']+\b", lowercase=True)
        self.clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")

    def fit(self, texts, y):
        X = hstack([self.vec.fit_transform(texts), fmt_feats(texts)]).tocsr()
        self.clf.fit(X, y)
        return self

    def score(self, texts):
        X = hstack([self.vec.transform(texts), fmt_feats(texts)]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


def bern_loglik(Xbin, p):
    """Xbin: docs x vocab {0,1}; p: vocab probabilities. Returns log P(doc) under independent Bernoullis."""
    lp, l1p = np.log(p), np.log1p(-p)
    return np.asarray(Xbin @ (lp - l1p)).ravel() + l1p.sum()


def mle_alpha(lp, lq):
    d = lq - lp
    d = np.clip(d, -60, 60)

    def nll(a):
        return -np.sum(np.log((1 - a) + a * np.exp(d)))
    r = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
    return float(r.x)


def main(panel_path, synth_path, outdir):
    df = pd.read_parquet(panel_path)
    df = df[df.desc.notna() & df.start_local_dt.notna()].copy()
    df["text"] = df.desc.map(norm)
    synth = [json.loads(l) for l in open(synth_path) if l.strip()]
    S = pd.DataFrame(synth)
    S["text"] = S.ai_text.map(norm)
    S = S[S.text.str.len() > 0]
    print("synthetic rewrites:", len(S), S.groupby(["variant", "model"]).size().to_dict())
    human = df[(df.start_local_dt >= "2019-01-01") & (df.start_local_dt < "2022-11-30")].copy()
    orig_ids = set(S["item"])
    # ---------------- C1 ----------------
    H = human[["item", "text", "agency"]].copy()
    H["y"] = 0
    H["grp"] = H["item"]
    A = S[["item", "text"]].copy()
    A["y"] = 1
    A["grp"] = A["item"]  # same group as its human original
    A["agency"] = None
    T = pd.concat([H, A], ignore_index=True)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(T))
    models = []
    for k, (tr, te) in enumerate(gkf.split(T, T.y, T.grp)):
        m = C1().fit(T.text.iloc[tr].tolist(), T.y.iloc[tr].values)
        oof[te] = m.score(T.text.iloc[te].tolist())
        models.append(m)
    T["oof"] = oof
    auc = roc_auc_score(T.y, T.oof)
    tau = float(np.quantile(T.loc[T.y == 0, "oof"], 0.99))
    tpr = float((T.loc[T.y == 1, "oof"] >= tau).mean())
    print("\nC1 cross-fitted AUC %.4f | tau(FPR 1%% on pre-ChatGPT human) %.4f | TPR at tau %.3f" % (auc, tau, tpr))
    Sx = S.merge(T[T.y == 1][["item", "oof"]], on="item", how="left")
    print("TPR by generator:", Sx.groupby(["variant", "model"]).oof.apply(lambda s: round(float((s >= tau).mean()), 3)).to_dict())
    print("paired check: share of rewrites scoring above their own original:",
          round(float((Sx.set_index("item").oof > T[(T.y == 0) & T["item"].isin(orig_ids)].set_index("item").oof.reindex(Sx["item"]).values).mean()), 3))
    # score everything with the average of the fold models (training docs get their out-of-fold score)
    allsc = np.mean([m.score(df.text.tolist()) for m in models], axis=0)
    df["ai_score"] = allsc
    hs = dict(zip(T.loc[T.y == 0, "item"], T.loc[T.y == 0, "oof"]))
    df.loc[df["item"].isin(hs), "ai_score"] = df.loc[df["item"].isin(hs), "item"].map(hs)
    df["ai_flag"] = (df.ai_score >= tau).astype(int)
    p1 = df[(df.start_local_dt >= "2022-12-01") & (df.start_local_dt < "2025-01-29")]
    print("out-of-time FPR proxy, listings started 2022-12..2025-01 (ChatGPT existed, no tool): %.4f (n=%d)" % (p1.ai_flag.mean(), len(p1)))
    for y in range(2019, 2027):
        g = df[df.start_local_dt.dt.year == y]
        print("  flag share %d: %.4f (n=%d)" % (y, g.ai_flag.mean(), len(g)))
    # top words of C1
    m0 = models[0]
    names = np.array(m0.vec.get_feature_names_out().tolist() + ["FMT_logw", "FMT_label", "FMT_emdash", "FMT_newline", "FMT_caps"])
    coef = m0.clf.coef_.ravel()
    o = np.argsort(coef)
    print("C1 most AI-indicative features:", list(names[o[-40:]][::-1]))
    print("C1 most human-indicative features:", list(names[o[:30]]))
    # ---------------- Liang distributional estimator ----------------
    cv = CountVectorizer(binary=True, min_df=5, token_pattern=r"(?u)\b[A-Za-z][A-Za-z']+\b", lowercase=True)
    rng = np.random.default_rng(1)
    hum_ids = human["item"].values
    rng.shuffle(hum_ids)
    hold_h = set(hum_ids[: len(hum_ids) // 5])
    s_ids = S["item"].unique().copy()
    rng.shuffle(s_ids)
    hold_s = set(s_ids[: len(s_ids) // 5])
    h_tr = human[~human["item"].isin(hold_h)].text.tolist()
    s_tr = S[~S["item"].isin(hold_s)].text.tolist()
    cv.fit(h_tr + s_tr)
    Xh = cv.transform(h_tr)
    Xs = cv.transform(s_tr)
    dfh = np.asarray(Xh.sum(0)).ravel() / Xh.shape[0]
    dfs = np.asarray(Xs.sum(0)).ravel() / Xs.shape[0]
    # Liang et al. use a vocabulary shared by P and Q. Rare words otherwise receive asymmetric
    # smoothing across corpora of very different size (first run: true alpha 0 estimated at 0.098).
    keep = np.where((dfh >= 0.01) & (dfs >= 0.01))[0]
    vocab = [str(x) for x in np.array(cv.get_feature_names_out())[keep]]
    cv = CountVectorizer(binary=True, vocabulary=vocab, token_pattern=r"(?u)\b[A-Za-z][A-Za-z']+\b", lowercase=True)
    Xh = cv.transform(h_tr)
    Xs = cv.transform(s_tr)
    p = (np.asarray(Xh.sum(0)).ravel() + 1.0) / (Xh.shape[0] + 2.0)
    q = (np.asarray(Xs.sum(0)).ravel() + 1.0) / (Xs.shape[0] + 2.0)
    print("Liang vocabulary (>= 1%% of documents in both corpora): %d words" % len(vocab))

    def ll(texts):
        X = cv.transform(texts)
        return bern_loglik(X, p), bern_loglik(X, q)
    hh = human[human["item"].isin(hold_h)].text.tolist()
    ss = S[S["item"].isin(hold_s)].text.tolist()
    lph, lqh = ll(hh)
    lps, lqs = ll(ss)
    print("\nLiang validation on held-out mixtures (true alpha -> estimate):")
    for a in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]:
        est = []
        for b in range(30):
            nh = 1000
            na = int(round(a * nh / (1 - a))) if a < 1 else nh
            ih = rng.integers(0, len(hh), nh)
            ia = rng.integers(0, len(ss), na) if na > 0 else np.array([], dtype=int)
            lp_ = np.concatenate([lph[ih], lps[ia]])
            lq_ = np.concatenate([lqh[ih], lqs[ia]])
            est.append(mle_alpha(lp_, lq_))
        print("  %.2f -> mean %.4f (sd %.4f)" % (a, np.mean(est), np.std(est)))
    lpa, lqa = ll(df.text.tolist())
    df["liang_llr"] = lqa - lpa
    df["m"] = df.start_local_dt.dt.to_period("M").astype(str)
    rows = []
    for m, g in df[df.start_local_dt >= "2019-01-01"].groupby("m"):
        lp_, lq_ = lpa[g.index.map(df.index.get_loc)], lqa[g.index.map(df.index.get_loc)]
        a = mle_alpha(lp_, lq_)
        bs = []
        for b in range(200):
            ii = rng.integers(0, len(g), len(g))
            bs.append(mle_alpha(lp_[ii], lq_[ii]))
        rows.append((m, len(g), a, np.quantile(bs, 0.025), np.quantile(bs, 0.975), g.ai_flag.mean()))
    M = pd.DataFrame(rows, columns=["month", "n", "alpha", "lo", "hi", "c1_flag_share"])
    M.to_csv(outdir + "/monthly_alpha.csv", index=False)
    print("\nmonthly alpha (Liang) and C1 flag share, 2022-06 onward:")
    print(M[M.month >= "2022-06"].round(4).to_string(index=False))
    # ---------------- NU robustness ----------------
    pre = df[(df.start_local_dt >= "2022-12-01") & (df.start_local_dt < "2025-01-29")].copy()
    post = df[(df.start_local_dt >= "2025-02-01") & (df.start_local_dt < "2025-10-01")].copy()
    pre["y"] = 0
    post["y"] = 1
    U = pd.concat([pre, post])
    oofu = np.zeros(len(U))
    for tr, te in GroupKFold(n_splits=5).split(U, U.y, U.agency):
        m = C1().fit(U.text.iloc[tr].tolist(), U.y.iloc[tr].values)
        oofu[te] = m.score(U.text.iloc[te].tolist())
    U["nu"] = oofu
    print("\nNU (pre-tool vs tool-era, grouped by agency) AUC %.4f" % roc_auc_score(U.y, U.nu))
    pp = U[U.y == 1]
    print("correlation of NU and C1 scores among tool-era listings: spearman %.3f" % pp[["nu", "ai_score"]].corr(method="spearman").iloc[0, 1])
    df = df.merge(U[["item", "nu"]], on="item", how="left")
    df[["item", "ai_score", "ai_flag", "liang_llr", "nu"]].to_parquet(outdir + "/ai_scores.parquet", index=False)
    json.dump({"tau": tau, "auc_c1": auc, "tpr_at_tau": tpr, "n_synth": int(len(S))}, open(outdir + "/detector_meta.json", "w"))
    print("wrote", outdir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
