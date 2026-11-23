#!/usr/bin/env python3
"""Session 8, analysis E (v3): how much price information does the seller's text carry, by period,
and does the pre-ChatGPT market's own text-to-price mapping still hold for AI-written listings?

r  = log closing high bid (>= 1 bid) minus the subcategory x end-quarter mean (subcategory mean, then
     grand mean, where a cell is empty), computed on training folds only; folds grouped by AGENCY.
s  = out-of-fold prediction of r from the text.
D  = mean r in the top fifth of s minus mean r in the bottom fifth (log points).
Uncertainty: B draws; in each draw every period is resampled by AGENCY with replacement (cluster
bootstrap), subsampled to the same n, and the whole cross-fitting is re-run. rho = D_t / D_P0 per draw.
Text representations: full (TF-IDF 1-2 grams, ridge); facts (log counts of numbers, units, years,
identifiers, condition and defect statements); style (function-word rates and log-transformed style
counts, clipped at |z| <= 4; object words never enter).
Frozen: the P0 full-text model and P0 cutpoints applied to P1 and P2; for each group the calibration
slope of r on s, the mean of (s - r) (over-prediction, log points), and the top/bottom-fifth shares,
with agency-bootstrap intervals.
Usage: python3 informativeness.py PANEL FEATURES AISCORES OUTDIR [B]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from analysis_core import load  # noqa: E402

FUNC = sorted(ENGLISH_STOP_WORDS)
STYLE_CNT = ["s_w", "s_sent", "s_n_label", "s_n_excl", "s_emdash", "s_n_mark", "s_n_puff", "s_n_you", "s_n_typo"]
STYLE_RATE = ["s_caps_share", "boiler_share"]
FACT_NUM = ["s_n_num", "s_n_unit", "s_n_year", "s_n_ident", "s_n_pos", "s_n_defect"]
ALPHAS = np.logspace(-2, 4, 14)


def resid(train, test):
    mu = train.groupby("subcat_q").log_price.mean()
    mu2 = train.groupby("subcat").log_price.mean()
    g = train.log_price.mean()

    def f(x):
        return x.log_price - x.subcat_q.map(mu).fillna(x.subcat.map(mu2)).fillna(g)
    return f(train), f(test)


def style_raw(x, cv):
    C = cv.transform(x.fulltext).toarray().astype(float)
    w = np.maximum(1.0, x.s_w.values.astype(float))[:, None]
    A = np.log1p(np.nan_to_num(x[STYLE_CNT].values.astype(float)))
    R = np.nan_to_num(x[STYLE_RATE].values.astype(float))
    return np.hstack([C / w, A, R])


def make_X(kind, tr, te):
    if kind == "full":
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        return v.fit_transform(tr.fulltext), v.transform(te.fulltext)
    if kind == "facts":
        A = np.log1p(tr[FACT_NUM].values.astype(float))
        B = np.log1p(te[FACT_NUM].values.astype(float))
    else:
        cv = CountVectorizer(vocabulary=FUNC, lowercase=True)
        A, B = style_raw(tr, cv), style_raw(te, cv)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    return np.clip((A - mu) / sd, -4, 4), np.clip((B - mu) / sd, -4, 4)


def crossfit(g, kind):
    g = g.reset_index(drop=True)
    s = np.full(len(g), np.nan)
    r_out = np.full(len(g), np.nan)
    # folds by the ORIGINAL agency: copies of an agency drawn twice by the bootstrap must stay in the
    # same fold, otherwise its price level leaks from training to test (bug found in v2, job 817628)
    k = min(5, g.agency.nunique())
    for tr, te in GroupKFold(n_splits=k).split(g, groups=g.agency):
        a, b = g.iloc[tr], g.iloc[te]
        rtr, rte = resid(a, b)
        Xtr, Xte = make_X(kind, a, b)
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, rtr.values)
        s[te] = m.predict(Xte)
        r_out[te] = rte.values
    return s, r_out


def spread(s, r):
    q = pd.qcut(pd.Series(s).rank(method="first"), 5, labels=False).values
    return r[q == 4].mean() - r[q == 0].mean(), 1 - np.mean((r - s) ** 2) / np.var(r)


def cluster_resample(g, rng):
    ags = g.agency.unique()
    pick = rng.choice(ags, size=len(ags), replace=True)
    parts = []
    for j, a in enumerate(pick):
        h = g[g.agency == a].copy()
        h["boot_cluster"] = h.agency + "#%d" % j  # bootstrap copy id (for reference only)
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def main(panel, feats, scores, outdir, B="20"):
    B = int(B)
    d = load(panel, feats, scores)
    d = d[d.log_price.notna()].copy()
    d["fulltext"] = d.desc.fillna("") + "\n" + d.updates_text.fillna("")
    n_by = d.period.value_counts().to_dict()
    nmin = min(n_by.values())
    print("priced listings by period", n_by, "| equal n per draw:", nmin, "| draws:", B)
    rng = np.random.default_rng(11)
    rows = []
    if os.environ.get("SKIP_SPREADS") == "1":
        B = 0
        print("SKIP_SPREADS=1: within-period spreads skipped; frozen-model section only")
    for b in range(B):
        for per in ["P0", "P1", "P2"]:
            g = cluster_resample(d[d.period == per], rng)
            g = g.sample(nmin, random_state=int(rng.integers(0, 1e9)), replace=len(g) < nmin)
            for kind in ["full", "facts", "style"]:
                s, r = crossfit(g, kind)
                D, r2 = spread(s, r)
                rows.append({"draw": b, "period": per, "kind": kind, "D": D, "oos_r2": r2})
    R = pd.DataFrame(rows)
    if len(R):
        R.to_csv(outdir + "/E_spreads_draws.csv", index=False)
    if not len(R):
        S = None
    else:
        S = R.groupby(["kind", "period"]).agg(D=("D", "mean"), D_lo=("D", lambda x: np.quantile(x, 0.025)),
                                              D_hi=("D", lambda x: np.quantile(x, 0.975)), r2=("oos_r2", "mean")).reset_index()
        S["D_pct"] = 100 * (np.exp(S.D) - 1)
        print("\nWithin-period spreads, agency-bootstrap means and 95% intervals")
        print(S.round(4).to_string(index=False))
        print("\nrho = D_t / D_P0 and D_t - D_P0 across draws (2.5-97.5%):")
        for kind in ["full", "facts", "style"]:
            w = R[R.kind == kind].pivot(index="draw", columns="period", values="D")
            for per in ["P1", "P2"]:
                rho = w[per] / w["P0"]
                dd = w[per] - w["P0"]
                print("  %-6s %s: rho %.3f [%.3f, %.3f]   diff %.4f [%.4f, %.4f]" % (
                    kind, per, rho.mean(), rho.quantile(.025), rho.quantile(.975), dd.mean(), dd.quantile(.025), dd.quantile(.975)))
    # ---------------- frozen P0 model ----------------
    print("\nFrozen P0 full-text model (fit on all P0 listings) applied to P1 and P2")
    p0 = d[d.period == "P0"].copy()
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    X0 = v.fit_transform(p0.fulltext)
    r0 = p0.log_price - p0.subcat_q.map(p0.groupby("subcat_q").log_price.mean())
    m = RidgeCV(alphas=ALPHAS).fit(X0, r0.values)
    s0 = cross_s0 = None
    cut = None
    # P0 cutpoints from cross-fitted P0 scores (in-sample scores would be overconfident)
    s0, r0x = crossfit(p0, "full")
    cut = np.quantile(s0, [0.2, 0.8])
    groups = {}
    for per in ["P1", "P2"]:
        g = d[d.period == per].copy()
        g["r"] = g.log_price - g.subcat_q.map(g.groupby("subcat_q").log_price.mean())
        g["s"] = m.predict(v.transform(g.fulltext))
        groups[per + "_all"] = g
        if per == "P2":
            groups["P2_human"] = g[g.ai == 0]
            groups["P2_AI"] = g[g.ai == 1]
    def stats(h):
        slope = np.polyfit(h.s, h.r, 1)[0] if len(h) > 10 else np.nan
        return slope, float((h.s - h.r).mean()), float((h.s >= cut[1]).mean()), float((h.s <= cut[0]).mean()), float(h.s.mean()), float(h.r.mean())
    print("  P0 cross-fitted reference: top-fifth cut %.3f, bottom-fifth cut %.3f" % (cut[1], cut[0]))
    for k, h in groups.items():
        sl, over, top, bot, ms, mr = stats(h)
        bs = []
        ags = h.agency.unique()
        av = h.agency.values
        idx = {a: np.where(av == a)[0] for a in ags}
        for b in range(300):
            pick = rng.choice(ags, size=len(ags), replace=True)
            hh = h.iloc[np.concatenate([idx[a] for a in pick])]
            bs.append(stats(hh))
        bs = np.array(bs)
        print("  %-9s n=%5d agencies=%4d | slope %.3f [%.3f, %.3f] | over-prediction (s-r) %+.3f [%+.3f, %+.3f] | mean s %.3f mean r %.3f | in P0 top fifth %.3f, bottom fifth %.3f" % (
            k, len(h), len(ags), sl, np.nanquantile(bs[:, 0], .025), np.nanquantile(bs[:, 0], .975), over,
            np.quantile(bs[:, 1], .025), np.quantile(bs[:, 1], .975), ms, mr, top, bot))
    a, hmn = groups["P2_AI"], groups["P2_human"]
    diff = []
    for b in range(500):
        aa = a.sample(len(a), replace=True, random_state=b)
        hh = hmn.sample(len(hmn), replace=True, random_state=10000 + b)
        diff.append(((aa.s - aa.r).mean()) - ((hh.s - hh.r).mean()))
    print("  over-prediction AI minus human: %+.3f [%+.3f, %+.3f] log points (listing bootstrap)" % (
        float((a.s - a.r).mean() - (hmn.s - hmn.r).mean()), np.quantile(diff, .025), np.quantile(diff, .975)))
    from regs import ols_fe
    p2 = groups["P2_all"].copy()
    p2["over"] = p2.s - p2.r
    print("  within agency (agency FE + subcategory FE, agency-clustered): AI coefficient on")
    for y, lab in [("over", "over-prediction s - r"), ("s", "predicted s (how pricey the text looks)"), ("r", "realized r")]:
        o, mm = ols_fe(p2, y, ["ai", "log_photos", "reserve_any"], fes=["agency", "subcat"], cluster="agency")
        if "ai" in o.index:
            print("    %-40s %+.4f (se %.4f, t %5.2f) n=%d clusters=%d" % (lab, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai", "t"], mm["n"], mm["n_clusters"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
