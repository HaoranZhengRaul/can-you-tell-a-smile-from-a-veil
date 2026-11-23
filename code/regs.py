#!/usr/bin/env python3
"""Session 8 regression utilities (numpy only; Mercury has no statsmodels).

ols_fe(df, y, xs, fes, cluster): OLS of y on xs with any number of absorbed fixed effects
(alternating projections), iterative removal of singleton fixed-effect groups, and cluster-robust
(CR1) standard errors: V = (G/(G-1)) * ((N-1)/(N-K)) * (X'X)^-1 (sum_g X_g'u_g u_g'X_g) (X'X)^-1,
with K = number of slope coefficients + absorbed fixed-effect parameters (approximated by the
number of FE levels minus the number of FE dimensions + 1).

Self-test: python3 regs.py selftest  (compares with an explicit dummy-variable regression).
"""
import sys

import numpy as np
import pandas as pd


def _demean(M, groups, tol=1e-10, maxit=10000):
    """Project the columns of M off the span of the indicator matrices in `groups`."""
    M = M.astype(float).copy()
    if not groups:
        return M - M.mean(axis=0)
    for it in range(maxit):
        maxchg = 0.0
        for g, ng in groups:
            s = np.zeros((ng, M.shape[1]))
            np.add.at(s, g, M)
            c = np.bincount(g, minlength=ng).astype(float)
            mu = s / c[:, None]
            M -= mu[g]
            maxchg = max(maxchg, float(np.abs(mu).max()) if mu.size else 0.0)
        if maxchg < tol:
            break
    return M


def _codes(series):
    c, u = pd.factorize(series, sort=False)
    return c, len(u)


def ols_fe(df, y, xs, fes=(), cluster=None, drop_singletons=True):
    cols = []
    for c in [y] + list(xs) + list(fes) + ([cluster] if cluster else []):
        if c not in cols:
            cols.append(c)
    d = df[cols].dropna().copy()
    n0 = len(d)
    if drop_singletons and fes:
        while True:
            keep = np.ones(len(d), dtype=bool)
            for f in fes:
                vc = d[f].map(d[f].value_counts())
                keep &= (vc > 1).values
            if keep.all():
                break
            d = d[keep]
    groups = [_codes(d[f]) for f in fes]
    Y = d[[y]].values
    X = d[list(xs)].values.astype(float)
    if fes:
        Z = _demean(np.hstack([Y, X]), groups)
    else:
        Z = np.hstack([Y, X])
        Z = Z - Z.mean(axis=0)
    yt, Xt = Z[:, 0], Z[:, 1:]
    # drop slope columns that are (numerically) absorbed by the fixed effects
    ok = np.where(np.abs(Xt).sum(axis=0) > 1e-8 * max(1.0, np.abs(X).sum()))[0]
    names = [xs[i] for i in ok]
    Xt = Xt[:, ok]
    XtX = Xt.T @ Xt
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (Xt.T @ yt)
    u = yt - Xt @ b
    N = len(d)
    k_fe = sum(ng for _, ng in groups) - (len(groups) - 1 if groups else 0)
    K = len(names) + k_fe if groups else len(names) + 1
    if cluster:
        cl, G = _codes(d[cluster])
        S = np.zeros((G, len(names)))
        np.add.at(S, cl, Xt * u[:, None])
        meat = S.T @ S
        adj = (G / (G - 1.0)) * ((N - 1.0) / max(1.0, N - K))
        V = adj * XtXi @ meat @ XtXi
    else:
        G = None
        s2 = (u @ u) / max(1.0, N - K)
        V = s2 * XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tss = ((yt - yt.mean()) ** 2).sum()
    out = pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=names)
    meta = {"n_input": n0, "n": N, "n_clusters": G, "k_fe": k_fe, "within_r2": 1.0 - (u @ u) / tss if tss > 0 else np.nan,
            "y_mean": float(d[y].mean()), "dropped_singletons": n0 - N}
    return out, meta


def fmt(out, meta, keep=None):
    rows = out if keep is None else out.loc[[k for k in keep if k in out.index]]
    s = []
    for k, r in rows.iterrows():
        s.append("  %-28s %9.4f  (se %.4f, t %6.2f)" % (k, r.coef, r.se, r.t))
    s.append("  n=%d clusters=%s within_R2=%.3f y_mean=%.3f singletons_dropped=%d" % (
        meta["n"], meta["n_clusters"], meta["within_r2"], meta["y_mean"], meta["dropped_singletons"]))
    return "\n".join(s)


def selftest():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 30, n)
    c = rng.integers(0, 7, n)
    x1 = rng.normal(size=n) + 0.3 * a / 30
    x2 = rng.normal(size=n)
    y = 1.5 * x1 - 0.7 * x2 + 0.2 * a - 0.5 * c + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "a": a, "c": c, "g": a})
    out, meta = ols_fe(df, "y", ["x1", "x2"], fes=["a", "c"], cluster="g")
    out_same, _ = ols_fe(df, "y", ["x1", "x2"], fes=["a", "c"], cluster="a")  # FE == cluster column
    ok0 = np.allclose(out_same.coef.values, out.coef.values)
    D = np.column_stack([x1, x2, pd.get_dummies(a, drop_first=False).values, pd.get_dummies(c, drop_first=True).values]).astype(float)
    bb, *_ = np.linalg.lstsq(D, y, rcond=None)
    ok1 = np.allclose(out.coef.values, bb[:2], atol=1e-7)
    # explicit CR1 sandwich on the dummy regression
    u = y - D @ bb
    XtXi = np.linalg.pinv(D.T @ D)
    G = 30
    S = np.zeros((G, D.shape[1]))
    np.add.at(S, a, D * u[:, None])
    K = np.linalg.matrix_rank(D)
    V = (G / (G - 1)) * ((n - 1) / (n - K)) * XtXi @ (S.T @ S) @ XtXi
    ok2 = np.allclose(out.se.values, np.sqrt(np.diag(V))[:2], rtol=1e-6)
    # a mutation must fail: wrong FE set gives different coefficients
    out2, _ = ols_fe(df, "y", ["x1", "x2"], fes=["c"], cluster="g")
    ok3 = not np.allclose(out2.coef.values, bb[:2], atol=1e-3)
    print("coef match dummy OLS:", ok1, "| CR1 se match:", ok2, "| mutation detected:", ok3, "| FE==cluster ok:", ok0)
    print(fmt(out, meta))
    return ok0 and ok1 and ok2 and ok3


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(0 if selftest() else 1)
