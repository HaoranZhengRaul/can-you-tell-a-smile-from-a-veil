#!/usr/bin/env python3
"""Session 9: summarize an informativeness bootstrap (E_spreads_draws.csv from code/informativeness.py) as JSON.

Spreads by text index and period (mean and 2.5/97.5 percentiles over draws) and the ratio of each later
period to the pre-ChatGPT period, computed draw by draw (so the reported ratio is the mean of per-draw ratios,
not the ratio of mean spreads).
Usage: python3 inform_summary.py E_SPREADS_DRAWS.csv OUT.json
"""
import json
import sys

import numpy as np
import pandas as pd


def main(src, out):
    E = pd.read_csv(src)
    S = {}
    for (kind, per), g in E.groupby(["kind", "period"]):
        S["%s|%s" % (kind, per)] = {"mean": float(g.D.mean()), "lo": float(g.D.quantile(0.025)), "hi": float(g.D.quantile(0.975)),
                                    "oos_r2": float(g.oos_r2.mean()), "draws": int(len(g))}
    Rt = {}
    for kind, g in E.groupby("kind"):
        w = g.pivot_table(index="draw", columns="period", values="D")
        for num in ("P1", "P2"):
            q = (w[num] / w["P0"]).replace([np.inf, -np.inf], np.nan).dropna()
            Rt["%s|%s/P0" % (kind, num)] = {"mean": float(q.mean()), "median": float(q.median()), "lo": float(q.quantile(0.025)),
                                            "hi": float(q.quantile(0.975)), "draws": int(len(q))}
    json.dump({"source": src, "spreads": S, "ratios": Rt}, open(out, "w"), indent=1)
    print("wrote", out, "| draws per cell:", sorted(set(v["draws"] for v in S.values())))


if __name__ == "__main__":
    main(*sys.argv[1:])
