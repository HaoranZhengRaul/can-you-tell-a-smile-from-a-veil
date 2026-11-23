#!/usr/bin/env python3
"""Session 8: recompute every number shown in ai_cheap_talk.html into one small JSON of aggregates
(no listing text, no identities). Run on Mercury; copy the JSON to the repo root as report_data.json.

Usage: python3 report_data.py PANEL FEATURES AISCORES OUT.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
import analysis_core  # noqa: E402
from regs import ols_fe  # noqa: E402


def coef(d, y, xs, fes, k="ai"):
    o, m = ols_fe(d, y, xs, fes=fes, cluster="agency")
    if k not in o.index:
        return None
    return {"b": float(o.loc[k, "coef"]), "se": float(o.loc[k, "se"]), "n": int(m["n"]), "clusters": int(m["n_clusters"]),
            "mean": float(m["y_mean"])}


def main(panel, feats, scores, out):
    R = {"generated_by": "code/report_data.py", "rules": {}}
    for rule in ["union", "lex", "c1"]:
        os.environ["AI_RULE"] = rule
        d = analysis_core.load(panel, feats, scores)
        d["sm"] = d.start_local_dt.dt.to_period("M").astype(str)
        mon = d[d.start_local_dt >= "2022-06-01"].groupby("sm").agg(n=("ai", "size"), ai=("ai", "mean"))
        p2 = d[d.period == "P2"].copy()
        rr = {"share_by_period": d.groupby("period").ai.mean().round(4).to_dict(),
              "n_by_period": d.period.value_counts().to_dict(),
              "monthly": {k: {"n": int(v.n), "ai": round(float(v.ai), 4)} for k, v in mon.iterrows()},
              "p2_ai": int(p2.ai.sum()), "p2_n": int(len(p2)), "p2_agencies_using": int(p2[p2.ai == 1].agency.nunique()),
              "p2_agencies": int(p2.agency.nunique())}
        content = {}
        for y in ["log_w", "s_w", "facts", "s_n_num", "s_n_unit", "s_n_defect_strict", "has_defect_strict", "s_n_quality",
                  "has_quality", "defect_and_praise", "s_n_puff", "s_n_you", "s_n_label", "has_typo"]:
            c = coef(p2, y, ["ai"], ["agency", "subcat_m"])
            if c:
                c["human_mean"] = float(p2.loc[p2.ai == 0, y].mean())
                content[y] = c
        rr["content"] = content
        market = {}
        for y in ["sold", "log_price", "log_bids", "log_views", "any_q", "n_q_before_end"]:
            xs = ["ai", "log_photos", "reserve_any"] + (["log_caplag"] if y == "log_views" else [])
            market[y] = {"subcat_m": coef(p2, y, xs, ["agency", "subcat_m"]), "subcat+month": coef(p2, y, xs, ["agency", "subcat", "endm"])}
        rr["market"] = market
        rr["median_price_p2"] = float(np.exp(p2.log_price.median()))
        R["rules"][rule] = rr
    # period-level descriptive means (union rule irrelevant here)
    os.environ["AI_RULE"] = "union"
    d = analysis_core.load(panel, feats, scores)
    R["period_means"] = d.groupby("period")[["s_w", "s_n_num", "s_n_defect_strict", "s_n_puff", "s_n_label", "has_typo", "sold",
                                             "n_q_before_end"]].mean().round(4).to_dict(orient="index")
    R["sample"] = {"listings_analysis": int(len(d)), "agencies": int(d.agency.nunique()),
                   "priced": int(d.log_price.notna().sum()), "start_min": str(d.start_local_dt.min())[:10],
                   "start_max": str(d.start_local_dt.max())[:10]}
    json.dump(R, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
