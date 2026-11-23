#!/usr/bin/env python3
"""Compare per-listing AI rules: recall/precision on the blind-read sample (main-loop labels) and
false-positive rate on guaranteed-human text (listings started 2019-01-01..2022-11-29) plus the
out-of-time no-tool period (2022-12-01..2025-01-28)."""
import json, sys
import numpy as np, pandas as pd
P = "/project/dmv_project/projects/ai_pricing_kbb/results/"
s = pd.read_parquet(P + "detector_s8/ai_scores.parquet")
f = pd.read_parquet(P + "panel_s8/features.parquet")[["item", "s_n_mark", "f_n_mark", "s_n_label", "s_n_puff"]]
p = pd.read_parquet(P + "panel_s8/panel.parquet")[["item", "start_local_dt", "desc"]]
d = p.merge(s, on="item").merge(f, on="item")
d = d[d.desc.notna()]
pre = d[(d.start_local_dt >= "2019-01-01") & (d.start_local_dt < "2022-11-30")]
p1 = d[(d.start_local_dt >= "2022-12-01") & (d.start_local_dt < "2025-01-29")]
lab = json.load(open(P + "blind_s8/my_labels.json"))["labels"]
key = pd.read_csv(P + "blind_s8/KEY_do_not_open.csv")
key["me"] = key.bid.map(lab)
key = key.merge(d[["item", "ai_score", "f_n_mark"]].rename(columns={"ai_score": "sc"}), left_on=key["item"].astype(str), right_on="item", how="left")
taus = {q: float(np.quantile(pre.ai_score, q)) for q in (0.95, 0.97, 0.98, 0.99)}
rules = {}
for q, t in taus.items():
    rules["C1 >= pre q%.2f" % q] = lambda x, t=t: x.ai_score >= t
rules["markers >= 2"] = lambda x: x.f_n_mark >= 2
rules["markers >= 3"] = lambda x: x.f_n_mark >= 3
for q in (0.97, 0.99):
    t = taus[q]
    rules["C1 >= pre q%.2f OR markers >= 2" % q] = lambda x, t=t: (x.ai_score >= t) | (x.f_n_mark >= 2)
print("%-36s %8s %8s | %6s %6s %6s %6s" % ("rule", "FPR_pre", "FPR_P1", "flagA", "flagU", "flagH", "recallA"))
k = key.rename(columns={"sc": "ai_score_x"})
k["ai_score"] = k["ai_score_x"]
for name, fn in rules.items():
    fpr0 = float(fn(pre).mean()); fpr1 = float(fn(p1).mean())
    fl = fn(k)
    a = int((fl & (k.me == "A")).sum()); u = int((fl & (k.me == "U")).sum()); h = int((fl & (k.me == "H")).sum())
    print("%-36s %8.4f %8.4f | %6d %6d %6d %6.2f" % (name, fpr0, fpr1, a, u, h, a / max(1, (k.me == "A").sum())))
print("blind sample: A=%d U=%d H=%d (25 per stratum; post strata = tool-era terciles of score)" % ((k.me == "A").sum(), (k.me == "U").sum(), (k.me == "H").sum()))
