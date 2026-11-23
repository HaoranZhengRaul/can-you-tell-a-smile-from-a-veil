#!/usr/bin/env python3
"""Stratified sample of pre-ChatGPT listings (start < 2022-11-30) for synthetic AI rewrites.
Strata: cat1 (proportional, min 3). Keeps descriptions of 3..400 words. Fixed seed."""
import sys, json, re
import pandas as pd
df = pd.read_parquet(sys.argv[1]); n = int(sys.argv[2]); seed = int(sys.argv[3])
d = df[(df.start_local_dt >= "2019-01-01") & (df.start_local_dt < "2022-11-30") & df.desc.notna()].copy()
d["w"] = d.desc.str.split().str.len()
d = d[(d.w >= 3) & (d.w <= 400)]
share = d.cat1.value_counts(normalize=True)
parts = []
for c, s in share.items():
    k = max(3, int(round(s * n)))
    g = d[d.cat1 == c]
    parts.append(g.sample(min(k, len(g)), random_state=seed))
s = pd.concat(parts).sample(frac=1, random_state=seed).head(n)
for _, r in s.iterrows():
    print(json.dumps({"item": r["item"], "title": r.title, "category": " > ".join([x for x in [r.cat1, r.cat2] if x]),
                      "original": r.desc}, ensure_ascii=False))
