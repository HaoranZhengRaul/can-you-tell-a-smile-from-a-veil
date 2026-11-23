#!/usr/bin/env python3
"""Blind validation sample for the AI detector.

Draws listings started 2025-02-01..2026-03-31 stratified by detector score (top / middle / bottom
thirds, k each) plus k pre-ChatGPT listings as known-human controls, shuffles them, and writes
  TEXTS: id, title, description (no score, no date, no agency)  -> for blind reading
  KEY:   id, item, start date, stratum, ai_score, ai_flag        -> opened only after labelling
Usage: python3 blind_sample.py PANEL AISCORES TEXTS_OUT KEY_OUT k seed
       python3 blind_sample.py --score KEY_CSV AISCORES FEATURES OUT_JSON LABELS_JSON [LABELS_JSON ...]
Session 10 (PI, 2026-10-30: "poilot done in download"): --score scores every labeller the same way (the volunteer's
human labels beside the two LLM judges'): label counts, the recall of each test on the labeller's AI labels, the
precision of each test's tool-era flags, the pre-ChatGPT controls labelled AI, labels by stratum, and agreement with
Cohen's kappa for each pair of labellers. Each label file was saved before the key was opened; counts only.
"""
import os
import json
import sys

import numpy as np
import pandas as pd


def main(panel, scores, texts_out, key_out, k, seed):
    k = int(k)
    rng = np.random.default_rng(int(seed))
    p = pd.read_parquet(panel)
    s = pd.read_parquet(scores)
    d = p.merge(s, on="item")
    d = d[d.desc.notna() & (d.desc.str.split().str.len() >= 3)]
    post = d[(d.start_local_dt >= "2025-02-01") & (d.start_local_dt < "2026-04-01")].copy()
    post["tercile"] = pd.qcut(post.ai_score.rank(method="first"), 3, labels=["low", "mid", "high"])
    parts = []
    for t in ["low", "mid", "high"]:
        g = post[post.tercile == t]
        parts.append(g.sample(k, random_state=int(rng.integers(0, 1e9))).assign(stratum="post_" + t))
    pre = d[(d.start_local_dt >= "2019-01-01") & (d.start_local_dt < "2022-11-30")]
    parts.append(pre.sample(k, random_state=int(rng.integers(0, 1e9))).assign(stratum="pre_control"))
    S = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(0, 1e9))).reset_index(drop=True)
    S["bid"] = ["B%03d" % i for i in range(len(S))]
    with open(texts_out, "w") as f:
        for _, r in S.iterrows():
            f.write(json.dumps({"id": r.bid, "title": r.title, "description": r.desc}, ensure_ascii=False) + "\n")
    S[["bid", "item", "start_local", "stratum", "ai_score", "ai_flag"]].to_csv(key_out, index=False)
    print("wrote", len(S), "texts")


def kappa(a, b, cats):
    """Cohen's kappa for two label series over the categories cats."""
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score_selftest():
    assert abs(kappa("AAHH", "AAHH", "AUH") - 1) < 1e-12
    assert abs(kappa("AH", "HA", "AUH") - (-1.0)) < 1e-12      # complete disagreement on two balanced labels
    assert abs(kappa("AAHH", "AHAH", "AUH") - 0.0) < 1e-12     # agreement at chance


def score_main(key_csv, scores, feats, out_json, *label_files):
    score_selftest()
    key = pd.read_csv(key_csv)
    s = pd.read_parquet(scores)[["item", "ai_flag"]]
    f = pd.read_parquet(feats)[["item", "f_n_mark"]]
    for t in (key, s, f):
        t["item"] = t["item"].astype(str)
    k = key[["bid", "item", "stratum"]].merge(s, on="item", how="left").merge(f, on="item", how="left")
    assert len(k) == 100 and k.ai_flag.notna().all() and k.f_n_mark.notna().all(), "the key and the scores do not join"
    k["c1"] = k.ai_flag.astype(bool)
    k["lex"] = k.f_n_mark >= 2
    k["union"] = k.c1 | k.lex
    k["tool"] = k.stratum.str.startswith("post")
    L = {}
    for path in label_files:
        d = json.load(open(path))
        lab = d.get("labels", d)
        name = os.path.basename(path).replace(".json", "")
        L[name] = k.bid.map(lab)
        assert L[name].notna().all() and set(L[name]) <= {"A", "U", "H"}, name
    out = {"n_listings": int(len(k)), "strata": {st: int((k.stratum == st).sum()) for st in sorted(k.stratum.unique())},
           "flags": {r: [int((k[r] & k.tool).sum()), int((k[r] & ~k.tool).sum())] for r in ("union", "lex", "c1")}, "labellers": {}}
    for name, lab in L.items():
        A = lab == "A"
        rec = {"n": {c: int((lab == c).sum()) for c in "AUH"},
               "controls_labelled_ai": [int((A & ~k.tool).sum()), int((~k.tool).sum())],
               "by_stratum": {st: {c: int(((lab == c) & (k.stratum == st)).sum()) for c in "AUH"} for st in out["strata"]}}
        for r in ("union", "lex", "c1"):
            rec["recall_" + r] = [int((k[r] & A).sum()), int(A.sum())]
            fl = k[r] & k.tool
            rec["precision_tool_" + r] = [int((fl & A).sum()), int(fl.sum())]
        out["labellers"][name] = rec
        print("%s: labels %s; controls labelled AI %d of %d; recall union %d/%d, marker words %d/%d, classifier %d/%d; "
              "precision of tool-era flags union %d/%d, marker words %d/%d, classifier %d/%d" % (
                  name, rec["n"], rec["controls_labelled_ai"][0], rec["controls_labelled_ai"][1],
                  *rec["recall_union"], *rec["recall_lex"], *rec["recall_c1"],
                  *rec["precision_tool_union"], *rec["precision_tool_lex"], *rec["precision_tool_c1"]))
        print("   by stratum:", rec["by_stratum"])
    names = list(L)
    out["agreement"] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = L[names[i]], L[names[j]]
            rec = {"agree_of_100": int((a.values == b.values).sum()), "kappa3": kappa(a, b, "AUH"),
                   "kappa_ai_vs_not": kappa(a == "A", b == "A", [True, False]),
                   "cross": {x + y: int(((a == x) & (b == y)).sum()) for x in "AUH" for y in "AUH"}}
            out["agreement"][names[i] + "|" + names[j]] = rec
            print("agreement %s vs %s: %d of 100; kappa (three labels) %.3f, (AI vs not) %.3f; cross %s" % (
                names[i], names[j], rec["agree_of_100"], rec["kappa3"], rec["kappa_ai_vs_not"], rec["cross"]))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        score_main(*sys.argv[2:])
        sys.exit(0)
    main(*sys.argv[1:])
