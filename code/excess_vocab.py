#!/usr/bin/env python3
"""Excess-vocabulary scan (Kobak et al. 2025 logic) on Municibid descriptions.

For each word w and listing start-year t: document frequency p_t(w) = share of descriptions that
contain w at least once. Baseline b(w) = mean of p_2019..p_2022 (all text written before ChatGPT;
listings whose start date is before 2022-11-30). Excess gap d_t(w) = p_t(w) - b(w); excess ratio
r_t(w) = p_t(w) / b(w). Words are lowercased alphabetic tokens (apostrophes kept).

Usage: python3 excess_vocab.py PANEL.parquet [YEAR ...]
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")


def docs_tokens(texts):
    return [set(TOK.findall(t.lower())) for t in texts]


def main(panel, years):
    df = pd.read_parquet(panel)
    df = df[df.desc.notna() & df.start_local_dt.notna()].copy()
    df["full"] = df.desc.fillna("") + "\n" + df.updates_text.fillna("")
    df["sy"] = df.start_local_dt.dt.year
    pre = df[(df.start_local_dt < "2022-11-30") & (df.sy >= 2019)]
    base_years = [2019, 2020, 2021, 2022]
    freq = {}
    n = {}
    for y in base_years + years:
        d = pre[pre.sy == y] if y in base_years else df[df.sy == y]
        toks = docs_tokens(d.full.tolist())
        c = Counter()
        for s in toks:
            c.update(s)
        freq[y] = c
        n[y] = len(toks)
    print("docs per year:", n)
    vocab = set()
    for y in freq:
        vocab |= set(w for w, k in freq[y].items() if k >= 5)
    base = {w: np.mean([freq[y][w] / n[y] for y in base_years]) for w in vocab}
    base_sd = {w: np.std([freq[y][w] / n[y] for y in base_years]) for w in vocab}
    for y in years:
        rows = []
        for w in vocab:
            p = freq[y][w] / n[y]
            b = base[w]
            rows.append((w, p, b, p - b, (p / b) if b > 0 else np.inf, base_sd[w], freq[y][w]))
        t = pd.DataFrame(rows, columns=["word", "p_t", "base", "gap", "ratio", "base_sd", "count_t"])
        t = t[t.count_t >= 10]
        print("\n==== %d: top 60 by excess gap (share of listings containing the word)" % y)
        for _, r in t.sort_values("gap", ascending=False).head(60).iterrows():
            print("  %-16s %6.2f%% vs base %6.2f%% (sd %.2f)  gap %+6.2fpp  ratio %6.1f  n=%d" % (
                r.word, 100 * r.p_t, 100 * r.base, 100 * r.base_sd, 100 * r.gap, r.ratio, r.count_t))
        print("\n==== %d: top 50 by ratio (p_t >= 1%%)" % y)
        for _, r in t[t.p_t >= 0.01].sort_values("ratio", ascending=False).head(50).iterrows():
            print("  %-16s %6.2f%% vs base %6.3f%%  ratio %7.1f  n=%d" % (r.word, 100 * r.p_t, 100 * r.base, r.ratio, r.count_t))
        print("\n==== %d: words that FELL most (gap)" % y)
        for _, r in t.sort_values("gap").head(25).iterrows():
            print("  %-16s %6.2f%% vs base %6.2f%%  gap %+6.2fpp" % (r.word, 100 * r.p_t, 100 * r.base, 100 * r.gap))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [2023, 2024, 2025, 2026])
