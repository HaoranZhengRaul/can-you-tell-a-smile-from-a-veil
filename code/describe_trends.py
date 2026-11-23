#!/usr/bin/env python3
"""Session 8: monthly series and period contrasts of text features and market outcomes.

Periods by listing START date (when the seller wrote the text):
  P0 2019-01-01..2022-11-29  before ChatGPT
  P1 2022-12-01..2025-01-28  ChatGPT available, no on-platform tool
  P2a 2025-02-01..2025-09-30 tool live, heavy use
  P2b 2025-10-01..2026-03-31 tool live, light use
Sample: ended auctions, US, start >= 2019-01-01, excluding real estate.
Within-agency contrasts: OLS of feature on period dummies + agency FE, agency-clustered SEs.

Usage: python3 describe_trends.py PANEL FEATURES OUTDIR
       python3 describe_trends.py --monthly PANEL FEATURES AISCORES OUT_JSON
Session 10 (PI, 2026-10-30: misspellings "might worth a graph in appendix"; the promotional words "is def going into the
paper"): --monthly gives, by start month and on the paper's own sample (analysis_core.load), the listings, those with a
misspelling (and those whose misspelling count is missing, never counted as zero), and the mean and standard deviation of
promotional words, numbers and words per listing, for all listings and for those the AI_RULE does not flag. Counts and
means only.
       python3 describe_trends.py --markers PANEL FEATURES AISCORES OUT_JSON
       python3 describe_trends.py --examples PANEL FEATURES AISCORES [N_PER] [SEED]
Session 10 (PI, 2026-10-30: "for marker word show the statistical distribution, these are improtant tis the foundation
of our arguments"; "the few type detailed actaul example from data def should be in appendix"): --markers gives, on the
paper's sample and by period, the listings that use each of the 38 marker words (in the full text, as the rule reads
it; checked against the stored count for every listing), the distribution of marker words per listing and of
classifier scores, and the four outcomes of the two tests (counts only). --examples prints, for the log only, a seeded
draw of tool-era listings with at least one bid in each outcome (both tests, classifier only, marker words only,
neither), with the words the rule found and the first 240 characters of the text (e-mails and phones masked).
       python3 describe_trends.py --photos PANEL FEATURES AISCORES PAGES_DIR OUT_JSON
Session 11 (PI, 2026-11-02: "Submit the job"): --photos counts, on the paper's sample, the listings whose photo count is
missing, which analysis_core.load reads as no photos (the photos control of the regressions and the balance table): by
period, by the main rule in the assistant era, among priced listings and vehicles, by status, page layout and capture
year; the photos control's group means as loaded and on observed counts only; and, for the three earliest and three
latest such listings, what their banked page carries where the parser looks for photos (counts of markup, no text).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from regs import ols_fe, fmt  # noqa: E402

CUTS = [("P0", "2019-01-01", "2022-11-30"), ("P1", "2022-12-01", "2025-01-29"),
        ("P2a", "2025-02-01", "2025-10-01"), ("P2b", "2025-10-01", "2026-04-01")]


def load(panel, feats):
    p = pd.read_parquet(panel)
    f = pd.read_parquet(feats).drop(columns=["spec"])
    d = p.merge(f, on="item", how="inner")
    d = d[(d.auction_state == "ended") & (d.country == "US") & (d.start_local_dt >= "2019-01-01")]
    d = d[~d.cat1.fillna("").str.lower().str.contains("real estate")]
    d["period"] = None
    for name, lo, hi in CUTS:
        d.loc[(d.start_local_dt >= lo) & (d.start_local_dt < hi), "period"] = name
    d = d[d.period.notna()].copy()
    d["ai2"] = (d.s_n_mark >= 2).astype(float)
    d["any_bid"] = (d.n_bids >= 1).astype(float)
    d["log_price"] = np.where((d.n_bids >= 1) & (d.final_bid > 0), np.log(d.final_bid.clip(lower=0.01)), np.nan)
    d["log_views"] = np.log1p(d.page_views)
    d["any_q"] = (d.n_q_before_end >= 1).astype(float)
    d["log_w"] = np.log1p(d.s_w)
    d["has_defect"] = (d.s_n_defect >= 1).astype(float)
    d["has_typo"] = (d.s_n_typo >= 1).astype(float)
    d["m"] = d.start_local_dt.dt.to_period("M").astype(str)
    return d


def main(panel, feats, outdir):
    d = load(panel, feats)
    print("sample", len(d), d.period.value_counts().to_dict())
    cols = ["ai2", "s_w", "log_w", "boiler_share", "s_n_num", "s_n_unit", "s_n_pos", "s_n_defect", "has_defect",
            "s_n_puff", "s_n_you", "typo_per100" if "typo_per100" in d else "s_typo_per100", "has_typo",
            "s_caps_share", "s_n_label", "n_photos", "any_bid", "n_bids", "log_price", "log_views", "any_q",
            "n_q_before_end"]
    cols = [c for c in cols if c in d.columns]
    mon = d.groupby("m")[cols].mean()
    mon["n"] = d.groupby("m").size()
    mon.to_csv(outdir + "/monthly_series.csv")
    pd.set_option("display.width", 250)
    print("\nPERIOD MEANS (raw)")
    print(d.groupby("period")[cols].mean().T.round(3).to_string())
    print("\nWITHIN-AGENCY period contrasts vs P0 (agency FE, agency-clustered SE)")
    for c in ["P1", "P2a", "P2b"]:
        d["is_" + c] = (d.period == c).astype(float)
    for y in cols:
        out, meta = ols_fe(d, y, ["is_P1", "is_P2a", "is_P2b"], fes=["agency"], cluster="agency")
        print("--", y, "(P0 mean %.3f)" % d.loc[d.period == "P0", y].mean())
        print(fmt(out, meta))


def month_record(g):
    """One month's counts and means; a missing misspelling count is reported, never read as zero."""
    known = g.s_n_typo.notna()
    rec = {"n": int(len(g)), "n_typo_known": int(known.sum()), "k_typo": int((g.s_n_typo[known] >= 1).sum()),
           "periods": sorted(set(str(p) for p in g.period))}
    for c in ("s_n_puff", "s_n_num", "s_w"):
        x = g[c].dropna()
        rec[c] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                  "sd": float(x.std(ddof=1)) if len(x) > 1 else None}
    return rec


def monthly_selftest():
    t = pd.DataFrame({"s_n_typo": [0, 1, 3, np.nan], "s_n_puff": [0, 2, 1, 0], "s_n_num": [1, 1, 1, np.nan],
                      "s_w": [10, 20, 30, 40], "period": ["P0"] * 4})
    r = month_record(t)
    assert (r["n"], r["n_typo_known"], r["k_typo"]) == (4, 3, 2), r      # the missing count is not a zero
    assert r["s_n_num"]["n"] == 3 and abs(r["s_n_puff"]["mean"] - 0.75) < 1e-12, r


def monthly_main(panel, feats, scores, out_json):
    import json
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analysis_core
    monthly_selftest()
    d = analysis_core.load(panel, feats, scores)
    d["m"] = d.start_local_dt.dt.to_period("M").astype(str)
    out = {"sample": "analysis_core.load (ended, US, started 2019-01-01 to 2026-03-31, no real estate)",
           "ai_rule": d.attrs.get("ai_rule"), "months": {}, "months_noflag": {}, "period": {}}
    for m, g in d.groupby("m"):
        out["months"][m] = month_record(g)
        out["months_noflag"][m] = month_record(g[g.ai == 0])
    for p, g in d.groupby("period"):
        out["period"][p] = month_record(g)
    tot = sum(r["n"] for r in out["months"].values())
    assert tot == len(d), (tot, len(d))
    miss = sum(r["n"] - r["n_typo_known"] for r in out["months"].values())
    print("listings %d in %d months; misspelling count missing for %d" % (len(d), len(out["months"]), miss))
    for p, r in out["period"].items():
        print("  %s: n %d, with a misspelling %.3f, promotional words %.3f, numbers %.3f, words %.1f" % (
            p, r["n"], r["k_typo"] / r["n_typo_known"], r["s_n_puff"]["mean"], r["s_n_num"]["mean"], r["s_w"]["mean"]))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def full_text(d):
    """The text the marker rule reads (textfeat.py main): description, then the seller's updates when present."""
    return d.desc.fillna("") + np.where(d.updates_text.notna(), "\n" + d.updates_text.fillna(""), "")


def outcome(c1, nm):
    return np.where(c1 & (nm >= 2), "both", np.where(c1, "classifier only", np.where(nm >= 2, "marker words only", "neither")))


def markers_main(panel, feats, scores, out_json):
    import json
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analysis_core
    from textfeat import MARK, TOK
    d = analysis_core.load(panel, feats, scores)
    toks = full_text(d).map(lambda t: set(w.lower() for w in TOK.findall(t)))
    found = toks.map(lambda s_: s_ & MARK)
    n_found = found.map(len)
    ok = n_found == d.f_n_mark.fillna(-1)
    print("listings %d; marker count recomputed from the full text equals the stored f_n_mark for %d (%.2f%%)" % (
        len(d), int(ok.sum()), 100.0 * ok.mean()))
    assert ok.mean() > 0.999, "the recomputed marker count does not match the rule's"
    c1 = d.ai_flag.fillna(0).astype(bool)
    d["outcome"] = outcome(c1.values, d.f_n_mark.fillna(0).values)
    out = {"sample": "analysis_core.load", "n": {}, "words": {}, "n_mark_dist": {}, "score_hist": {}, "outcomes": {},
           "tau_share": {}, "bins": [round(0.05 * i, 2) for i in range(21)]}
    for p, g in d.groupby("period"):
        idx = g.index
        out["n"][p] = int(len(g))
        out["n_mark_dist"][p] = {str(k): int(v) for k, v in g.f_n_mark.fillna(0).clip(upper=5).astype(int).value_counts().sort_index().items()}
        h, _ = np.histogram(g.ai_score.dropna().clip(0, 1), bins=out["bins"])
        out["score_hist"][p] = [int(x) for x in h]
        out["tau_share"][p] = float(g.ai_flag.fillna(0).mean())
        out["outcomes"][p] = {str(k): int(v) for k, v in g.outcome.value_counts().items()}
        for w in sorted(MARK):
            out["words"].setdefault(w, {})[p] = int(found.loc[idx].map(lambda s_: w in s_).sum())
    for p in out["n"]:
        print("%s: n %d; outcomes %s; marker words per listing %s; classifier flag %.4f" % (
            p, out["n"][p], out["outcomes"][p], out["n_mark_dist"][p], out["tau_share"][p]))
    top = sorted(out["words"].items(), key=lambda kv: -kv[1].get("P2", 0))
    print("words by tool-era use (listings P0 / P1 / P2):")
    for w, v in top:
        print("  %-14s %4d %4d %4d" % (w, v.get("P0", 0), v.get("P1", 0), v.get("P2", 0)))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def examples_main(panel, feats, scores, n_per=2, seed=30):
    import os
    import re
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analysis_core
    from textfeat import MARK, TOK
    n_per, seed = int(n_per), int(seed)
    d = analysis_core.load(panel, feats, scores)
    d["full"] = full_text(d)
    d["nw"] = d.full.map(lambda t: len(TOK.findall(t)))
    c1 = d.ai_flag.fillna(0).astype(bool)
    d["outcome"] = outcome(c1.values, d.f_n_mark.fillna(0).values)
    red = lambda t: re.sub(r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", "[phone]", re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", t))
    pool = d[(d.period == "P2") & (d.n_bids >= 1) & (d.nw >= 25) & (d.nw <= 150)]
    rng = np.random.default_rng(seed)
    print("tool-era listings with a bid and 25 to 150 words: %d; by outcome %s" % (len(pool), pool.outcome.value_counts().to_dict()))
    for oc in ("both", "classifier only", "marker words only", "neither"):
        g = pool[pool.outcome == oc]
        pick = g.iloc[rng.permutation(len(g))[:n_per]]
        for _, x in pick.iterrows():
            mw = sorted(set(w.lower() for w in TOK.findall(x.full)) & MARK)
            print("== %s | item %s | %s | started %s | $%s after %d bids | classifier %.3f | %d marker words %s | %d words" % (
                oc, x["item"], str(x.title)[:60], str(x.start_local_dt)[:10], "{:,.0f}".format(x.final_bid), int(x.n_bids),
                x.ai_score, len(mw), mw, x.nw))
            print("   " + red(" ".join(x.full.split()))[:240])


def photo_coverage(d):
    """The listings whose photo count is missing. mb_parse.py counts the photos of a page's gallery and leaves the count
    None, with n_photos_status "og_image_only" or "absent", when it finds no gallery, so an observed count is at least 1;
    analysis_core.load turns a missing count into no photos (log_photos = log1p(n_photos.fillna(0)))."""
    miss = d.n_photos.isna()

    def tab(mask):
        n, k = int(mask.sum()), int((miss & mask).sum())
        return {"listings": n, "missing": k, "share_pct": round(100.0 * k / n, 2) if n else None}

    G = {"all": tab(miss | ~miss)}
    for p in ("P0", "P1", "P2"):
        per = d.period == p
        G[p] = tab(per)
        G[p + " priced"] = tab(per & d.log_price.notna())
        G[p + " vehicles"] = tab(per & d.veh)
    for a, nm in ((1.0, "AI written"), (0.0, "human written")):
        g = (d.period == "P2") & (d.ai == a)
        G["P2 " + nm] = tab(g)
        G["P2 " + nm + " priced"] = tab(g & d.log_price.notna())
    lay = d.layout.fillna("(none)").astype(str)
    cy = d.capture_ts.fillna("").astype(str).str[:4].replace("", "(none)")
    obs = d.n_photos.dropna()
    bal = {}
    for a, nm in ((1.0, "AI written"), (0.0, "human written")):
        g = d[(d.period == "P2") & (d.ai == a)]
        o = g.n_photos.dropna()
        bal[nm] = {"listings": int(len(g)),
                   "mean_log_photos_as_loaded": round(float(g.log_photos.mean()), 4) if len(g) else None,
                   "mean_log_photos_observed_only": round(float(np.log1p(o).mean()), 4) if len(o) else None}
    return {"ai_rule": d.attrs.get("ai_rule"), "groups": G,
            "status_missing": {str(k): int(v) for k, v in d.loc[miss, "n_photos_status"].fillna("(none)").value_counts().items()},
            "layout": {k: tab(lay == k) for k in sorted(lay.unique())},
            "capture_year": {k: tab(cy == k) for k in sorted(cy.unique())},
            "observed": {"listings": int(len(obs)), "min": int(obs.min()) if len(obs) else None,
                         "median": float(obs.median()) if len(obs) else None, "max": int(obs.max()) if len(obs) else None},
            "balance_P2": bal}


def page_markup(path):
    """What a banked page carries where mb_parse.py looks for photos: counts and yes/no only, never text."""
    import gzip
    import os
    import re
    if not os.path.exists(path):
        return {"page": "not banked"}
    t = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    return {"page_chars": len(t), "img_tags": len(re.findall(r"<img\b", t, re.I)), "es_slides": "es-slides" in t,
            "gallery_div": bool(re.search(r'class="[^"]*\bgallery\b', t)), "alt_listing_image": 'alt="Listing Image"' in t,
            "og_image": "og:image" in t}


def photos_main(panel, feats, scores, pages, out_json):
    import json
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analysis_core
    d = analysis_core.load(panel, feats, scores)
    out = photo_coverage(d)
    out["sample"] = "analysis_core.load"
    m = d[d.n_photos.isna()].sort_values(["start_local_dt", "item"])
    peek = pd.concat([m.head(3), m.tail(3)]).drop_duplicates("item")
    out["pages_read"] = [dict(item=str(x["item"]), period=x["period"], start=str(x["start_local_dt"])[:10],
                              capture=str(x["capture_ts"])[:8], layout=x["layout"], status=x["n_photos_status"],
                              **page_markup(os.path.join(pages, str(x["item"])[:3], str(x["item"]) + ".html.gz")))
                         for _, x in peek.iterrows()]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=1)
    print("AI rule:", out["ai_rule"])
    for k, v in out["groups"].items():
        print("%-26s %6d listings; photo count missing in %5d (%s%%)" % (k, v["listings"], v["missing"], v["share_pct"]))
    print("status of the missing:", out["status_missing"])
    print("by page layout (missing, listings):", {k: (v["missing"], v["listings"]) for k, v in out["layout"].items()})
    print("by capture year (missing, listings):", {k: (v["missing"], v["listings"]) for k, v in out["capture_year"].items()})
    print("observed counts:", out["observed"])
    print("photos control in the assistant era:", out["balance_P2"])
    for r in out["pages_read"]:
        print("page:", r)
    print("wrote", out_json)


if __name__ == "__main__":
    if sys.argv[1] == "--photos":
        photos_main(*sys.argv[2:7])
    elif sys.argv[1] == "--markers":
        markers_main(*sys.argv[2:6])
    elif sys.argv[1] == "--examples":
        examples_main(*sys.argv[2:7])
    elif sys.argv[1] == "--monthly":
        monthly_main(*sys.argv[2:6])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
