#!/usr/bin/env python3
"""Session 8: tighten the AI-vs-human price comparison with vehicle hedonics.

Vehicles = listings whose agency table reports a model year and a make (ended, >=1 bid, US, no
real estate). Outcome: log closing high bid. AI = validated indicator (AI_RULE; union by default).

  V1  pooled 2019-2026, agency FE + make x model FE + body-style FE + subcategory x end-quarter FE,
      hedonics (age, age^2, log miles + missing flag, starts / drivable / title / inspected each as
      yes / no / not reported), photos, reserve. The AI coefficient compares an agency's AI-written
      vehicles with all its other vehicles, adjusted for time and for what the vehicle is.
  V2  the same but the comparison group is limited to listings started from 2022-12-01 (ChatGPT era).
  V3  the same, tool era only (the earlier, less precise design).
  V4  out-of-time hedonic residual: a ridge hedonic model fit on pre-tool vehicles predicts each
      tool-era vehicle's log price; residual = log price - prediction; AI vs human residual within
      agency (tool era).
  G1  all goods (non-vehicles): pooled 2019-2026, agency FE + subcategory x end-quarter FE.
Reported: coefficient, 95% interval in percent and in dollars at the tool-era median vehicle price,
minimum detectable effect (2.8 x SE: 80% power, 5% two-sided) and the hedonic fit (within R^2).
Session 10 switches (environment, like AI_RULE and AI_SCOPE; PI, 2026-10-29, "run the vehicle designs without the 55
parts listings; run the vehicle design with the Texas same-type price"):
  VEH_NO_PARTS=1   every vehicle design (V1 to V4c, V1-balance) without "Automotive > Parts" listings: a tailgate or a
                   seat with a year and a make in its form table is not a vehicle (Session 9 WORKLOG section 24: 55 of
                   the 4,331 in the design)
  VEH_DMV=PARQUET  three more designs on the vehicles with 10 or more Texas DMV sales of the same make, model and model
                   year within 3 months of the start month (code/dmv_benchmark.py, job 874347; n_ex3, med_ex3):
                   T0 V1 on that subset (the subset's own estimate), T1 T0 + log Texas median as a control, T2 the
                   closing bid relative to the Texas median (log price - log median) with V1's hedonics and fixed
                   effects except make x model and body (the benchmark prices the type)
                   With VEH_DMV the run also saves veh_tx_bins (PI, 2026-10-30: "the car part ... I want at least my PI
                   to take a look at that figure"): the matched vehicles in equal-count bins of the log Texas median, per
                   bin the count and the mean log Texas median and log closing bid, for all of them and, in the tool era,
                   for AI written and other vehicles; the median ratio and its quartiles by group (aggregates only)
Session 10 (PI, 2026-10-30, "lets exclude the wreck?"):
  VEH_NO_WRECK=title  every vehicle design without the vehicles whose title names a wreck (WRECK_RX: wrecked, totaled,
                      salvage, fire or flood damage, rolled over, crashed, accident or collision damage, "in an
                      accident"); the title is the clerk's own field, which the description button does not rewrite
  VEH_NO_WRECK=text   the same rule on the title, the description and the seller's updates; an AI rewrite may word a
                      wreck differently, so the log gives each group's share of wrecks under both rules
Usage: python3 price_hedonics.py PANEL FEATURES AISCORES [OUT_JSON]
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from analysis_core import load, hedonic_cols  # noqa: E402
from regs import ols_fe  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


OUT = {}


def report(name, o, m, med, k="ai", key=None, n_ai=None):
    if k not in o.index:
        print("  %-62s (absorbed) n=%d" % (name, m["n"]))
        return
    if key:
        OUT[key] = {"b": float(o.loc[k, "coef"]), "se": float(o.loc[k, "se"]), "n": int(m["n"]), "clusters": int(m["n_clusters"]),
                    "within_r2": float(m["within_r2"]), "median_price": med, "label": name}
        if n_ai is not None:
            OUT[key]["n_ai"] = int(n_ai)
    b, se = o.loc[k, "coef"], o.loc[k, "se"]
    lo, hi = b - 1.96 * se, b + 1.96 * se
    print("  %-62s %+.3f (se %.3f) = %+.1f%% [%+.1f%%, %+.1f%%] = $%+.0f [$%+.0f, $%+.0f] | MDE %.3f (%.0f%%) | n=%d clusters=%d within-R2=%.3f" % (
        name, b, se, 100 * (np.exp(b) - 1), 100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1),
        med * (np.exp(b) - 1), med * (np.exp(lo) - 1), med * (np.exp(hi) - 1), 2.8 * se, 100 * (np.exp(2.8 * se) - 1),
        m["n"], m["n_clusters"], m["within_r2"]))


def ref_level(frame, key, y, price=False):
    """Figure 3's reference bar (Session 9, PI: "non-AI written versus AI written"): the outcome's level among tool-era
    listings the rule does not flag, in the rows the design uses (prices: the median closing price; otherwise the mean
    of y, so log1p outcomes keep their log scale). Stored beside the estimate with its n and the tool-era AI n, which
    must add up to the tool-era rows."""
    p2 = frame[frame.period == "P2"]
    r = p2[p2.ai == 0]
    assert len(r) + int((p2.ai == 1).sum()) == len(p2), key
    OUT[key]["ref"] = {"group": "tool-era listings not flagged", "n": int(len(r)), "n_ai_tool_era": int((p2.ai == 1).sum()),
                       ("median_price" if price else "mean"): float(np.exp(r[y].median()) if price else r[y].mean())}
    print("       reference for %-24s tool era, not flagged: n=%d (AI n=%d) %s=%.4f" % (
        key, len(r), int((p2.ai == 1).sum()), "median price" if price else "mean", OUT[key]["ref"]["median_price" if price else "mean"]))


# a wreck, not a damaged part: "wrecker" (a tow truck), "no accidents", "collision warning" and "rollover protection" do
# not match; "totaed" is the spelling of a listing read in the Texas comparison (job 877521). "salvage" alone left the
# rule after its first run (jobs 884256 to 884259): read in context it was mostly an office ("Contact Salvage Control at
# (312) ...", Chicago), boilerplate ("Salvage Removal Order") or "parts salvage"; a salvage title or certificate stays
WRECK_RX = re.compile(r"(?i)\b(?:wreck(?:s|ed|age)?|totall?ed|totaed|total loss|salvage (?:title|certificate)|fire[- ]damaged?|"
                      r"flood[- ]damaged?|flooded|rolled over|crashed|crash damage|(?:accident|collision)[- ]damaged?|"
                      r"(?:in|from) an? (?:accident|collision|crash)|damaged in an? (?:accident|collision|crash))\b")


def wreck_selftest(rx=None):
    """Fails if the rule misses a wreck or flags a non-wreck; returns the failures (empty when the rule passes)."""
    rx = rx or WRECK_RX
    pos = ["2012 DODGE CHARGER HEAVILY WRECKED", "2012 DODGE CHARGER TOTAED, NO ENGINE", "Vehicle was totaled",
           "totalled in 2019", "SALVAGE TITLE", "fire damaged bus", "Flood-damaged sedan", "it rolled over",
           "was in an accident", "damaged in a collision", "crashed into a pole", "front end wreck",
           "accident damage to the rear", "total loss per insurer", "Sold with Salvage Certificate",
           "Winning bidder will get salvage title and KEYS"]
    neg = ["Ford F-450 wrecker", "tow truck / wrecker body", "no accidents", "accident free", "accident-free history",
           "Forward Collision Warning", "collision avoidance system", "rollover protection structure", "burns oil",
           "Fire Department pumper", "runs and drives", "minor dents", "total of 3 keys", "totally rebuilt engine",
           "Contact Salvage Control at (000) 000-0000", "Salvage Removal Order. All items must be removed",
           "the frame is rotted, making it suitable for parts salvage", "title history may be marked flood, salvage, TMU"]
    return [("missed", t) for t in pos if not rx.search(t)] + [("false hit", t) for t in neg if rx.search(t)]


def tx_ratio_by(v, path, col):
    """Median closing bid against the Texas median, by col (descriptive; before any exclusion)."""
    B = pd.read_parquet(path, columns=["item", "n_ex3", "med_ex3"])
    B["item"] = B["item"].astype(str)
    t = v[["item", "log_price", "period", "ai", col]].copy()
    t["item"] = t["item"].astype(str)
    t = t.merge(B, on="item", how="left")
    t = t[(t.n_ex3 >= 10) & (t.med_ex3 > 0)]
    r = np.exp(t.log_price - np.log(t.med_ex3))
    for k, g in r.groupby(t[col]):
        print("  Texas-benchmark vehicles with %s=%s: n %d (tool era %d), median closing bid / Texas median %.3f" % (
            col, bool(k), len(g), int((t.loc[g.index, "period"] == "P2").sum()), float(g.median())))


def tx_bins(t, nb_all=20, nb_grp=4, min_n=8):
    """The Texas figure's data: equal-count bins of the log Texas median (ties broken by row order), per bin the count
    and the mean log Texas median and log closing bid; a group with fewer than nb x min_n vehicles is not binned (None),
    so no bin describes fewer than min_n listings. Also the OLS slope of log closing bid on log Texas median and, by
    group, the median ratio of closing bid to Texas median with its quartiles."""
    def bins(x, nb):
        if len(x) < nb * min_n:
            return None
        q = pd.qcut(x.log_tx.rank(method="first"), nb, labels=False)
        rows = [{"n": int(len(b)), "log_tx": float(b.log_tx.mean()), "log_price": float(b.log_price.mean())}
                for _, b in x.groupby(q)]
        assert sum(r["n"] for r in rows) == len(x) and min(r["n"] for r in rows) >= min_n
        return rows

    def ratio(x):
        return {"n": int(len(x)), "median": float(np.exp(x.rel.median())), "q25": float(np.exp(x.rel.quantile(0.25))),
                "q75": float(np.exp(x.rel.quantile(0.75)))}
    p2 = t[t.period == "P2"]
    groups = {"all": t, "pre_tool": t[t.period != "P2"], "tool_ai": p2[p2.ai == 1], "tool_other": p2[p2.ai == 0]}
    return {"bins": {"all": bins(t, nb_all), "tool_ai": bins(groups["tool_ai"], nb_grp),
                     "tool_other": bins(groups["tool_other"], nb_grp)},
            "ratio": {k: ratio(g) for k, g in groups.items()}, "slope_log": float(np.polyfit(t.log_tx, t.log_price, 1)[0]),
            "nb": [nb_all, nb_grp], "min_n": min_n}


def tx_bins_selftest():
    """Fails unless the bins recover a known world (closing bid 75% of the Texas median: slope 1, ratio 0.75) and show
    no slope in a world where the bid ignores the Texas median (so a slope cannot come from the binning itself, as it
    would if the bins were cut on the closing bid); a group too small to bin must give None. Returns the failures."""
    rng = np.random.default_rng(7)
    n = 600
    lt = rng.normal(np.log(8000), 0.8, n)
    base = pd.DataFrame({"log_tx": lt, "period": np.where(rng.random(n) < 0.3, "P2", "P0"),
                         "ai": (rng.random(n) < 0.4).astype(float)})
    bad = []
    for name, lp in (("good", lt + np.log(0.75) + rng.normal(0, 0.3, n)), ("flat", np.log(6000) + rng.normal(0, 0.3, n))):
        w = base.assign(log_price=lp)
        w["rel"] = w.log_price - w.log_tx
        o = tx_bins(w, nb_all=10, nb_grp=3, min_n=5)
        b = o["bins"]["all"]
        s = (b[-1]["log_price"] - b[0]["log_price"]) / (b[-1]["log_tx"] - b[0]["log_tx"])
        if name == "good" and not (0.85 < s < 1.15 and 0.70 < o["ratio"]["all"]["median"] < 0.80 and 0.9 < o["slope_log"] < 1.1):
            bad.append(("good world not recovered", s, o["ratio"]["all"]["median"], o["slope_log"]))
        if name == "flat" and not (abs(s) < 0.15 and abs(o["slope_log"]) < 0.1):
            bad.append(("flat world shows a slope", s, o["slope_log"]))
        if o["bins"]["tool_ai"] is None or sum(r["n"] for r in o["bins"]["tool_ai"]) != int(((w.period == "P2") & (w.ai == 1)).sum()):
            bad.append(("tool-era AI bins do not add up", name))
    tiny = base.head(10).assign(log_price=lt[:10])
    tiny["rel"] = 0.0
    if tx_bins(tiny, nb_all=4, nb_grp=4, min_n=5)["bins"]["all"] is not None:
        bad.append(("a group smaller than nb x min_n was binned",))
    return bad


def dmv_designs(v, xs, fes, hc, PRE, med, path):
    """T0, T1, T2 of the docstring on the vehicles with a Texas median from 10 or more same-type sales within 3 months."""
    bad = tx_bins_selftest()
    assert not bad, ("the Texas bins fail their own test", bad)
    B = pd.read_parquet(path, columns=["item", "n_ex3", "med_ex3"])
    B["item"] = B["item"].astype(str)
    assert B["item"].is_unique, "benchmark: one row per listing expected"
    t = v.copy()
    t["item"] = t["item"].astype(str)
    n0 = len(t)
    t = t.merge(B, on="item", how="left")
    assert len(t) == n0, "the merge changed the row count"
    keep = (t.n_ex3 >= 10) & (t.med_ex3 > 0)
    print("VEH_DMV: %d of %d priced vehicles have 10+ Texas sales of their make, model and year within 3 months "
          "(tool era %d, tool-era AI %d)" % (int(keep.sum()), n0, int((keep & (t.period == "P2")).sum()),
                                             int((keep & (t.period == "P2") & (t.ai == 1)).sum())))
    t = t[keep].copy()
    t["log_tx"] = np.log(t.med_ex3)
    t["rel"] = t.log_price - t.log_tx
    r = np.corrcoef(t.log_price, t.log_tx)[0, 1]
    print("  correlation of log closing bid and log Texas median: %.3f; median ratio %.3f" % (r, float(np.exp(t.rel.median()))))
    show = ["title", "subcat", "f_year", "f_make", "f_model", "period", "ai"]
    for label, part in (("lowest", t.nsmallest(5, "rel")), ("highest", t.nlargest(5, "rel"))):
        print("  %s closing bid against the Texas median:" % label)
        for _, x in part.iterrows():
            print("    ratio %.3f  $%.0f vs $%.0f  %s | %s | %s %s %s | %s ai=%d" % (
                np.exp(x.rel), np.exp(x.log_price), x.med_ex3, str(x.title)[:45], x.subcat, x.f_year, x.f_make, x.f_model,
                x.period, int(x.ai)))
    o, m = ols_fe(t, "log_price", xs, fes=fes, cluster="agency")
    report("T0 V1 on the Texas-benchmark subset (no Texas control)", o, m, med, key="veh_tx_subset")
    o, m = ols_fe(t, "log_price", xs + ["log_tx"], fes=fes, cluster="agency")
    report("T1 T0 + log Texas median of the same make, model and year", o, m, med, key="veh_tx_control")
    OUT["veh_tx_control"]["log_tx"] = {"b": float(o.loc["log_tx", "coef"]), "se": float(o.loc["log_tx", "se"])}
    print("       control log Texas median %+.3f (se %.3f)" % (o.loc["log_tx", "coef"], o.loc["log_tx", "se"]))
    o, m = ols_fe(t, "rel", ["ai"] + PRE + hc, fes=["agency", "subcat_q"], cluster="agency")
    report("T2 log price relative to the Texas median, hedonics, agency + subcat x quarter", o, m, med, key="veh_tx_relative")
    OUT["veh_tx"] = {"n": int(len(t)), "tool_era": int((t.period == "P2").sum()), "tool_era_ai": int(((t.period == "P2") & (t.ai == 1)).sum()),
                     "corr_log": float(r), "median_ratio": float(np.exp(t.rel.median())), "benchmark": path}
    OUT["veh_tx_bins"] = tx_bins(t)
    pts = os.environ.get("VEH_TX_POINTS")
    if pts:
        # (Session 10, the PI's figure options: "isnt a scatter of both color be better ... show me 5 options") each
        # Texas-matched vehicle's log Texas median, log closing bid, period and AI flag; no item id, rows shuffled, kept
        # in our results folder and the session scratchpad, never in report_inputs
        import json as _json
        P = t[["log_tx", "log_price", "period", "ai", "start_local_dt"]].sample(frac=1.0, random_state=11)
        sd = pd.to_datetime(P.start_local_dt)
        _json.dump({"n": int(len(P)), "log_tx": [round(float(x), 3) for x in P.log_tx],
                    "log_price": [round(float(x), 3) for x in P.log_price], "period": [str(x) for x in P.period],
                    "ai": [int(x) for x in P.ai],
                    "half": ["%dH%d" % (d.year, 1 if d.month <= 6 else 2) for d in sd]}, open(pts, "w"))
        print("  wrote %d Texas-matched vehicles (no ids, shuffled) to %s" % (len(P), pts))
    bx = OUT["veh_tx_bins"]
    print("  Texas bins: slope of log closing bid on log Texas median %.3f; median ratio (q25, q75) by group: %s" % (
        bx["slope_log"], {k: "%.3f (%.3f, %.3f) n=%d" % (g["median"], g["q25"], g["q75"], g["n"]) for k, g in bx["ratio"].items()}))
    for k, rows in bx["bins"].items():
        print("    bins %-10s %s" % (k, "none (too few)" if rows is None else
                                     " ".join("%d:$%.0f->$%.0f" % (b["n"], np.exp(b["log_tx"]), np.exp(b["log_price"])) for b in rows)))


def main(panel, feats, scores, out_json=None):
    d = load(panel, feats, scores)
    print("AI rule:", d.attrs.get("ai_rule"))
    # AI_SCOPE=tool (Session 9): AI written = flagged AND started in the tool era. A flag on a listing started
    # before the tool is a detector false positive; it gets its own control (flag_pre), so it sits neither in
    # the AI group nor in the comparison group. AI_SCOPE=all (default) reproduces the Session 8 runs.
    scope = os.environ.get("AI_SCOPE", "all")
    if scope == "tool":
        d["flag_pre"] = (d.ai * (d.period != "P2")).astype(float)
        d["ai"] = (d.ai * (d.period == "P2")).astype(float)
    PRE = ["flag_pre"] if scope == "tool" else []
    print("AI scope:", scope, "| AI = 1 by period:", d.groupby("period").ai.sum().to_dict(),
          "| pre-tool flags:", int(d["flag_pre"].sum()) if PRE else "n/a")
    OUT["ai_scope"] = scope
    v = d[d.veh & d.log_price.notna() & d.age.notna()].copy()
    if os.environ.get("VEH_NO_PARTS") == "1":
        parts = v.subcat == "Automotive > Parts"
        OUT["parts_dropped"] = {"n": int(parts.sum()), "tool_era": int((parts & (v.period == "P2")).sum()),
                                "tool_era_ai": int((parts & (v.period == "P2") & (v.ai == 1)).sum()), "n_before": int(len(v))}
        print("VEH_NO_PARTS: dropping %d 'Automotive > Parts' listings of %d (tool era %d, tool-era AI %d)" % (
            OUT["parts_dropped"]["n"], len(v), OUT["parts_dropped"]["tool_era"], OUT["parts_dropped"]["tool_era_ai"]))
        for t in sorted(v[parts].title.astype(str).str.slice(0, 60)):
            print("    part: %s" % t)
        assert parts.sum() > 0, "no parts listings found: the subcategory name changed?"
        v = v[~parts].copy()
        assert not (v.subcat == "Automotive > Parts").any()
    wk = os.environ.get("VEH_NO_WRECK")
    if wk:
        assert wk in ("title", "text"), "VEH_NO_WRECK must be title or text: %s" % wk
        bad = wreck_selftest()
        assert not bad, ("the wreck rule fails its own test", bad)
        cols = [c for c in ("title", "desc", "updates_text") if c in v.columns]
        assert "title" in cols and "desc" in cols, ("text columns missing", list(v.columns)[:40])
        tt = v.title.fillna("").astype(str)
        tx = tt + " \n " + v.desc.fillna("").astype(str) + (" \n " + v.updates_text.fillna("").astype(str) if "updates_text" in cols else "")
        v["wreck_title"] = tt.map(lambda x: bool(WRECK_RX.search(x)))
        v["wreck_text"] = tx.map(lambda x: bool(WRECK_RX.search(x)))
        print("VEH_NO_WRECK=%s (rule read on %s): wrecks among %d priced vehicles" % (wk, "+".join(cols), len(v)))
        for c in ("wreck_title", "wreck_text"):
            g = v.groupby(["period", "ai"])[c].agg(["sum", "count"])
            print("  %s by (period, AI): %s" % (c, {"%s|%d" % k: "%d of %d" % (r["sum"], r["count"]) for k, r in g.iterrows()}))
            p2 = v[v.period == "P2"]
            for a in (1, 0):
                x = p2.loc[p2.ai == a, c]
                print("    tool era, AI=%d: %d of %d (%.1f%%)" % (a, int(x.sum()), len(x), 100 * x.mean() if len(x) else float("nan")))
        print("  every wreck in the tool era, and the first 40 before it (title; the matched words in the text):")
        hit = v[v.wreck_text | v.wreck_title].sort_values(["period", "ai"], ascending=[False, False])
        shown = 0
        for _, x in hit.iterrows():
            if x.period != "P2" and shown >= 40:
                continue
            shown += x.period != "P2"
            m = sorted(set(w.lower() for w in WRECK_RX.findall(str(x.title) + " " + str(x.desc) + " " + str(x.get("updates_text") or ""))))
            print("    %s ai=%d title=%d text=%d | %s | %s" % (x.period, int(x.ai), int(x.wreck_title), int(x.wreck_text),
                                                          str(x.title)[:55], ", ".join(m)[:80]))
        if os.environ.get("VEH_DMV"):
            tx_ratio_by(v, os.environ["VEH_DMV"], "wreck_" + wk)
        drop = v["wreck_" + wk]
        OUT["wreck_dropped"] = {"rule": wk, "n": int(drop.sum()), "n_before": int(len(v)),
                                "tool_era": int((drop & (v.period == "P2")).sum()),
                                "tool_era_ai": int((drop & (v.period == "P2") & (v.ai == 1)).sum()),
                                "tool_era_ai_n": int(((v.period == "P2") & (v.ai == 1)).sum()),
                                "tool_era_other_n": int(((v.period == "P2") & (v.ai == 0)).sum()),
                                "tool_era_other": int((drop & (v.period == "P2") & (v.ai == 0)).sum())}
        print("VEH_NO_WRECK=%s: dropping %d of %d (tool era %d, tool-era AI %d)" % (
            wk, OUT["wreck_dropped"]["n"], len(v), OUT["wreck_dropped"]["tool_era"], OUT["wreck_dropped"]["tool_era_ai"]))
        assert drop.sum() > 0, "no wrecks found: the rule or the text columns are wrong"
        v = v[~drop].copy()
    v, hc = hedonic_cols(v, veh_only=True)
    v["mm"] = v.f_make.map(norm) + "|" + v.f_model.map(norm)
    v["body"] = v.f_body.map(norm).replace("", "none")
    p2v = v[v.period == "P2"]
    med = float(np.exp(p2v.log_price.median()))
    print("vehicles priced: %d (P0 %d, P1 %d, P2 %d); tool-era AI vehicles %d; make x model cells %d; tool-era median $%.0f" % (
        len(v), (v.period == "P0").sum(), (v.period == "P1").sum(), (v.period == "P2").sum(), int(p2v.ai.sum()), v.mm.nunique(), med))
    xs = ["ai"] + PRE + hc
    xs_p2 = ["ai"] + hc
    fes = ["agency", "mm", "body", "subcat_q"]
    o, m = ols_fe(v, "log_price", xs, fes=fes, cluster="agency")
    report("V1 pooled 2019-26, agency+make x model+body+subcat x quarter", o, m, med, key="veh_pooled")
    ref_level(v, "veh_pooled", "log_price", price=True)
    for k in ["age", "log_miles0", "f_starts_x_0", "f_drivable_x_0", "f_title_x_0", "log_photos"] + PRE:
        if k in o.index:
            print("       hedonic %-14s %+.3f (se %.3f)" % (k, o.loc[k, "coef"], o.loc[k, "se"]))
    o, m = ols_fe(v[v.start_local_dt >= "2022-12-01"], "log_price", xs, fes=fes, cluster="agency")
    report("V2 pooled from Dec 2022 (ChatGPT era only)", o, m, med, key="veh_from_dec2022")
    o, m = ols_fe(v[v.period == "P2"], "log_price", xs_p2, fes=fes, cluster="agency")
    report("V3 tool era only", o, m, med, key="veh_tool_era")
    o, m = ols_fe(v, "log_price", xs, fes=["agency", "mm", "subcat_q"], cluster="agency")
    report("V1b pooled, without body-style FE", o, m, med, key="veh_nobody")
    o, m = ols_fe(v, "log_price", xs + ["log_start"], fes=fes, cluster="agency")
    report("V1c pooled, + log opening bid (seller's own price signal)", o, m, med, key="veh_pooled_start")
    # V4 out-of-time hedonic residual
    pre = v[v.start_local_dt < "2025-01-29"].copy()
    post = v[v.period == "P2"].copy()
    num = ["age", "age2", "log_miles0", "miss_miles", "log_photos", "reserve_any"] + [c for c in hc if c.startswith("f_")]
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    def X(df, fit=False):
        cat = df[["mm", "body", "make", "subcat"]].astype(str)
        C = enc.fit_transform(cat) if fit else enc.transform(cat)
        from scipy.sparse import hstack, csr_matrix
        return hstack([C, csr_matrix(df[num].fillna(0).values.astype(float))]).tocsr()
    m4 = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X(pre, True), pre.log_price.values)
    post["pred"] = m4.predict(X(post))
    post["hres"] = post.log_price - post.pred
    r2 = 1 - np.var(post.hres) / np.var(post.log_price)
    print("  V4 hedonic model fit on %d pre-tool vehicles; out-of-time R2 on tool-era vehicles %.3f" % (len(pre), r2))
    o, m = ols_fe(post, "hres", ["ai"], fes=["agency"], cluster="agency")
    report("V4 tool era, hedonic residual, agency FE", o, m, med, key="veh_residual", n_ai=post.ai.sum())
    # Session 9 (review round 4): flagged listings cluster in May-Aug 2025, and V4 has no time effects, so a
    # tool-era drift in prices relative to the pre-tool hedonic prediction would load on the AI coefficient
    o, m = ols_fe(post, "hres", ["ai"], fes=["agency", "endm"], cluster="agency")
    report("V4b tool era, hedonic residual, agency + end-month FE", o, m, med, key="veh_residual_month", n_ai=post.ai.sum())
    o, m = ols_fe(post, "hres", ["ai"], fes=["agency", "endq"], cluster="agency")
    report("V4c tool era, hedonic residual, agency + end-quarter FE", o, m, med, key="veh_residual_quarter", n_ai=post.ai.sum())
    # G1 all goods
    g = d[(~d.veh) & d.log_price.notna()].copy()
    gmed = float(np.exp(g[g.period == "P2"].log_price.median()))
    print("non-vehicle goods priced: %d; tool-era AI %d; tool-era median $%.0f" % (len(g), int(g[g.period == "P2"].ai.sum()), gmed))
    o, m = ols_fe(g, "log_price", ["ai"] + PRE + ["log_photos", "reserve_any"], fes=["agency", "subcat_q"], cluster="agency")
    report("G1 other goods, pooled 2019-26, agency + subcat x quarter", o, m, gmed, key="goods_pooled")
    o, m = ols_fe(g, "log_price", ["ai"] + PRE + ["log_photos", "reserve_any", "log_start"], fes=["agency", "subcat_q"], cluster="agency")
    report("G1c other goods, pooled, + log opening bid", o, m, gmed, key="goods_pooled_start")
    o, m = ols_fe(g, "log_start", ["ai"] + PRE + ["log_photos", "reserve_any"], fes=["agency", "subcat_q"], cluster="agency")
    print("  G1-balance: AI coefficient on log opening bid (other goods, pooled) %+.3f (se %.3f)" % (o.loc["ai", "coef"], o.loc["ai", "se"]))
    OUT["balance_goods_log_start"] = {"b": float(o.loc["ai", "coef"]), "se": float(o.loc["ai", "se"]), "n": int(m["n"])}
    allg = d[d.log_price.notna()].copy()
    amed = float(np.exp(allg[allg.period == "P2"].log_price.median()))
    o, m = ols_fe(allg, "log_price", ["ai"] + PRE + ["log_photos", "reserve_any"], fes=["agency", "subcat_q"], cluster="agency")
    report("G2 all listings, pooled 2019-26, agency + subcat x quarter", o, m, amed, key="all_pooled")
    o, m = ols_fe(allg, "log_price", ["ai"] + PRE + ["log_photos", "reserve_any", "log_start"], fes=["agency", "subcat_q"], cluster="agency")
    report("G2c all listings, pooled, + log opening bid", o, m, amed, key="all_pooled_start")
    ref_level(allg[allg.log_start.notna()], "all_pooled_start", "log_price", price=True)
    o, m = ols_fe(v, "log_start", ["ai"] + PRE + hc, fes=fes, cluster="agency")
    print("  V1-balance: AI coefficient on log opening bid (vehicles, pooled with hedonics) %+.3f (se %.3f)" % (o.loc["ai", "coef"], o.loc["ai", "se"]))
    OUT["balance_veh_log_start"] = {"b": float(o.loc["ai", "coef"]), "se": float(o.loc["ai", "se"]), "n": int(m["n"])}
    if os.environ.get("VEH_DMV"):
        dmv_designs(v, xs, fes, hc, PRE, med, os.environ["VEH_DMV"])
    # same pooled design for participation outcomes (all listings)
    for y in ["sold", "log_bids", "log_views", "n_q_before_end"]:
        xx = ["ai"] + PRE + ["log_photos", "reserve_any"] + (["log_caplag"] if y == "log_views" else [])
        o, m = ols_fe(d, y, xx, fes=["agency", "subcat_q"], cluster="agency")
        b, se = o.loc["ai", "coef"], o.loc["ai", "se"]
        print("  pooled %-15s AI %+.4f (se %.4f) [%+.4f, %+.4f] n=%d clusters=%d mean=%.3f" % (y, b, se, b - 1.96 * se, b + 1.96 * se, m["n"], m["n_clusters"], m["y_mean"]))
        OUT["pooled_" + y] = {"b": float(b), "se": float(se), "n": int(m["n"]), "clusters": int(m["n_clusters"]), "mean": float(m["y_mean"])}
        ref_level(d[d[y].notna()], "pooled_" + y, y)
    OUT["n_vehicles_priced"] = int(len(v))
    OUT["n_ai_vehicles_tool_era"] = int(p2v.ai.sum())
    OUT["vehicle_median_price_tool_era"] = med
    OUT["hedonic_out_of_time_r2"] = float(r2)
    OUT["ai_rule"] = d.attrs.get("ai_rule")
    if out_json:
        import json
        json.dump(OUT, open(out_json, "w"), indent=1)
        print("wrote", out_json)


if __name__ == "__main__":
    main(*sys.argv[1:])
