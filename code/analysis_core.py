#!/usr/bin/env python3
"""Session 8 core analyses on the Municibid panel.

Loader merges panel + text features + AI scores. Sample: ended auctions, US, listing start
2019-01-01..2026-03-31, real estate excluded. Periods by START date (when the text was written):
P0 < 2022-11-30 (no ChatGPT) | P1 2022-12-01..2025-01-28 (ChatGPT, no tool) | P2 >= 2025-02-01 (tool).

Analyses (run all with: python3 analysis_core.py PANEL FEATURES AISCORES OUTDIR):
  A  premium for words / facts over time (half-year windows), within agency x subcategory-quarter
  B  AI-written vs human listings in P2: content and market outcomes, within agency, subcat x month
  C  adopter vs non-adopter agencies: event study by quarter around the rollout
  D  buyer questions: counts, topics, and whether the description already contained the answer
  G  winners: repeat-buyer share of winning bidders
Units: log points for log outcomes (x100 ~ percent), percentage points for 0/1 outcomes.
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from regs import ols_fe, fmt  # noqa: E402

QTOPICS = [
    ("runs", r"\b(run|runs|running|start|starts|starting|drive|drives|driv(?:e|able)|operat|work|working|function)"),
    ("miles_hours", r"\b(mile|miles|mileage|odometer|hours|hrs|hour meter)"),
    ("title_keys", r"\b(title|titles|key|keys|registration|paperwork|bill of sale)"),
    ("condition", r"\b(rust|leak|leaks|damage|damaged|crack|broken|issue|issues|problem|condition|frame|engine|transmission|brakes?|tires?|battery)"),
    ("specs", r"\b(size|dimension|dimensions|weight|how big|width|length|height|model|year|serial|capacity|voltage|volt|phase|specs?|gvw|cc|hp)"),
    ("included", r"\b(come with|comes with|included|include|includes|accessor|attachments?|all items|how many|quantity|each|lot)"),
    ("logistics", r"\b(pick ?up|pickup|load|loading|forklift|ship|shipping|deliver|delivery|location|where|address|remove|removal|tow)"),
    ("inspect", r"\b(inspect|inspection|see it|look at|view|come see|visit|appointment|in person|preview)"),
    ("payment_price", r"\b(pay|payment|cash|check|credit|reserve|minimum|price|fee|tax|buyer'?s premium)"),
]


def load(panel, feats, scores):
    p = pd.read_parquet(panel)
    f = pd.read_parquet(feats)
    s = pd.read_parquet(scores)
    d = p.merge(f, on="item", how="inner").merge(s, on="item", how="left")
    d = d[(d.auction_state == "ended") & (d.country == "US") & (d.start_local_dt >= "2019-01-01") & (d.start_local_dt < "2026-04-01")]
    d = d[~d.cat1.fillna("").str.lower().str.contains("real estate")].copy()
    d["period"] = np.select([d.start_local_dt < "2022-11-30", d.start_local_dt < "2025-01-29", d.start_local_dt >= "2025-02-01"], ["P0", "P1", "P2"], "gap")
    d = d[d.period != "gap"].copy()
    d["subcat"] = d.cat1.fillna("?") + " > " + d.cat2.fillna("?")
    d["endq"] = d.end_local_dt.dt.to_period("Q").astype(str)
    d["endm"] = d.end_local_dt.dt.to_period("M").astype(str)
    d["subcat_q"] = d.subcat + "|" + d.endq
    d["subcat_m"] = d.subcat + "|" + d.endm
    d["sold"] = (d.n_bids >= 1).astype(float)
    d["log_price"] = np.where((d.n_bids >= 1) & (d.final_bid > 0), np.log(d.final_bid.clip(lower=0.01)), np.nan)
    d["log_bids"] = np.log1p(d.n_bids)
    d["log_views"] = np.log1p(d.page_views)
    d["log_caplag"] = np.log1p(d.capture_lag_days.clip(lower=0))
    d["any_q"] = (d.n_q_before_end >= 1).astype(float)
    d["log_w"] = np.log1p(d.s_w)
    d["facts"] = d.s_n_num + d.s_n_unit + d.s_n_pos + d.s_n_defect
    d["log_facts"] = np.log1p(d.facts)
    d["has_typo"] = (d.s_n_typo >= 1).astype(float)
    d["has_defect"] = (d.s_n_defect >= 1).astype(float)
    d["has_defect_strict"] = (d.s_n_defect_strict >= 1).astype(float) if "s_n_defect_strict" in d else np.nan
    d["has_quality"] = (d.s_n_quality >= 1).astype(float) if "s_n_quality" in d else np.nan
    d["defect_and_praise"] = ((d.s_n_defect_strict >= 1) & ((d.s_n_quality >= 1) | (d.s_n_puff >= 1))).astype(float) if "s_n_defect_strict" in d else np.nan
    d["log_start"] = np.log(d.starting_bid.clip(lower=0.01))
    d["log_photos"] = np.log1p(d.n_photos.fillna(0))
    d["reserve_any"] = (d.reserve.isin(["met", "not_met"])).astype(float)
    # vehicle hedonics: only where the seller's structured table reports them
    d["veh"] = d.f_year.notna() & d.f_make.notna()
    d["age"] = np.where(d.veh, d.end_local_dt.dt.year - d.f_year, np.nan)
    d.loc[(d.age < -1) | (d.age > 80), "age"] = np.nan
    d["log_miles"] = np.where((d.f_miles > 0) & (d.f_miles < 2e6), np.log(d.f_miles), np.nan)
    for c in ["f_starts", "f_drivable", "f_title", "f_inspected"]:
        d[c + "_x"] = d[c].fillna(-1)
    d["make"] = d.f_make.fillna("").str.lower().str.strip()
    # AI indicator (validated in WORKLOG section 11): AI_RULE=union (default; classifier at the 1%%-FPR
    # threshold OR >= 2 LLM-marker words: FPR 1.1%% pre-ChatGPT, recall 0.73 on the blind read),
    # lex (>= 2 marker words: FPR 0.13%%, recall 0.64), c1 (classifier only: FPR 1.0%%, recall 0.36)
    import os
    rule = os.environ.get("AI_RULE", "union")
    c1 = d.ai_flag.fillna(0).astype(bool)
    lex = d.f_n_mark.fillna(0) >= 2
    d["ai"] = {"union": (c1 | lex), "lex": lex, "c1": c1}[rule].astype(float)
    d.attrs["ai_rule"] = rule
    return d


def hedonic_cols(d, veh_only=False):
    cols = ["log_photos", "reserve_any"]
    if veh_only:
        dd = d.copy()
        dd["age2"] = dd.age ** 2
        dd["miss_miles"] = dd.log_miles.isna().astype(float)
        dd["log_miles0"] = dd.log_miles.fillna(0)
        for c in ["f_starts_x", "f_drivable_x", "f_title_x", "f_inspected_x"]:
            for v in (0, 1):
                dd["%s_%d" % (c, v)] = (dd[c] == v).astype(float)
        return dd, cols + ["age", "age2", "log_miles0", "miss_miles"] + ["%s_%d" % (c, v) for c in ["f_starts_x", "f_drivable_x", "f_title_x", "f_inspected_x"] for v in (0, 1)]
    return d, cols


def analysis_A(d, out):
    print("\n" + "=" * 100 + "\nA. Premium for words and facts over time (log price, within agency and subcategory x quarter)")
    d = d[d.log_price.notna()].copy()
    d["half"] = d.start_local_dt.dt.year.astype(str) + "H" + np.where(d.start_local_dt.dt.month <= 6, "1", "2")
    rows = []
    for h, g in d.groupby("half"):
        if len(g) < 300:
            continue
        o, m = ols_fe(g, "log_price", ["log_w", "log_facts", "has_typo", "log_photos", "reserve_any"], fes=["agency", "subcat_q"], cluster="agency")
        r = {"half": h, "n": m["n"], "clusters": m["n_clusters"]}
        for k in ["log_w", "log_facts", "has_typo", "log_photos"]:
            if k in o.index:
                r[k] = o.loc[k, "coef"]
                r[k + "_se"] = o.loc[k, "se"]
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(out + "/A_premia_by_half.csv", index=False)
    print(R.round(4).to_string(index=False))
    print("\nA2. Pooled with period interactions (P1, P2 vs P0); vehicles with hedonics separately")
    for c in ["P1", "P2"]:
        d["is_" + c] = (d.period == c).astype(float)
        for v in ["log_w", "log_facts", "has_typo"]:
            d["%s_x_%s" % (v, c)] = d[v] * d["is_" + c]
    xs = ["log_w", "log_facts", "has_typo", "log_photos", "reserve_any"] + ["%s_x_%s" % (v, c) for v in ["log_w", "log_facts", "has_typo"] for c in ["P1", "P2"]]
    o, m = ols_fe(d, "log_price", xs, fes=["agency", "subcat_q"], cluster="agency")
    print("all items:\n" + fmt(o, m))
    vv, hc = hedonic_cols(d[d.veh], veh_only=True)
    vv["make_q"] = vv.make + "|" + vv.endq
    o, m = ols_fe(vv, "log_price", xs + hc[2:], fes=["agency", "make_q", "subcat"], cluster="agency")
    print("vehicles (age, miles, starts/drivable/title/inspected, make x quarter FE):\n" + fmt(o, m, keep=xs))
    # AI interaction inside P2
    p2 = d[d.period == "P2"].copy()
    p2["ai_x_logw"] = p2.ai * p2.log_w
    p2["ai_x_logfacts"] = p2.ai * p2.log_facts
    o, m = ols_fe(p2, "log_price", ["log_w", "log_facts", "ai", "ai_x_logw", "ai_x_logfacts", "has_typo", "log_photos", "reserve_any"], fes=["agency", "subcat_q"], cluster="agency")
    print("P2 only, AI interactions:\n" + fmt(o, m))


def analysis_B(d, out):
    print("\n" + "=" * 100 + "\nB. AI-written vs human listings in the tool era (P2): content and outcomes")
    p2 = d[d.period == "P2"].copy()
    print("P2 listings %d, AI-flagged %d (%.1f%%), agencies %d, agencies with >=1 AI %d" % (
        len(p2), p2.ai.sum(), 100 * p2.ai.mean(), p2.agency.nunique(), p2[p2.ai == 1].agency.nunique()))
    # balance on predetermined item characteristics within agency
    print("\nB0. Balance (within agency x subcategory): AI vs human on item characteristics")
    for y in ["log_start", "age", "log_miles", "f_starts", "f_drivable", "log_photos", "reserve_any"]:
        o, m = ols_fe(p2, y, ["ai"], fes=["agency", "subcat"], cluster="agency")
        if "ai" in o.index:
            print("  %-14s AI-human %8.4f (se %.4f, t %5.2f) n=%d mean=%.3f" % (y, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai", "t"], m["n"], m["y_mean"]))
    print("\nB1. Content: AI vs human (agency FE + subcategory x month FE)")
    for y in ["s_w", "log_w", "facts", "s_n_num", "s_n_unit", "s_n_pos", "s_n_defect", "has_defect", "s_n_defect_strict", "has_defect_strict",
              "s_n_quality", "has_quality", "defect_and_praise", "s_n_puff", "s_n_you", "has_typo", "s_n_label", "boiler_share"]:
        o, m = ols_fe(p2, y, ["ai"], fes=["agency", "subcat_m"], cluster="agency")
        if "ai" in o.index:
            print("  %-12s AI-human %8.4f (se %.4f, t %5.2f) n=%d human-mean=%.3f" % (y, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai", "t"], m["n"], p2.loc[p2.ai == 0, y].mean()))
    print("\nB2. Market outcomes: AI vs human, with and without fact/length controls")
    specs = [("baseline", ["ai", "log_photos", "reserve_any"]),
             ("+facts", ["ai", "log_facts", "log_photos", "reserve_any"]),
             ("+facts+start", ["ai", "log_facts", "log_start", "log_photos", "reserve_any"])]
    for y in ["sold", "log_price", "log_bids", "log_views", "any_q", "n_q_before_end"]:
        for name, xs in specs:
            xx = xs + (["log_caplag"] if y == "log_views" else [])
            o, m = ols_fe(p2, y, xx, fes=["agency", "subcat_m"], cluster="agency")
            if "ai" in o.index:
                print("  %-14s %-13s AI %8.4f (se %.4f, t %5.2f) n=%d clusters=%d mean=%.3f" % (y, name, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai", "t"], m["n"], m["n_clusters"], m["y_mean"]))
    vv, hc = hedonic_cols(p2[p2.veh], veh_only=True)
    vv["make_q"] = vv.make + "|" + vv.endq
    print("  vehicles with hedonics (n=%d):" % len(vv))
    for y in ["sold", "log_price", "log_bids", "any_q"]:
        o, m = ols_fe(vv, y, ["ai", "log_facts"] + hc, fes=["agency", "make_q"], cluster="agency")
        if "ai" in o.index:
            print("  %-14s AI %8.4f (se %.4f, t %5.2f) n=%d clusters=%d" % (y, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai", "t"], m["n"], m["n_clusters"]))


def analysis_C(d, out):
    print("\n" + "=" * 100 + "\nC. Adopter vs non-adopter agencies: event study by quarter (start date), agency FE + subcat x quarter FE")
    p2a = d[(d.start_local_dt >= "2025-02-01") & (d.start_local_dt < "2025-10-01")]
    ag = p2a.groupby("agency").agg(n=("ai", "size"), share=("ai", "mean"))
    ag = ag[ag.n >= 3]
    adopt = set(ag[ag.share >= 0.5].index)
    never = set(ag[ag.share == 0].index)
    print("agencies with >=3 listings Feb-Sep 2025: %d | adopters (>=50%% AI) %d | non-adopters (0%%) %d" % (len(ag), len(adopt), len(never)))
    e = d[d.agency.isin(adopt | never) & (d.start_local_dt >= "2023-01-01")].copy()
    e["adopter"] = e.agency.isin(adopt).astype(float)
    e["sq"] = e.start_local_dt.dt.to_period("Q").astype(str)
    qs = sorted(e.sq.unique())
    ref = "2024Q4"
    xs = []
    for q in qs:
        if q == ref:
            continue
        e["D_" + q] = ((e.sq == q) & (e.adopter == 1)).astype(float)
        xs.append("D_" + q)
    e["sq_fe"] = e.sq
    for y in ["ai", "log_w", "s_n_puff", "facts", "has_defect", "sold", "log_price", "log_bids", "any_q"]:
        o, m = ols_fe(e, y, xs, fes=["agency", "subcat_q"], cluster="agency")
        line = " ".join("%s:%+.3f(%.3f)" % (k[2:], o.loc[k, "coef"], o.loc[k, "se"]) for k in xs if k in o.index)
        print("  %-10s n=%d clusters=%d | %s" % (y, m["n"], m["n_clusters"], line))
    # pooled DiD
    e["post"] = (e.start_local_dt >= "2025-02-01").astype(float)
    e["did"] = e.post * e.adopter
    print("pooled DiD (adopter x post), agency FE + subcat x quarter FE:")
    for y in ["ai", "log_w", "s_n_puff", "facts", "has_defect", "sold", "log_price", "log_bids", "any_q", "n_q_before_end"]:
        o, m = ols_fe(e, y, ["did"], fes=["agency", "subcat_q"], cluster="agency")
        if "did" in o.index:
            print("  %-14s DiD %8.4f (se %.4f, t %5.2f) n=%d clusters=%d" % (y, o.loc["did", "coef"], o.loc["did", "se"], o.loc["did", "t"], m["n"], m["n_clusters"]))


def analysis_D(d, panel_qa, out):
    print("\n" + "=" * 100 + "\nD. Buyer questions")
    q = panel_qa
    rows = []
    for _, r in q.iterrows():
        for x in r.qa or []:
            t = (x.get("q") or "").lower()
            tops = [name for name, pat in QTOPICS if re.search(pat, t)]
            rows.append({"item": r["item"], "q": t, "topics": tops or ["other"]})
    Q = pd.DataFrame(rows)
    Q = Q.merge(d[["item", "period", "ai", "spec", "s_n_num"]], on="item", how="inner")
    print("questions linked to sample listings: %d on %d listings" % (len(Q), Q["item"].nunique()))
    ex = Q.explode("topics")
    tab = ex.groupby(["period", "topics"]).size().unstack(0).fillna(0)
    tab = tab / tab.sum()
    print("topic shares by period (share of question-topic mentions):\n" + tab.round(3).to_string())
    p2 = ex[ex.period == "P2"]
    t2 = p2.groupby(["ai", "topics"]).size().unstack(0).fillna(0)
    t2 = t2 / t2.sum()
    print("P2 topic shares, human (0) vs AI (1):\n" + t2.round(3).to_string())
    # was the answer already in the description? (runs / miles questions vs description mentions)
    has_runs_txt = d.set_index("item").spec.str.contains(r"(?i)\b(runs?|running|starts?|drives?|does not start|won'?t start|inoperable|operational)\b")
    has_miles_txt = d.set_index("item").spec.str.contains(r"(?i)\b(miles?|mileage|odometer|hours|hrs)\b")
    Q["desc_runs"] = Q["item"].map(has_runs_txt)
    Q["desc_miles"] = Q["item"].map(has_miles_txt)
    for top, col in [("runs", "desc_runs"), ("miles_hours", "desc_miles")]:
        z = Q[Q.topics.map(lambda t: top in t)]
        print("questions about %-11s: share whose description already mentioned it, by period x AI:" % top)
        print(z.groupby(["period", "ai"])[col].agg(["size", "mean"]).round(3).to_string())


def analysis_G(d, out):
    print("\n" + "=" * 100 + "\nG. Winners: repeat buyers")
    w = d[(d.sold == 1) & d.high_bidder_id.notna()].copy()
    wins = w.groupby("high_bidder_id").size()
    w["winner_total_wins"] = w.high_bidder_id.map(wins)
    w["repeat5"] = (w.winner_total_wins >= 5).astype(float)
    print("winning bidders %d; wins per winner: median %.0f, share of auctions won by 5+ win buyers %.3f" % (
        len(wins), wins.median(), w.repeat5.mean()))
    print(w.groupby("period").repeat5.mean().round(3).to_string())
    p2 = w[w.period == "P2"]
    o, m = ols_fe(p2, "repeat5", ["ai", "log_photos", "reserve_any"], fes=["agency", "subcat_m"], cluster="agency")
    print("P2: repeat-buyer (5+ wins) winner, AI vs human:\n" + fmt(o, m, keep=["ai"]))


def main(panel, feats, scores, outdir, pages_jsonl=None):
    d = load(panel, feats, scores)
    print("AI_RULE =", d.attrs.get("ai_rule"), "| analysis sample", len(d), d.period.value_counts().to_dict(), "| priced", int(d.log_price.notna().sum()))
    print("AI share by period:", d.groupby("period").ai.mean().round(4).to_dict())
    analysis_A(d, outdir)
    analysis_B(d, outdir)
    analysis_C(d, outdir)
    if pages_jsonl:
        import json
        qa = []
        for l in open(pages_jsonl):
            r = json.loads(l)
            if r.get("page_kind") == "listing" and r.get("qa"):
                qa.append({"item": r["item"], "qa": r["qa"]})
        analysis_D(d, pd.DataFrame(qa), outdir)
    analysis_G(d, outdir)


if __name__ == "__main__":
    main(*sys.argv[1:])
