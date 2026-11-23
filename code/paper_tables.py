#!/usr/bin/env python3
"""Session 9: the paper's tables as LaTeX tabulars (paper/tables/*.tex), from report_inputs/ only.

  tab_detector   detector validation: false positives (in and out of time), recall on controlled rewrites
                 by instruction x generator, blind-read recall and controls flagged, for the three rules
  tab_outcomes   (appendix) every buyer design under every AI definition, in the outcome's own units
  tab_content    (appendix) what changes in the text, tool era, under every AI definition, with the
                 controlled-rewrite change when paper_numbers.json exists
  tab_rewrites   (appendix) the controlled rewrites by instruction x generator
  tab_balance    (appendix) balance on characteristics set before or apart from the text (paper_numbers.json)
  tab_stddet     (appendix) two standard detectors (Binoculars, Fast-DetectGPT; job 842934) beside the main rule:
                 share flagged by period, recall on the controlled rewrites, blind read (stddet_summary.json,
                 stddet_blind.csv and reader 1's saved labels)
  tab_wrecks     (appendix; PI, 2026-10-30, "show with wreck and with out wrck both table"): the vehicle designs on all
                 priced vehicles, without the Parts listings, and also without wrecks named in the title or in the
                 title or description (hedonic_<rule>_tool_noparts.json, _nowreck_title.json, _nowreck_text.json: jobs
                 877521 to 877523 and 884267 to 884272)
  tab_examples   (appendix; --examples, PI 2026-10-30: "the few type detailed actaul example from data def should be in
                 appendix"): two assistant-era listings per test outcome (a seeded draw, job 884643), a span of each
                 listing's own words, its classifier score, AI marker words and closing price. The excerpts are listing
                 text, so their file stays in the session scratchpad, never in report_inputs/
Cells that need paper_numbers.json print as \\pend{} until it exists.
Usage: python3 paper_tables.py REPORT_INPUTS_DIR OUT_DIR
"""
import json
import math
import os
import sys

RULES = [("union", "Main rule"), ("lex", "AI marker words only"), ("c1", "Classifier only")]
GENS = [("appealing|opus", "``Appealing'', Opus"), ("appealing|sonnet", "``Appealing'', Sonnet"), ("appealing|haiku", "``Appealing'', Haiku"),
        ("concise|opus", "``Concise'', Opus"), ("concise|sonnet", "``Concise'', Sonnet")]


def r2s(x, d):
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def sg(x, d=1):
    s = r2s(abs(x), d)
    return s if float(s) == 0 else ("+" if x > 0 else "\\ensuremath{-}") + s


def ci_lin(b, se, scale=1.0, d=1):
    return "%s [%s, %s]" % (sg(scale * b, d), sg(scale * (b - 1.96 * se), d), sg(scale * (b + 1.96 * se), d))


def ci_exp(b, se, d=1):
    f = lambda z: 100 * (math.exp(z) - 1)
    return "%s [%s, %s]" % (sg(f(b), d), sg(f(b - 1.96 * se), d), sg(f(b + 1.96 * se), d))


def gen_key(variant, model):
    v = {"V1": "appealing", "V2": "concise"}.get(str(variant), str(variant).lower())
    m = str(model).lower()
    for k in ("opus", "sonnet", "haiku"):
        if k in m:
            return "%s|%s" % ("appealing" if "1" in v or "appeal" in v or "enh" in v else "concise", k)
    return None


def blind_flags(inp):
    """Reader 1's blind-read labels (saved before the key was opened) joined to the standard detectors' per-text flags:
    per flag, (reader 1's AI labels flagged, reader 1's AI labels, pre-ChatGPT controls flagged, controls). None when a
    file is missing."""
    import csv
    bp = os.path.join(inp, "stddet_blind.csv")
    lp = os.path.join(inp, "blind_labels_s8.json")   # reader 1's labels only (scratchpad/machine_records_s8/blind_validation/my_labels.json without its notes)
    if not (os.path.exists(bp) and os.path.exists(lp)):
        return None
    rows = list(csv.DictReader(open(bp)))
    lab = json.load(open(lp))["labels"]
    assert {r["bid"] for r in rows} == set(lab), "blind ids differ between the detector run and reader 1"
    ai = [r for r in rows if lab[r["bid"]] == "A"]
    ctrl = [r for r in rows if r["stratum"] == "pre_control"]
    return {f: (sum(r[f] == "True" for r in ai), len(ai), sum(r[f] == "True" for r in ctrl), len(ctrl))
            for f in ("bino_ai", "bino_cal", "fdg_ai", "fdg_cal")}


def tab_pilot(BL, AG, RULES, out, wilson):
    """Appendix: the two LLM judges and the one-person human pilot on the same 100 blind listings (blind_sample.py
    --score, job 884704). Rows: labels given, pre-ChatGPT controls labeled AI, assistant-era listings labeled AI by
    classifier-score third, each rule's recall of the judge's AI labels and precision among its assistant-era flags
    (95% Wilson intervals), and agreement with each LLM judge (count of 100 and Cohen's kappa for three labels)."""
    J = (("llm_judge1", "LLM judge 1"), ("llm_judge2", "LLM judge 2"), ("human_pilot", "Human pilot"))
    for j, _ in J:
        n = BL[j]["n"]
        assert n["A"] + n["U"] + n["H"] == 100, (j, n)
        assert sum(BL[j]["by_stratum"][s_]["A"] for s_ in BL[j]["by_stratum"]) == n["A"], j
        assert BL[j]["by_stratum"]["pre_control"]["A"] == BL[j]["controls_labelled_ai"][0], j
    def ci(kn):
        k, n = kn
        lo, hi = wilson(k, n)
        return "%s [%s, %s]" % (r2s(k / n, 2), r2s(lo, 2), r2s(hi, 2))
    L = ["\\begin{tabular}{lccc}", "\\toprule", " & " + " & ".join(lab for _, lab in J) + " \\\\", "\\midrule"]
    L.append("Labels: AI / unsure / human (of 100) & " + " & ".join("%d / %d / %d" % (BL[j]["n"]["A"], BL[j]["n"]["U"], BL[j]["n"]["H"]) for j, _ in J) + " \\\\")
    L.append("Pre-ChatGPT listings labeled AI (of 25) & " + " & ".join(str(BL[j]["controls_labelled_ai"][0]) for j, _ in J) + " \\\\")
    L.append("\\multicolumn{4}{l}{\\emph{Assistant-era listings labeled AI, by third of classifier score (of 25 each)}} \\\\")
    for st, lab in (("post_high", "Highest third"), ("post_mid", "Middle third"), ("post_low", "Lowest third")):
        L.append("\\quad %s & %s \\\\" % (lab, " & ".join(str(BL[j]["by_stratum"][st]["A"]) for j, _ in J)))
    names = dict(RULES)
    L.append("\\multicolumn{4}{l}{\\emph{Recall: share of the judge's AI labels that the rule flags}} \\\\")
    for r, _ in RULES:
        L.append("\\quad %s & %s \\\\" % (names[r], " & ".join(ci(BL[j]["recall_" + r]) for j, _ in J)))
    L.append("\\multicolumn{4}{l}{\\emph{Precision: share of the rule's assistant-era flags that the judge labels AI}} \\\\")
    for r, _ in RULES:
        L.append("\\quad %s & %s \\\\" % (names[r], " & ".join(ci(BL[j]["precision_tool_" + r]) for j, _ in J)))
    L.append("\\multicolumn{4}{l}{\\emph{Agreement on 100 listings: count, Cohen's $\\kappa$ for three labels}} \\\\")
    for j0, lab0 in J[:2]:
        cells = []
        for j, _ in J:
            if j == j0:
                cells.append("--")
                continue
            a_ = AG.get("%s|%s" % (j0, j)) or AG.get("%s|%s" % (j, j0))
            cells.append("%d, %s" % (a_["agree_of_100"], r2s(a_["kappa3"], 2)))
        L.append("\\quad With %s & %s \\\\" % (lab0, " & ".join(cells)))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_pilot.tex"), "w").write("\n".join(L) + "\n")


def main(inp, out):
    os.makedirs(out, exist_ok=True)
    J = lambda f: json.load(open(os.path.join(inp, f)))
    R, K = J("report_data.json"), J("paper_constants.json")
    tool = all(os.path.exists(os.path.join(inp, "hedonic_%s_tool.json" % r)) for r, _ in RULES)
    H = {r: J(("hedonic_%s_tool.json" if tool else "hedonic_%s.json") % r) for r, _ in RULES}
    PN = J("paper_numbers.json") if os.path.exists(os.path.join(inp, "paper_numbers.json")) else None
    D = K["detector"]
    # ---------------- tab_detector ----------------
    rec = {}
    if PN:
        for r, _ in RULES:
            rec[r] = {}
            for k, v in PN["recall_rewrites"][r].items():
                if "|" in k:
                    g = gen_key(*k.split("|"))
                    if g:
                        rec[r][g] = v
                elif k == "all":
                    rec[r]["all"] = v
    def wilson(k, n, z=1.96):
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    L = ["\\begin{tabular}{lccc}", "\\toprule", " & Main rule & AI marker words only & Classifier only \\\\", "\\midrule",
         "\\multicolumn{4}{l}{\\emph{Share flagged in the analysis sample}} \\\\"]
    sp = {r: R["rules"][r]["share_by_period"] for r, _ in RULES}
    L.append("\\quad Before ChatGPT (false positives) & " + " & ".join("%.2f\\%%" % (100 * sp[r]["P0"]) for r, _ in RULES) + " \\\\")
    L.append("\\quad Chatbots only & " + " & ".join("%.2f\\%%" % (100 * sp[r]["P1"]) for r, _ in RULES) + " \\\\")
    L.append("\\quad Optional AI assistant & " + " & ".join("%.1f\\%%" % (100 * sp[r]["P2"]) for r, _ in RULES) + " \\\\")
    L.append("\\multicolumn{4}{l}{\\emph{Recall on controlled rewrites}} \\\\")
    # the field flags come from the saved detector (detector_s8); paper_numbers.py rebuilds the classifier, and when the
    # rebuild is not identical (Session 9: threshold 0.664 against the saved 0.679) only the marker rule, which does not
    # use the classifier, is taken from the rebuild
    same = bool(PN and PN.get("checks", {}).get("B_detector_rebuild", {}).get("identical"))
    for g, lab in GENS + [("all", "All 600")]:
        cells = []
        for r, _ in RULES:
            if PN and g in rec.get(r, {}) and (same or r == "lex"):
                cells.append(r2s(rec[r][g]["recall"], 2))
            elif r == "c1":
                cells.append(r2s(D["recall_rewrites_c1"][g], 2))
            elif r == "union":
                # the main rule flags every listing the classifier flags, so its recall is at least the classifier's
                cells.append("$\\geq$ %s" % r2s(D["recall_rewrites_c1"][g], 2))
            else:
                cells.append("\\pend{}")
        n = (" (%d)" % rec["union"][g]["n"]) if PN and g in rec.get("union", {}) else ""
        L.append("\\quad %s%s & %s \\\\" % (lab, n, " & ".join(cells)))
    L.append("\\multicolumn{4}{l}{\\emph{Blind labels on 100 listings: two LLM judges}} \\\\")
    rc1 = {"union": (16, 22), "lex": (14, 22), "c1": (8, 22)}
    rc2 = {"union": (16, 21), "lex": (15, 21), "c1": (7, 21)}
    pr1 = {"union": (16, 20), "lex": (14, 15), "c1": (8, 11)}
    for lab, dd in (("Recall, LLM judge 1", rc1), ("Recall, LLM judge 2", rc2), ("Precision of assistant-era flags (LLM judge 1)", pr1)):
        cells = []
        for r, _ in RULES:
            k, n = dd[r]
            lo, hi = wilson(k, n)
            cells.append("%s [%s, %s]" % (r2s(k / n, 2), r2s(lo, 2), r2s(hi, 2)))
        L.append("\\quad %s & %s \\\\" % (lab, " & ".join(cells)))
    L.append("\\quad Pre-ChatGPT listings flagged (of 25) & 2 & 0 & 2 \\\\")
    # Session 10 (PI, 2026-10-30: "lets include it for now"): the one-person human pilot, scored like the LLM judges by
    # blind_sample.py --score (job 884704); the same job reproduces the LLM judge rows above, checked here
    bs = os.path.join(inp, "blind_scores_s10.json")
    if os.path.exists(bs):
        BL = json.load(open(bs))["labellers"]
        for dd, key in ((rc1, "recall_"), (pr1, "precision_tool_")):
            assert all(tuple(BL["llm_judge1"][key + r]) == dd[r] for r, _ in RULES), ("LLM judge 1 rows differ from job 884704", key)
        assert all(tuple(BL["llm_judge2"]["recall_" + r]) == rc2[r] for r, _ in RULES), "LLM judge 2 rows differ from job 884704"
        # Session 10 rewrite (PI, 2026-10-30: "the human agreement, we can leave them [in the appendix] ... Being just
        # one person pilot can't be in the main paper"): the pilot leaves Table 1 for its own appendix table, tab_pilot,
        # with every judge side by side (counts, controls, terciles, recall and precision by rule, agreement)
        tab_pilot(BL, json.load(open(bs))["agreement"], RULES, out, wilson)
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_detector.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_outcomes ----------------
    def cell(e, kind):
        return ci_lin(e["b"], e["se"], 100) if kind == "pp" else (ci_lin(e["b"], e["se"], 1, 2) if kind == "cnt" else ci_exp(e["b"], e["se"]))
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Outcome (units) & Main rule & AI marker words only & Classifier only & $n$ (sellers) \\\\", "\\midrule",
         "\\multicolumn{5}{l}{\\emph{Pooled 2019 to 2026: seller and subcategory $\\times$ quarter fixed effects, photos, reserve}} \\\\"]
    rows = [("Received a bid (percentage points)", "pooled_sold", "pp"), ("Bids (\\%)", "pooled_log_bids", "exp"), ("Page views (\\%)", "pooled_log_views", "exp"),
            ("Buyer questions (count)", "pooled_n_q_before_end", "cnt"), ("Price, vehicles, vehicle controls (\\%)", "veh_pooled", "exp"),
            ("Price, vehicles, characteristics and opening bid held fixed (\\%)", "veh_pooled_start", "exp"), ("Price, all goods (\\%)", "all_pooled", "exp"),
            ("Price, all goods, opening bid held fixed (\\%)", "all_pooled_start", "exp"),
            # written by the Session 9 re-run (AI_SCOPE=tool); every design price_hedonics.py runs is shown
            ("Price, vehicles, comparison group from Dec 2022 (\\%)", "veh_from_dec2022", "exp"),
            ("Price, vehicles, no body-style fixed effects (\\%)", "veh_nobody", "exp"),
            ("Price, non-vehicle goods (\\%)", "goods_pooled", "exp"),
            ("Price, non-vehicle goods, opening bid held fixed (\\%)", "goods_pooled_start", "exp")]
    for lab, key, kind in rows:
        if key not in H["union"]:
            continue
        e0 = H["union"][key]
        L.append("\\quad %s & %s & %s (%s) \\\\" % (lab, " & ".join(cell(H[r][key], kind) for r, _ in RULES), "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    L.append("\\midrule")
    L.append("\\multicolumn{5}{l}{\\emph{Assistant era only: seller and subcategory $\\times$ month fixed effects, photos, reserve}} \\\\")
    for lab, key, kind in [("Received a bid (percentage points)", "sold", "pp"), ("Price (\\%)", "log_price", "exp"), ("Bids (\\%)", "log_bids", "exp"),
                           ("Page views (\\%)", "log_views", "exp"), ("Any buyer question (percentage points)", "any_q", "pp"), ("Buyer questions (count)", "n_q_before_end", "cnt")]:
        e0 = R["rules"]["union"]["market"][key]["subcat_m"]
        L.append("\\quad %s & %s & %s (%s) \\\\" % (lab, " & ".join(cell(R["rules"][r]["market"][key]["subcat_m"], kind) for r, _ in RULES),
                                                 "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    e0 = H["union"]["veh_residual"]
    L.append("\\quad Price, vehicles, net of a prediction from earlier vehicles' characteristics (\\%%), seller fixed effects only & %s & %s (%s) \\\\" % (
        " & ".join(ci_exp(H[r]["veh_residual"]["b"], H[r]["veh_residual"]["se"]) for r, _ in RULES), "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    for key, lab in (("veh_residual_month", "The same, seller and end-month fixed effects (\\%)"),
                     ("veh_residual_quarter", "The same, seller and end-quarter fixed effects (\\%)")):
        if key in H["union"]:
            e0 = H["union"][key]
            L.append("\\quad %s & %s & %s (%s) \\\\" % (lab, " & ".join(ci_exp(H[r][key]["b"], H[r][key]["se"]) for r, _ in RULES),
                                                         "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    if "veh_tool_era" in H["union"]:
        e0 = H["union"]["veh_tool_era"]
        L.append("\\quad Price, vehicles, characteristics held fixed, subcategory $\\times$ quarter (\\%%) & %s & %s (%s) \\\\" % (
            " & ".join(ci_exp(H[r]["veh_tool_era"]["b"], H[r]["veh_tool_era"]["se"]) for r, _ in RULES), "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    SD = PN.get("same_day") if PN else None
    if SD and "error" not in SD:
        L.append("\\midrule")
        L.append("\\multicolumn{5}{l}{\\emph{Assistant era, same seller and start day: seller $\\times$ start-day and subcategory fixed effects, photos, reserve (%s seller-days)}} \\\\"
                 % "{:,}".format(SD["union"]["batches"]))
        for lab, key, kind in [("Received a bid (percentage points)", "sold", "pp"), ("Price (\\%)", "log_price", "exp"), ("Bids (\\%)", "log_bids", "exp"),
                               ("Page views (\\%)", "log_views", "exp"), ("Any buyer question (percentage points)", "any_q", "pp"), ("Buyer questions (count)", "nq", "cnt")]:
            e0 = SD["union"]["base|" + key]
            if not e0:
                continue
            L.append("\\quad %s & %s & %s (%s) \\\\" % (lab, " & ".join(cell(SD[r]["base|" + key], kind) if SD[r]["base|" + key] else "--" for r, _ in RULES),
                                                     "{:,}".format(e0["n"]), "{:,}".format(e0["clusters"])))
    L.append("\\multicolumn{5}{l}{\\footnotesize $n$: main rule, after dropping groups that hold a single listing. Page views also control for log(1 + capture lag in days).} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_outcomes.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_content ----------------
    feats = [("Log of one plus words", "log_w", 2), ("Words", "s_w", 1), ("Promotional words", "s_n_puff", 2), ("``you'' or ``your''", "s_n_you", 2),
             ("``Label: text'' lines", "s_n_label", 2), ("Numbers", "s_n_num", 2), ("Numbers with units", "s_n_unit", 2),
             ("Defect statements", "s_n_defect_strict", 2), ("Any quality phrase (percentage points)", "has_quality", 1), ("Defect stated, item praised (percentage points)", "defect_and_praise", 1)]
    rw = PN["rewrites"]["all"] if PN else None
    rwkey = {"log_w": "log_w", "s_w": "w", "s_n_puff": "n_puff", "s_n_you": "n_you", "s_n_label": "n_label", "s_n_num": "n_num", "s_n_unit": "n_unit",
             "s_n_defect_strict": "n_defect_strict", "has_quality": "has_quality", "defect_and_praise": "defect_and_praise"}
    L = ["\\begin{tabular}{lccccc}", "\\toprule",
         "Feature (per listing) & Human-written mean (main / marker / classifier) & Main rule & AI marker words & Classifier & Rewrites \\\\", "\\midrule"]
    for lab, key, d in feats:
        pp = key in ("has_quality", "defect_and_praise")
        sc = 100 if pp else 1
        # each definition has its own human group (review round 4: one column hid the classifier's higher base)
        hms = [R["rules"][r]["content"][key]["human_mean"] for r, _ in RULES]
        cells = [" / ".join(("%.2f" % h) if not pp else ("%.1f\\%%" % (100 * h)) for h in hms)]
        for r, _ in RULES:
            e = R["rules"][r]["content"][key]
            cells.append("%s (%.*f)" % (sg(sc * e["b"], d), d, sc * e["se"]))
        if rw:
            q = rw[rwkey[key]]
            cells.append("%s [%s, %s]" % (sg(sc * q["diff"], d), sg(sc * q["lo"], d), sg(sc * q["hi"], d)))
        else:
            cells.append("\\pend{}")
        L.append("%s & %s \\\\" % (lab, " & ".join(cells)))
    e = R["rules"]["union"]["content"]["s_w"]
    L.append("\\multicolumn{6}{l}{\\footnotesize Main rule: %s assistant-era listings, %s sellers. Rule columns: coefficient (standard error).} \\\\" % (
        "{:,}".format(e["n"]), "{:,}".format(e["clusters"])))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_content.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_rewrites ----------------
    A, Bm = K["rewrite_audit"]["by_generator"], K["benchmark_rewrites"]["by_generator"]
    CI = K["benchmark_rewrites"]["by_generator_ci"]
    L = ["\\begin{tabular}{lccccccc}", "\\toprule",
         "AI model & $n$ & Numbers & Defects & Adds quality & Promotional words & Words & Boost \\\\",
         " & & kept & kept$^a$ & phrase & added & $\\times$ & (log points, 95\\% CI) \\\\", "\\midrule"]
    for g, lab in GENS:
        a = A[g]
        L.append("%s & %d & %.3f & %.3f & %d\\%% & %s & %.2f & %s [%s, %s] \\\\" % (lab, a["n"], a["numbers_kept"], a["defects_kept"], round(100 * a["adds_quality"]),
                 sg(a["promo_added"], 2), a["words_ratio"], sg(100 * Bm[g], 1), sg(100 * CI[g][0], 1), sg(100 * CI[g][1], 1)))
    ra = K["rewrite_audit"]
    bb = K["benchmark_rewrites"]
    L.append("\\midrule")
    L.append("All 600 & 600 & %.3f & %.3f & %.1f\\%% & %s & %.2f & %s [%s, %s] \\\\" % (ra["numbers_kept"], ra["defects_kept"], 100 * ra["adds_quality_claim"],
             sg(ra["promo_added"], 2), ra["words_ratio"], sg(100 * bb["b"], 1), sg(100 * bb["lo"], 1), sg(100 * bb["hi"], 1)))
    L.append("\\multicolumn{8}{l}{\\footnotesize $^a$ Loose defect pattern of the audit (includes ``needs'', ``requires''); real listings use the strict pattern.} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_rewrites.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_balance ----------------
    L = ["\\begin{tabular}{lcccc}", "\\toprule", "Characteristic & Comparison mean & Assistant era & Pooled 2019 to 2026 & $n$ (pooled) \\\\", "\\midrule"]
    if PN:
        b = PN["balance"]["union"]
        for lab, key, d in [("Log opening bid", "log_start", 2), ("Log photos", "log_photos", 2), ("Reserve set", "reserve_any", 3),
                            ("Has characteristics table", "veh_f", 3), ("Vehicle age, years", "age", 2), ("Log miles", "log_miles", 2),
                            ("Starts (reported yes)", "starts_yes", 3), ("Drives (reported yes)", "drivable_yes", 3), ("Title (reported yes)", "title_yes", 3)]:
            t, p = b["tool_era"].get(key), b["pooled"].get(key)
            if not (t and p):
                continue
            L.append("%s & %.*f & %s (%.*f) & %s (%.*f) & %s \\\\" % (lab, d, p["human_mean"], sg(t["b"], d), d, t["se"], sg(p["b"], d), d, p["se"], "{:,}".format(p["n"])))
    else:
        L.append("\\multicolumn{5}{c}{\\pend{}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_balance.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_stddet (appendix): two standard detectors beside the main rule ----------------
    if os.path.exists(os.path.join(inp, "stddet_summary.json")):
        SD, bl, br = J("stddet_summary.json"), blind_flags(inp), D["blind_read"]
        pc1 = lambda x: "%.1f\\%%" % (100 * x)
        pc0 = lambda x: "%s\\%%" % r2s(100 * x, 0)
        L = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Share of listings flagged} & Rewrites & \\multicolumn{2}{c}{LLM judge 1} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}",
             "Rule & Before ChatGPT & Chatbots only & Optional AI assistant & flagged & AI labels found & Pre-ChatGPT listings flagged \\\\", "\\midrule"]
        spu = R["rules"]["union"]["share_by_period"]
        # recall from the rebuild only when it is identical to the saved detector, as in tab_detector
        our_rw = pc0(rec["union"]["all"]["recall"]) if (PN and same and "all" in rec.get("union", {})) else \
            "$\\geq$ " + pc0(D["recall_rewrites_c1"]["all"])
        L.append("Main rule & %s & %s & %s & %s & %d of %d & %d of %d \\\\" % (
            pc1(spu["P0"]), pc1(spu["P1"]), pc1(spu["P2"]), our_rw, br["recall_union_vs_reader1"][0], br["recall_union_vs_reader1"][1],
            br["controls_flagged_by_rule"][0], br["controls_flagged_by_rule"][1]))
        for k, lab in (("bino_ai", "Binoculars, published threshold"), ("bino_cal", "Binoculars, 1\\% before ChatGPT"),
                       ("fdg_ai", "Fast-DetectGPT, published rule"), ("fdg_cal", "Fast-DetectGPT, 1\\% before ChatGPT")):
            sp_ = SD[k]["share_by_period"]
            b_ = ("%d of %d & %d of %d" % bl[k]) if bl else "\\pend{} & \\pend{}"
            L.append("%s & %s & %s & %s & %s & %s \\\\" % (lab, pc1(sp_["P0"]), pc1(sp_["P1"]), pc1(sp_["P2"]),
                                                          pc0(SD[k]["recall_rewrites"]["all"]), b_))
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(out, "tab_stddet.tex"), "w").write("\n".join(L) + "\n")
    # ---------------- tab_wrecks ----------------
    vs = [("tool_noparts", "noparts"), ("tool_nowreck_title", "title"), ("tool_nowreck_text", "text")]
    if all(os.path.exists(os.path.join(inp, "hedonic_%s_%s.json" % (r, v))) for r, _ in RULES for v, _ in vs):
        V = {"all": H}
        V.update({k: {r: J("hedonic_%s_%s.json" % (r, v)) for r, _ in RULES} for v, k in vs})
        cols = ["all", "noparts", "title", "text"]
        for r, _ in RULES:
            assert V["noparts"][r]["parts_dropped"]["n"] == V["title"][r]["parts_dropped"]["n"] == V["text"][r]["parts_dropped"]["n"]
            assert V["title"][r]["wreck_dropped"]["rule"] == "title" and V["text"][r]["wreck_dropped"]["rule"] == "text"
        u = {k: V[k]["union"] for k in cols}
        ce = lambda e: ci_exp(e["b"], e["se"])
        L = ["\\begin{tabular}{lcccc}", "\\toprule",
             "& All priced & Without the %d & \\multicolumn{2}{c}{Also without wrecks named in} \\\\" % u["noparts"]["parts_dropped"]["n"],
             "\\cmidrule(lr){4-5}",
             "Design and AI definition (price, \\%%) & vehicles & Parts listings & the title (%d) & title or description (%d) \\\\" % (
                 u["title"]["wreck_dropped"]["n"], u["text"]["wreck_dropped"]["n"]),
             "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Pooled 2019 to 2026, vehicle characteristics held fixed (seller, make $\\times$ model, body and subcategory $\\times$ quarter fixed effects)}} \\\\"]
        for r, lab in RULES:
            L.append("\\quad %s & %s \\\\" % (lab, " & ".join(ce(V[k][r]["veh_pooled"]) for k in cols)))
        L.append("\\multicolumn{5}{l}{\\emph{Main rule, two more vehicle comparisons}} \\\\")
        L.append("\\quad Assistant era, price left after the prediction from earlier vehicles, seller fixed effects & %s \\\\" % " & ".join(ce(u[k]["veh_residual"]) for k in cols))
        L.append("\\quad Pooled, log Texas price held fixed & -- & %s \\\\" % " & ".join(
            ce(u[k]["veh_tx_control"]) for k in cols[1:]))
        L.append("\\midrule")
        L.append("Vehicles against all other listings (AI-written) & %s \\\\" % " & ".join(
            "%s (%d)" % ("{:,}".format(u[k]["veh_pooled"]["n"]), u[k]["n_ai_vehicles_tool_era"]) for k in cols))
        L.append("Median closing price as a share of the Texas price & -- & %s \\\\" % " & ".join(
            "%.1f\\%%" % (100 * u[k]["veh_tx"]["median_ratio"]) for k in cols[1:]))
        L.append("\\multicolumn{5}{l}{\\footnotesize Texas rows: vehicles with 10 or more Texas sales of their make, model and year within 3 months; --: not run on all priced vehicles; AI-written counted before dropping groups that hold a single listing.} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(out, "tab_wrecks.tex"), "w").write("\n".join(L) + "\n")
    print("wrote tables to %s (pooled scope %s; paper_numbers %s)" % (out, "tool" if tool else "any", "used" if PN else "absent"))


SHORT = {"73811655": "1996 Seagrave aerial fire truck", "80638514": "2015 Gillig 30-foot diesel bus",
         "79932209": "2009 Dodge Charger", "76019965": "Ten desk chairs", "73216929": "2006 Chevy 3500 rack-body pickup",
         "80984472": "Confined-space tripod and winch", "74057509": "John Deere 7500 fairway mower",
         "79339133": "Motorola mobile radio"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def tex_escape(t):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        t = t.replace(a, b)
    return t


def tab_examples(path, out):
    E = json.load(open(path, encoding="utf-8"))["examples"]
    lab = {"both": "Both", "classifier only": "Classifier only", "AI marker words only": "AI marker words only", "neither": "Neither"}
    L = ["\\begin{tabular}{p{1.55cm}p{2.6cm}p{6.6cm}cp{2.0cm}c}", "\\toprule",
         "\\raggedright Tests that flag it & \\raggedright Listing (started) & \\raggedright The listing's words & Classifier & "
         "\\raggedright AI marker words & Closing price \\\\", "\\midrule"]
    prev = None
    for e in E:
        y, m = e["start"].split("-")
        mw = ", ".join("\\emph{%s}" % w for w in e["marker_words"]) or "none"
        if len(e["marker_words"]) == 1:
            mw += " (one)"
        # ragged right in the paragraph columns (justified, the narrow ones left wide gaps); the last column is not one
        L.append("\\raggedright %s & \\raggedright %s (%s %s) & \\raggedright ``%s'' & %.2f & \\raggedright %s & \\$%s (%d %s) \\\\" % (
            lab[e["outcome"]] if e["outcome"] != prev else "", SHORT[e["item"]], MONTHS[int(m) - 1], y, tex_escape(e["excerpt"]),
            e["score"], mw, "{:,}".format(e["price"]), e["bids"], "bid" if e["bids"] == 1 else "bids"))
        prev = e["outcome"]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_examples.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote tab_examples (%d listings)" % len(E))


if __name__ == "__main__":
    if sys.argv[1] == "--examples":
        tab_examples(sys.argv[2], sys.argv[3])
        sys.exit(0)
    main(*sys.argv[1:])
