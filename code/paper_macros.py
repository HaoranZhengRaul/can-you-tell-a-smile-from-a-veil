#!/usr/bin/env python3
"""Session 9: every estimate the paper's text quotes, written as LaTeX macros (paper/numbers.tex).

Reads report_inputs/ only: report_data.json (job 817678), hedonic_<rule>_tool.json when it exists (pooled
designs with AI = flagged and started in the tool era; Session 9) else hedonic_<rule>.json (jobs
820155-820157), paper_constants.json (logged values with sources) and paper_numbers.json when it exists.
Re-run after any input changes; the text then updates everywhere at once.
Usage: python3 paper_macros.py REPORT_INPUTS_DIR OUT_TEX
"""
import csv
import json
import math
import os
import sys


def r2s(x, d):
    """Round half up on the decimal value as stored (0.235 -> 0.24), not on its binary approximation."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def sg(x, d=1):
    """Signed number with a math minus, e.g. +1.2 or $-$1.2."""
    s = r2s(abs(x), d)
    if float(s) == 0:
        return s
    return ("+" if x > 0 else "\\ensuremath{-}") + s


def un(x, d=1):
    s = r2s(abs(x), d)
    return ("\\ensuremath{-}" if x < 0 and float(s) != 0 else "") + s


def thou(n):
    return "{:,}".format(int(n))


def pexp(b, se):
    return 100 * (math.exp(b) - 1), 100 * (math.exp(b - 1.96 * se) - 1), 100 * (math.exp(b + 1.96 * se) - 1)


def pci_fmt(e):
    """Percent change with its 95% interval, e.g. +4.1% [-18.4%, +32.7%], and whether the interval excludes zero."""
    p_, lo_, hi_ = pexp(e["b"], e["se"])
    return "%s\\%% [%s\\%%, %s\\%%]" % (sg(p_), sg(lo_), sg(hi_)), lo_ > 0 or hi_ < 0


def phi_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def power(delta, se):
    """Power of a two-sided 5% z-test of zero when the true effect is delta (normal approximation)."""
    z = delta / se
    return phi_cdf(z - 1.96) + phi_cdf(-z - 1.96)


def wilson_hi(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def wilson_ci(k, n, z=1.96):
    """Two-sided Wilson interval for k of n."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def lam_pooled(s, f, phi, n0, n1, n2, k0, k1, k2, tool):
    """Attenuation P(A|AI group) - P(A|comparison) in a pooled design whose comparison group is every unflagged
    listing of all periods; with tool=False the AI group also holds the k0 + k1 pre-tool flags (taken as human)."""
    pi = (phi - f) / (s - f)
    ppv2 = s * pi / phi
    pa = (1 - s) * pi * n2 / ((n0 - k0) + (n1 - k1) + (n2 - k2))
    return (ppv2 - pa) if tool else (ppv2 * k2 / (k0 + k1 + k2) - pa)


def pct_of_base(units, est_units, base):
    """A predicted gap as a percent change of its outcome: a percent as is; percentage points over the share base;
    a count over its mean (Session 12, the abstract's bound)."""
    if units == "pct":
        return est_units
    if units == "pp":
        return est_units / (100.0 * base) * 100.0
    return est_units / base * 100.0


def check_below(changes, bound):
    """The abstract's "by less than bound%": every change strictly inside the bound, or the run stops."""
    bad = [c for c in changes if not abs(c) < bound]
    if bad:
        raise AssertionError("a predicted gap breaks the abstract's bound of %s%%: %s" % (bound, bad))


def check_more_often(b, lo, hi):
    """The abstract's "did so ... more often" with its interval: the gap is above zero and the interval holds zero (a
    null), or the run stops and the sentence must change (Session 12, the PI's read of the abstract)."""
    if not (b > 0 and lo < 0 < hi):
        raise AssertionError("the abstract's bid sentence no longer fits: gap %s, interval [%s, %s]" % (b, lo, hi))


def ordinal(n):
    """An English ordinal for a whole number (22 -> "22nd"); a number that is not whole stops the run (Session 13: the
    introduction says the first stated problem comes after the 22nd word, so the median must be a whole word count)."""
    if n != int(n):
        raise AssertionError("an ordinal needs a whole number, got %s" % n)
    n = int(n)
    suf = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return "%d%s" % (n, suf)


def selftest():
    """Failing-world checks: each line fails if the formula is wrong (one-sided test, a missing term)."""
    assert abs(power(0, 1) - 0.05) < 1e-4, "power at zero must be the test size"
    assert abs(power(2.8, 1) - 0.80) < 0.005, "an effect of 2.8 standard errors must have 80% power"
    assert abs(power(-2.8, 1) - power(2.8, 1)) < 1e-12, "two-sided power is symmetric"
    assert abs(wilson_hi(0, 10) - 0.2775) < 1e-3, "Wilson upper bound for 0 of 10"
    assert abs(wilson_hi(50, 100) - 0.5964) < 1e-3, "Wilson upper bound for 50 of 100"
    assert abs(lam_pooled(1, 0, 0.2, 1000, 500, 1000, 0, 0, 200, True) - 1) < 1e-12, "a perfect detector does not attenuate"
    assert abs(lam_pooled(1, 0, 0.2, 1000, 500, 1000, 100, 100, 200, False) - 0.5) < 1e-12, "half the AI group human: factor 0.5"
    assert lam_pooled(0.7, 0.01, 0.2, 1000, 500, 1000, 10, 5, 200, True) < 1, "missed AI text in the comparison group attenuates"
    assert abs(pct_of_base("pp", 0.5, 0.5) - 1.0) < 1e-12, "half a point on a 50% base is a 1% change"
    assert abs(pct_of_base("count", 0.03, 1.5) - 2.0) < 1e-12, "0.03 questions on a mean of 1.5 is a 2% change"
    assert abs(pct_of_base("pct", -2.5, 9.9) + 2.5) < 1e-12, "a percent stays a percent"
    check_below([4.97, -4.99, 0.0], 5.0)
    for planted in ([4.6, 5.0], [-5.2, 1.0]):
        try:
            check_below(planted, 5.0)
        except AssertionError:
            continue
        raise RuntimeError("a change at or beyond the bound must stop the run: %s" % planted)
    check_more_often(0.012, -0.049, 0.074)
    for planted in ((-0.01, -0.06, 0.04), (0.05, 0.01, 0.09), (0.0, -0.05, 0.05)):
        try:
            check_more_often(*planted)
        except AssertionError:
            continue
        raise RuntimeError("a gap at or below zero, or an interval without zero, must stop the run: %s" % (planted,))
    for n_, want in ((1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"),
                     (21, "21st"), (22, "22nd"), (72, "72nd"), (101, "101st"), (111, "111th"), (112, "112th")):
        assert ordinal(n_) == want, (n_, ordinal(n_), want)
    try:
        ordinal(21.5)
    except AssertionError:
        pass
    else:
        raise RuntimeError("a median of 21.5 words must not become an ordinal")


def main(inp, out):
    selftest()
    J = lambda f: json.load(open(os.path.join(inp, f)))
    R = J("report_data.json")
    K = J("paper_constants.json")
    tool = all(os.path.exists(os.path.join(inp, "hedonic_%s_tool.json" % r)) for r in ("union", "lex", "c1"))
    H = {r: J(("hedonic_%s_tool.json" if tool else "hedonic_%s.json") % r) for r in ("union", "lex", "c1")}
    PN = J("paper_numbers.json") if os.path.exists(os.path.join(inp, "paper_numbers.json")) else None
    M = []

    def mac(name, val, why=""):
        M.append("\\newcommand{\\%s}{%s}%s" % (name, val, ("  %% " + why) if why else ""))

    M.append("%% Generated by code/paper_macros.py; do not edit by hand. Pooled designs: %s." %
             ("AI-written = flagged and started in the tool era (hedonic_<rule>_tool.json)" if tool else
              "Session 8 runs, AI = flagged in any period (hedonic_<rule>.json); re-run pending"))
    mac("pooledScope", "tool" if tool else "any")
    # printed in every build (not a hideable note): until the re-run, no copy may describe these estimates as tool-era ones
    mac("scopeNote", "" if tool else "{\\color{BrickRed}[Pooled estimates: Session 8 run, where a flagged pre-tool listing also counts as AI-written; tool-era re-run pending.]}")
    # ---- sample and setting
    S = K["sample"]
    mac("nPages", thou(K["parse_audit"]["pages"]))
    mac("nListings", thou(K["parse_audit"]["listings"]))
    mac("nNonListings", thou(K["parse_audit"]["non_listings"]))
    mac("nAgenciesAll", thou(K["parse_audit"]["agencies_all"]))
    mac("nAnalysis", thou(S["analysis_listings"]))
    mac("nAnalysisAg", thou(S["analysis_agencies"]))
    mac("nPriced", thou(S["priced"]))
    mac("nPzero", thou(S["periods"]["P0"]))
    mac("nPone", thou(S["periods"]["P1"]))
    mac("nPtwo", thou(S["periods"]["P2"]))
    # ---- adoption
    for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
        sp = R["rules"][r]["share_by_period"]
        mac("shareP" + tag + "zero", "%.1f\\%%" % (100 * sp["P0"]))
        mac("shareP" + tag + "one", "%.1f\\%%" % (100 * sp["P1"]))
        mac("shareP" + tag + "two", "%.1f\\%%" % (100 * sp["P2"]))
        mac("nAI" + tag, thou(R["rules"][r]["p2_ai"]))
        mac("agUsing" + tag, thou(R["rules"][r]["p2_agencies_using"]))
    # assistant-era flags that only the AI marker words raise (the main rule is the classifier OR the AI marker words, so these
    # are union minus classifier; Session 10 review: two in five)
    lo_only = R["rules"]["union"]["p2_ai"] - R["rules"]["c1"]["p2_ai"]
    assert 0 < lo_only <= R["rules"]["lex"]["p2_ai"]
    mac("lexOnlyN", thou(lo_only), "assistant-era flags from the AI marker words alone")
    mac("lexOnlyPct", "%d\\%%" % round(100 * lo_only / R["rules"]["union"]["p2_ai"]), "of the main rule's assistant-era flags")
    u = R["rules"]["union"]
    mac("agToolEra", thou(u["p2_agencies"]))
    mac("agUsingPct", "%d\\%%" % round(100 * u["p2_agencies_using"] / u["p2_agencies"]))
    mo = u["monthly"]
    pk = max(mo, key=lambda k: mo[k]["ai"])
    names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    mac("peakShare", "%.1f\\%%" % (100 * mo[pk]["ai"]))
    mac("peakN", thou(mo[pk]["n"]))
    mac("peakMonth", "%s %s" % (names[int(pk[5:7]) - 1], pk[:4]))
    for k, nm in (("2025-03", "Mar"), ("2025-04", "Apr"), ("2025-05", "May"), ("2025-07", "Jul"), ("2025-08", "Aug"), ("2025-09", "Sep"),
                  ("2025-10", "Oct"), ("2025-11", "Nov"), ("2025-12", "Dec"), ("2026-01", "JanSix"), ("2026-02", "FebSix")):
        mac("share" + nm, "%.1f\\%%" % (100 * mo[k]["ai"]))
    A = K["adoption"]
    mac("agJune", thou(A["agencies_june_2025"]))
    mac("stayers", thou(A["stayers_used_early"]))
    mac("stayersStopped", thou(A["stayers_stopped"]))
    mac("stayersUse", "%d\\%% to %d\\%%" % (round(100 * A["stayers_use"][0]), round(100 * A["stayers_use"][1])))
    mac("fallFrom", "%.1f\\%%" % (100 * A["fall_windows"]["may_aug_2025"]))
    mac("fallTo", "%.1f\\%%" % (100 * A["fall_windows"]["oct2025_feb2026"]))
    q = A["ever_use_by_pretool_length_quartile"]
    mac("everUseQuartiles", " / ".join("%d\\%%" % round(100 * x) for x in q))
    f5 = A["ai_share_by_opening_bid_fifth"]
    mac("shareTopFifth", "%d\\%%" % round(100 * f5[-1]))
    mac("shareBottomFifth", "%d\\%%" % round(100 * f5[0]))
    T = K["tool_dating"]
    mac("janPages", "%d of %d" % tuple(T["jan2025_pages_with_code"]))
    mac("febPages", "%d of %d" % tuple(T["feb2025_pages_with_code"]))
    bdp = os.path.join(inp, "button_dating.json")
    if os.path.exists(bdp):
        # the monthly series behind the word-sized line (code/button_dating.py, job 874348); its January and February
        # counts must equal Session 8's tool_dating.py, an independently written count over the same pages
        BD = J("button_dating.json")
        BM = BD["months"]
        assert [BM["2025-01"]["box"], BM["2025-01"]["pages"]] == T["jan2025_pages_with_code"], "January counts disagree"
        assert [BM["2025-02"]["box"], BM["2025-02"]["pages"]] == T["feb2025_pages_with_code"], "February counts disagree"
        full = [m for m in BM if BM[m]["pages"] >= 10]
        pre = [m for m in full if m < "2025-01"]
        post = [m for m in full if m >= "2025-02"]
        assert all(BM[m]["box"] == 0 for m in BM if m < "2025-01"), "a page captured before 2025 carries the box"
        lo = min(BM[m]["box"] / BM[m]["pages"] for m in post)
        hi = max(BM[m]["box"] / BM[m]["pages"] for m in post)
        mac("btnAfterLo", "%d" % math.floor(100 * lo))
        mac("btnAfterHi", "%d" % math.floor(100 * hi))
        mac("btnPreMonths", "%d" % len(pre))
        mac("btnPrePages", thou(sum(BM[m]["pages"] for m in BM if m < "2025-01")))
        from datetime import datetime
        t1 = datetime.strptime(BD["first_with_box"]["capture_ts"], "%Y%m%d%H%M%S")
        t0 = datetime.strptime(BD["last_before_first"]["capture_ts"], "%Y%m%d%H%M%S")
        hrs = (t1 - t0).total_seconds() / 3600
        assert t0.date() == t1.date() and 0 < hrs < 24
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
        mac("btnGapHours", words[int(hrs)] if hrs < 13 else "%d" % int(hrs))
        mac("btnStale", "%d of %s" % (BD["n_without_box_from_2025-02"], thou(sum(BM[m]["pages"] for m in BM if m >= "2025-02"))))
    # ---- detector
    D = K["detector"]
    for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
        sp = R["rules"][r]["share_by_period"]
        mac("fpr" + tag + "Pre", ("%." + str(2 if sp["P0"] < 0.005 else 1) + "f\\%%") % (100 * sp["P0"]))
        mac("fpr" + tag + "Oot", ("%." + str(2 if sp["P1"] < 0.005 else 1) + "f\\%%") % (100 * sp["P1"]))
    # ChatGPT-era use bound: P1 share minus P0 share (main rule), 95% upper bound, divided by an assumed recall
    sp, nb = R["rules"]["union"]["share_by_period"], R["rules"]["union"]["n_by_period"]
    p0, p1, n0, n1 = sp["P0"], sp["P1"], nb["P0"], nb["P1"]
    dif = p1 - p0
    se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    up = dif + 1.96 * se
    mac("chatDiff", "%s~percentage points [%s, %s]" % (sg(100 * dif, 2), sg(100 * (dif - 1.96 * se), 2), sg(100 * up, 2)))
    for nm, rc in (("chatBoundMain", 0.73), ("chatBoundHalf", 0.5), ("chatBoundThird", 0.3)):
        mac(nm, "%s\\%%" % r2s(100 * up / rc, 1))
    # bound that assumes nothing about the period's false-positive rate: flags = false positives + caught AI text,
    # so use <= (flag share) / recall; flag count recovered from the 4-decimal share and checked to be unique
    k1 = int(round(p1 * n1))
    assert [k for k in (k1 - 1, k1, k1 + 1) if round(k / n1, 4) == round(p1, 4)] == [k1], "P1 flag count not unique"
    up1 = wilson_hi(k1, n1)
    mac("fprOotUp", "%s\\%%" % r2s(100 * up1, 1), "95%% Wilson upper bound, %d of %d" % (k1, n1))
    # the period's false-positive rate f = (share - recall * use) / (1 - use) falls as use rises (share < recall),
    # so its cap is the share itself, at use = 0: \fprOotUp (review round 4 corrected an earlier, looser cap)
    mac("chatBoundFree", "%s\\%%" % r2s(100 * up1 / 0.73, 1))
    mac("chatBoundFreeThird", "%s\\%%" % r2s(100 * up1 / 0.3, 1))
    mac("aucClf", "%.3f" % D["auc_crossfit"])
    mac("detTau", "%.3f" % D["tau"])
    rc = PN["recall_rewrites"] if PN else None
    same = bool(PN and PN.get("checks", {}).get("B_detector_rebuild", {}).get("identical"))
    mac("recallClfAll", "%.2f" % (rc["c1"]["all"]["recall"] if (rc and same) else D["recall_rewrites_c1"]["all"]))
    mac("recallUnionRw", ("%.2f" % rc["union"]["all"]["recall"]) if (rc and same) else ("$\\geq$ %.2f" % D["recall_rewrites_c1"]["all"]))
    mac("recallLexRw", ("%.2f" % rc["lex"]["all"]["recall"]) if rc else "\\textbf{[pending]}")
    mac("aboveOwn", "%.1f\\%%" % (100 * D["rewrites_above_own_original"]))
    br = D["blind_read"]
    mac("blindRecall", "%.2f" % br["recall_vs_reader1"]["union"])
    mac("blindRecallTwo", "%d of %d" % tuple(br["recall_union_vs_reader2"]))
    mac("blindRecallOne", "%d of %d" % tuple(br["recall_union_vs_reader1"]))
    mac("kappaThree", "%s [%s, %s]" % tuple(r2s(x, 2) for x in br["kappa3"]))
    mac("kappaTwo", "%s [%s, %s]" % tuple(r2s(x, 2) for x in br["kappa_ai_vs_not"]))
    mac("blindAgree", str(br["agree_of_100"]))
    mac("blindAUC", "%.3f" % br["score_auc_vs_reader1_A_vs_H"])
    # ---- what changes in the text (tool era, within agency x subcategory x month, main rule)
    c = u["content"]
    mac("nContent", thou(c["s_w"]["n"]))
    mac("nContentAg", thou(c["s_w"]["clusters"]))
    mac("wordsDiff", sg(c["s_w"]["b"], 0))
    mac("wordsBase", "%.0f" % c["s_w"]["human_mean"])
    mac("wordsTimes", "%.1f" % math.exp(c["log_w"]["b"]))
    mac("wordsLog", r2s(100 * c["log_w"]["b"], 0))
    mac("wordsT", "%.1f" % (c["log_w"]["b"] / c["log_w"]["se"]))
    # Figure 2 shows counts per listing, so the text quotes the word gap in words (s_w), not as a ratio
    mac("wordsLevT", "%.1f" % (c["s_w"]["b"] / c["s_w"]["se"]))
    mac("promoClfDiff", sg(R["rules"]["c1"]["content"]["s_n_puff"]["b"], 2))
    for key, nm, d in (("s_n_puff", "promo", 2), ("s_n_you", "you", 2), ("s_n_label", "label", 1), ("s_n_num", "num", 1),
                       ("s_n_unit", "unit", 2), ("s_n_defect_strict", "def", 2)):
        mac(nm + "Diff", sg(c[key]["b"], d))
        mac(nm + "Base", "%.2f" % c[key]["human_mean"])
        mac(nm + "T", "%.1f" % (c[key]["b"] / c[key]["se"]))
    mac("promoTimes", "%.0f" % ((c["s_n_puff"]["human_mean"] + c["s_n_puff"]["b"]) / c["s_n_puff"]["human_mean"]))
    for key, nm in (("has_quality", "qual"), ("defect_and_praise", "dp")):
        mac(nm + "PP", sg(100 * c[key]["b"], 1))
        mac(nm + "Base", "%.1f\\%%" % (100 * c[key]["human_mean"]))
        mac(nm + "T", "%.1f" % (c[key]["b"] / c[key]["se"]))
    cl = R["rules"]["lex"]["content"]
    mac("dpLexPP", sg(100 * cl["defect_and_praise"]["b"], 1))
    mac("dpLexT", "%.1f" % (cl["defect_and_praise"]["b"] / cl["defect_and_praise"]["se"]))
    mac("numLexDiff", sg(cl["s_n_num"]["b"], 1))
    mac("numLexBase", "%.1f" % cl["s_n_num"]["human_mean"])
    # ---- controlled rewrites
    ra = K["rewrite_audit"]
    rwc = PN["rewrites"]["all"] if PN else None
    mac("rwNumKept", "%.1f\\%%" % (100 * (rwc["numbers_kept"]["mean"] if rwc else ra["numbers_kept"])))
    mac("rwDefKept", "%.1f\\%%" % (100 * (rwc["defects_kept"]["mean"] if rwc else ra["defects_kept"])))
    mac("rwQual", "%.1f\\%%" % (100 * ra["adds_quality_claim"]))
    mac("rwWords", "%.1f" % ra["words_ratio"])
    bg = ra["by_generator"]
    mac("rwWordsConcise", "%.1f" % min(bg["concise|opus"]["words_ratio"], bg["concise|sonnet"]["words_ratio"]))
    mac("rwWordsMax", "%.1f" % max(v["words_ratio"] for v in bg.values()))
    # Session 13 (PI: "just choose one number"): the 200 "concise" rewrites pooled, as a count and a share
    kq_ = [bg[g]["adds_quality"] * bg[g]["n"] for g in ("concise|opus", "concise|sonnet")]
    assert all(abs(x - round(x)) < 1e-9 for x in kq_), kq_
    nq_ = bg["concise|opus"]["n"] + bg["concise|sonnet"]["n"]
    assert nq_ == 200, nq_
    mac("rwQualConciseK", "%d" % round(sum(kq_)), "concise rewrites with a new quality phrase, of 200")
    mac("rwQualConcisePct", "%s\\%%" % r2s(100.0 * round(sum(kq_)) / nq_, 1))
    mac("rwQualAppealing", "%d--%d\\%%" % (round(100 * min(bg[k]["adds_quality"] for k in bg if k.startswith("appealing"))),
                                         round(100 * max(bg[k]["adds_quality"] for k in bg if k.startswith("appealing")))))
    B = K["benchmark_rewrites"]
    mac("bench", "+%s" % r2s(100 * B["b"], 1))
    mac("benchCI", "[+%s, +%s]" % (r2s(100 * B["lo"], 1), r2s(100 * B["hi"], 1)))
    mac("benchCIPct", "[%s\\%%, %s\\%%]" % (sg(100 * (math.exp(B["lo"]) - 1), 1), sg(100 * (math.exp(B["hi"]) - 1), 1)), "the interval in percent")
    mac("benchPct", "%.1f\\%%" % (100 * (math.exp(B["b"]) - 1)))
    mac("benchTab", "%s [%s, %s]" % (sg(100 * (math.exp(B["b"]) - 1), 1), sg(100 * (math.exp(B["lo"]) - 1), 1), sg(100 * (math.exp(B["hi"]) - 1), 1)),
        "the premium for a table in percent, unit in the row label")
    mac("benchShare", "%d\\%%" % round(100 * B["share_raising"]))
    # the expected price before ChatGPT for vehicles (PI, 2026-10-31: "Just say, expected price"): the human written median
    # of the main vehicle design raised by the frozen model's premium for a rewrite; and the frozen model's training count
    vref = H["union"]["veh_pooled"]["ref"]["median_price"]
    mac("vehHumanMedian", "\\$%s" % thou(round(vref)), "human-written median closing price, main vehicle design")
    mac("vehExpected", "\\$%s" % thou(round(vref * math.exp(B["b"]))), "that median raised by the frozen model's premium for a rewrite")
    mac("benchFitN", thou(B["n_fit_p0"]), "priced pre-ChatGPT listings the frozen model is fit on (the 600 rewritten ones left out)")
    g = B["by_generator"]
    mac("benchAppealing", "+%s to +%s" % (r2s(100 * min(g[k] for k in g if k.startswith("appealing")), 1), r2s(100 * max(g[k] for k in g if k.startswith("appealing")), 1)))
    mac("benchConcise", "+%s to +%s" % (r2s(100 * min(g[k] for k in g if k.startswith("concise")), 1), r2s(100 * max(g[k] for k in g if k.startswith("concise")), 1)))
    for nm_, pre_ in (("benchAppealingPct", "appealing"), ("benchConcisePct", "concise")):
        vv = [g[k] for k in g if k.startswith(pre_)]
        mac(nm_, "+%s\\%% to +%s\\%%" % (r2s(100 * (math.exp(min(vv)) - 1), 1), r2s(100 * (math.exp(max(vv)) - 1), 1)), "the premium in percent")
    # Session 13 (PI: one number, not a range by model): the premium per prompt, the per-model means weighted by the design's
    # rewrites per model (Table 2: Opus 200, Sonnet 100, Haiku 100 under "appealing"; Opus 100, Sonnet 100 under "concise");
    # the weighted mean of all five must give back the pooled premium
    w_ = {"appealing|opus": 200, "appealing|sonnet": 100, "appealing|haiku": 100, "concise|opus": 100, "concise|sonnet": 100}
    assert set(w_) == set(g), sorted(g)
    assert abs(sum(w_[k] * g[k] for k in w_) / 600 - B["b"]) < 0.0015, (sum(w_[k] * g[k] for k in w_) / 600, B["b"])
    for nm_, pre_ in (("benchAppPoolPct", "appealing"), ("benchConPoolPct", "concise")):
        ks_ = [k for k in w_ if k.startswith(pre_)]
        m_ = sum(w_[k] * g[k] for k in ks_) / sum(w_[k] for k in ks_)
        mac(nm_, "%s\\%%" % sg(100 * (math.exp(m_) - 1), 1), "the premium for this prompt's rewrites")
    # ---- buyer response, tool era only (analysis_core B, same agency x subcategory x month)
    mk = u["market"]
    e = mk["sold"]["subcat_m"]
    mac("tSold", "%s~percentage points [%s, %s]" % (sg(100 * e["b"]), sg(100 * (e["b"] - 1.96 * e["se"])), sg(100 * (e["b"] + 1.96 * e["se"]))))
    # Session 12, the abstract (PI, 2026-11-20): the same gap with its point and interval apart, for "did so 1.2
    # percentage points more often (95% CI [-4.9, +7.4])"; the sentence holds only for a positive gap
    # whose interval contains zero
    check_more_often(e["b"], e["b"] - 1.96 * e["se"], e["b"] + 1.96 * e["se"])
    mac("tSoldPt", r2s(100 * e["b"], 1), "the assistant-era gap in received a bid, percentage points (abstract)")
    mac("tSoldCI", "[%s, %s]" % (sg(100 * (e["b"] - 1.96 * e["se"])), sg(100 * (e["b"] + 1.96 * e["se"]))),
        "its 95% CI, percentage points (abstract)")
    # Session 13 (PI, 2026-11-20: "Yes, keep it"): the bound on falls, led by its direction and sized by this design; the
    # largest fall each assistant-era interval allows, as a positive number that must match the printed lower end
    lo_t = 100 * (e["b"] - 1.96 * e["se"])
    assert lo_t < 0, "the assistant-era interval for received a bid no longer allows a fall"
    assert sg(lo_t) == "\\ensuremath{-}" + r2s(-lo_t, 1)
    mac("tSoldFallMax", "%s~percentage points" % r2s(-lo_t, 1), "the largest fall in received a bid the assistant-era interval allows")
    for key, nm in (("log_price", "tPrice"), ("log_bids", "tBids"), ("log_views", "tViews")):
        e = mk[key]["subcat_m"]
        p, lo, hi = pexp(e["b"], e["se"])
        mac(nm, "%s\\%% [%s\\%%, %s\\%%]" % (sg(p), sg(lo), sg(hi)))
        if nm == "tPrice":
            # Session 13 (PI, 2026-11-20): "3.8% higher on average, but not significant", and indistinguishable from the
            # swayed buyer's premium; the introduction says both, so both are checked here
            bench_ = 100 * (math.exp(B["b"]) - 1)
            assert p > 0 and lo < 0 < hi and lo < bench_ < hi, "the assistant-era price interval no longer holds both zero and the premium"
            mac("tPricePct", "%s\\%%" % r2s(p, 1), "the assistant-era closing-price gap in percent (introduction)")
            mac("tPriceCIPct", "[%s\\%%, %s\\%%]" % (sg(lo), sg(hi)), "its 95% CI in percent, for (95% CI ..., p = ...)")
        if nm in ("tBids", "tViews"):
            assert lo < 0 and sg(lo) == "\\ensuremath{-}" + r2s(-lo, 1), nm
            mac(nm + "FallMax", "%s\\%%" % r2s(-lo, 1), "the largest fall the assistant-era interval allows (Session 13)")
    # Session 13 (PI, 2026-11-22: "maybe just a p-value ... everywhere"): two-sided normal p-values of the assistant-era gaps
    def pfmt(p_):
        return "<0.001" if p_ < 0.001 else ("%.3f" % p_ if p_ < 0.1 else "%.2f" % p_)
    for key, nm in (("sold", "tSoldP"), ("log_price", "tPriceP"), ("log_bids", "tBidsP"), ("log_views", "tViewsP"),
                    ("n_q_before_end", "tQP")):
        e = mk[key]["subcat_m"]
        mac(nm, pfmt(math.erfc(abs(e["b"] / e["se"]) / math.sqrt(2))), "two-sided p of the assistant-era gap, %s" % key)
    e = mk["n_q_before_end"]["subcat_m"]
    mac("tQ", "%s [%s, %s] on a base of %.2f" % (sg(e["b"], 2), sg(e["b"] - 1.96 * e["se"], 2), sg(e["b"] + 1.96 * e["se"], 2), e["mean"]))
    # same agency, same start day (paper_numbers.py block H): a sentence in Section 5 once it exists, a note until then
    SDm = PN.get("same_day") if PN else None
    if SDm and "error" not in SDm and SDm["union"].get("base|sold"):
        e, p_ = SDm["union"]["base|sold"], SDm["union"].get("base|log_price")
        pr = (", and price %s\\%% [%s\\%%, %s\\%%]" % tuple(sg(x) for x in pexp(p_["b"], p_["se"]))) if p_ else ""
        mac("sameDay", " A third design, the \\emph{same-day design}, compares assistant-era listings that an agency started on the same day (%s agency-days with "
            "both kinds of listing, an effect for each agency-day): received a bid %s~percentage points [%s, %s]%s (Appendix Table~\\ref{tab:outcomes})." % (
                thou(SDm["union"]["batches"]), sg(100 * e["b"]), sg(100 * (e["b"] - 1.96 * e["se"])), sg(100 * (e["b"] + 1.96 * e["se"])), pr))
        mac("sameDayPending", "")
    else:
        mac("sameDay", "")
        mac("sameDayPending", "\\note{Same agency, same start day: Session 8 ran this design (70 agency-days with both kinds of "
            "listing under the main rule; questions $-$0.06) without saving intervals; paper\\_numbers.py block H re-runs it "
            "for every outcome and definition.}")
    mac("nTool", thou(mk["sold"]["subcat_m"]["n"]))
    mac("nToolAg", thou(mk["sold"]["subcat_m"]["clusters"]))
    # ---- buyer response, pooled 2019-26 (price_hedonics)
    h = H["union"]
    e = h["pooled_sold"]
    mac("pSold", "%s~percentage points [%s, %s]" % (sg(100 * e["b"]), sg(100 * (e["b"] - 1.96 * e["se"])), sg(100 * (e["b"] + 1.96 * e["se"]))))
    # bases are the reference bar, assistant-era listings the rule does not flag (price_hedonics.py ref_level; Session 9);
    # required (Session 12: the fallback to the whole sample's mean is removed)
    mac("pSoldBase", "%.0f\\%%" % (100 * e["ref"]["mean"]))
    # Session 12, buyers first: the largest fall each interval allows, as a positive number ("rules out a fall of more than")
    lo_ = 100 * (e["b"] - 1.96 * e["se"])
    assert lo_ < 0
    mac("pSoldFallMax", "%s~percentage points" % r2s(-lo_, 1), "the largest fall in received a bid the interval allows")
    mac("nPooled", thou(e["n"]))
    mac("nPooledAg", thou(e["clusters"]))
    for key, nm in (("pooled_log_bids", "pBids"), ("pooled_log_views", "pViews"), ("veh_pooled", "pVeh"), ("veh_residual", "pVehResid"),
                    ("all_pooled", "pAll"), ("all_pooled_start", "pAllStart")):
        e = h[key]
        p, lo, hi = pexp(e["b"], e["se"])
        mac(nm, "%s\\%% [%s\\%%, %s\\%%]" % (sg(p), sg(lo), sg(hi)))
        mac(nm + "Hi", "%s\\%%" % sg(hi))
        if lo < 0:
            mac(nm + "FallMax", "%s\\%%" % r2s(-lo, 1), "the largest fall the interval allows (Session 12)")
    e = h["pooled_n_q_before_end"]
    # the base is Figure 5's reference bar, as for \pSoldBase (2026-11-01; it had been the whole pooled sample's 1.21)
    mac("pQ", "%s [%s, %s] on a base of %.2f" % (sg(e["b"], 2), sg(e["b"] - 1.96 * e["se"], 2), sg(e["b"] + 1.96 * e["se"], 2),
                                              e["ref"]["mean"]))
    mac("pQHi", sg(e["b"] + 1.96 * e["se"], 2), "upper end of the questions interval (Session 12)")
    mac("pQBase", "%.2f" % e["ref"]["mean"])
    e = h["veh_pooled"]
    med = e["ref"]["median_price"]
    p, lo, hi = pexp(e["b"], e["se"])
    mac("pVehDollar", "%s [%s, %s] on a vehicle at the %s of \\$%s" % (
        ("+\\$" if p >= 0 else "\\ensuremath{-}\\$") + thou(abs(round(med * p / 100))), ("\\ensuremath{-}\\$" if lo < 0 else "+\\$") + thou(abs(round(med * lo / 100))),
        ("+\\$" if hi >= 0 else "\\ensuremath{-}\\$") + thou(abs(round(med * hi / 100))),
        "median price", thou(round(med))))
    mac("pVehMDE", "%d\\%%" % round(100 * (math.exp(2.8 * e["se"]) - 1)))
    mac("nVeh", thou(e["n"]))
    mac("nVehAg", thou(e["clusters"]))
    mac("vehWithinR", "%.2f" % e["within_r2"])
    mac("nVehPriced", thou(h["n_vehicles_priced"]))
    mac("nVehAI", thou(h["n_ai_vehicles_tool_era"]))
    mac("vehOotR", "%d\\%%" % round(100 * h["hedonic_out_of_time_r2"]))
    e = h["all_pooled_start"]
    mac("pAllMDE", "%d\\%%" % round(100 * (math.exp(2.8 * e["se"]) - 1)))
    mac("vehToolEraNote", "" if "veh_tool_era" in h else
        " A third vehicle design, assistant-era listings only with the full vehicle controls, has a standard error of 0.31 log"
        " points (a 95\\% interval of about $\\pm$61 log points) and is not shown.", "WORKLOG s8 section 15")
    m1, m2 = round(100 * (math.exp(2.8 * h["veh_pooled"]["se"]) - 1)), round(100 * (math.exp(2.8 * h["all_pooled_start"]["se"]) - 1))
    mac("mdePair", ("both %d\\%%" % m1) if m1 == m2 else ("%d\\%% and %d\\%%" % (m1, m2)))
    mac("pResidMDE", "%d\\%%" % round(100 * (math.exp(2.8 * h["veh_residual"]["se"]) - 1)))
    # power of each price design against the form reader's premium (two-sided 5% test, normal approximation)
    for key, nm in (("veh_pooled", "powVeh"), ("all_pooled_start", "powAll"), ("veh_residual", "powResid")):
        mac(nm, "%d\\%%" % round(100 * power(K["benchmark_rewrites"]["b"], h[key]["se"])))
    for key, nm in (("balance_goods_log_start", "balGoods"), ("balance_veh_log_start", "balVeh")):
        if key in h:
            mac(nm, "%s (standard error %s)" % (sg(100 * h[key]["b"], 0), r2s(100 * h[key]["se"], 0)))
            mac(nm + "B", sg(100 * h[key]["b"], 0), "log points")
            mac(nm + "Pct", sg(100 * (math.exp(h[key]["b"]) - 1), 0) + "\\%", "the same in percent")
            mac(nm + "T", r2s(h[key]["b"] / h[key]["se"], 2))
        else:
            mac(nm, {"balGoods": "+26 (standard error 16)", "balVeh": "+14 (standard error 22)"}[nm], "Session 8 values (WORKLOG s8 section 15)")
            mac(nm + "T", {"balGoods": "1.65", "balVeh": "0.63"}[nm])
    # ---- price information in the text and the frozen model
    I = K["inform_spreads"]
    if PN:
        sp, ra2 = PN["inform"]["spreads"], PN["inform"]["ratios"]
        I = {k: {p: [sp["%s|%s" % (k, p)][x] for x in ("mean", "lo", "hi")] for p in ("P0", "P1", "P2")} for k in ("full", "facts", "style")}
        for k in I:
            I[k]["P2/P0"] = [ra2["%s|P2/P0" % k][x] for x in ("mean", "lo", "hi")]
    b200 = os.path.join(inp, "inform_b200.json")
    if os.path.exists(b200):
        J2 = json.load(open(b200))
        sp, ra2 = J2["spreads"], J2["ratios"]
        I = {k: {p: [sp["%s|%s" % (k, p)][x] for x in ("mean", "lo", "hi")] for p in ("P0", "P1", "P2")} for k in ("full", "facts", "style")}
        for k in I:
            I[k]["P2/P0"] = [ra2["%s|P2/P0" % k][x] for x in ("mean", "lo", "hi")]
        mac("infDraws", str(sp["full|P0"]["draws"]))
        mac("infDrawsNote", "")
    else:
        mac("infDraws", "20")
        mac("infDrawsNote", "; with 20 draws these percentiles sit close to the most extreme draws")
    for k, nm in (("full", "Full"), ("facts", "Facts"), ("style", "Style")):
        for p, pn in (("P0", "Zero"), ("P1", "One"), ("P2", "Two")):
            mac("inf" + nm + pn, "%s [%s, %s]" % tuple(r2s(100 * x, 0) for x in I[k][p]))
        mac("inf" + nm + "Ratio", "%.2f [%.2f, %.2f]" % tuple(I[k]["P2/P0"]))
    mac("infFullTimes", "%.1f" % math.exp(I["full"]["P0"][0]))
    mac("infStyleTimes", "%.1f" % math.exp(I["style"]["P0"][0]), "the style-only gap before ChatGPT as a ratio")
    mac("infFactsTimes", "%.1f" % math.exp(I["facts"]["P0"][0]), "the facts-only gap before ChatGPT as a ratio")
    mac("infFullZeroPt", r2s(100 * I["full"]["P0"][0], 0))
    F = K["frozen_field"]
    for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
        for key, nm in (("s", "S"), ("r", "R"), ("s_minus_r", "D")):
            v, t = F[r][key]
            mac("fz" + tag + nm, sg(100 * v, 1))
            mac("fz" + tag + nm + "t", un(t, 2))
            mac("fz" + tag + nm + "Pct", sg(100 * (math.exp(v) - 1), 1) + "\\%", "the same gap in percent")
            if nm == "R" and not tag:
                se_ = abs(v / t)
                mac("fzRci", "[%s, %s]" % (sg(100 * (v - 1.96 * se_), 1), sg(100 * (v + 1.96 * se_), 1)), "realized gap, 95%, from b and t")
                pc_ = lambda x: sg(100 * (math.exp(x) - 1), 1) + "\\%"
                mac("fzRPctCI", "%s [%s, %s]" % (pc_(v), pc_(v - 1.96 * se_), pc_(v + 1.96 * se_)), "realized gap in percent, 95%")
    # the comparison's size (PI, 2026-10-31: "we need N here"), from the jobs' logs (paper_constants frozen_field.n)
    if "n" in F:
        mac("fzN", thou(F["n"]["union"]["listings"]), "assistant-era listings in the within-agency comparison (job 817639 log)")
        mac("fzAg", thou(F["n"]["union"]["agencies"]), "agencies in it")
    # the price-gap figure in dollars per $1,000 of human written price (the figure prints the same numbers)
    vr_, sr_ = F["union"]["r"][0], abs(F["union"]["r"][0] / F["union"]["r"][1])
    d_ = lambda x: "\\$%s" % thou(round(1000 * math.exp(x)))
    mac("gapDollar", "%s [%s, %s]" % (d_(vr_), d_(vr_ - 1.96 * sr_), d_(vr_ + 1.96 * sr_)), "AI-written closing price per $1,000 of human-written price, 95%")
    mac("gapExpDollar", d_(F["union"]["s"][0]), "the expected price before ChatGPT per $1,000 of human-written price")
    V = K["vehicles"]
    mac("vehAgePct", "%s\\%%" % sg(V["age_effect_pct_per_year"]))
    mac("vehNoStart", sg(V["does_not_start_lp"], 2))
    mac("vehNoDrive", sg(V["not_drivable_lp"], 2))
    # ---- review round 1 additions
    src = PN["monthly"] if PN else {r: R["rules"][r]["monthly"] for r in ("union", "lex", "c1")}
    first = "2019-01" if PN else "2022-06"
    plotted = [k for k in src["union"] if k >= first and src["union"][k]["n"] >= 10]
    mac("nFigAdopt", thou(sum(src["union"][k]["n"] for k in plotted)))
    mac("figAdoptFrom", "%s %s" % (names[int(min(plotted)[5:7]) - 1], min(plotted)[:4]))
    mac("figAdoptTo", "%s %s" % (names[int(max(plotted)[5:7]) - 1], max(plotted)[:4]))
    mac("febShare", "%d of %d" % (round(mo["2025-02"]["ai"] * mo["2025-02"]["n"]), mo["2025-02"]["n"]))
    # Figure 1's bars: listings both tests flag before ChatGPT. report_data.json keeps shares to 4 decimals, so each
    # count is the one whole number whose share rounds to the stored value (checked unique), as for \fprOotUp
    n0_ = R["rules"]["union"]["n_by_period"]["P0"]
    assert all(R["rules"][r]["n_by_period"]["P0"] == n0_ for r in ("lex", "c1"))
    def flagged0(r):
        s_ = R["rules"][r]["share_by_period"]["P0"]
        ks = [k for k in range(max(0, round(s_ * n0_) - 3), round(s_ * n0_) + 4) if round(k / n0_, 4) == round(s_, 4)]
        assert len(ks) == 1, "%s P0: flag count not unique %s" % (r, ks)
        return ks[0]
    b0_ = flagged0("lex") + flagged0("c1") - flagged0("union")
    assert 0 <= b0_ <= min(flagged0("lex"), flagged0("c1"))
    mac("bothPzero", thou(b0_), "of %d listings; marker %d, classifier %d, either %d" % (n0_, flagged0("lex"), flagged0("c1"), flagged0("union")))
    mac("fifthsList", ", ".join("%d\\%%" % round(100 * x) for x in f5))
    # largest decline each 95% interval allows, as a share of the human mean
    for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
        cc = R["rules"][r]["content"]
        for key, nm in (("s_n_num", "num"), ("s_n_unit", "unit"), ("s_n_defect_strict", "def")):
            e = cc[key]
            lo = (e["b"] - 1.96 * e["se"]) / e["human_mean"]
            mac(nm + tag + "MaxDecline", "%d\\%%" % round(-100 * lo) if lo < 0 else "0\\%")
            if not tag and nm in ("num", "def"):
                # the largest rise the same interval allows (Session 10 review: a null ships both ends of its bound)
                mac(nm + "MaxRise", "%d\\%%" % round(100 * (e["b"] + 1.96 * e["se"]) / e["human_mean"]))
        e = cc["s_n_puff"]
        if tag:
            mac("promo" + tag + "Times", r2s((e["human_mean"] + e["b"]) / e["human_mean"], 1))
        e = cc["defect_and_praise"]
        mac("dp" + tag + "PPx", sg(100 * e["b"], 1))
        mac("dp" + tag + "Tx", r2s(e["b"] / e["se"], 2))
    # misclassification: attenuation lambda = P(A|F) - P(A|not F) at recall s, flag share phi, false-positive rate f
    phi, f0 = u["share_by_period"]["P2"], u["share_by_period"]["P0"]

    def lam(s_):
        pi = (phi - f0) / (s_ - f0)
        return s_ * pi / phi - (1 - s_) * pi / (1 - phi), pi
    l73, pi73 = lam(0.73)
    mac("lamMain", "%d\\%%" % round(100 * (1 - l73)))
    mac("lamLo", "%d\\%%" % round(100 * (1 - lam(0.87)[0])))
    mac("lamHi", "%d\\%%" % round(100 * (1 - lam(0.52)[0])))
    mac("piMain", r2s(pi73, 2))
    mac("humanShare", "%d\\%%" % round(100 * (1 - pi73)))
    mac("lossRatio", r2s(1 - pi73, 2))
    ppv80 = 0.80
    mac("fprImplied", "%s\\%%" % r2s(100 * (1 - ppv80) * phi / (1 - ppv80 * phi / 0.73), 1))
    mac("ppvMain", r2s(0.73 * pi73 / phi, 2))
    mac("npvMain", r2s((1 - 0.73) * pi73 / (1 - phi), 2))
    pi80 = 0.80 * phi / 0.73
    mac("lamBlindPrec", "%d\\%%" % round(100 * (1 - (0.80 - (1 - 0.73) * pi80 / (1 - phi)))))
    # the form reader's premium as the flagged-versus-unflagged gap would show it: Delta_g times lambda
    bb = K["benchmark_rewrites"]["b"]
    mac("benchLam", "%.1f\\%%" % (100 * (math.exp(l73 * bb) - 1)))
    mac("benchLamRange", "%.1f\\%% to %.1f\\%%" % (100 * (math.exp(lam(0.52)[0] * bb) - 1), 100 * (math.exp(lam(0.87)[0] * bb) - 1)))
    # attenuation in the pooled designs: the comparison group is every unflagged listing of every period, so
    # P(A | not F) is small; under AI_SCOPE=all (the Session 8 run) the AI group also holds the pre-tool flags
    nbp = R["rules"]["union"]["n_by_period"]
    kk = {}
    for per in ("P0", "P1"):
        kc = int(round(u["share_by_period"][per] * nbp[per]))
        assert [k for k in (kc - 1, kc, kc + 1) if round(k / nbp[per], 4) == round(u["share_by_period"][per], 4)] == [kc], per
        kk[per] = kc
    k2, n2 = u["p2_ai"], u["p2_n"]
    assert n2 == nbp["P2"], "tool-era n differs between report_data fields"
    unfl = (nbp["P0"] - kk["P0"]) + (nbp["P1"] - kk["P1"]) + (n2 - k2)
    lp_args = (0.73, f0, phi, nbp["P0"], nbp["P1"], n2, kk["P0"], kk["P1"], k2)
    # lam_pooled() is a difference in means against every unflagged listing of every period (0.95 here). The pooled
    # regressions carry subcategory-by-quarter effects, so the flag's gap is identified within assistant-era quarters,
    # where unflagged listings are about 9% truly AI written, and agency effects absorb part of the rest of the variation
    # in true authorship: their factor is at most the assistant-era one (2026-11-01: lit_check_C's derivation and the
    # main loop's simulation, scratchpad/machine_records_s10/sim_lambda.py and WORKLOG s10 §71: 0.85 in data like ours
    # against the formula's 0.95; 0.87 when every agency adopts alike). The assistant-era factor is used as that bound.
    lam_pool = min(lam_pooled(*lp_args, tool=tool), l73)
    mac("lamPooled", r2s(lam_pool, 2), "upper bound, the assistant-era factor; AI scope %s; pre-tool flags %d, tool-era flags %d, unflagged %d" % ("tool" if tool else "any", kk["P0"] + kk["P1"], k2, unfl))
    mac("lamPooledInv", r2s(1.0 / float(r2s(lam_pool, 2)), 2), "Session 12: 1 over the printed factor; how much detector error widens a gap at least")
    LAMP = float(r2s(lam_pool, 2))
    mac("lamPooledTool", r2s(lam_pooled(*lp_args, tool=True), 2), "a difference in means against every unflagged listing of every period (not the regressions' factor)")
    mac("benchLamPooled", "%.1f\\%%" % (100 * (math.exp(lam_pool * bb) - 1)))
    mac("powVehLam", "%d\\%%" % round(100 * power(lam_pool * bb, H["union"]["veh_pooled"]["se"])))
    ev = H["union"]["veh_pooled"]
    mac("pVehHiTrue", "%d\\%%" % round(100 * (math.exp((ev["b"] + 1.96 * ev["se"]) / float(r2s(lam_pool, 2))) - 1)),
        "upper end of the vehicle interval divided by the printed detector factor, in percent (Session 12, round 3)")
    # the tool-era residual design against the benchmark: z of the gap from Delta_g, by definition
    for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
        e = H[r]["veh_residual"]
        mac("zResid" + tag, r2s((bb - e["b"]) / e["se"], 2))
    mac("zResidLam", r2s((l73 * bb - H["union"]["veh_residual"]["b"]) / H["union"]["veh_residual"]["se"], 2))
    # how many price intervals, across designs and definitions, exclude Delta_g
    pk = [k for k in ("veh_pooled", "veh_pooled_start", "veh_from_dec2022", "veh_nobody", "veh_tool_era", "all_pooled", "all_pooled_start",
                      "goods_pooled", "goods_pooled_start", "veh_residual", "veh_residual_month", "veh_residual_quarter") if k in H["union"]]
    cells = [H[r][k] for r in ("union", "lex", "c1") for k in pk] + [R["rules"][r]["market"]["log_price"]["subcat_m"] for r in ("union", "lex", "c1")]
    mac("nPriceIntervals", str(len(cells)))
    mac("nExclBench", str(sum((e["b"] + 1.96 * e["se"] < bb) or (e["b"] - 1.96 * e["se"] > bb) for e in cells)))
    # queued results switch on a sentence; until then a review note says what is queued
    if "veh_residual_month" in H["union"]:
        em = H["union"]["veh_residual_month"]
        survives = (em["b"] + 1.96 * em["se"] < bb) or (em["b"] - 1.96 * em["se"] > bb)
        for r, tag in (("union", ""), ("lex", "Lex"), ("c1", "Clf")):
            mac("pVehResidMonth" + tag, pci_fmt(H[r]["veh_residual_month"])[0])
        mac("residMonth", " With end-month effects, which remove the bunching of flags in May to August 2025, it is %s under the "
            "main rule, %s." % (pci_fmt(em)[0], "and the exclusion survives" if survives else "and the exclusion does not survive"))
        mac("residMonthPending", "")
    else:
        mac("residMonth", "")
        mac("residMonthPending", "\\note{The residual design with end-month and end-quarter effects (price\\_hedonics.py V4b, V4c) is queued with the re-run.}")
    F19 = PN.get("fpr_pre2019") if PN else None
    if F19 and "error" not in F19:
        mac("fprPreNineteen", " Listings started before 2019, which entered neither the classifier's training, its threshold nor the "
            "marker list, were flagged at %s (%d of %d)." % ("%.1f\\%%" % (100 * F19["union"]["share"]), F19["union"]["flagged"], F19["union"]["n"]))
        mac("fprPreNineteenPending", "")
    else:
        mac("fprPreNineteen", "")
        mac("fprPreNineteenPending", "\\note{Out-of-sample false positives on listings started before 2019, which never entered the "
            "detector's training, threshold or marker list, are queued (paper\\_numbers.py block I).}")
    # ---- review round 4 additions
    mac("nPreFlags", thou(kk["P0"] + kk["P1"]))
    mac("lamPooledText", ("" if tool else
        " In the run reported here, whose AI group also holds the %s pre-tool flags (taken as human), the factor is about %s; once "
        "only tool-era flags count as AI-written it is about %s." % (thou(kk["P0"] + kk["P1"]), r2s(lam_pool, 2), r2s(lam_pooled(*lp_args, tool=True), 2))))
    # the fall in measured use, decomposed (job 817663): leavers, entrants, and the agencies in both windows
    fd = K["adoption"]["fall_decomposition"]
    mac("fallTotal", r2s(100 * (fd["from"] - fd["to"]), 1))
    mac("fallLeavers", r2s(-100 * fd["leavers"], 1))
    mac("fallEntrants", r2s(100 * fd["entrants"], 1))
    mac("fallStayers", r2s(-100 * (fd["mix_among_stayers"] + fd["use_among_stayers"]), 1))
    mac("nBothWindows", str(fd["agencies_both_windows"]))
    LV = K["late_vocabulary"]
    mac("lateCarefully", "%d" % LV["ratio_to_2019_2022"]["carefully"])
    mac("lateReliable", "%.1f\\%%" % (100 * LV["share_of_listings"]["reliable"]))
    mac("reliableTwentyFive", "%.1f\\%%" % (100 * LV["share_2025_for_comparison"]["reliable"]))
    # tool-era listings that started in 2025, the text the AI marker words were chosen on
    mac("nToolTwentyFive", thou(sum(v["n"] for k, v in mo.items() if "2025-02" <= k <= "2025-12")))
    # price point estimates across every design and definition, standard errors of the vehicle designs
    allp = [100 * (math.exp(e["b"]) - 1) for e in cells]
    mac("pRangeAll", "%s\\%% to %s\\%%" % (sg(min(allp)), sg(max(allp))))
    mac("seVeh", "%.1f" % (100 * H["union"]["veh_pooled"]["se"]), "log points (natural-log units until 2026-10-31)")
    mac("seResid", "%.0f" % (100 * H["union"]["veh_residual"]["se"]), "log points")
    mac("seVThree", "%.0f" % (100 * H["union"]["veh_tool_era"]["se"]) if "veh_tool_era" in H["union"] else "31", "" if "veh_tool_era" in H["union"] else "WORKLOG s8 section 15")
    # one-sided 5% tests of the premium: the upper end of the 90% interval against Delta_g
    ev = H["union"]["veh_pooled"]
    mac("pVehHiOne", "%s\\%%" % sg(100 * (math.exp(ev["b"] + 1.645 * ev["se"]) - 1)))
    mac("nExclBenchOne", str(sum(e["b"] + 1.645 * e["se"] < bb for e in cells)))
    # against the main rule's misclassification-scaled premium: pooled designs use the pooled factor, tool-era designs
    # the tool-era factor (the other definitions' recall is too poorly measured to scale them)
    tool_keys = {"veh_residual", "veh_residual_month", "veh_residual_quarter", "veh_tool_era"}
    main_cells = [(H["union"][k], l73 if k in tool_keys else lam_pool) for k in pk] + [(R["rules"]["union"]["market"]["log_price"]["subcat_m"], l73)]
    mac("nMainPriceIntervals", str(len(main_cells)))
    mac("nExclBenchLamOne", str(sum(e["b"] + 1.645 * e["se"] < lam_ * bb for e, lam_ in main_cells)))
    names_ = {"veh_pooled": "pooled vehicles", "veh_pooled_start": "pooled vehicles with the opening bid",
              "veh_from_dec2022": "vehicles against listings from December 2022", "veh_nobody": "vehicles without body-style effects",
              "veh_tool_era": "tool-era vehicles with full controls", "all_pooled": "all goods",
              "all_pooled_start": "all goods with the opening bid held fixed", "goods_pooled": "non-vehicle goods",
              "goods_pooled_start": "non-vehicle goods with the opening bid", "veh_residual": "the vehicle residual with agency effects alone",
              "veh_residual_month": "the vehicle residual with month effects", "veh_residual_quarter": "the vehicle residual with quarter effects"}
    hits = [names_[k] for k in pk if H["union"][k]["b"] + 1.645 * H["union"][k]["se"] < (l73 if k in tool_keys else lam_pool) * bb]
    e_t = R["rules"]["union"]["market"]["log_price"]["subcat_m"]
    if e_t["b"] + 1.645 * e_t["se"] < l73 * bb:
        hits.append("tool-era price")
    mac("exclLamNames", " and ".join(hits) if len(hits) <= 2 else ", ".join(hits[:-1]) + " and " + hits[-1])
    mac("expFalseFlags", "%d" % round(f0 * u["p2_n"] * (1 - pi73)))
    fpr80 = (1 - 0.80) * phi / (1 - pi80)
    mac("expFalseFlagsHi", "%d" % round(fpr80 * u["p2_n"] * (1 - pi80)))
    # price and attention under the other definitions, and how many intervals exclude zero
    def pci(e):
        p_, lo_, hi_ = pexp(e["b"], e["se"])
        return "%s\\%% [%s\\%%, %s\\%%]" % (sg(p_), sg(lo_), sg(hi_)), lo_ > 0 or hi_ < 0
    mac("pAllClf", pci(H["c1"]["all_pooled"])[0])
    mac("pVehResidLex", pci(H["lex"]["veh_residual"])[0])
    mac("pVehResidClf", pci(H["c1"]["veh_residual"])[0])
    mac("pVehLex", pci(H["lex"]["veh_pooled"])[0])
    mac("pViewsClf", pci(H["c1"]["pooled_log_views"])[0])
    mac("pViewsT", r2s(H["union"]["pooled_log_views"]["b"] / H["union"]["pooled_log_views"]["se"], 1))
    lo_all = min(100 * (math.exp(H[r][k]["b"]) - 1) for r in ("union", "lex", "c1") for k in ("veh_pooled", "all_pooled", "all_pooled_start"))
    hi_all = max(100 * (math.exp(H[r][k]["b"]) - 1) for r in ("union", "lex", "c1") for k in ("veh_pooled", "all_pooled", "all_pooled_start"))
    mac("pRange", "%s\\%% to %s\\%%" % (sg(lo_all), sg(hi_all)))
    cnt, excl = 0, 0
    for r in ("union", "lex", "c1"):
        for k in ("pooled_sold", "pooled_log_bids", "pooled_log_views", "pooled_n_q_before_end", "veh_pooled", "veh_residual", "all_pooled",
                  "all_pooled_start", "veh_pooled_start", "veh_from_dec2022", "veh_nobody", "goods_pooled", "goods_pooled_start",
                  "veh_tool_era", "veh_residual_month", "veh_residual_quarter"):
            if k in H[r]:
                e = H[r][k]
                cnt += 1
                excl += (e["b"] - 1.96 * e["se"] > 0) or (e["b"] + 1.96 * e["se"] < 0)
        for k in ("sold", "log_price", "log_bids", "log_views", "any_q", "n_q_before_end"):
            e = R["rules"][r]["market"][k]["subcat_m"]
            cnt += 1
            excl += (e["b"] - 1.96 * e["se"] > 0) or (e["b"] + 1.96 * e["se"] < 0)
    mac("nIntervals", str(cnt))
    mac("nExclZero", str(excl))
    mac("nExclZeroChance", r2s(0.05 * cnt, 1))
    # first-pass questions result (classifier flag, job 817620): +0.55 (t 2.43) on the tool-era mean
    eq_ = R["rules"]["c1"]["market"]["n_q_before_end"]["subcat_m"]
    mac("qClf", "%s ($t=%s$) on a base of %s" % (sg(eq_["b"], 2), r2s(eq_["b"] / eq_["se"], 2), r2s(eq_["mean"], 2)))
    mac("qClfPct", "%d\\%%" % round(100 * eq_["b"] / eq_["mean"]))
    for r, tag in (("union", ""), ("lex", "Lex")):
        e2 = R["rules"][r]["market"]["n_q_before_end"]["subcat_m"]
        mac("qTool" + tag, "%s ($t=%s$)" % (sg(e2["b"], 2), r2s(e2["b"] / e2["se"], 2)))
    # the experiment's own effect on "defect stated, item praised" and on counts of numbers (paper_numbers.py block C)
    if PN and "defect_and_praise" in PN["rewrites"]["all"]:
        dp_, nn_ = PN["rewrites"]["all"]["defect_and_praise"], PN["rewrites"]["all"]["n_num"]
        mac("rwDP", "%s~percentage points [%s, %s] on a base of %.1f\\%%" % (sg(100 * dp_["diff"], 1), sg(100 * dp_["lo"], 1), sg(100 * dp_["hi"], 1), 100 * dp_["base"]))
        mac("rwNumAdd", "%s [%s, %s] on %.2f" % (sg(nn_["diff"], 2), sg(nn_["lo"], 2), sg(nn_["hi"], 2), nn_["base"]))
    else:
        mac("rwDP", "\\pend{}")
        mac("rwNumAdd", "\\pend{}")
    # appendix sentences from paper_numbers.json (empty until it exists)
    if PN:
        G_, F_ = PN["review_r1_detector"], PN["review_r1"]
        c2s, c2r = G_["c2_share_by_period"], G_["c2_recall_rewrites"]["all"]
        lg = G_["leave_one_generator_out"]
        assert lg, "leave-one-model-out recalls missing"
        gname = {"V1_enhance|claude-opus-5-5": "Opus ``appealing''", "V1_enhance|claude-sonnet-5-5": "Sonnet ``appealing''",
                 "V1_enhance|claude-haiku-5-5": "Haiku ``appealing''", "V2_concise|claude-opus-5-5": "Opus ``concise''",
                 "V2_concise|claude-sonnet-5-5": "Sonnet ``concise''"}
        mac("detIndep", " A second classifier with no format features and no promotional or AI marker words flagged %.1f\\%%, %.1f\\%% "
            "and %.1f\\%% of listings in the three periods and flagged %d\\%% of the rewrites. With the classifier retrained without one "
            "AI model's rewrites under one prompt (its rewrites under the other prompt stayed in training), the main rule flagged %s "
            "of the rewrites left out." % (100 * c2s["P0"], 100 * c2s["P1"], 100 * c2s["P2"],
            round(100 * c2r), (lambda it: ", ".join(it[:-1]) + " and " + it[-1])(["%d\\%% (%s)" % (round(100 * v["union_recall"]), gname[k]) for k, v in lg.items()])))
        W_ = PN.get("detector_words", {})
        if W_.get("top_ai"):
            mac("detWords", " The classifier's heaviest phrases toward AI are %s, and toward human %s. Among the format features, words in capitals "
                "weigh toward human and em dashes toward AI." % (", ".join("``%s''" % t["term"] for t in W_["top_ai"][:8]),
                ", ".join("``%s''" % t["term"] for t in W_["top_human"][:8])))
        else:
            mac("detWords", "")
        br = F_.get("benchmark_robustness", {})
        if br and "ngram12_cv" in br:
            a_cv = br["ngram12_cv"]["alpha"]
            near = [100 * v["diff"] for k, v in br.items() if isinstance(v, dict) and "diff" in v and k != "ngram12_cv" and v["alpha"] <= 2 * a_cv]
            far = [(v["alpha"], 100 * v["diff"]) for k, v in br.items() if isinstance(v, dict) and "diff" in v and v["alpha"] > 2 * a_cv]
            mac("benchRobRange", "%s to %s" % (r2s(min(near), 1), r2s(max(near), 1)), "the benchmark under other settings, log points")
            mac("benchRobPct", "+%s\\%% to +%s\\%%" % (r2s(100 * (math.exp(min(near) / 100) - 1), 1), r2s(100 * (math.exp(max(near) / 100) - 1), 1)), "the same in percent")
            assert len(far) == 1, far
            mac("benchRobFar", r2s(far[0][1], 1), "under heavy shrinkage, log points")
            mac("benchRobFarPct", r2s(100 * (math.exp(far[0][1] / 100) - 1), 1) + "\\%", "the same in percent")
            mac("benchRobust", " With the penalty at or near the one cross-validation chooses (%s), single words only, or a "
                "higher minimum number of listings per word or phrase, the premium ran from %s to %s log points (%s to %s\\%%)%s." % (
                r2s(a_cv, 1), r2s(min(near), 1), r2s(max(near), 1), r2s(100 * (math.exp(min(near) / 100) - 1), 1),
                r2s(100 * (math.exp(max(near) / 100) - 1), 1),
                "".join(", and a penalty of %g, which shrinks every prediction toward the mean, gave %s log points (%s\\%%)" % (
                    al, r2s(d, 1), r2s(100 * (math.exp(d / 100) - 1), 1)) for al, d in far)))
        else:
            mac("benchRobust", "")
        lay = F_.get("tool_era_flag_by_layout", {})
        if "page_title_2019" in lay and "title_text_2024" in lay:
            mac("layoutShares", " In the assistant era the main rule flagged %.1f\\%% of listings on the older page layout (%s) and %.1f\\%% on the "
                "redesign (%s). The redesign's listings came later, when use had fallen, so the comparison mixes layout and time." % (
                100 * lay["page_title_2019"]["ai"], thou(lay["page_title_2019"]["n"]), 100 * lay["title_text_2024"]["ai"], thou(lay["title_text_2024"]["n"])))
        else:
            mac("layoutShares", "")
        bl = PN["balance"]["union"]
        lab_ = {"log_start": "log opening bid", "log_photos": "log photos", "reserve_any": "reserve", "veh_f": "vehicle share",
                "age": "vehicle age", "log_miles": "log miles", "starts_yes": "starts", "drivable_yes": "drives", "title_yes": "title"}
        def big(d):
            out = [(lab_[k], v) for k, v in bl[d].items() if v and abs(v["b"] / v["se"]) >= 1.96]
            return ", ".join("%s (%s, standard error %s)" % (n, sg(v["b"], 2), r2s(v["se"], 2)) for n, v in out) or "none"
        def says(d, where):
            b_ = big(d)
            if b_ == "none":
                return "in the %s no characteristic did" % where
            k_ = b_.count(" (")
            return "in the %s %s %s" % (where, "one characteristic differed by two standard errors or more," if k_ == 1 else
                                        "these differed by two standard errors or more:", b_)
        first = says("tool_era", "assistant era")
        second = says("pooled", "comparison with all other listings")
        mac("balanceSummary", " %s. %s." % (first[0].upper() + first[1:], second[0].upper() + second[1:]))
    else:
        for n_ in ("detIndep", "detWords", "benchRobust", "layoutShares", "balanceSummary"):
            mac(n_, "")
    # ---- two standard detectors (job 842934, code/std_detectors.py): appendix B and one sentence in Section 4. Blind-read
    # counts join the job's per-text flags (stddet_blind.csv) with reader 1's labels, saved before the key was opened
    # (report_inputs/blind_labels_s8.json: the labels of scratchpad/machine_records_s8/blind_validation/my_labels.json without its notes)
    SDN = ("sdScored", "sdScoredAnalysis", "sdTooShort", "sdCut", "sdBinoPre", "sdFdgPre", "sdBinoRw", "sdFdgRw", "sdOurRw", "sdBinoRwCal", "sdFdgRwCal",
           "sdRwMoreAI", "sdSpearman", "sdBinoCalPzero", "sdBinoCalPone", "sdBinoCalPtwo", "sdFdgCalPzero", "sdFdgCalPone",
           "sdFdgCalPtwo", "sdFdgMar", "sdFdgPeak", "sdFdgSep", "sdFdgLate", "sdBinoBlind", "sdFdgBlind", "sdFdgBlindPost")
    if os.path.exists(os.path.join(inp, "stddet_summary.json")):
        SD = J("stddet_summary.json")
        p1 = lambda x: "%.1f\\%%" % (100 * x)
        p0 = lambda x: "%s\\%%" % r2s(100 * x, 0)
        mac("sdScored", thou(SD["coverage"]["listing"]["scored"]))
        mac("sdScoredAnalysis", thou(sum(SD["analysis_sample_scored"].values())), "analysis sample, all three periods")
        mac("sdTooShort", thou(SD["coverage"]["listing"]["too_short"]))
        mac("sdCut", thou(SD["truncated"]))
        for k, nm in (("bino_ai", "Bino"), ("fdg_ai", "Fdg")):
            mac("sd%sPre" % nm, p1(SD[k]["share_by_period"]["P0"]))
            mac("sd%sRw" % nm, p0(SD[k]["recall_rewrites"]["all"]))
        for k, nm in (("bino_cal", "Bino"), ("fdg_cal", "Fdg")):
            mac("sd%sRwCal" % nm, p0(SD[k]["recall_rewrites"]["all"]))
            for per, tag in (("P0", "Pzero"), ("P1", "Pone"), ("P2", "Ptwo")):
                mac("sd%sCal%s" % (nm, tag), p1(SD[k]["share_by_period"][per]))
        # the main rule's recall as Table 1 takes it: from the rebuild only when the rebuild is identical
        mac("sdOurRw", p0(rc["union"]["all"]["recall"]) if (rc and same) else "at least " + p0(D["recall_rewrites_c1"]["all"]))
        mm, sc = SD["rewrite_more_ai_like_than_original"], SD["spearman_with_our_classifier_tool_era"]
        mac("sdRwMoreAI", "%s and %s" % (p0(mm["binoculars"]), p0(mm["fdg"])), "of %d pairs" % mm["pairs"])
        mac("sdSpearman", "%s and %s" % (r2s(sc["binoculars"], 2), r2s(sc["fdg"], 2)), "tool era, n = %d" % sc["n"])
        fm = SD["fdg_cal"]["monthly"]
        pk_ = max((k for k in fm if "2025-02" <= k <= "2026-02"), key=lambda k: fm[k]["ai"])
        # results.tex (Section 6.1) says Fast-DetectGPT "peaks in the same month" as the main rule (\peakMonth, from the
        # same report_data.json series): fail loudly if not
        um_ = R["rules"]["union"]["monthly"]
        pk_main = max(um_, key=lambda k: um_[k]["ai"])
        assert pk_ == pk_main, "Fast-DetectGPT at 1%% peaks in %s, the main rule in %s: fix Section 6.1" % (pk_, pk_main)
        mac("sdFdgMar", p1(fm["2025-03"]["ai"]))
        mac("sdFdgPeak", "%s in %s" % (p1(fm[pk_]["ai"]), names[int(pk_[5:7]) - 1]), "%s, %d listings" % (pk_, fm[pk_]["n"]))
        mac("sdFdgSep", p1(fm["2025-09"]["ai"]))
        late = [k for k in fm if "2025-10" <= k <= "2026-02"]
        kl, nl = sum(round(fm[k]["ai"] * fm[k]["n"]) for k in late), sum(fm[k]["n"] for k in late)
        mac("sdFdgLate", p1(kl / nl), "%d of %d listings, October 2025 to February 2026" % (kl, nl))
        bp = os.path.join(inp, "stddet_blind.csv")
        lp = os.path.join(inp, "blind_labels_s8.json")   # reader 1's labels only (scratchpad/machine_records_s8/blind_validation/my_labels.json without its notes)
        if os.path.exists(bp) and os.path.exists(lp):
            rows_ = list(csv.DictReader(open(bp)))
            lab1 = json.load(open(lp))["labels"]
            assert {r["bid"] for r in rows_} == set(lab1), "blind ids differ between the detector run and reader 1"
            ai1 = [r for r in rows_ if lab1[r["bid"]] == "A"]
            post = [r for r in rows_ if r["stratum"] != "pre_control" and r["fdg_ai"] == "True"]
            mac("sdBinoBlind", "%d of the %d" % (sum(r["bino_ai"] == "True" for r in ai1), len(ai1)))
            mac("sdFdgBlind", "%d of the %d" % (sum(r["fdg_ai"] == "True" for r in ai1), len(ai1)))
            mac("sdFdgBlindPost", "%d of its %d" % (sum(lab1[r["bid"]] == "A" for r in post), len(post)))
        else:
            for n_ in ("sdBinoBlind", "sdFdgBlind", "sdFdgBlindPost"):
                mac(n_, "\\pend{}")
    else:
        for n_ in SDN:
            mac(n_, "\\pend{}")
    mac("rwPromo", sg(K["rewrite_audit"]["promo_added"], 2))
    mac("rwPromoOpus", sg(K["rewrite_audit"]["by_generator"]["appealing|opus"]["promo_added"], 2))
    # Session 10 (PI, 2026-10-29: "a great section to show how buyers respond ... lets make a major section with fig"):
    # what buyers ask, from the blind coding (questions_kinds.json, analysis_questions.py --kinds) and every question
    # that names the AI text (questions_ai_mentions.json, job 878457)
    qk, qm = os.path.join(inp, "questions_kinds.json"), os.path.join(inp, "questions_ai_mentions.json")
    if os.path.exists(qk) and os.path.exists(qm):
        Qk, Qm = json.load(open(qk)), json.load(open(qm))
        q1 = lambda x: "%.1f\\%%" % (100 * x)
        qd = lambda d: "%s~percentage points [%s, %s]" % (sg(100 * d["diff"]), sg(100 * d["lo"]), sg(100 * d["hi"]))
        g1, g0 = Qk["groups"]["1"], Qk["groups"]["0"]
        mac("qNai", str(g1["n_questions"]))
        mac("qLai", str(g1["n_listings"]))
        mac("qNoth", str(g0["n_questions"]))
        mac("qLoth", str(g0["n_listings"]))
        for k, nm in (("never_mentioned", "Never"), ("verify_claim", "Verify"), ("already_stated", "Already"), ("logistics", "Log")):
            mac("q%sAI" % nm, q1(g1[k]["share"]))
            mac("q%sOth" % nm, q1(g0[k]["share"]))
            mac("q%sDiff" % nm, qd(Qk["diff"][k]))
        for t, nm in (("Condition and function", "Cond"), ("Specifications and features", "Spec"),
                      ("Hours, miles and history", "Hist"), ("Photos", "Photo"), ("What comes with it", "Incl")):
            mac("q%sAI" % nm, q1(Qk["topics"][t]["1"]["share"]))
            mac("q%sOth" % nm, q1(Qk["topics"][t]["0"]["share"]))
            mac("q%sDiff" % nm, qd(Qk["topics"][t]["diff"]))
        ag = Qk["agreement_with_main_loop"]
        mac("qAgree", "%d of %d" % (round(ag["agree"] * ag["n"]), ag["n"]))
        QQ = Qm["questions"]
        mac("qTotal", thou(sum(QQ.values())), "questions asked before the close, 2019 to 2026")
        mac("qOnFlagged", thou(QQ["P2|1"]))
        mac("qOnOther", thou(QQ["P2|0"]))
        named = sum(Qm["matches"].values())
        assert set(Qm["matches"]) <= {"P2|1"}, "a question naming the AI text outside flagged tool-era listings"
        mac("qNamed", str(named))
        mac("qNamedListings", str(sum(Qm["listings_matched"].values())))
        mac("qNamedShare", "%.1f\\%%" % (100.0 * named / QQ["P2|1"]))
        KQ = K["questions"]
        mac("qToolListings", thou(KQ["tool_era_listings"]))
        mac("qToolWithQ", thou(KQ["tool_era_with_question"]))
        mac("qFacts", thou(KQ["facts"]))
        mac("qSampleListings", str(KQ["sample_listings"]))
    # Session 10 (PI, 2026-10-30: "the puffer is def going into the paper that is how we start talk abt the
    # characteristics of the listings"; misspellings "might worth a graph in appendix"): period averages over every
    # listing of the analysis sample (report_data.json period_means, analysis_core.load), so no flag selects them
    pm = R["period_means"]
    for p, nm in (("P0", "Pzero"), ("P1", "Pone"), ("P2", "Ptwo")):
        mac("puffHundred" + nm, "%.1f" % (100 * pm[p]["s_n_puff"]), "promotional words per 100 listings, " + p)
        mac("num" + nm, "%.1f" % pm[p]["s_n_num"], "numbers per listing, " + p)
        mac("typo" + nm, "%.1f\\%%" % (100 * pm[p]["has_typo"]), "listings with a misspelling, " + p)
    mac("puffRatio", "%.1f" % (pm["P2"]["s_n_puff"] / pm["P0"]["s_n_puff"]), "tool era over before ChatGPT")
    # the marker list against its stated cut (markers_union.json, job 884642): the paper had said every word was in at most
    # 0.12% of pre-ChatGPT listings; three were not
    mk = os.path.join(inp, "markers_union.json")
    if os.path.exists(mk):
        MK = json.load(open(mk))
        n0 = MK["n"]["P0"]
        sh = {w: v.get("P0", 0) / n0 for w, v in MK["words"].items()}
        over = sorted([w for w in sh if sh[w] > 0.0012], key=lambda w: -sh[w])
        mac("markersRareN", str(len(sh) - len(over)), "of the 38 AI marker words in at most 0.12% of pre-ChatGPT listings")
        join = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]
        mac("markersOverWords", join(["\\emph{%s}" % w for w in over]))
        mac("markersOverShares", join(["%.2f\\%%" % (100 * sh[w]) for w in over]))
    # the one-person human pilot (blind_scores_s10.json, job 884704; the PI, 2026-10-30: include it as a pilot)
    bs = os.path.join(inp, "blind_scores_s10.json")
    if os.path.exists(bs):
        B = json.load(open(bs))
        HB, AG = B["labellers"]["human_pilot"], B["agreement"]
        k, n = HB["precision_tool_union"]
        lo, hi = wilson_ci(k, n)
        mac("hjPrec", "%d of %d" % (k, n), "assistant-era main-rule flags the human judge labelled AI")
        mac("hjPrecK", str(k))
        mac("hjPrecN", str(n))
        mac("hjPrecPct", "%.0f\\%% [%.0f, %.0f]" % (100.0 * k / n, 100 * lo, 100 * hi))
        kr, nr = HB["recall_union"]
        lo_r, hi_r = wilson_ci(kr, nr)
        mac("hjRecall", "%s [%s, %s]" % (r2s(kr / nr, 2), r2s(lo_r, 2), r2s(hi_r, 2)), "main rule's recall of the human pilot's AI labels, %d of %d" % (kr, nr))
        mac("hjRecallPt", r2s(kr / nr, 2), "the same recall without its interval (the introduction, Session 13)")
        k1, n1 = B["labellers"]["llm_judge1"]["precision_tool_union"]
        mac("ljPrecPct", "%.0f\\%%" % (100.0 * k1 / n1), "LLM judge 1, the same flags")
        mac("hjCtrl", "%d of %d" % tuple(HB["controls_labelled_ai"]), "pre-ChatGPT listings the human judge labelled AI")
        mac("hjAI", str(HB["n"]["A"]))
        mac("hjKappaOne", "%.2f" % AG["llm_judge1|human_pilot"]["kappa3"])
        mac("hjKappaTwo", "%.2f" % AG["llm_judge2|human_pilot"]["kappa3"])
        mac("ljKappa", "%.2f" % AG["llm_judge1|llm_judge2"]["kappa3"])
    # the spelling graph (trends_monthly.json, job 884255): its months are Figure 1's, the first and last with 10 or more
    # listings; the caption's gap is the largest quarterly difference between all listings and those neither test flags
    tm = os.path.join(inp, "trends_monthly.json")
    if os.path.exists(tm):
        T = json.load(open(tm))
        ok = sorted(m for m, r in T["months"].items() if r["n"] >= 10)
        assert (ok[0], ok[-1]) == ("2019-04", "2026-02"), (ok[0], ok[-1])
        Q = {}
        for m in sorted(T["months"]):
            if ok[0] <= m <= ok[-1]:
                q = Q.setdefault(m[:4] + "Q%d" % ((int(m[5:7]) - 1) // 3 + 1), [0, 0, 0, 0])
                a_, b_ = T["months"][m], T["months_noflag"][m]
                q[0] += a_["k_typo"]; q[1] += a_["n_typo_known"]; q[2] += b_["k_typo"]; q[3] += b_["n_typo_known"]
        gap = max(abs(100.0 * v[0] / v[1] - 100.0 * v[2] / v[3]) for v in Q.values() if v[1] >= 10)
        mac("typoFlagGap", "%.1f" % gap, "points: all listings against those neither test flags, any quarter")
    # parts and wrecks (tab_wrecks; jobs 877521 to 877523 and 884267 to 884272)
    vw = {v: {r: os.path.join(inp, "hedonic_%s_tool_%s.json" % (r, v)) for r in ("union", "lex", "c1")}
          for v in ("noparts", "nowreck_title", "nowreck_text")}
    if all(os.path.exists(p) for d in vw.values() for p in d.values()):
        VW = {v: {r: json.load(open(p)) for r, p in d.items()} for v, d in vw.items()}
        mac("vehParts", str(VW["noparts"]["union"]["parts_dropped"]["n"]))
        mac("vehWreckTitle", str(VW["nowreck_title"]["union"]["wreck_dropped"]["n"]))
        mac("vehWreckText", str(VW["nowreck_text"]["union"]["wreck_dropped"]["n"]))
        pc = lambda e: 100 * (math.exp(e["b"]) - 1)
        moves = [abs(pc(VW[v][r]["veh_pooled"]) - pc(H[r]["veh_pooled"])) for v in VW for r in ("union", "lex", "c1")]
        moves += [abs(pc(VW[v]["union"]["veh_residual"]) - pc(H["union"]["veh_residual"])) for v in VW]
        mac("vehWreckMaxMove", "%.1f" % max(moves), "points, largest move from all priced vehicles in tab_wrecks")
        lo = [pc({"b": VW[v][r]["veh_pooled"]["b"] - 1.96 * VW[v][r]["veh_pooled"]["se"]}) for v in VW for r in ("union", "lex", "c1")]
        assert min(lo) < 0, "an interval in tab_wrecks excludes zero: the text says none does"
        # the Texas check (Session 10; PI, 2026-10-30: "the car part ... I want at least my PI to take a look"): the
        # no-parts union run (job 877521); texas_union.json, when present, adds the figure's bins from the same design
        U = VW["noparts"]["union"]
        tx = U["veh_tx"]
        mac("txN", thou(tx["n"]), "priced vehicles (no Parts) with 10+ Texas sales of their make, model and year within 3 months")
        mac("txOf", thou(U["n_vehicles_priced"]), "priced vehicles without the Parts listings")
        mac("txAI", str(tx["tool_era_ai"]), "of them assistant-era AI-written (main rule)")
        mac("txToolEra", str(tx["tool_era"]), "of them assistant era")
        mac("txRatio", "%.1f\\%%" % (100 * tx["median_ratio"]), "median closing price / Texas median")
        mac("txCorr", "%.2f" % tx["corr_log"], "correlation of log closing price and log Texas median")
        for k, key in (("txSubset", "veh_tx_subset"), ("txControl", "veh_tx_control"), ("txRel", "veh_tx_relative"),
                       ("vehNoPartsVOne", "veh_pooled")):
            mac(k, pci_fmt(U[key])[0])
            assert not pci_fmt(U[key])[1], (key, "excludes zero: the text says no Texas design does")
            assert pexp(U[key]["b"], U[key]["se"])[2] > 100 * (math.exp(K["benchmark_rewrites"]["b"]) - 1), (key, "excludes the premium")
        c = U["veh_tx_control"]["log_tx"]
        mac("txCtrlCoef", "%s (standard error %s)" % (sg(c["b"], 2), r2s(c["se"], 2)), "coefficient on the log Texas median in T1")
        se_t = U["veh_tx_control"]["se"]
        mac("txSE", "%.0f" % (100 * se_t), "standard error of the AI-written gap with the Texas control, log points (main vehicle design: seVeh)")
        mac("txMDE", "%d\\%%" % round(100 * (math.exp(2.8 * se_t) - 1)), "minimum detectable effect, Texas control design")
        mac("txPow", "%d\\%%" % round(100 * power(K["benchmark_rewrites"]["b"], se_t)), "power against the premium")
        tj = os.path.join(inp, "texas_union.json")
        if os.path.exists(tj):
            TB = json.load(open(tj))
            for key in ("veh_tx_subset", "veh_tx_control", "veh_tx_relative"):
                assert abs(TB[key]["b"] - U[key]["b"]) < 1e-9, (key, "the bins run does not reproduce job 877521")
            rb = TB["veh_tx_bins"]["ratio"]
            mac("txRatioAI", "%.1f\\%%" % (100 * rb["tool_ai"]["median"]), "assistant-era AI-written, median ratio")
            mac("txRatioOther", "%.1f\\%%" % (100 * rb["tool_other"]["median"]), "assistant-era human-written, median ratio")
            mac("txSlope", "%.2f" % TB["veh_tx_bins"]["slope_log"], "OLS slope, log closing price on log Texas median")
        # the Texas figure's panel (b) keeps its first axis, 4% to 900% of the Texas price (paper_figures.fig_texas, which
        # prints the same count); its caption says how many vehicles that axis leaves out
        ts = os.path.join(inp, "texas_shares.json")
        if os.path.exists(ts):
            SH = json.load(open(ts))["shares"]
            n_all = sum(len(v) for v in SH.values())
            n_off = sum(1 for v in SH.values() for x in v if not 0.04 <= x <= 9)
            assert n_all == 1989 and 0 < n_off < n_all, (n_all, n_off)
            mac("txAllVeh", thou(n_all), "vehicles in the Texas figure (texas_shares.json, job 887835)")
            mac("txOffAxis", str(n_off), "of them outside panel (b)'s 4% to 900% axis")
    # rewrite fidelity (Session 10, WORKLOG s10 section 14): four blind LLM coders (claude-opus-5-5, 150 pairs each) compared
    # every rewrite with its original; counts only (rewrite_audit.py --fidelity)
    fj = os.path.join(inp, "fidelity_summary.json")
    if os.path.exists(fj):
        FD = json.load(open(fj))
        assert FD["n_coded"] == FD["n_key"] == 600 and FD["arithmetic_errors"] == 0, "fidelity: not every rewrite coded"
        A = FD["cells"]["all|all"]
        mac("fidFactsN", thou(A["facts"]["n"]), "facts in the 600 originals")
        mac("fidFactsKept", "%.1f\\%%" % (100 * A["facts"]["kept"]))
        mac("fidDefN", thou(A["defects"]["n"]))
        soft = round(A["defects"]["softened"] * A["defects"]["n"])
        assert abs(soft - A["defects"]["softened"] * A["defects"]["n"]) < 1e-6 and A["defects"]["dropped"] == 0
        mac("fidDefSoft", str(soft), "defect statements softened; none dropped")
        for nm, key in (("fidAdded", "any_added_specific"), ("fidAddedStrict", "any_added_specific_strict")):
            e = A[key]
            mac(nm, "%.1f\\%% [%.1f, %.1f]" % (100 * e["share"], 100 * e["lo"], 100 * e["hi"]), "rewrites adding an unsupported claim about the item")
        o = FD["cells"]["claude-opus-5-5|V1_enhance"]["any_added_specific"]
        mac("fidAddedOpus", "%.0f\\%%" % (100 * o["share"]), "Opus appealing")
        o2 = FD["cells"]["claude-opus-5-5|V1_enhance"]["any_added_specific_strict"]
        mac("fidAddedOpusStrict", "%.0f\\%%" % (100 * o2["share"]), "Opus appealing, bare reliability and quality words left out")
    # the agencies' answers (analysis_questions.py --answers, job 884260; Session 10 review: the gaps ship their intervals)
    aj = os.path.join(inp, "answers_union.json")
    if os.path.exists(aj):
        AQ = json.load(open(aj))
        bp = AQ["by_period"]
        nq = sum(bp[p_]["questions"] for p_ in bp)
        na = sum(round(bp[p_]["questions"] * bp[p_]["answered"]) for p_ in bp)
        q_tot = next(int(m.split("}{")[1].split("}")[0].replace(",", "")) for m in M if m.startswith("\\newcommand{\\qTotal}"))
        assert nq == q_tot, ("answers file and questions disagree", nq, q_tot)
        mac("ansN", thou(na), "questions answered, of qTotal")
        mac("ansShare", "%.1f\\%%" % (100 * na / nq))
        mac("ansDelays", "%s, %s and %s" % tuple(r2s(bp[p_]["delay_h_median"], 1) for p_ in ("P0", "P1", "P2")), "median hours to the answer by period")
        mac("ansWords", "%d" % round(bp["P2"]["words_median"]))
        T = AQ["tool_era_by_ai"]
        mac("ansAI", "%.1f\\%%" % (100 * T["1"]["answered"]))
        mac("ansOth", "%.1f\\%%" % (100 * T["0"]["answered"]))
        for nm, key in (("ansGapAg", "answered|agency"), ("ansGapSm", "answered|agency+subcat_m")):
            e = AQ["gaps"][key]
            mac(nm, "%s~percentage points [%s, %s]" % (sg(100 * e["b"]), sg(100 * (e["b"] - 1.96 * e["se"])), sg(100 * (e["b"] + 1.96 * e["se"]))))
        m2 = {p_: round(bp[p_]["mark2_share"] * round(bp[p_]["questions"] * bp[p_]["answered"])) for p_ in bp}
        assert m2["P2"] == 0 and m2["P0"] == 0, m2
        mac("ansMarkTwo", str(m2["P1"]), "answers with two or more AI marker words, all in the chatbots-only period")
    # ---- Session 12 (PI, 2026-11-18; the advisor's C3 and C4). Every input below is required (no fallback).
    # (a) The rewrite test against item quality (job 988391, code/frozen_rewrites.py --quality): does rewriting make
    # cheap items read like expensive ones to the pricing model?
    QS = J("quality_s12.json")
    assert QS["n_pairs"] == 600 and abs(QS["mean_lift_all"] - bb) < 0.0015, ("quality job must reproduce the benchmark", QS["mean_lift_all"], bb)
    lpct = lambda x: 100 * (math.exp(x) - 1)
    mac("qualN", thou(QS["n_quality"]), "rewrite pairs whose original drew a bid in a cell with training listings")
    mac("qualNPriced", thou(QS["n_priced"]), "of the 600 originals, those that drew a bid")
    for grp, tag in (("all", ""), ("appealing", "App"), ("concise", "Con")):
        G = QS["groups"][grp]
        mac("qualNGroup" + tag, thou(G["n"]))
        k = G["gap_kept"]
        mac("qualGapKept" + tag, "%d\\%% [%d\\%%, %d\\%%]" % (round(100 * k["est"]), round(100 * k["lo"]), round(100 * k["hi"])),
            "share of the pricing model's gap between the top and bottom thirds of item quality that the rewrite keeps")
        mac("qualGapKeptLo" + tag, "%d\\%%" % round(100 * k["lo"]))
        for part in ("low", "mid", "high"):
            mac("qualLift" + part.capitalize() + tag, "%s\\%%" % sg(lpct(G["lift_" + part]["est"])), "rewrite lift, %s third of item quality" % part)
        s = G["sd_ratio"]
        mac("qualSdRatio" + tag, "%.2f [%.2f, %.2f]" % (s["est"], s["lo"], s["hi"]), "spread of rewrite scores over spread of original scores")
        c = G["corr_diff"]
        mac("qualCorrDiff" + tag, "%s [%s, %s]" % (sg(c["est"], 2), sg(c["lo"], 2), sg(c["hi"], 2)), "change in the correlation of score and item quality")
        mac("qualCorrOrig" + tag, "%.2f" % G["corr_orig_r"]["est"])
        mac("qualCorrAI" + tag, "%.2f" % G["corr_ai_r"]["est"])
        mac("qualPermP" + tag, "%.3f" % G["perm_p_slope"], "permutation p of the slope of the lift on item quality")
        sl = G["slope_lift_on_r"]
        mac("qualSlope" + tag, "%s [%s, %s]" % (sg(sl["est"], 3), sg(sl["lo"], 3), sg(sl["hi"], 3)), "slope of the boost on item quality (Session 12, review round 2)")
    A0 = QS["groups"]["all"]
    mac("qualGapOrigPct", "%s\\%%" % sg(lpct(A0["gap_orig"]["est"]), 0), "pricing model, top third over bottom third, originals")
    mac("qualGapAIPct", "%s\\%%" % sg(lpct(A0["gap_ai"]["est"]), 0), "the same, rewrites")
    mac("qualGapOrigTimes", "%s" % r2s(math.exp(A0["gap_orig"]["est"]), 2), "the same as a ratio (a gap of 50 log points or more is given as times)")
    mac("qualGapAITimes", "%s" % r2s(math.exp(A0["gap_ai"]["est"]), 2))
    mac("qualSpreadNarrowApp", "%d\\%%" % round(100 * (1 - QS["groups"]["appealing"]["sd_ratio"]["est"])))
    # Session 12, review round 3: the narrowing with its interval (1 minus the spread ratio), the largest fall in the
    # correlation the interval allows, and the slope decoded for an item that closed at half the price of similar items
    sr = QS["groups"]["appealing"]["sd_ratio"]
    mac("qualSpreadNarrowAppCI", "%d\\%% [%d\\%%, %d\\%%]" % (round(100 * (1 - sr["est"])), round(100 * (1 - sr["hi"])), round(100 * (1 - sr["lo"]))),
        "how much less spread out the appealing rewrites' predictions are than their originals'")
    cd = QS["groups"]["all"]["corr_diff"]
    assert cd["lo"] < 0
    mac("qualCorrFallMax", r2s(-cd["lo"], 2), "the largest fall in the correlation with item quality that the interval allows")
    sl = QS["groups"]["all"]["slope_lift_on_r"]
    assert sl["est"] < 0 and sl["lo"] < 0
    half = math.log(2)
    mac("qualSlopeHalf", r2s(-100 * sl["est"] * half, 1), "log points more boost for an item at half the price of similar items")
    mac("qualSlopeHalfMax", r2s(-100 * sl["lo"] * half, 1), "the same at the slope's lower 95% end, log points")
    mac("qualSlopeHalfMaxPct", "%s\\%%" % r2s(100 * (math.exp(-sl["lo"] * half) - 1), 1), "the same in percent")
    mac("qualSlopeHalfPct", "%s\\%%" % r2s(100 * (math.exp(-sl["est"] * half) - 1), 1))
    # the slope without the lots priced more than 500 log points below their cell (code/frozen_rewrites.py --trimmed)
    QT = J("quality_trim_s12.json")
    assert QT["cut_r"] == -5.0 and len(QT["dropped"]) == 4, QT["dropped"]
    mac("qualTrimN", str(len(QT["dropped"])), "lots with quality below -5")
    for grp, tag in (("all", ""), ("appealing", "App"), ("concise", "Con")):
        g = QT["groups"][grp]
        assert abs(g["slope_full"] - QS["groups"][grp]["slope_lift_on_r"]["est"]) < 1e-9
        mac("qualSlopeTrim" + tag, "%s [%s, %s]" % (sg(g["slope"]["est"], 3), sg(g["slope"]["lo"], 3), sg(g["slope"]["hi"], 3)),
            "slope without the %d lots below -5" % g["n_dropped"])
        mac("qualPermPTrim" + tag, "%.2f" % g["perm_p"])
        mac("qualTrimDropped" + tag, str(g["n_dropped"]))
    # (b) Promotional words other than the 15 AI marker words (job 842907, paper_numbers.json review_r1): the flag cannot
    # have put them there.
    R1n = J("paper_numbers.json")["review_r1"]
    e = R1n["promo_nomark_tool_era"]["union"]
    mac("promoNoMark", "%s (standard error %s) on a base of %s" % (sg(e["b"], 2), r2s(e["se"], 2), r2s(e["human_mean"], 2)),
        "assistant-era design, main rule; %d listings, %d agencies" % (e["n"], e["clusters"]))
    mac("promoNoMarkTimes", "%.0f" % ((e["human_mean"] + e["b"]) / e["human_mean"]), "AI-written over human-written, the same agency, subcategory and month")
    e = R1n["promo_nomark_rewrites"]
    mac("promoNoMarkRw", "%s [%s, %s] on %s" % (sg(e["diff"], 2), sg(e["lo"], 2), sg(e["hi"], 2), r2s(e["base"], 3)), "the 600 rewrites")
    # (c) What it would take: the standard error a vehicle price test needs to detect the pre-ChatGPT premium with
    # 80% power at the 5% level (2.8 standard errors), and the sample that implies if precision grows with the square
    # root of the number of AI-written vehicles.
    se_v = H["union"]["veh_pooled"]["se"]
    se_need = bb / 2.8
    n_ai_veh = int(next(m.split("}{")[1].split("}")[0].replace(",", "") for m in M if m.startswith("\\newcommand{\\nVehAI}")))
    fac = (se_v / se_need) ** 2
    assert 3 < fac < 6, fac
    mac("needSE", "%.1f" % (100 * se_need), "log points: the vehicle standard error that gives 80% power against the premium")
    mac("needFactor", "%.1f" % fac, "times as many AI-written vehicles, if precision grows with their square root")
    mac("needVehAI", thou(round(fac * n_ai_veh / 10.0) * 10), "AI-written vehicles a powered test needs, rounded to tens")
    lam_p = float(r2s(lam_pool, 2))
    se_need_l = math.log(1 + lam_p * (math.exp(bb) - 1)) / 2.8          # the premium the flag gap should show, at most
    fac_l = (se_v / se_need_l) ** 2
    mac("needFactorLam", "%.1f" % fac_l, "the same against the premium scaled by the detector factor")
    mac("needVehAILam", thou(round(fac_l * n_ai_veh / 10.0) * 10))
    e_s = H["union"]["pooled_sold"]
    mac("pSoldTrue", "[%s, %s]" % (sg(100 * (e_s["b"] - 1.96 * e_s["se"]) / lam_p), sg(100 * (e_s["b"] + 1.96 * e_s["se"]) / lam_p)),
        "the received-a-bid interval divided by the detector factor: the bound for true authorship, percentage points")
    # Session 12, review round 8: the largest falls the intervals allow for true authorship (each lower end over the factor)
    mac("pSoldFallMaxTrue", "%s~percentage points" % r2s(-100 * (e_s["b"] - 1.96 * e_s["se"]) / lam_p, 1),
        "the largest fall in received a bid for true authorship")
    for key_, nm_ in (("pooled_log_views", "pViews"), ("pooled_log_bids", "pBids")):
        e_ = H["union"][key_]
        lo_t = (e_["b"] - 1.96 * e_["se"]) / lam_p
        assert lo_t < 0
        mac(nm_ + "FallMaxTrue", "%s\\%%" % r2s(-100 * (math.exp(lo_t) - 1), 1), "the largest fall for true authorship")
    # (d) The re-coding of the 600 pairs by six blind coders (code/rewrite_audit.py --fidelity-s12; counts only)
    FS = J("fidelity_s12.json")
    assert FS["n_coded"] == 600 and FS["arithmetic_errors"] == 0, (FS["n_coded"], FS["arithmetic_errors"])
    assert set(FS["coder_models"].values()) == {"claude-opus-5-5"}, FS["coder_models"]
    fa, de = FS["facts"], FS["defects"]
    mac("fidNewFactsN", thou(fa["n"]))
    mac("fidNewFactsKept", "%.1f\\%%" % (100 * fa["kept"] / fa["n"]))
    mac("fidNewFactsDropped", thou(fa["dropped"]))
    mac("fidNewFactsChanged", thou(fa["changed"]))
    mac("fidNewDefN", thou(de["n"]))
    mac("fidNewDefDropped", thou(de["dropped"]))
    mac("fidNewDefDroppedWords", "none" if de["dropped"] == 0 else thou(de["dropped"]), "defect statements dropped, in words")
    mac("fidNewDefSoft", thou(de["softened"]))
    # shares from the exact counts (115 of 200 is 57.5, not 57.4999...), rounded half up like every other macro
    sh = lambda e: "%s\\%%" % r2s(100.0 * e["k"] / e["n"], 0)
    shci = lambda e: "%s\\%% [%s, %s]" % (r2s(100.0 * e["k"] / e["n"], 0), r2s(100 * e["lo"], 0), r2s(100 * e["hi"], 0))
    for grp, tag in (("appealing", "App"), ("concise", "Con"), ("all", "All")):
        G = FS[grp]
        mac("fidNew%sN" % tag, thou(G["any_claim"]["n"]))
        for k, nm in (("any_claim", "Any"), ("unit_claim", "Unit"), ("unit_strict", "UnitStrict"), ("condition", "Cond"),
                      ("specification", "Spec"), ("suitability", "Suit"), ("contradicts", "Contra"), ("generic", "Generic")):
            mac("fidNew%s%sK" % (tag, nm), thou(G[k]["k"]))
            mac("fidNew%s%sPct" % (tag, nm), shci(G[k]))
            mac("fidNew%s%sSh" % (tag, nm), sh(G[k]))
    for c, tag in (("claude-opus-5-5|V1_enhance", "OpusApp"), ("claude-sonnet-5-5|V1_enhance", "SonnetApp"),
                   ("claude-haiku-5-5|V1_enhance", "HaikuApp"), ("claude-opus-5-5|V2_concise", "OpusCon"),
                   ("claude-sonnet-5-5|V2_concise", "SonnetCon")):
        B = FS["cells"][c]
        for k, nm in (("any_claim", "Any"), ("unit_claim", "Unit"), ("condition", "Cond"), ("suitability", "Suit")):
            mac("fidNew%s%s" % (tag, nm), sh(B[k]), "%d of %d" % (B[k]["k"], B[k]["n"]))
    pp = lambda e: "%s~percentage points [%s, %s]" % (sg(100 * e["diff"]), sg(100 * e["lo"]), sg(100 * e["hi"]))
    for s, tag in (("defect", "Def"), ("short", "Short"), ("vehicle", "Veh")):
        S = FS["splits_appealing"][s]
        for side, st in (("yes", "Yes"), ("no", "No")):
            mac("fidNew%s%sUnit" % (tag, st), sh(S[side]["unit_claim"]), "%d of %d" % (S[side]["unit_claim"]["k"], S[side]["unit_claim"]["n"]))
            mac("fidNew%s%sCond" % (tag, st), sh(S[side]["condition"]))
            mac("fidNew%s%sN" % (tag, st), thou(S[side]["unit_claim"]["n"]))
            if s == "defect":
                # Session 13 (PI): contradictions by whether the original stated a defect (all came from those that did)
                mac("fidNewDef%sContraK" % st, thou(S[side]["contradicts"]["k"]), "rewrites with a claim that contradicts the original")
                mac("fidNewDef%sContraSh" % st, sh(S[side]["contradicts"]))
                if side == "no":
                    assert S[side]["contradicts"]["k"] == 0, "the text says none of these rewrites contradicted the original"
        mac("fidNew%sDiffUnit" % tag, pp(S["diff_unit"]), "permutation p %.3f" % S["diff_unit"]["perm_p"])
        mac("fidNew%sDiffCond" % tag, pp(S["diff_condition"]), "permutation p %.3f" % S["diff_condition"]["perm_p"])
        mac("fidNew%sPermP" % tag, "%.3f" % S["diff_unit"]["perm_p"])
        if s == "defect":
            mac("fidNewDefAnyPermP", "%.3f" % S["diff_any"]["perm_p"], "permutation p of the any-claim gap by defect")
        mac("fidNew%sDiffUnitHi" % tag, r2s(100 * S["diff_unit"]["hi"], 1), "upper end, percentage points")
    Q = FS["splits_appealing"]["quality"]
    for side, st in (("low", "Low"), ("mid", "Mid"), ("high", "High")):
        mac("fidNewQual%sUnit" % st, sh(Q[side]["unit_claim"]), "%d of %d" % (Q[side]["unit_claim"]["k"], Q[side]["unit_claim"]["n"]))
        mac("fidNewQual%sN" % st, thou(Q[side]["unit_claim"]["n"]))
    mac("fidNewQualDiffUnit", pp(Q["diff_unit"]), "bottom minus top third; permutation p %.3f" % Q["diff_unit"]["perm_p"])
    mac("fidNewQualPermP", "%.2f" % Q["diff_unit"]["perm_p"])
    mac("fidNewQualDiffUnitHi", r2s(100 * Q["diff_unit"]["hi"], 1), "upper end, percentage points")
    CA = FS["claims_appealing"]
    for k, nm in (("total", "Total"), ("condition", "Cond"), ("specification", "Spec"), ("suitability", "Suit"),
                  ("other", "Other"), ("contradicts", "Contra")):
        mac("fidNewClaims%s" % nm, thou(CA[k]), "claims in the appealing rewrites")
    mac("fidNewClaimsCon", thou(FS["claims_concise"]["total"]), "claims in the concise rewrites")
    mac("fidNewMedWords", "%d" % FS["median_original_words"])
    S_def = FS["splits_appealing"]["defect"]
    mac("fidNewDefDiffAny", pp(S_def["diff_any"]), "permutation p %.3f" % S_def["diff_any"]["perm_p"])
    mac("fidNewDefYesAny", sh(S_def["yes"]["any_claim"])); mac("fidNewDefNoAny", sh(S_def["no"]["any_claim"]))
    mac("fidNewDefPermPAny", "%.3f" % S_def["diff_any"]["perm_p"])
    eq = lambda k: sum(100.0 * FS["cells"][c][k]["k"] / FS["cells"][c][k]["n"] for c in ("claude-opus-5-5|V1_enhance", "claude-sonnet-5-5|V1_enhance", "claude-haiku-5-5|V1_enhance")) / 3
    mac("fidEqAny", "%s\\%%" % r2s(eq("any_claim"), 0), "appealing, the three models weighted equally")
    mac("fidEqUnit", "%s\\%%" % r2s(eq("unit_claim"), 0))
    for nm, tag in (("first", "One"), ("second", "Two")):
        Bt = FS["opus_appealing_batches"][nm]
        mac("fidOpusBatch%sAny" % tag, thou(Bt["any_claim"]["k"]), "of 100, Opus appealing batch %s" % nm)
        mac("fidOpusBatch%sUnit" % tag, thou(Bt["unit_claim"]["k"]))
    # (e) Review round 1 (the marketing referee): a market in which an unmarked 27% of items carry their rewrite, and the
    # boost for originals that drew no bid (code/frozen_rewrites.py --mixed on job 988391's per-pair scores)
    QM = J("quality_mixed_s12.json")
    m27 = next(m for m in QM["mixes"]["all"] if abs(m["share"] - 0.27) < 1e-9)
    c = m27["corr_change"]
    mac("mixCorrChange", "%s [%s, %s]" % (sg(c["mean"], 2), sg(c["lo"], 2), sg(c["hi"], 2)), "27%% of items rewritten at random, %d draws" % 2000)
    g = m27["gap_kept"]
    mac("mixGapKept", "%s\\%% [%s\\%%, %s\\%%]" % (r2s(100 * g["mean"], 0), r2s(100 * g["lo"], 0), r2s(100 * g["hi"], 0)))
    mac("mixShare", "%d\\%%" % round(100 * m27["share"]))
    B = QM["boost_unpriced"]
    mac("boostUnpriced", "%s\\%%" % sg(100 * (math.exp(B["mean_unpriced"]) - 1)), "%d originals with no bid" % B["n_unpriced"])
    mac("boostPriced", "%s\\%%" % sg(100 * (math.exp(B["mean_priced"]) - 1)), "%d priced originals" % B["n_priced"])
    mac("boostUnpricedN", thou(B["n_unpriced"]))
    pc = lambda x: 100 * (math.exp(x) - 1)
    mac("boostDiff", "%s\\%% [%s\\%%, %s\\%%]" % (sg(pc(B["diff"])), sg(pc(B["lo"])), sg(pc(B["hi"]))), "unpriced over priced, %s log points" % sg(100 * B["diff"]))
    # (f) Review round 1: a second coder (Claude Sonnet 5.5) coded 100 of the pairs (P101 to P200) blind; agreement with
    # the first coding (report_inputs/fidelity_agreement_s12.json, computed in the Session 12 WORKLOG section 13)
    AG = J("fidelity_agreement_s12.json")
    assert AG["n"] == 100
    for k, nm in (("any_claim", "Any"), ("unit_claim", "Unit"), ("condition", "Cond"), ("suitability", "Suit"), ("any_drop", "Drop")):
        e = AG["flags"][k]
        mac("agree%s" % nm, "%d of %d" % (round(e["agree"] * AG["n"]), AG["n"]), "pairs on which the two codings agree")
        mac("agree%sKappa" % nm, r2s(e["kappa"], 2), "Cohen's kappa")
    e = AG["flags"]["any_drop"]
    mac("agreeDropFlagged", "%d and %d" % (e["first"], e["second"]), "pairs with a dropped fact, first and second coding")
    # (g) Review round 1 (both outside referees: the instruction confound): the "appealing" instruction plus the
    # sentence "Use only the information in the title and the original description", Claude Opus 5.5, the same 100
    # listings it rewrote under "appealing", coded blind (code/rewrite_audit.py --arm-s12)
    AR = J("arm_s12.json")
    assert AR["n"] == 100 and AR["arithmetic_errors"] == 0 and AR["coder_model"] == "claude-opus-5-5", (AR["n"], AR["arithmetic_errors"])
    for k, nm in (("any_claim", "Any"), ("unit_claim", "Unit"), ("suitability", "Suit"), ("generic", "Generic"),
                  ("contradicts", "Contra"), ("any_drop", "Drop")):
        e = AR["flags"][k]
        mac("arm%sApp" % nm, thou(e["appealing"]["k"]), "of 100, appealing")
        mac("arm%sOnly" % nm, thou(e["appealing_only_info"]["k"]), "of 100, appealing plus use only")
    mac("armLenApp", r2s(AR["length_ratio_appealing"], 1), "median words of the rewrite over the original")
    mac("armLenOnly", r2s(AR["length_ratio_only_info"], 1))
    mac("armFactsKept", "%s\\%%" % r2s(100 * AR["facts_kept_only_info"], 1))
    # (h) Review round 2 (the marketing referee): the quality test can fail on the real model, and the share of added
    # words the model can price (job 989075, code/frozen_rewrites.py --quality, results/quality_s12b/)
    QB = J("quality_s12b.json")
    assert abs(QB["mean_lift_all"] - bb) < 0.0015
    pg = QB["planted"]["groups"]["all"]
    mac("plantedGapKept", "%s\\%% [%s\\%%, %s\\%%]" % (un(100 * pg["gap_kept"]["est"], 0), un(100 * pg["gap_kept"]["lo"], 0),
        un(100 * pg["gap_kept"]["hi"], 0)), "share of the quality gap kept had the top ten price-raising words been appended to the bottom third")
    mac("plantedCorrDrop", r2s(-pg["corr_diff"]["est"], 2), "fall in the correlation with quality in that planted test")
    mac("coverageShare", "%s\\%%" % r2s(100 * QB["coverage"]["mean_share_with_weight"], 0), "share of the words and phrases rewrites add that the model has a weight for")
    # (j) The PI's blind check of rewrite pairs (code/rewrite_audit.py --pi-check-s12; report_inputs/pi_check_s12.json)
    PC = J("pi_check_s12.json")
    assert PC["variants"] == ["V1_enhance"], PC["variants"]
    mac("piN", str(PC["n"]), "pairs the author coded blind")
    md = PC["models"]
    mac("piModels", "%d Opus, %d Sonnet and %d Haiku" % (md["claude-opus-5-5"], md["claude-sonnet-5-5"], md["claude-haiku-5-5"]))
    for q, nm in (("claim", "Claim"), ("fact", "Fact"), ("soft", "Soft")):
        g = PC["questions"][q]
        mac("piAgree" + nm, "%d of %d" % (g["agree"], g["n"]), "pairs on which the author and the coders agree")
        mac("piAgree" + nm + "Pct", "%s\\%% [%s, %s]" % (r2s(100.0 * g["agree"] / g["n"], 0), r2s(100 * g["lo"], 0), r2s(100 * g["hi"], 0)))
        mac("pi" + nm + "PI", str(g["pi"]))
        mac("pi" + nm + "Coders", str(g["coders"]))
    mac("piClaimKappa", r2s(PC["questions"]["claim"]["kappa"], 2), "low because nearly every pair has a claim")
    # the sentence "the author's words name a suggested use in every pair the author marks" holds only if they do
    assert PC["pi_claim_words_name_a_use"] == PC["pi_claims"], (PC["pi_claim_words_name_a_use"], PC["pi_claims"])
    mac("piUnitPairs", str(len(PC["coders_unit_claim_pairs"])), "pairs where the coders found a claim about the unit")
    mac("piProblemPairs", str(len(PC["coders_stated_problem_pairs"])), "pairs whose original states a problem, by the coders")
    # (k) Where the first stated problem sits, original against rewrite (code/rewrite_audit.py --burying; the main version
    # passes over negated mentions such as "no major leaks": report_inputs/burying_s12b.json; the panel's flag selects)
    BU = J("burying_s12b.json")
    assert BU["skip_negated"] is True
    S_ = BU["selections"]["panel"]
    num = lambda x: ("%g" % x)
    sgn = lambda x: ("+" if x > 0 else ("\\ensuremath{-}" if x < 0 else "")) + ("%g" % abs(x))
    mac("buryN", str(S_["n_pairs"]), "pairs whose original states a problem (panel flag)")
    mac("buryBoth", str(S_["groups"]["all"]["found_in_both"]), "of them, the problem found in both texts")
    mac("buryNotFound", str(S_["groups"]["all"]["rewrite_not_found"]), "rewrites in which the pattern finds no problem")
    for g, tag in (("all", ""), ("appealing", "App"), ("concise", "Con"), ("appealing|claude-opus-5-5", "Opus"),
                   ("appealing|claude-sonnet-5-5", "Sonnet"), ("appealing|claude-haiku-5-5", "Haiku")):
        x = S_["groups"][g]
        mac("buryWordsOrig" + tag, num(x["words_before_orig"]["median"]), "median words before the first stated problem, original")
        mac("buryWordsAI" + tag, num(x["words_before_rewrite"]["median"]), "the same, rewrite")
        if tag == "App":
            mac("buryWordsOrigAppOrd", ordinal(x["words_before_orig"]["median"]), "the same as an ordinal: the problem comes after this word")
            mac("buryWordsAIAppOrd", ordinal(x["words_before_rewrite"]["median"]), "the same, rewrite, as an ordinal")
            mac("buryStartOrigAppOrd", ordinal(x["words_before_orig"]["median"] + 1), "the word the problem starts at, original (median)")
            mac("buryStartAIAppOrd", ordinal(x["words_before_rewrite"]["median"] + 1), "the same, rewrite")
        if tag == "":
            mac("buryStartOrigOrd", ordinal(x["words_before_orig"]["median"] + 1), "the word the problem starts at, original (median), all prompts")
            mac("buryStartAIOrd", ordinal(x["words_before_rewrite"]["median"] + 1), "the same, rewrite")
        mac("buryShift" + tag, "%s [%s, %s]" % (sgn(x["diff_median"]), sgn(x["diff_lo"]), sgn(x["diff_hi"])), "paired median shift, words, pair bootstrap")
        mac("buryShift" + tag + "Med", num(abs(x["diff_median"])), "the same, the median alone, in words")
        mac("buryN" + tag + "Grp", str(x["n"]), "pairs in this group whose original states a problem")
        mac("buryBoth" + tag + "Grp", str(x["found_in_both"]), "of them, the problem found in both texts")
        mac("buryStated" + tag + "Grp", str(x["n"] - x["orig_not_found"]),
            "of them, originals in which the pattern finds a problem once negated mentions are passed over")
        assert x["n"] - x["orig_not_found"] - x["rewrite_not_found"] == x["found_in_both"]
        mac("buryShareOrig" + tag, "%d\\%%" % round(100 * x["share_before_orig_median"]), "median share of the words before it, original")
        mac("buryShareAI" + tag, "%d\\%%" % round(100 * x["share_before_rewrite_median"]))
        mac("buryLinesAI" + tag, num(x["lines_before_rewrite_median"]), "median lines before it, rewrite")
        mac("buryWithinOrig" + tag, "%d\\%%" % round(100 * x["within_25_orig"]), "share with the problem within the first 25 words, original")
        mac("buryWithinAI" + tag, "%d\\%%" % round(100 * x["within_25_rewrite"]))
    R_ = BU["selections"]["raw"]
    mac("buryRawN", str(R_["n_pairs"]), "the check: the strict pattern on the raw original")
    mac("buryRawShiftApp", sgn(R_["groups"]["appealing"]["diff_median"]))
    B0 = J("burying_s12.json")["selections"]["panel"]["groups"]["appealing"]
    mac("buryAllMatchShiftApp", sgn(B0["diff_median"]), "the first computation, negated mentions kept")
    # (l) Predicted gaps for the non-price outcomes (Session 12, buyers first; code/frozen_rewrites.py --reference, job
    # 993128; report_inputs/reference_s12.json): what buyers who kept the pre-ChatGPT relation between a listing's text and
    # each outcome would have done differently for the controlled rewrite, set beside the market gaps of Table 2
    RF = J("reference_s12c.json")   # job 993139: job 993129's run plus the title variant and the promotional-word check (round 8)
    assert abs(RF["reproduce_log_price"]["mean_delta"] - 0.172) < 0.0015, "the reference job must reproduce job 817664"
    for y, tag in (("sold", "Sold"), ("log_bids", "Bids"), ("log_views", "Views"), ("n_q_before_end", "Q")):
        o = RF["outcomes"][y]
        a_ = o["groups"]["all"]
        if o["units"] == "pp":
            fmt = lambda e, l, h: "%s~percentage points [%s, %s]" % (sg(e, 1), sg(l, 1), sg(h, 1))
            short = lambda e: "%s~percentage points" % sg(e, 1)
        elif o["units"] == "pct":
            fmt = lambda e, l, h: "%s\\%% [%s\\%%, %s\\%%]" % (sg(e, 1), sg(l, 1), sg(h, 1))
            short = lambda e: "%s\\%%" % sg(e, 1)
        else:
            fmt = lambda e, l, h: "%s [%s, %s]" % (sg(e, 2), sg(l, 2), sg(h, 2))
            short = lambda e: sg(e, 2)
        mac("ref" + tag, fmt(a_["est_units"], a_["lo_units"], a_["hi_units"]), "predicted gap over the 600 pairs: " + o["label"])
        mac("ref" + tag + "App", short(o["groups"]["appealing"]["est_units"]), "the same, appealing rewrites")
        mac("ref" + tag + "Con", short(o["groups"]["concise"]["est_units"]), "the same, concise rewrites")
        for g_, t_ in (("appealing", "App"), ("concise", "Con")):
            G_ = o["groups"][g_]
            mac("ref" + tag + t_ + "CI", fmt(G_["est_units"], G_["lo_units"], G_["hi_units"]), "with its interval")
        P_ = o["planted_top"]
        assert P_["lo"] > 0 and o["planted_bottom"]["hi"] < 0, "the planted check must pass"
        if o["units"] == "pct" and P_["est"] >= 0.5:
            mac("ref" + tag + "PlantTop", "about %s times" % r2s(math.exp(P_["est"]), 1), "the model's ten highest-weight words planted in every original")
        else:
            mac("ref" + tag + "PlantTop", short(P_["est_units"]), "the model's ten highest-weight words planted in every original")
        mac("ref" + tag + "PlantBottom", short(o["planted_bottom"]["est_units"]), "its ten lowest-weight words")
        sc = o["scale"]
        if o["units"] == "pp":
            mac("ref" + tag + "Abs", "%s~percentage points" % r2s(100 * sc["mean_abs_change"], 1), "mean absolute change over the pairs")
            mac("ref" + tag + "SD", "%s~percentage points" % r2s(100 * sc["sd_score_originals"], 1), "spread of the scores across the originals")
        mac("ref" + tag + "OOS", "%.2f" % o["oos"]["est"], "out-of-sample correlation with the originals' own outcomes")
        mac("ref" + tag + "Pt", short(a_["est_units"]), "the pooled predicted gap without its interval")
        PP = o["promo_planted"]
        mac("ref" + tag + "Promo", fmt(PP["est_units"], PP["lo_units"], PP["hi_units"]), "the 32 promotional words planted in every original")
        mac("ref" + tag + "Slope", "%.2f" % o["slope_rewrite_on_original"], "slope of the rewrite's score on the original's")
        TV = RF["title_variant"]["outcomes"][y]
        mac("ref" + tag + "Title", fmt(TV["groups"]["all"]["est_units"], TV["groups"]["all"]["lo_units"], TV["groups"]["all"]["hi_units"]),
            "with the title in every text")
        mac("ref" + tag + "TitleApp", short(TV["groups"]["appealing"]["est_units"]), "with the title, appealing")
    oos = [RF["outcomes"][y]["oos"]["est"] for y in RF["outcomes"]]
    mac("refNTrain", thou(RF["outcomes"]["sold"]["n_train"]), "pre-ChatGPT listings each outcome model is fit on")
    mac("refQBase", "%.2f" % RF["outcomes"]["n_q_before_end"]["train_mean"], "mean questions per listing in the training listings")
    LP = RF["log_price_checks"]
    mac("benchPromo", "+%s\\%%" % r2s(LP["promo_planted"]["est_units"], 1), "the pricing model: the 32 promotional words planted in every original")
    mac("benchPromoVocab", str(LP["promo_planted"]["words_in_vocabulary"]), "of the 32, the words the pricing model has a weight for")
    mac("benchSlope", "%.2f" % LP["slope_rewrite_on_original"], "the pricing model: slope of the rewrite's score on the original's")
    mac("refPromoVocab", str(RF["outcomes"]["sold"]["promo_planted"]["words_in_vocabulary"]), "of the 32, the words the outcome models have a weight for")
    sl = [RF["outcomes"][y]["slope_rewrite_on_original"] for y in RF["outcomes"]]
    mac("refSlopeMin", "%.2f" % min(sl)); mac("refSlopeMax", "%.2f" % max(sl))
    TP = RF["title_variant"]["outcomes"]["log_price"]["groups"]
    mac("benchTitlePct", "+%s\\%% [%s\\%%, %s\\%%]" % (r2s(TP["all"]["est_units"], 1), sg(TP["all"]["lo_units"], 1), sg(TP["all"]["hi_units"], 1)),
        "the pre-ChatGPT premium with the title in every text")
    mac("benchTitleApp", "+%s\\%%" % r2s(TP["appealing"]["est_units"], 1)); mac("benchTitleCon", "+%s\\%%" % r2s(TP["concise"]["est_units"], 1))
    RC = J("reference_counts_s12.json")
    assert RC["p0_listings"] - RC["originals_in_p0"] == RC["p0_outside_originals"] == RF["outcomes"]["sold"]["n_train"]
    mac("refNOrigIn", thou(RC["originals_in_p0"]), "of the 600 originals, those in the analysis sample's pre-ChatGPT listings")
    mac("refNOOS", thou(RF["outcomes"]["sold"]["oos"]["n"]), "originals with the outcome in a training cell")
    mac("refOOSMin", "%.2f" % min(oos))
    mac("refOOSMax", "%.2f" % max(oos))
    # The abstract's bound (Session 12, the PI's read of the abstract, 2026-11-20): every predicted gap for received
    # a bid, bids, page views and buyer questions, with and without the title in the text, moves its outcome by less
    # than ABS_BOUND percent (received a bid and questions over their training means). A re-run that breaks the bound
    # stops the build, and the abstract's last sentence must change with it.
    ABS_BOUND = 5
    ch_ = []
    for y in ("sold", "log_bids", "log_views", "n_q_before_end"):
        o_ = RF["outcomes"][y]
        for g_ in (o_["groups"]["all"], RF["title_variant"]["outcomes"][y]["groups"]["all"]):
            ch_.append(pct_of_base(o_["units"], g_["est_units"], o_["train_mean"]))
    check_below(ch_, ABS_BOUND)
    mac("refBoundPct", "%d\\%%" % ABS_BOUND, "the abstract's bound on the participation predicted gaps (largest: %.2f%%)" % max(abs(c) for c in ch_))
    # (m) The outcome models on real assistant-era listings (Session 12, review round 8; code/frozen_rewrites.py
    # --reference-real, job 993140; report_inputs/reference_real_s12.json): the gap the pre-ChatGPT relation between text
    # and each outcome predicts for AI-written against the same seller's human-written listings, beside the realized gap,
    # which must reproduce the assistant-era design of Table 2 (report_data.json)
    RR = J("reference_real_s12.json")
    mk_ = u["market"]
    for y, tag in (("sold", "Sold"), ("log_bids", "Bids"), ("log_views", "Views"), ("n_q_before_end", "Q"), ("log_price", "Price")):
        r_ = RR["variants"]["description"][y]
        e_ = mk_[y]["subcat_m"]
        assert abs(r_["realized"]["b"] - e_["b"]) < 1e-6 and r_["n"] == e_["n"], ("the realized gap must reproduce Table 2", y)
        x_ = r_["expected"]
        if r_["units"] == "pp":
            txt = "%s~percentage points [%s, %s]" % (sg(x_["est_units"], 1), sg(x_["lo_units"], 1), sg(x_["hi_units"], 1))
        elif r_["units"] == "pct":
            txt = "%s\\%% [%s\\%%, %s\\%%]" % (sg(x_["est_units"], 1), sg(x_["lo_units"], 1), sg(x_["hi_units"], 1))
        else:
            txt = "%s [%s, %s]" % (sg(x_["est_units"], 2), sg(x_["lo_units"], 2), sg(x_["hi_units"], 2))
        mac("real" + tag, txt, "expected gap from the text of real assistant-era listings, %s" % y)
    # Session 13 (PI, 2026-11-22, after Kraft and Rao 2026): the running text keeps an interval only where the bound is the
    # finding; elsewhere it prints the estimate alone. For every macro whose value carries a bracketed interval, a twin
    # with the suffix "Est" prints the same value with the interval cut (the tables and the appendix keep the full one).
    import re as _re
    names_ = set(_re.match(r"\\newcommand\{\\([A-Za-z]+)\}", x).group(1) for x in M if x.startswith("\\newcommand"))
    est_ = []
    for x in M:
        m_ = _re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}(  %%.*)?$", x)
        if not m_ or " [" not in m_.group(2):
            continue
        v_ = _re.sub(r" \[[^\]]*\]", "", m_.group(2), count=1)
        if v_ == m_.group(2) or not v_.strip() or "[" in v_:
            continue
        assert m_.group(1) + "Est" not in names_, m_.group(1)
        est_.append("\\newcommand{\\%sEst}{%s}  %% %s without its interval" % (m_.group(1), v_, m_.group(1)))
        # a twin with the suffix "Tab" for tables whose row label carries the unit: the interval kept, units and bases cut
        t_ = _re.sub(r" on a base of .*$", "", m_.group(2)).replace("\\%", "").replace("~percentage points", "")
        assert m_.group(1) + "Tab" not in names_, m_.group(1)
        est_.append("\\newcommand{\\%sTab}{%s}  %% %s for a table, unit in the row label" % (m_.group(1), t_, m_.group(1)))
        # and a twin with the suffix "Txt" for the running text, the interval labeled: "+3.8\% (95\% CI [...])"
        b_ = _re.search(r" (\[[^\]]*\])", m_.group(2)).group(1)
        assert m_.group(1) + "Txt" not in names_, m_.group(1)
        est_.append("\\newcommand{\\%sTxt}{%s}  %% %s with its interval labeled" % (
            m_.group(1), m_.group(2).replace(" " + b_, " (95\\%% CI %s)" % b_, 1), m_.group(1)))
    M.extend(est_)
    open(out, "w").write("\n".join(M) + "\n")
    print("wrote %s: %d macros; pooled scope = %s; paper_numbers %s" % (out, len(M) - 1, "tool" if tool else "any", "used" if PN else "absent"))


if __name__ == "__main__":
    main(*sys.argv[1:])
