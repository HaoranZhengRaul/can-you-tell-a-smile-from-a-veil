#!/usr/bin/env python3
"""Session 8: adoption of the on-platform AI writing tool (AI = validated union indicator by default).

1 monthly AI share of new listings (by start month) and number of agencies using it
2 who adopts: agency-level linear probability of using AI at least once in the tool era, on the
  agency's own pre-tool writing (median item-specific words, misspelling share), volume and goods mix
3 intensity: distribution of each agency's AI share of its tool-era listings (all-or-nothing vs mixed)
4 the fall: AI share in heavy months (2025-05..2025-08) vs later months (2025-10..2026-02), split into
  a mix term (which agencies list) and a use term (how much the same agencies use it):
  share = sum_a w_a u_a ; d(share) = sum_a dw_a u_a(early) + sum_a w_a(late) du_a  (+ entry/exit)
5 (Session 10) the stayers that used AI early: stoppers vs continuers on the outcomes of their own early AI written
  listings against their other early listings (received a bid, log bids, log views, log price)
Usage: python3 adoption.py PANEL FEATURES AISCORES
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from analysis_core import load  # noqa: E402
from regs import ols_fe  # noqa: E402


def main(panel, feats, scores):
    d = load(panel, feats, scores)
    d["sm"] = d.start_local_dt.dt.to_period("M").astype(str)
    print("AI rule:", d.attrs.get("ai_rule"))
    g = d[d.start_local_dt >= "2024-06-01"].groupby("sm")
    T = pd.DataFrame({"n": g.size(), "ai_share": g.ai.mean().round(3), "n_ai": g.ai.sum().astype(int),
                      "agencies": g.agency.nunique(), "agencies_using": g.apply(lambda x: x[x.ai == 1].agency.nunique())})
    print("1. monthly (listing start month):\n" + T.to_string())
    p2 = d[d.period == "P2"]
    pre = d[d.period.isin(["P0", "P1"])]
    A = p2.groupby("agency").agg(n2=("ai", "size"), ai2=("ai", "mean"))
    B = pre.groupby("agency").agg(n_pre=("ai", "size"), w_pre=("s_w", "median"), typo_pre=("has_typo", "mean"),
                                   veh_pre=("veh", "mean"), photos_pre=("n_photos", "mean"), puff_pre=("s_n_puff", "mean"))
    M = A.join(B, how="left")
    M["ever"] = (M.ai2 > 0).astype(float)
    M["has_pre"] = M.n_pre.notna().astype(float)
    print("\n2. agencies with tool-era listings: %d; used AI at least once: %d (%.1f%%); with pre-tool history: %d" % (
        len(M), int(M.ever.sum()), 100 * M.ever.mean(), int(M.has_pre.sum())))
    H = M[M.n_pre.notna()].copy()
    H["log_n_pre"] = np.log(H.n_pre)
    H["log_n2"] = np.log(H.n2)
    H["log_w_pre"] = np.log1p(H.w_pre)
    H["one"] = 1.0
    H["const_fe"] = "all"
    o, m = ols_fe(H, "ever", ["log_n2", "log_n_pre", "log_w_pre", "typo_pre", "veh_pre", "photos_pre"], fes=["const_fe"], cluster=None)
    print("   linear probability of ever using AI (agency level, n=%d):" % m["n"])
    for k, r in o.iterrows():
        print("     %-12s %8.4f (se %.4f, t %5.2f)" % (k, r.coef, r.se, r.t))
    q = H.groupby(pd.qcut(H.w_pre.rank(method="first"), 4, labels=["Q1 terse", "Q2", "Q3", "Q4 verbose"])).ever.agg(["size", "mean"])
    print("   ever-use rate by quartile of the agency's own pre-tool median item-specific words:\n" + q.round(3).to_string())
    print("\n3. intensity among agencies with >=3 tool-era listings:")
    I = M[M.n2 >= 3]
    print("   agencies %d | AI share 0: %d | (0,0.5): %d | [0.5,1): %d | 1: %d" % (
        len(I), int((I.ai2 == 0).sum()), int(((I.ai2 > 0) & (I.ai2 < 0.5)).sum()), int(((I.ai2 >= 0.5) & (I.ai2 < 1)).sum()), int((I.ai2 == 1).sum())))
    early = d[(d.start_local_dt >= "2025-05-01") & (d.start_local_dt < "2025-09-01")]
    late = d[(d.start_local_dt >= "2025-10-01") & (d.start_local_dt < "2026-03-01")]
    se, sl = early.ai.mean(), late.ai.mean()
    we = early.agency.value_counts(normalize=True)
    wl = late.agency.value_counts(normalize=True)
    ue = early.groupby("agency").ai.mean()
    ul = late.groupby("agency").ai.mean()
    both = we.index.intersection(wl.index)
    mix = float(((wl.reindex(both) - we.reindex(both)) * ue.reindex(both)).sum())
    use = float((wl.reindex(both) * (ul.reindex(both) - ue.reindex(both))).sum())
    entry = float((wl.drop(both, errors="ignore") * ul.drop(both, errors="ignore")).sum())
    exitt = float(-(we.drop(both, errors="ignore") * ue.drop(both, errors="ignore")).sum())
    print("\n4. AI share May-Aug 2025 %.3f (n=%d, agencies %d) -> Oct 2025-Feb 2026 %.3f (n=%d, agencies %d); change %.3f" % (
        se, len(early), early.agency.nunique(), sl, len(late), late.agency.nunique(), sl - se))
    print("   agencies present in both windows: %d; their listing share early %.3f, late %.3f" % (len(both), we.reindex(both).sum(), wl.reindex(both).sum()))
    print("   decomposition: mix among stayers %+.3f | same agencies' use %+.3f | entrants %+.3f | leavers %+.3f | sum %+.3f" % (
        mix, use, entry, exitt, mix + use + entry + exitt))
    st = pd.DataFrame({"early": ue.reindex(both), "late": ul.reindex(both), "n_late": late.groupby("agency").size().reindex(both)})
    users = st[st.early > 0]
    print("   among stayers that used AI early (n=%d): still use late (share>0) %d, stopped %d; mean use early %.3f late %.3f" % (
        len(users), int((users.late > 0).sum()), int((users.late == 0).sum()), users.early.mean(), users.late.mean()))
    # 5 (Session 10; PI, 2026-10-29, "why use fell"): did the stayers that stopped see worse outcomes on their own AI
    # written listings, early (May-Aug 2025), than the stayers that kept using it? Within agency, AI written vs not.
    # 25 agencies at most, so read the intervals, not the signs; cluster-robust errors with this few clusters are rough
    stop_ix, cont_ix = users[users.late == 0].index, users[users.late > 0].index
    e = early[early.agency.isin(users.index)].copy()
    e["stopper"] = e.agency.isin(stop_ix).astype(float)
    e["ai_x_stop"] = e.ai * e.stopper
    print("\n5. early listings (May-Aug 2025) of the %d early users that kept listing: %d listings, %d AI written; "
          "stoppers %d agencies (%d listings, %d AI), continuers %d agencies (%d listings, %d AI)" % (
              len(users), len(e), int(e.ai.sum()), len(stop_ix), int(e.stopper.sum()), int((e.ai * e.stopper).sum()),
              len(cont_ix), int((1 - e.stopper).sum()), int((e.ai * (1 - e.stopper)).sum())))
    print("   distinct agencies: early window %d, late window %d, both %d" % (
        early.agency.nunique(), late.agency.nunique(), len(set(early.agency) & set(late.agency))))
    rng = np.random.default_rng(29)
    MIN_AG = 5
    for y in ["sold", "log_bids", "log_views", "log_price"]:
        z = e[e[y].notna()]
        per = z.groupby(["agency", "ai"])[y].mean().unstack()
        per = per.dropna() if (0.0 in per.columns and 1.0 in per.columns) else per.iloc[0:0]
        gap = (per[1.0] - per[0.0]) if len(per) else pd.Series(dtype=float)
        out = []
        for name, ix in (("stoppers", stop_ix), ("continuers", cont_ix)):
            g = gap.reindex(ix).dropna().values
            if len(g) >= MIN_AG:
                bs = [rng.choice(g, len(g)).mean() for _ in range(2000)]
                out.append("%s %+.3f [%+.3f, %+.3f] (n=%d agencies with both)" % (name, g.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(g)))
            else:
                out.append("%s: %d agencies with both kinds of early listings (fewer than %d: no estimate)" % (name, len(g), MIN_AG))
        print("   %-10s per-agency AI minus non-AI mean, agency bootstrap 95%%: %s" % (y, "; ".join(out)))
        # the stoppers-vs-continuers regression needs within-agency variation in both groups; with agency fixed effects
        # a group whose agencies wrote only AI (or only non-AI) early listings gives none, and the two coefficients then
        # come out identical (877524 to 877526, first run): refuse it rather than print an unidentified number
        n_both = [int(gap.reindex(ix).notna().sum()) for ix in (stop_ix, cont_ix)]
        if min(n_both) < MIN_AG:
            print("   %-10s stoppers-vs-continuers regression not identified: agencies with both kinds, stoppers %d, continuers %d (need %d each)" % (
                y, n_both[0], n_both[1], MIN_AG))
            continue
        o, m = ols_fe(z, y, ["ai", "ai_x_stop"], fes=["agency", "subcat"], cluster="agency")
        print("   %-10s regression (agency + subcategory FE): AI gap at continuers %+.3f (se %.3f); stoppers minus "
              "continuers %+.3f (se %.3f); n=%d clusters=%d" % (
                  y, o.loc["ai", "coef"], o.loc["ai", "se"], o.loc["ai_x_stop", "coef"], o.loc["ai_x_stop", "se"], m["n"], m["n_clusters"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
