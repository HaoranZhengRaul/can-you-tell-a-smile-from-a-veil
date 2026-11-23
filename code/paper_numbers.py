#!/usr/bin/env python3
"""Session 9: the numbers the paper needs that no earlier job saved.

  A  monthly share of new listings flagged AI written, 2019-01 to 2026-03, under each AI definition
     (analysis sample of analysis_core.load), with the listing count of each month
  B  detector validation for the three AI definitions: false positives on text written before ChatGPT
     and out of time, and recall on the 600 controlled rewrites by instruction and generator. The
     cross-fitted classifier is rebuilt exactly as in detector.py (same texts, same folds, same model);
     the check records whether its threshold equals the saved one in detector_meta.json.
  C  controlled rewrites: item-specific text features (boilerplate removed as in textfeat.py) of each
     original and its rewrite; paired differences overall and by instruction x generator; numbers and
     defect statements kept, as in rewrite_audit.py
  D  balance: AI written vs human listings on characteristics set before, or apart from, the text, in
     the tool-era design (agency + subcategory x month) and the pooled 2019-26 design (agency +
     subcategory x quarter); vehicles add age, miles and the seller's condition table
  E  price information in the text: within-period spreads and the tool-era / pre-ChatGPT ratio from the
     20 agency-bootstrap draws of job 817639 (informativeness.py v3)
  B2 what our classifier keys on: the strongest AI-side and human-side phrases (weights averaged over the five
     cross-fitted models) with their shares of pre-ChatGPT and tool-era listings, and per-word contributions for the
     paper's example listings (raw text, examples file only)
  H  same agency, same start day: tool-era batches with both AI written and human listings (batch and
     subcategory effects), every buyer outcome, under each AI definition, with and without the opening bid
  I  false positives on listings started before 2019, which the detector never saw (training, threshold and
     marker words all used 2019-2022 text), under each AI definition, with input coverage
  X  example listings (raw text, for the paper's example table; written to a separate file that stays
     out of git)
Every block checks itself against an earlier saved output; the checks are stored in the JSON.
Usage: python3 paper_numbers.py RESULTS_DIR OUT_JSON EXAMPLES_JSON
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
import detector  # noqa: E402
from analysis_core import load  # noqa: E402
from regs import ols_fe  # noqa: E402
from rewrite_audit import nums, stems  # noqa: E402
from textfeat import DEFECT, MARK, boilerplate_set, features, norm_sent, split_sents  # noqa: E402

RULES = ["union", "lex", "c1"]
EXAMPLE_ITEMS = [75374770, 76648142, 78549518, 76076922, 75771948, 74478922, 29219566, 80212265, 77650336]
WORD_VIEW_ITEMS = [75374770, 76648142, 21346726]          # the paper's bottles, hybrid bus and 2019 pickup
WORD_VIEW_SNIPPET = "great running vehicle, but has a few oil leaks"   # the paper's rewrite example


def paired(a, b):
    """Mean of b - a with a normal 95% interval (paired listings)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    m, s = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"diff": m, "lo": m - 1.96 * s, "hi": m + 1.96 * s, "base": float(np.nanmean(a)), "n": int(len(d))}


def coef(o, m, k="ai"):
    if k not in o.index:
        return None
    return {"b": float(o.loc[k, "coef"]), "se": float(o.loc[k, "se"]), "n": int(m["n"]), "clusters": int(m["n_clusters"])}


def fold_word_weights(fold_models, n_fmt=5):
    """Average weight of every phrase over the fold models (0 where a fold's vocabulary lacks it), of the format
    features, and of the intercept. Returns (names per model, {phrase: weight}, format weights, intercept)."""
    names = [m.vec.get_feature_names_out() for m in fold_models]
    acc, fmt_w = {}, np.zeros(n_fmt)
    for m, nm in zip(fold_models, names):
        coef = m.clf.coef_[0]
        for j, term in enumerate(nm):
            acc[term] = acc.get(term, 0.0) + coef[j] / len(fold_models)
        fmt_w += coef[len(nm):] / len(fold_models)
    return names, acc, fmt_w, float(np.mean([m.clf.intercept_[0] for m in fold_models]))


def word_contrib(text, fold_models, names, fmt_feats, n_fmt=5):
    """Each phrase's contribution to one text's logit (weight x tf-idf value, averaged over the folds) and the
    format features' contributions. With the mean intercept they add up to the mean of the folds' logits."""
    terms, fmt_c = {}, np.zeros(n_fmt)
    fv = fmt_feats([text]).toarray()[0]
    for m, nm in zip(fold_models, names):
        coef = m.clf.coef_[0]
        x = m.vec.transform([text]).tocoo()
        for j, v in zip(x.col, x.data):
            terms[nm[j]] = terms.get(nm[j], 0.0) + float(coef[j] * v) / len(fold_models)
        fmt_c += coef[len(nm):] * fv / len(fold_models)
    return terms, fmt_c


def main(res, out_json, ex_json):
    P = res.rstrip("/") + "/"
    panel, feats, scores = P + "panel_s8/panel.parquet", P + "panel_s8/features.parquet", P + "detector_s8/ai_scores.parquet"
    OUT = {"generated_by": "code/paper_numbers.py", "checks": {}}
    rd = json.load(open(P + "report_s8/report_data.json"))

    # ---------------- A: monthly series and per-rule samples ----------------
    D = {}
    OUT["monthly"], OUT["period_share"] = {}, {}
    for rule in RULES:
        os.environ["AI_RULE"] = rule
        d = load(panel, feats, scores)
        D[rule] = d
        d["m"] = d.start_local_dt.dt.to_period("M").astype(str)
        g = d.groupby("m").ai.agg(["size", "mean"])
        OUT["monthly"][rule] = {k: {"n": int(r["size"]), "ai": float(r["mean"])} for k, r in g.iterrows()}
        OUT["period_share"][rule] = {k: float(v) for k, v in d.groupby("period").ai.mean().items()}
        ref = rd["rules"][rule]["monthly"]
        common = [k for k in ref if k in OUT["monthly"][rule]]
        OUT["checks"]["A_monthly_vs_report_" + rule] = {
            "months_compared": len(common),
            "max_abs_share_diff": max(abs(OUT["monthly"][rule][k]["ai"] - ref[k]["ai"]) for k in common),
            "months_with_different_n": int(sum(OUT["monthly"][rule][k]["n"] != ref[k]["n"] for k in common))}
    print("A done:", {r: OUT["checks"]["A_monthly_vs_report_" + r] for r in RULES})

    # ---------------- B: detector validation ----------------
    df = pd.read_parquet(panel)
    df = df[df.desc.notna() & df.start_local_dt.notna()].copy()
    df["text"] = df.desc.map(detector.norm)
    S = pd.DataFrame([json.loads(l) for l in open(P + "synth_s8/synthetic_600.jsonl") if l.strip()])
    S["text"] = S.ai_text.map(detector.norm)
    S = S[S.text.str.len() > 0].reset_index(drop=True)
    human = df[(df.start_local_dt >= "2019-01-01") & (df.start_local_dt < "2022-11-30")].copy()
    H = human[["item", "text", "agency"]].copy()
    H["y"], H["grp"] = 0, H["item"]
    A = S[["item", "text"]].copy()
    A["y"], A["grp"], A["agency"] = 1, A["item"], None
    T = pd.concat([H, A], ignore_index=True)
    oof = np.zeros(len(T))
    fold_models = []
    for tr, te in GroupKFold(n_splits=5).split(T, T.y, T.grp):
        mdl = detector.C1().fit(T.text.iloc[tr].tolist(), T.y.iloc[tr].values)
        oof[te] = mdl.score(T.text.iloc[te].tolist())
        fold_models.append(mdl)
    T["oof"] = oof
    tau = float(np.quantile(T.loc[T.y == 0, "oof"], 0.99))
    tpr = float((T.loc[T.y == 1, "oof"] >= tau).mean())
    meta = json.load(open(P + "detector_s8/detector_meta.json"))
    OUT["checks"]["B_detector_rebuild"] = {"tau": tau, "tau_saved": meta["tau"], "tpr": tpr, "tpr_saved": meta["tpr_at_tau"],
                                           "identical": bool(abs(tau - meta["tau"]) < 1e-9 and abs(tpr - meta["tpr_at_tau"]) < 1e-9)}
    S["oof"] = T.loc[T.y == 1, "oof"].values
    S["n_mark"] = S.ai_text.map(lambda t: features(t or "", None)["n_mark"])
    S["c1"] = S.oof >= tau
    S["lex"] = S.n_mark >= 2
    S["union"] = S.c1 | S.lex
    rec = {}
    for rule in RULES:
        rec[rule] = {"all": {"recall": float(S[rule].mean()), "n": int(len(S))}}
        for (v, mo), g in S.groupby(["variant", "model"]):
            rec[rule]["%s|%s" % (v, mo)] = {"recall": float(g[rule].mean()), "n": int(len(g))}
        for v, g in S.groupby("variant"):
            rec[rule][str(v)] = {"recall": float(g[rule].mean()), "n": int(len(g))}
    OUT["recall_rewrites"] = rec
    sc = pd.read_parquet(scores)[["item", "ai_flag"]]
    fe = pd.read_parquet(feats)[["item", "f_n_mark"]]
    z = df[["item", "start_local_dt"]].merge(sc, on="item", how="left").merge(fe, on="item", how="left")
    z["c1"] = z.ai_flag.fillna(0).astype(bool)
    z["lex"] = z.f_n_mark.fillna(0) >= 2
    z["union"] = z.c1 | z.lex
    pre = z[(z.start_local_dt >= "2019-01-01") & (z.start_local_dt < "2022-11-30")]
    oot = z[(z.start_local_dt >= "2022-12-01") & (z.start_local_dt < "2025-01-29")]
    OUT["fpr"] = {rule: {"pre_chatgpt": float(pre[rule].mean()), "n_pre": int(len(pre)),
                         "out_of_time": float(oot[rule].mean()), "n_oot": int(len(oot))} for rule in RULES}
    print("B done:", OUT["checks"]["B_detector_rebuild"], {r: round(rec[r]["all"]["recall"], 3) for r in RULES})

    # ---------------- B2: what our classifier keys on, word by word (PI, 2026-10-26: illustrative views) ----------------
    # detector.py scores field listings with the average of the five cross-fitted models, so their weights are
    # averaged; a phrase's contribution to one listing is its weight times its tf-idf value there (the logit adds
    # these, five format terms and the intercept)
    WV, word_examples = {}, []
    try:
        names, acc, fmt_w, icpt = fold_word_weights(fold_models)
        an = fold_models[0].vec.build_analyzer()

        def docfreq(texts):
            c = {}
            for t in texts:
                for term in set(an(t)):
                    c[term] = c.get(term, 0) + 1
            return c, max(1, len(texts))
        f0, n0 = docfreq(df[(df.start_local_dt >= "2019-01-01") & (df.start_local_dt < "2022-11-30")].text.tolist())
        f2, n2 = docfreq(df[df.start_local_dt >= "2025-02-01"].text.tolist())
        ranked = sorted(acc.items(), key=lambda kv: kv[1])

        def row(t, w):
            return {"term": t, "weight": float(w), "share_pre_chatgpt": f0.get(t, 0) / n0, "share_tool_era": f2.get(t, 0) / n2,
                    "marker_word": t in MARK}
        WV = {"top_ai": [row(t, w) for t, w in ranked[::-1][:40]], "top_human": [row(t, w) for t, w in ranked[:40]],
              "format_weights": dict(zip(["log words", "label lines", "em dash", "line breaks", "capitals"], map(float, fmt_w))),
              "intercept": icpt, "n_folds": len(fold_models),
              "n_pre_chatgpt": n0, "n_tool_era": n2}

        Xw = {json.loads(l)["item"]: json.loads(l) for l in open(P + "synth_s8/inputs_600.jsonl") if l.strip()}
        ids_ = df["item"].astype(str)
        cases = [("listing", int(i), df.loc[ids_ == str(i), "text"].iloc[0]) for i in WORD_VIEW_ITEMS if (ids_ == str(i)).any()]
        for k, x in Xw.items():
            if WORD_VIEW_SNIPPET in (x.get("original") or ""):
                cases.append(("original", int(k), detector.norm(x["original"])))
                cases += [("rewrite", int(k), t) for t in S.loc[S["item"] == k, "text"]]
        for kind, it, text in cases:
            terms, fmt_c = word_contrib(text, fold_models, names, detector.fmt_feats)
            word_examples.append({"kind": kind, "item": it, "text": text, "terms": terms,
                                  "format": dict(zip(WV["format_weights"], map(float, fmt_c))), "intercept": WV["intercept"],
                                  "score": float(np.mean([m.score([text])[0] for m in fold_models])),
                                  "marker_words": sorted(set(an(text)) & MARK)})
        WV["examples_found"] = [(c[0], c[1]) for c in cases]
    except Exception as ex:
        WV = {"error": repr(ex)}
    OUT["detector_words"] = WV
    print("B2 done:", {k: (v[:5] if isinstance(v, list) else v) for k, v in WV.items() if k in ("top_ai", "examples_found", "error")})

    # ---------------- C: controlled rewrites, paired text features ----------------
    pan = df.copy()
    full = (pan.desc.fillna("") + np.where(pan.updates_text.notna(), "\n" + pan.updates_text.fillna(""), "")).tolist()
    boiler = boilerplate_set(full)

    def spec(t):
        return "\n".join(x for x in split_sents(t or "") if norm_sent(x) not in boiler)
    X = {json.loads(l)["item"]: json.loads(l) for l in open(P + "synth_s8/inputs_600.jsonl") if l.strip()}
    rows = []
    for _, r in S.iterrows():
        x = X[r["item"]]
        fo, fn = features(spec(x["original"]), None), features(spec(r["ai_text"]), None)
        orig_all = (x.get("title") or "") + "\n" + (x.get("original") or "")
        no, nn = nums(orig_all), nums(r["ai_text"] or "")
        do, dn = stems(DEFECT, orig_all), stems(DEFECT, r["ai_text"] or "")
        row = {"item": r["item"], "variant": r["variant"], "model": r["model"],
               "numbers_kept": (len(no & nn) / len(no)) if no else np.nan,
               "defects_kept": (len(do & dn) / len(do)) if do else np.nan}
        for tag, f in (("o", fo), ("r", fn)):
            row[tag + "_w"] = f["w"]
            row[tag + "_log_w"] = np.log1p(f["w"])
            for k in ["n_num", "n_unit", "n_defect_strict", "n_quality", "n_puff", "n_you", "n_label", "n_mark"]:
                row[tag + "_" + k] = f[k]
            row[tag + "_has_quality"] = float(f["n_quality"] >= 1)
            row[tag + "_has_defect_strict"] = float(f["n_defect_strict"] >= 1)
            row[tag + "_defect_and_praise"] = float(f["n_defect_strict"] >= 1 and (f["n_quality"] >= 1 or f["n_puff"] >= 1))
        rows.append(row)
    R = pd.DataFrame(rows)
    keys = ["w", "log_w", "n_num", "n_unit", "n_defect_strict", "has_defect_strict", "n_quality", "has_quality",
            "defect_and_praise", "n_puff", "n_you", "n_label", "n_mark"]
    C = {"all": {k: paired(R["o_" + k], R["r_" + k]) for k in keys}}
    C["all"]["numbers_kept"] = {"mean": float(R.numbers_kept.mean()), "n": int(R.numbers_kept.notna().sum())}
    C["all"]["defects_kept"] = {"mean": float(R.defects_kept.mean()), "n": int(R.defects_kept.notna().sum())}
    for (v, mo), g in R.groupby(["variant", "model"]):
        C["%s|%s" % (v, mo)] = {k: paired(g["o_" + k], g["r_" + k]) for k in keys}
        C["%s|%s" % (v, mo)]["numbers_kept"] = {"mean": float(g.numbers_kept.mean()), "n": int(g.numbers_kept.notna().sum())}
        C["%s|%s" % (v, mo)]["defects_kept"] = {"mean": float(g.defects_kept.mean()), "n": int(g.defects_kept.notna().sum())}
    OUT["rewrites"] = C
    OUT["checks"]["C_kept_vs_worklog"] = {"numbers_kept": C["all"]["numbers_kept"]["mean"], "worklog": 0.977,
                                          "defects_kept": C["all"]["defects_kept"]["mean"], "worklog_defects": 0.903}
    print("C done:", OUT["checks"]["C_kept_vs_worklog"])

    # ---------------- D: balance ----------------
    B = {}
    for rule in RULES:
        d = D[rule].copy()
        d["starts_yes"] = (d.f_starts_x == 1).astype(float)
        d["drivable_yes"] = (d.f_drivable_x == 1).astype(float)
        d["title_yes"] = (d.f_title_x == 1).astype(float)
        d["veh_f"] = d.veh.astype(float)
        # AI written = flagged and started in the tool era; a pre-tool flag is a false positive with its own control
        d["flag_pre"] = (d.ai * (d.period != "P2")).astype(float)
        d["ai_tool"] = (d.ai * (d.period == "P2")).astype(float)
        out = {}
        for design, sub, fes, xs in (("tool_era", d[d.period == "P2"], ["agency", "subcat_m"], ["ai_tool"]),
                                     ("pooled", d, ["agency", "subcat_q"], ["ai_tool", "flag_pre"])):
            res_d = {}
            for y, rows_ in (("log_start", sub), ("log_photos", sub), ("reserve_any", sub), ("veh_f", sub),
                             ("age", sub[sub.veh & sub.age.notna()]), ("log_miles", sub[sub.veh & sub.log_miles.notna()]),
                             ("starts_yes", sub[sub.veh]), ("drivable_yes", sub[sub.veh]), ("title_yes", sub[sub.veh])):
                rr = rows_[rows_[y].notna()]
                o, m = ols_fe(rr, y, xs, fes=fes, cluster="agency")
                c = coef(o, m, "ai_tool")
                if c:
                    c["human_mean"] = float(rr.loc[(rr.ai_tool == 0) & (rr.flag_pre == 0), y].mean())
                res_d[y] = c
            out[design] = res_d
        B[rule] = out
    OUT["balance"] = B
    print("D done: tool-era log_start (union):", B["union"]["tool_era"]["log_start"])

    # ---------------- E: price information in the text ----------------
    E = pd.read_csv(P + "inform_s8_v3/E_spreads_draws.csv")
    spreads = {}
    for (kind, per), g in E.groupby(["kind", "period"]):
        spreads["%s|%s" % (kind, per)] = {"mean": float(g.D.mean()), "lo": float(g.D.quantile(0.025)), "hi": float(g.D.quantile(0.975)),
                                          "oos_r2": float(g.oos_r2.mean()), "draws": int(len(g))}
    ratios = {}
    for kind, g in E.groupby("kind"):
        w = g.pivot_table(index="draw", columns="period", values="D")
        for num in ["P1", "P2"]:
            q = (w[num] / w["P0"]).replace([np.inf, -np.inf], np.nan).dropna()
            ratios["%s|%s/P0" % (kind, num)] = {"mean": float(q.mean()), "lo": float(q.quantile(0.025)), "hi": float(q.quantile(0.975)), "draws": int(len(q))}
    OUT["inform"] = {"spreads": spreads, "ratios": ratios}
    OUT["checks"]["E_vs_canonical"] = {"full_P0": spreads["full|P0"], "doc_full_P0": [0.92, 0.62, 1.17],
                                       "full_P2/P0": ratios["full|P2/P0"], "doc_ratio": [1.12, 0.40, 2.12]}
    print("E done:", OUT["checks"]["E_vs_canonical"])

    # ---------------- F: review round 1 additions (each block fails alone, recorded in checks) ----------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, RidgeCV
    from textfeat import PUFF, WORD  # MARK comes from the module-level import (a local import broke block B2)
    F = {}
    d = D["union"].copy()
    d["flag_pre"] = (d.ai * (d.period != "P2")).astype(float)
    d["ai_tool"] = (d.ai * (d.period == "P2")).astype(float)
    p2 = d[d.period == "P2"].copy()
    # F1 treated counts per design (main rule, tool-era AI)
    try:
        F["treated_counts"] = {
            "tool_era_listings": int(len(p2)), "tool_era_ai": int(p2.ai_tool.sum()),
            "pooled_all": int(len(d)), "pooled_ai": int(d.ai_tool.sum()), "pooled_preflag": int(d.flag_pre.sum()),
            "pooled_priced_ai": int(d[d.log_price.notna()].ai_tool.sum()),
            "vehicles_priced_ai": int(d[d.veh & d.log_price.notna() & d.age.notna()].ai_tool.sum()),
            "vehicles_priced_preflag": int(d[d.veh & d.log_price.notna() & d.age.notna()].flag_pre.sum())}
    except Exception as ex:
        F["treated_counts"] = {"error": repr(ex)}
    # F2 flag share by page layout within the tool era (detects a layout-driven artefact)
    try:
        lay = [c for c in ("layout", "page_layout", "layout_family") if c in p2.columns]
        if lay:
            g = p2.groupby(lay[0]).agg(n=("ai", "size"), ai=("ai", "mean"))
            F["tool_era_flag_by_layout"] = {str(k): {"n": int(r.n), "ai": float(r.ai)} for k, r in g.iterrows()}
        else:
            F["tool_era_flag_by_layout"] = {"error": "no layout column in panel", "columns_with_lay": [c for c in p2.columns if "lay" in c]}
    except Exception as ex:
        F["tool_era_flag_by_layout"] = {"error": repr(ex)}
    # F3 promotional words that are NOT marker words, and the overlap itself (mechanical-link check)
    try:
        nonmark = sorted(PUFF - MARK)
        F["puff_mark_overlap"] = {"overlap": sorted(PUFF & MARK), "puff_only": nonmark, "n_puff": len(PUFF), "n_mark": len(MARK)}
        p2["s_n_puff_nomark"] = p2["spec"].fillna("").map(lambda t: len(set(WORD.findall(t.lower())) & set(nonmark)))
        res_c = {}
        for rule in RULES:
            q = D[rule][D[rule].period == "P2"].copy()
            q["s_n_puff_nomark"] = q["spec"].fillna("").map(lambda t: len(set(WORD.findall(t.lower())) & set(nonmark)))
            o, m = ols_fe(q, "s_n_puff_nomark", ["ai"], fes=["agency", "subcat_m"], cluster="agency")
            c = coef(o, m)
            if c:
                c["human_mean"] = float(q.loc[q.ai == 0, "s_n_puff_nomark"].mean())
            res_c[rule] = c
        F["promo_nomark_tool_era"] = res_c
        rwp = []
        for _, r in S.iterrows():
            x = X[r["item"]]
            a0 = len(set(WORD.findall(spec(x["original"]).lower())) & set(nonmark))
            a1 = len(set(WORD.findall(spec(r["ai_text"]).lower())) & set(nonmark))
            rwp.append((a0, a1))
        F["promo_nomark_rewrites"] = paired([a for a, _ in rwp], [b for _, b in rwp])
    except Exception as ex:
        F["promo_nomark"] = {"error": repr(ex)}
    # F4 the text-implied benchmark under other text-model settings (penalty, vocabulary)
    try:
        dd = D["union"]
        pr = dd[dd.log_price.notna()].copy()
        pr["fulltext"] = pr.desc.fillna("") + "\n" + pr.updates_text.fillna("")
        p0 = pr[(pr.period == "P0") & (~pr["item"].isin(set(X)))].copy()
        r0 = (p0.log_price - p0.subcat_q.map(p0.groupby("subcat_q").log_price.mean())).values
        variants = {"ngram12_cv": dict(ngram_range=(1, 2), min_df=3, max_features=40000, alpha=None),
                    "ngram1_cv": dict(ngram_range=(1, 1), min_df=3, max_features=40000, alpha=None),
                    "ngram12_mindf10_cv": dict(ngram_range=(1, 2), min_df=10, max_features=40000, alpha=None)}
        for a in (0.3, 3.0, 30.0):
            variants["ngram12_alpha%g" % a] = dict(ngram_range=(1, 2), min_df=3, max_features=40000, alpha=a)
        BM = {}
        for name, v in variants.items():
            vec = TfidfVectorizer(ngram_range=v["ngram_range"], min_df=v["min_df"], max_features=v["max_features"], sublinear_tf=True)
            Xp0 = vec.fit_transform(p0.fulltext)
            mdl = (Ridge(alpha=v["alpha"]) if v["alpha"] else RidgeCV(alphas=np.logspace(-2, 4, 14))).fit(Xp0, r0)
            so = mdl.predict(vec.transform([X[i]["original"] or "" for i in S["item"]]))
            sa = mdl.predict(vec.transform(S.ai_text.fillna("").tolist()))
            BM[name] = paired(so, sa)
            BM[name]["alpha"] = float(getattr(mdl, "alpha_", v["alpha"]))
        F["benchmark_robustness"] = BM
        OUT["checks"]["F_benchmark_default_vs_817664"] = {"recomputed": BM["ngram12_cv"]["diff"], "job_817664": 0.172}
    except Exception as ex:
        F["benchmark_robustness"] = {"error": repr(ex)}
    # F5 sample flow
    try:
        F["sample_flow"] = {"analysis_sample": int(len(d)), "priced": int(d.log_price.notna().sum()),
                            "vehicles_with_year_make": int(d.veh.sum()), "vehicles_priced_with_age": int((d.veh & d.log_price.notna() & d.age.notna()).sum()),
                            "by_period": {k: int(v) for k, v in d.period.value_counts().items()}}
    except Exception as ex:
        F["sample_flow"] = {"error": repr(ex)}
    OUT["review_r1"] = F
    print("F done:", {k: (list(v)[:3] if isinstance(v, dict) else v) for k, v in F.items()})

    # ---------------- G: detector-independence checks asked for in review round 1 ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from textfeat import DEFECT_STRICT
    G = {}
    nonmark = sorted(PUFF - MARK)
    S["gen"] = S["variant"].astype(str) + "|" + S["model"].astype(str)
    # G1 a classifier with no format features and no promotional or marker words (C2); its flag cannot
    #    select on length, labels or promotional vocabulary, so form gaps under C2 are not built in
    try:
        stop = sorted(PUFF | MARK)
        def c2():
            return (TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=60000, sublinear_tf=True, stop_words=stop,
                                    token_pattern=r"(?u)\b[A-Za-z][A-Za-z']+\b", lowercase=True),
                    LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced"))
        oof2 = np.zeros(len(T))
        fits = []
        for tr, te in GroupKFold(n_splits=5).split(T, T.y, T.grp):
            vec, clf = c2()
            clf.fit(vec.fit_transform(T.text.iloc[tr].tolist()), T.y.iloc[tr].values)
            oof2[te] = clf.predict_proba(vec.transform(T.text.iloc[te].tolist()))[:, 1]
            fits.append((vec, clf))
        tau2 = float(np.quantile(oof2[T.y.values == 0], 0.99))
        S["c2"] = oof2[T.y.values == 1] >= tau2
        G["c2_recall_rewrites"] = {"all": float(S.c2.mean())}
        for (v, mo), g in S.groupby(["variant", "model"]):
            G["c2_recall_rewrites"]["%s|%s" % (v, mo)] = float(g.c2.mean())
        sc2 = np.mean([clf.predict_proba(vec.transform(df.text.tolist()))[:, 1] for vec, clf in fits], axis=0)
        df["c2_score"] = sc2
        hs2 = dict(zip(T.loc[T.y == 0, "item"], oof2[T.y.values == 0]))
        df.loc[df["item"].isin(hs2), "c2_score"] = df.loc[df["item"].isin(hs2), "item"].map(hs2)
        q = D["union"].merge(df[["item", "c2_score"]], on="item", how="left")
        q["ai_c2"] = (q.c2_score >= tau2).astype(float)
        G["c2_share_by_period"] = {k: float(v) for k, v in q.groupby("period").ai_c2.mean().items()}
        q2 = q[q.period == "P2"].copy()
        q2["s_n_puff_nomark"] = q2["spec"].fillna("").map(lambda t: len(set(WORD.findall(t.lower())) & set(nonmark)))
        res = {}
        for y in ["log_w", "s_w", "s_n_puff", "s_n_puff_nomark", "s_n_label", "s_n_you", "s_n_num", "s_n_unit", "s_n_defect_strict", "has_quality", "defect_and_praise"]:
            o, m = ols_fe(q2, y, ["ai_c2"], fes=["agency", "subcat_m"], cluster="agency")
            c = coef(o, m, "ai_c2")
            if c:
                c["human_mean"] = float(q2.loc[q2.ai_c2 == 0, y].mean())
            res[y] = c
        G["c2_content_tool_era"] = res
        G["c2_tau"] = tau2
        G["c2_auc"] = float(roc_auc_score(T.y, oof2))
    except Exception as ex:
        G["c2"] = {"error": repr(ex)}
    # G2 leave-one-generator-out recall of the main classifier: train without one generator's rewrites
    try:
        rng2 = np.random.default_rng(9)
        hid = T.loc[T.y == 0, "item"].values.copy()
        rng2.shuffle(hid)
        hold = set(hid[: len(hid) // 5])
        lo_ = {}
        for gname in sorted(S.gen.unique()):
            trn = T[((T.y == 0) & (~T["item"].isin(hold))) | ((T.y == 1) & (T["item"].isin(set(S.loc[S.gen != gname, "item"]))))]
            mdl = detector.C1().fit(trn.text.tolist(), trn.y.values)
            hh = mdl.score(T.loc[(T.y == 0) & (T["item"].isin(hold)), "text"].tolist())
            tau_g = float(np.quantile(hh, 0.99))
            sg_ = mdl.score(S.loc[S.gen == gname, "text"].tolist())
            ln = S.loc[S.gen == gname, "n_mark"].values >= 2
            lo_[gname] = {"n": int((S.gen == gname).sum()), "c1_recall": float((sg_ >= tau_g).mean()), "union_recall": float(((sg_ >= tau_g) | ln).mean())}
        G["leave_one_generator_out"] = lo_
    except Exception as ex:
        G["leave_one_generator_out"] = {"error": repr(ex)}
    # G3 defect statements kept in the rewrites under the STRICT pattern used in the field
    try:
        ks = []
        for _, r in S.iterrows():
            x = X[r["item"]]
            orig_all = (x.get("title") or "") + "\n" + (x.get("original") or "")
            do = stems(DEFECT_STRICT, orig_all)
            if do:
                ks.append((r["gen"], len(do & stems(DEFECT_STRICT, r["ai_text"] or "")) / len(do)))
        kk = pd.DataFrame(ks, columns=["gen", "kept"])
        G["strict_defects_kept"] = {"all": float(kk.kept.mean()), "n": int(len(kk)), "by_generator": {g: float(v) for g, v in kk.groupby("gen").kept.mean().items()}}
    except Exception as ex:
        G["strict_defects_kept"] = {"error": repr(ex)}
    # G4 how often listings mention inspection (evidence for 'checkable'), by period and AI flag
    try:
        import re as _re
        INSP = _re.compile(r"(?i)\binspect(?:ion|ions|ed|ing)?\b")
        z2 = D["union"].copy()
        txt = z2.desc.fillna("") + " " + z2.updates_text.fillna("") + " " + (z2["terms"].fillna("") if "terms" in z2.columns else "")
        z2["mentions_inspection"] = txt.map(lambda t: 1.0 if INSP.search(t) else 0.0)
        G["inspection_share"] = {"by_period": {k: float(v) for k, v in z2.groupby("period").mentions_inspection.mean().items()},
                                 "tool_era_by_ai": {str(int(k)): float(v) for k, v in z2[z2.period == "P2"].groupby("ai").mentions_inspection.mean().items()},
                                 "all": float(z2.mentions_inspection.mean())}
    except Exception as ex:
        G["inspection_share"] = {"error": repr(ex)}
    OUT["review_r1_detector"] = G
    print("G done:", {k: (list(v)[:3] if isinstance(v, dict) else v) for k, v in G.items()})

    # ---------------- H: same agency, same start day (review round 3) ----------------
    # Tool-era listings an agency started on the same day, in batches holding both AI written and human
    # listings: batch (agency x start day) and subcategory effects, as in analysis_questions.py section 3;
    # page views add log(1 + capture lag) as in the tool-era design of analysis_core.py.
    HS = {}
    try:
        for rule in RULES:
            p2 = D[rule][D[rule].period == "P2"].copy()
            p2["batch"] = p2.agency + "|" + p2.start_local_dt.dt.strftime("%Y-%m-%d")
            p2["nq"] = p2.n_q_before_end
            b = p2.groupby("batch").ai.agg(["mean", "size"])
            mixed = b[(b["mean"] > 0) & (b["mean"] < 1)].index
            bb = p2[p2.batch.isin(mixed)]
            res_h = {"batches": int(len(mixed)), "listings": int(len(bb)), "ai_listings": int(bb.ai.sum()),
                     "agencies": int(bb.agency.nunique())}
            for ctrl, xs in (("base", ["ai", "log_photos", "reserve_any"]), ("with_start", ["ai", "log_photos", "reserve_any", "log_start"])):
                for y in ("sold", "log_price", "log_bids", "log_views", "any_q", "nq"):
                    xx = xs + (["log_caplag"] if y == "log_views" else [])
                    o, m = ols_fe(bb, y, xx, fes=["batch", "subcat"], cluster="agency")
                    c = coef(o, m)
                    if c:
                        c["mean"] = float(bb[y].mean())
                    res_h["%s|%s" % (ctrl, y)] = c
            HS[rule] = res_h
        # Session 8 record (WORKLOG s8 section 12; jobs 817630, 817632): questions with the opening bid, no capture lag
        OUT["checks"]["H_sameday_vs_s8"] = {
            "union_batches": HS["union"]["batches"], "s8_union_batches": 70,
            "union_nq": (HS["union"]["with_start|nq"] or {}).get("b"), "s8_union_nq": -0.057,
            "lex_batches": HS["lex"]["batches"], "s8_lex_batches": 62,
            "lex_nq": (HS["lex"]["with_start|nq"] or {}).get("b"), "s8_lex_nq": -0.216}
    except Exception as ex:
        HS = {"error": repr(ex)}
    OUT["same_day"] = HS
    print("H done:", OUT["checks"].get("H_sameday_vs_s8"))

    # ---------------- I: false positives on guaranteed-human text that never touched the detector ----------------
    # detector.py trains the classifier and sets its threshold on listings started 2019-01-01 to 2022-11-29, and
    # the marker words were chosen against 2019-2022 text, so listings started before 2019 entered neither.
    pre19_texts = []
    try:
        pp = pd.read_parquet(panel)[["item", "start_local_dt", "desc", "title"]]
        pp = pp[pp.desc.notna() & pp.start_local_dt.notna() & (pp.start_local_dt < "2019-01-01")]
        q = pp.merge(pd.read_parquet(scores)[["item", "ai_flag", "ai_score"]], on="item", how="left").merge(
            pd.read_parquet(feats)[["item", "f_n_mark"]], on="item", how="left")
        c1f = q.ai_flag.fillna(0).astype(bool)
        lexf = q.f_n_mark.fillna(0) >= 2
        flags = {"union": c1f | lexf, "lex": lexf, "c1": c1f}
        OUT["fpr_pre2019"] = {r: {"flagged": int(flags[r].sum()), "n": int(len(q)), "share": float(flags[r].mean())} for r in RULES}
        OUT["fpr_pre2019"]["coverage"] = {"n": int(len(q)), "missing_score": int(q.ai_flag.isna().sum()),
                                          "missing_marker_count": int(q.f_n_mark.isna().sum()),
                                          "start_min": str(q.start_local_dt.min()), "start_max": str(q.start_local_dt.max())}
        # the flagged listings and the five highest-scoring unflagged ones, to be read as text (examples file only)
        top = q.sort_values("ai_score", ascending=False)
        pre19_texts = pd.concat([q[flags["union"]], top[~flags["union"].loc[top.index]].head(5)])[
            ["item", "title", "desc", "ai_score", "f_n_mark"]].to_dict(orient="records")
    except Exception as ex:
        OUT["fpr_pre2019"] = {"error": repr(ex)}
    print("I done:", OUT["fpr_pre2019"])

    # ---------------- X: example listings (raw text; kept out of git) ----------------
    p = pd.read_parquet(panel)
    hit = p[p.desc.fillna("").str.contains("Bad Motor, Engine knocking", regex=False)]
    want = {str(i) for i in EXAMPLE_ITEMS} | {str(i) for i in hit["item"].tolist()}   # item numbers are text in the panel
    ex = p[p["item"].astype(str).isin(want)][["item", "title", "desc", "updates_text", "agency", "start_local_dt", "final_bid", "n_bids", "page_views", "cat1", "cat2"]].copy()
    ex["start_local_dt"] = ex.start_local_dt.astype(str)
    sc_all = pd.read_parquet(scores)[["item", "ai_score", "ai_flag"]]
    fe_all = pd.read_parquet(feats)[["item", "f_n_mark", "s_w"]]
    ex = ex.merge(sc_all, on="item", how="left").merge(fe_all, on="item", how="left")
    rw = S[S["item"].astype(str).isin(want)][["item", "variant", "model", "ai_text"]].merge(
        pd.DataFrame([{"item": k, "orig_title": v.get("title"), "orig_text": v.get("original")} for k, v in X.items() if str(k) in want],
                     columns=["item", "orig_title", "orig_text"]), on="item", how="left")
    json.dump({"field": ex.to_dict(orient="records"), "rewrites": rw.to_dict(orient="records"), "pre2019_flagged_and_top": pre19_texts,
               "word_views": word_examples},
              open(ex_json, "w"), indent=1, default=str)
    OUT["checks"]["X_examples"] = {"field_found": int(len(ex)), "rewrites_found": int(len(rw)), "ram_hits": hit["item"].tolist()}

    json.dump(OUT, open(out_json, "w"), indent=1, default=float)
    print("wrote", out_json, "and", ex_json)
    print("CHECKS:", json.dumps(OUT["checks"], default=float)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:])
