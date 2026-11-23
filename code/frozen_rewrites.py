#!/usr/bin/env python3
"""Session 8: how much would an AI rewrite of the SAME listing have been 'worth' under the pre-ChatGPT
market's text-to-price mapping?

Fit the frozen P0 model (TF-IDF 1-2 grams, ridge; target = log closing bid minus subcategory x
end-quarter mean; P0 = listings started 2019-01-01..2022-11-29 with >= 1 bid), excluding the 600
listings whose descriptions were rewritten (so no original is scored by a model trained on it).
Score each original description and its rewrite(s); report the mean change in predicted log price
(rewrite minus original) by generator, with a paired bootstrap interval.

Session 12 (PI, 2026-11-18; the advisor's C4: "Does AI make low-quality products sound more similar to high-quality
products?"). --quality OUT_DIR also sets each pair beside the item's realized quality, r = its own log closing bid minus
the mean log closing bid of the model's training listings in its subcategory and end quarter (a price set before any
rewrite; None when the item drew no bid or its cell has no training listing). It reports, for all pairs and by
instruction, with pair-bootstrap 95% intervals: the slope of the rewrite's lift (s_ai - s_orig) on r (below zero:
cheap items gain more, so rewritten items read more alike); the correlation of each score with r; the gap in score
between the top and bottom thirds of r before and after rewriting, and the share of it the rewrite keeps; the spread
of the scores. A permutation of r across pairs gives the slope's null distribution on the real data. Writes
OUT_DIR/quality.json (aggregates only) and OUT_DIR/pair_scores.json (per pair: item number, instruction, model, scores,
r, category, a strict-defect flag, word count; no listing text). The mean lift must reproduce job 817664 (+0.172).
Session 12, buyers first (PI, 2026-11-20: "Go, with analysis 1"). --reference OUT_JSON gives the market's non-price
outcomes the reference point that +18.8% gives price: for received a bid, log(1 + bids), log(1 + page views) and buyer
questions before the close, the same text model (TF-IDF 1-2 grams, ridge, target = the outcome minus its subcategory x
end-quarter mean) is fit on the pre-ChatGPT listings, the 600 rewritten originals left out, and scores each original and
its rewrite; the mean change (rewrite minus original) with a pair-bootstrap interval is what buyers who responded to
text as before ChatGPT would have done differently for the rewrite. Each model's predictions are checked out of sample
against the 600 originals' own outcomes, the log-price model must reproduce job 817664's +0.172, and the top-weighted
words are printed. Writes aggregates only (no listing text) to OUT_JSON.
Usage: python3 frozen_rewrites.py PANEL FEATURES AISCORES INPUTS.jsonl SYNTH.jsonl [--quality OUT_DIR | --reference OUT_JSON]
       python3 frozen_rewrites.py --selftest   (the statistics on made-up worlds; each must come out as built)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from analysis_core import load  # noqa: E402


def quality_stats(h, cuts):
    """h: pairs with s_orig, s_ai and r all present; cuts: the two tercile cut points of r (fixed on all priced
    originals, since quality belongs to the item). Returns the statistics described in the module docstring."""
    x, so, sa = h.r.values, h.s_orig.values, h.s_ai.values
    lift = sa - so
    lo, hi = x <= cuts[0], x > cuts[1]
    gap_o = so[hi].mean() - so[lo].mean()
    gap_a = sa[hi].mean() - sa[lo].mean()
    return {"n": int(len(h)), "slope_lift_on_r": float(np.polyfit(x, lift, 1)[0]),
            "corr_orig_r": float(np.corrcoef(so, x)[0, 1]), "corr_ai_r": float(np.corrcoef(sa, x)[0, 1]),
            "corr_diff": float(np.corrcoef(sa, x)[0, 1] - np.corrcoef(so, x)[0, 1]),
            "gap_orig": float(gap_o), "gap_ai": float(gap_a), "gap_kept": float(gap_a / gap_o),
            "sd_orig": float(so.std(ddof=1)), "sd_ai": float(sa.std(ddof=1)),
            "sd_ratio": float(sa.std(ddof=1) / so.std(ddof=1)),
            "lift_low": float(lift[lo].mean()), "lift_mid": float(lift[~lo & ~hi].mean()),
            "lift_high": float(lift[hi].mean()), "r_gap": float(x[hi].mean() - x[lo].mean()),
            "n_low": int(lo.sum()), "n_mid": int((~lo & ~hi).sum()), "n_high": int(hi.sum())}


def with_intervals(h, cuts, rng, draws=2000):
    """quality_stats on h, each value with a pair-bootstrap 95% interval (pairs resampled with replacement)."""
    est = quality_stats(h, cuts)
    keys = [k for k in est if not k.startswith("n")]
    bs = {k: [] for k in keys}
    for _ in range(draws):
        b = quality_stats(h.sample(len(h), replace=True, random_state=int(rng.integers(0, 2**31 - 1))), cuts)
        for k in keys:
            bs[k].append(b[k])
    return {k: ({"est": est[k], "lo": float(np.quantile(bs[k], .025)), "hi": float(np.quantile(bs[k], .975))}
                if k in keys else est[k]) for k in est}


def permutation_p(h, rng, draws=2000):
    """Share of permutations of r across pairs whose |slope of lift on r| reaches the observed one."""
    obs = abs(np.polyfit(h.r.values, h.s_ai.values - h.s_orig.values, 1)[0])
    lift = h.s_ai.values - h.s_orig.values
    hits = sum(abs(np.polyfit(rng.permutation(h.r.values), lift, 1)[0]) >= obs for _ in range(draws))
    return float(hits / draws)


def selftest():
    """Three made-up worlds with a known answer; the statistics must find each one."""
    rng = np.random.default_rng(1)
    n = 600
    r = rng.normal(0, 1, n)
    s_orig = 0.4 * r + rng.normal(0, 0.4, n)
    cuts = np.quantile(r, [1 / 3, 2 / 3])
    uniform = quality_stats(pd.DataFrame({"r": r, "s_orig": s_orig, "s_ai": s_orig + 0.17 + rng.normal(0, .02, n)}), cuts)
    assert abs(uniform["slope_lift_on_r"]) < 0.02 and abs(uniform["gap_kept"] - 1) < 0.05, uniform
    halved = quality_stats(pd.DataFrame({"r": r, "s_orig": s_orig, "s_ai": 0.5 * s_orig + 0.17 + rng.normal(0, .02, n)}), cuts)
    assert halved["slope_lift_on_r"] < -0.15 and abs(halved["gap_kept"] - 0.5) < 0.05, halved
    blind = pd.DataFrame({"r": r, "s_orig": s_orig, "s_ai": 0.17 + rng.normal(0, .3, n)})   # rewrite ignores the item
    b = quality_stats(blind, cuts)
    assert abs(b["corr_ai_r"]) < 0.1 and b["gap_kept"] < 0.2 and b["slope_lift_on_r"] < -0.3, b
    p_none = permutation_p(pd.DataFrame({"r": rng.permutation(r), "s_orig": s_orig, "s_ai": 0.5 * s_orig}), rng, 500)
    p_real = permutation_p(pd.DataFrame({"r": r, "s_orig": s_orig, "s_ai": 0.5 * s_orig}), rng, 500)
    assert p_real < 0.01 < p_none, (p_real, p_none)
    print("selftest: a uniform lift reads as no compression (slope %+.3f, gap kept %.2f); halving is found (slope %+.3f,"
          " gap kept %.2f); a rewrite blind to the item keeps %.2f of the gap; permutation p %.3f with a link, %.3f without"
          % (uniform["slope_lift_on_r"], uniform["gap_kept"], halved["slope_lift_on_r"], halved["gap_kept"],
             b["gap_kept"], p_real, p_none))


def mixed_change(Q, share, rng, draws=2000):
    """A market in which a random `share` of items carry their rewrite and the rest their original (no page marks
    which): the change, against all originals, in the correlation of the pricing model's score with realized quality
    and in the gap between the top and bottom thirds of quality; mean and 2.5 and 97.5 percentiles over random mixes."""
    x, so, sa = Q.r.values, Q.s_orig.values, Q.s_ai.values
    c0 = np.corrcoef(so, x)[0, 1]
    cuts = np.quantile(x, [1 / 3, 2 / 3])
    lo, hi = x <= cuts[0], x > cuts[1]
    g0 = so[hi].mean() - so[lo].mean()
    dc, dg = [], []
    for _ in range(draws):
        pick = rng.random(len(x)) < share
        s = np.where(pick, sa, so)
        dc.append(np.corrcoef(s, x)[0, 1] - c0)
        dg.append((s[hi].mean() - s[lo].mean()) / g0)
    q = lambda v: {"mean": float(np.mean(v)), "lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975))}
    return {"share": share, "corr_all_original": float(c0), "corr_change": q(dc), "gap_kept": q(dg)}


def selftest_mixed():
    """Rewrites that add a fixed lift to a random part of the items lower the correlation of score and quality; no
    lift leaves it unchanged; and a lift aimed at cheap items lowers it more."""
    rng = np.random.default_rng(2)
    n = 600
    r = rng.normal(0, 1, n)
    so = 0.4 * r + rng.normal(0, 0.4, n)
    none = mixed_change(pd.DataFrame({"r": r, "s_orig": so, "s_ai": so}), 0.27, rng, 300)
    assert abs(none["corr_change"]["mean"]) < 1e-9, none
    flat = mixed_change(pd.DataFrame({"r": r, "s_orig": so, "s_ai": so + 0.5}), 0.27, rng, 300)
    aimed = mixed_change(pd.DataFrame({"r": r, "s_orig": so, "s_ai": so + 0.5 * (r < 0)}), 0.27, rng, 300)
    assert flat["corr_change"]["mean"] < -0.02 and aimed["corr_change"]["mean"] < flat["corr_change"]["mean"], (flat, aimed)
    print("selftest (mixed market): no lift %+.3f; a flat lift %+.3f; a lift aimed at cheap items %+.3f" % (
        none["corr_change"]["mean"], flat["corr_change"]["mean"], aimed["corr_change"]["mean"]))


def mixed(pair_scores, quality_json, out_json, share=0.27):
    """Session 12 (round 1 of review, the marketing referee): no page marks AI text, so a reader cannot subtract a
    boost it cannot see. Reads the quality job's per-pair scores (job 988391; no listing text) and writes, for markets
    in which a random 27% (the estimated truly AI-written share of the assistant era), half or all of the items carry
    their rewrite, the change in the pricing model's ranking of realized quality; and the boost for the originals that
    drew no bid, which the quality test cannot use, beside the priced ones."""
    selftest_mixed()
    P = pd.DataFrame(json.load(open(pair_scores)))
    assert len(P) == 600
    Q = P[P.r.notna()].copy()
    rng = np.random.default_rng(27)
    out = {"source": "code/frozen_rewrites.py --mixed (Session 12) on results/quality_s12/pair_scores.json (job 988391)",
           "n_quality": int(len(Q)), "mixes": {}}
    for nm, h in (("all", Q), ("appealing", Q[Q.variant == "V1_enhance"])):
        out["mixes"][nm] = [mixed_change(h, s, rng) for s in (share, 0.5, 1.0)]
    d = P.s_ai - P.s_orig
    pr, un = d[P.has_price], d[~P.has_price]
    bs = [rng.choice(un.values, len(un)).mean() - rng.choice(pr.values, len(pr)).mean() for _ in range(2000)]
    out["boost_unpriced"] = {"n_unpriced": int(len(un)), "n_priced": int(len(pr)), "mean_unpriced": float(un.mean()),
                             "mean_priced": float(pr.mean()), "diff": float(un.mean() - pr.mean()),
                             "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975))}
    json.dump(out, open(out_json, "w"), indent=1)
    for nm in out["mixes"]:
        for m in out["mixes"][nm]:
            print("%-9s share %.2f: correlation %.3f -> change %+.3f [%+.3f, %+.3f]; gap kept %.2f [%.2f, %.2f]" % (
                nm, m["share"], m["corr_all_original"], m["corr_change"]["mean"], m["corr_change"]["lo"], m["corr_change"]["hi"],
                m["gap_kept"]["mean"], m["gap_kept"]["lo"], m["gap_kept"]["hi"]))
    b = out["boost_unpriced"]
    print("boost: unpriced %d %+.3f, priced %d %+.3f, difference %+.3f [%+.3f, %+.3f]" % (
        b["n_unpriced"], b["mean_unpriced"], b["n_priced"], b["mean_priced"], b["diff"], b["lo"], b["hi"]))
    print("wrote", out_json)


def selftest_trimmed():
    """The trim can fail: a slope driven by four extreme lots is found before the trim and gone after it, and a slope
    spread over all pairs survives the trim."""
    rng = np.random.default_rng(3)
    n = 300
    r = rng.normal(0, 1, n)
    base = pd.DataFrame({"r": np.r_[r, [-7.0] * 4], "s_orig": 0.0})
    base["s_ai"] = 0.17 + rng.normal(0, 0.02, n + 4) + np.r_[np.zeros(n), [0.6] * 4]
    full = np.polyfit(base.r, base.s_ai - base.s_orig, 1)[0]
    t = base[base.r >= -5]
    trim = np.polyfit(t.r, t.s_ai - t.s_orig, 1)[0]
    assert full < -0.02 and abs(trim) < 0.01, (full, trim)
    real = base.copy()
    real["s_ai"] = 0.17 - 0.05 * real.r + rng.normal(0, 0.02, n + 4)
    t = real[real.r >= -5]
    assert np.polyfit(t.r, t.s_ai - t.s_orig, 1)[0] < -0.04
    print("selftest_trimmed: an outlier-driven slope is caught and removed; a real slope survives the trim")


def trimmed(pair_scores, quality_json, out_json, cut=-5.0):
    """Session 12 (review round 3): the "appealing" slope's permutation p of 0.041 rests on a few lots priced more than
    500 log points below their cell (r below -5: closing bids under a hundredth of the cell's geometric mean). The slope
    of the boost on quality without them, with its pair-bootstrap interval and permutation p, for all pairs and by
    instruction; the tercile cuts stay the quality job's (quality belongs to the item). Reads the per-pair scores of job
    988391 (no listing text) and checks that the untrimmed slopes reproduce that job's."""
    selftest_trimmed()
    P = pd.DataFrame(json.load(open(pair_scores)))
    assert len(P) == 600
    QJ = json.load(open(quality_json))
    cuts = QJ["r_cuts"]
    Q = P[P.r.notna()].copy()
    assert len(Q) == QJ["n_quality"], (len(Q), QJ["n_quality"])
    rng = np.random.default_rng(5)
    drop = Q[Q.r < cut]
    out = {"source": "code/frozen_rewrites.py --trimmed (Session 12) on results/quality_s12/pair_scores.json (job 988391)",
           "cut_r": cut, "n_quality": int(len(Q)), "dropped": [{"item": str(x.item), "variant": x.variant, "r": float(x.r)}
                                                                  for x in drop.itertuples()], "groups": {}}
    for nm, h in (("all", Q), ("appealing", Q[Q.variant == "V1_enhance"]), ("concise", Q[Q.variant == "V2_concise"])):
        full = float(np.polyfit(h.r.values, h.s_ai.values - h.s_orig.values, 1)[0])
        assert abs(full - QJ["groups"][nm]["slope_lift_on_r"]["est"]) < 1e-9, (nm, full)
        t = h[h.r >= cut]
        st = with_intervals(t, cuts, rng)
        out["groups"][nm] = {"n": int(len(t)), "n_dropped": int(len(h) - len(t)), "slope_full": full,
                             "slope": st["slope_lift_on_r"], "perm_p": permutation_p(t, rng), "gap_kept": st["gap_kept"]}
        g = out["groups"][nm]
        print("%-9s n %d (dropped %d): slope %+.4f -> %+.4f [%+.4f, %+.4f], permutation p %.3f" % (
            nm, g["n"], g["n_dropped"], full, g["slope"]["est"], g["slope"]["lo"], g["slope"]["hi"], g["perm_p"]))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


REF_OUTCOMES = [("sold", "received a bid", "pp"), ("log_bids", "bids, log of one plus the count", "pct"),
                ("log_views", "page views, log of one plus the count", "pct"),
                ("n_q_before_end", "buyer questions before the close, per listing", "count")]


# the 32 promotional words of textfeat.PUFF (checked equal at run time), planted in the round-8 check
PROMO_32 = ["amazing", "awesome", "beautiful", "comfortable", "convenient", "dependable", "durable", "efficient", "elegant",
            "excellent", "exceptional", "fantastic", "great", "ideal", "impressive", "nice", "outstanding", "perfect",
            "powerful", "premium", "pristine", "quality", "reliable", "robust", "sleek", "spacious", "sturdy", "stylish",
            "superb", "terrific", "versatile", "wonderful"]


def round8_checks(v, m, originals, R, units, rng):
    """Session 12, review round 8 (the Opus referee): does the model see selling language, and how much does it pull a
    rewrite's score toward the cell mean? Plant the 32 promotional words in every original; report the slope of the
    rewrite's score on the original's across pairs (below one: the rewrite's scores lie closer together)."""
    fake = [(i, "P", "promo", originals[i] + " " + " ".join(PROMO_32)) for i in sorted(originals)]
    Rp = reference_deltas(v, m, originals, fake)
    b_ = boot_mean(Rp.delta.values, rng)
    an = v.build_analyzer()
    known = sum(1 for w in PROMO_32 if w in v.vocabulary_)
    slope = float(np.polyfit(R.s_orig.values, R.s_ai.values, 1)[0])
    return {"promo_planted": dict(b_, est_units=float(to_units(b_["est"], units)), lo_units=float(to_units(b_["lo"], units)),
                                  hi_units=float(to_units(b_["hi"], units)), words_in_vocabulary=int(known)),
            "slope_rewrite_on_original": slope}


def fit_text_model(texts, y, cells):
    """The pricing model's design for any outcome: TF-IDF 1-2 grams and ridge on y minus its cell mean."""
    r0 = y - y.groupby(cells).transform("mean")
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    m = RidgeCV(alphas=np.logspace(-2, 4, 14)).fit(v.fit_transform(texts), r0.values)
    return v, m, y.groupby(cells).mean()


def boot_mean(x, rng, draws=2000):
    x = np.asarray(x, dtype=float)
    n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(draws)])
    return {"est": float(x.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n": int(n)}


def boot_corr(a, b, rng, draws=2000):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    bs = []
    for _ in range(draws):
        k = rng.integers(0, n, n)
        bs.append(np.corrcoef(a[k], b[k])[0, 1])
    return {"est": float(np.corrcoef(a, b)[0, 1]), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)),
            "n": int(n)}


def reference_deltas(v, m, originals, rewrites):
    """originals: {item: text}; rewrites: list of (item, variant, model, text). Returns per-pair scores and changes."""
    items = sorted(originals)
    so = dict(zip(items, m.predict(v.transform([originals[k] for k in items]))))
    sa = m.predict(v.transform([t for _, _, _, t in rewrites]))
    R = pd.DataFrame([{"item": i, "variant": va, "model": mo, "s_orig": float(so[i]), "s_ai": float(x)}
                      for (i, va, mo, _), x in zip(rewrites, sa)])
    R["delta"] = R.s_ai - R.s_orig
    return R


def to_units(x, units):
    return 100 * x if units == "pp" else (100 * (np.exp(x) - 1) if units == "pct" else x)


def selftest_reference():
    """Three made-up worlds the reference statistic must get right: an outcome that rises with a word the rewrites add
    (the change must be positive, its interval above zero); an outcome that is pure noise (the interval must contain
    zero); and rewrites that add only words the training texts never use (the change must be zero to rounding)."""
    rng = np.random.default_rng(31)
    vocab = ["w%d" % j for j in range(150)]
    def doc():
        return " ".join(rng.choice(vocab, 25)) + " great" * int(rng.poisson(1.0))
    train = [doc() for _ in range(2500)]
    cells = pd.Series(rng.integers(0, 15, len(train)).astype(str))
    great = pd.Series([t.split().count("great") for t in train], dtype=float)
    cell_fx = cells.map({str(c): rng.normal(0, 1) for c in range(15)}).astype(float)
    y_sig = 0.3 * great + cell_fx + pd.Series(rng.normal(0, 0.5, len(train)))
    y_noise = cell_fx + pd.Series(rng.normal(0, 0.5, len(train)))
    originals = {"i%d" % k: doc() for k in range(300)}
    add_great = [(k, "V1", "m", t + " great great") for k, t in originals.items()]
    add_unseen = [(k, "V1", "m", t + " zzqqx yyqqz") for k, t in originals.items()]
    out = {}
    for name, y, rw, test in (("planted", y_sig, add_great, lambda b: b["lo"] > 0.05),
                              ("noise", y_noise, add_great, lambda b: b["lo"] <= 0 <= b["hi"] or abs(b["est"]) < 0.02),
                              ("unseen", y_sig, add_unseen, lambda b: abs(b["est"]) < 1e-9)):
        v, m, _ = fit_text_model(pd.Series(train), y, cells)
        R = reference_deltas(v, m, originals, rw)
        b = boot_mean(R.delta.values, rng, 500)
        out[name] = b
        assert test(b), (name, b)
        print("selftest reference [%s]: mean change %+.4f [%+.4f, %+.4f]: as built" % (name, b["est"], b["lo"], b["hi"]))
    return out


def reference(panel, feats, scores, inputs, synth, out_json):
    d = load(panel, feats, scores)
    d["fulltext"] = d.desc.fillna("") + "\n" + d.updates_text.fillna("")
    X = {json.loads(l)["item"]: json.loads(l) for l in open(inputs)}
    S = [json.loads(l) for l in open(synth) if l.strip()]
    originals = {k: x["original"] for k, x in X.items()}
    rewrites = [(r["item"], r["variant"], r["model"], r["ai_text"]) for r in S]
    from textfeat import PUFF as PUFF_TF
    assert set(PROMO_32) == set(PUFF_TF), "PROMO_32 must equal textfeat.PUFF"
    # round 8: the title, which buyers see beside every description, written into both texts and into the training text
    assert all(x.get("title") is not None for x in X.values()), "every original needs its title"
    originals_t = {k: x["title"] + "\n" + x["original"] for k, x in X.items()}
    rewrites_t = [(i, va, mo, X[i]["title"] + "\n" + t) for i, va, mo, t in rewrites]
    d["titletext"] = d.title.fillna("") + "\n" + d.fulltext
    meta = d.set_index("item")
    rng = np.random.default_rng(41)
    res = {"source": "code/frozen_rewrites.py --reference (Session 12, buyers first); the pricing model's design (job 817664)"
                     " for each outcome; pair-bootstrap intervals hold each fitted model fixed",
           "n_pairs": len(rewrites), "outcomes": {}}
    # the machinery must reproduce the pricing model's lift
    pz = d[(d.period == "P0") & d.log_price.notna() & ~d["item"].isin(set(X))]
    v, m, _ = fit_text_model(pz.fulltext, pz.log_price, pz.subcat_q)
    R = reference_deltas(v, m, originals, rewrites)
    assert abs(R.delta.mean() - 0.172) < 0.0015, ("does not reproduce job 817664's +0.172", R.delta.mean())
    res["reproduce_log_price"] = {"n_train": int(len(pz)), "mean_delta": float(R.delta.mean())}
    res["log_price_checks"] = round8_checks(v, m, originals, R, "pct", rng)
    print("log price: promotional words planted %+.2f%% [%+.2f, %+.2f] (%d of 32 in the vocabulary); slope of the rewrite's"
          " score on the original's %.3f" % (res["log_price_checks"]["promo_planted"]["est_units"],
          res["log_price_checks"]["promo_planted"]["lo_units"], res["log_price_checks"]["promo_planted"]["hi_units"],
          res["log_price_checks"]["promo_planted"]["words_in_vocabulary"], res["log_price_checks"]["slope_rewrite_on_original"]))
    print("log price reproduced: %d training listings, mean change %+.4f (job 817664: +0.172)" % (len(pz), R.delta.mean()))
    for y, label, units in REF_OUTCOMES:
        tr = d[(d.period == "P0") & d[y].notna() & ~d["item"].isin(set(X))]
        v, m, cellmean = fit_text_model(tr.fulltext, tr[y], tr.subcat_q)
        R = reference_deltas(v, m, originals, rewrites)
        out = {"label": label, "units": units, "n_train": int(len(tr)), "alpha": float(m.alpha_),
               "train_mean": float(tr[y].mean()), "groups": {}}
        for g, h in (("all", R), ("appealing", R[R.variant == "V1_enhance"]), ("concise", R[R.variant == "V2_concise"])):
            b = boot_mean(h.delta.values, rng)
            out["groups"][g] = dict(b, est_units=float(to_units(b["est"], units)), lo_units=float(to_units(b["lo"], units)),
                                    hi_units=float(to_units(b["hi"], units)), share_positive=float((h.delta > 0).mean()))
        # out of sample: the 600 originals are not in the training data; their own outcomes net of the cell mean
        oo = [(R.drop_duplicates("item").set_index("item").s_orig[i], meta[y][i] - cellmean[meta.subcat_q[i]])
              for i in originals if i in meta.index and meta[y][i] == meta[y][i] and meta.subcat_q[i] in cellmean.index]
        out["oos"] = boot_corr([a for a, _ in oo], [b for _, b in oo], rng)
        vocab = v.get_feature_names_out()
        order = np.argsort(m.coef_)
        out["top_words"] = [str(vocab[j]) for j in order[::-1][:12]]
        out["bottom_words"] = [str(vocab[j]) for j in order[:12]]
        # Session 12, review round 7 (the Fable referee): a version of the statistic that can fail on the real model.
        # Planted: append the model's own ten highest-weight (and, apart, lowest-weight) single words to every original; the
        # mean change must come out clearly above (below) zero, so a near-zero change for the real rewrites is not a model
        # that cannot see wording. Scale: the spread of the model's scores across the originals and of the pairs' changes.
        uni = [j for j in order[::-1] if " " not in vocab[j]]
        for nm_, words_ in (("planted_top", [str(vocab[j]) for j in uni[:10]]),
                            ("planted_bottom", [str(vocab[j]) for j in uni[::-1][:10]])):
            fake = [(i, "P", "planted", originals[i] + " " + " ".join(words_)) for i in sorted(originals)]
            Rp = reference_deltas(v, m, originals, fake)
            b_ = boot_mean(Rp.delta.values, rng)
            out[nm_] = dict(b_, words=words_, est_units=float(to_units(b_["est"], units)),
                            lo_units=float(to_units(b_["lo"], units)), hi_units=float(to_units(b_["hi"], units)))
        so_ = R.drop_duplicates("item").s_orig.values
        out["scale"] = {"sd_score_originals": float(np.std(so_)), "sd_change_pairs": float(np.std(R.delta.values)),
                        "mean_abs_change": float(np.mean(np.abs(R.delta.values))),
                        "sd_score_originals_units": float(to_units(np.std(so_), units) if units != "pct" else 100 * np.std(so_)),
                        "sd_change_pairs_units": float(to_units(np.std(R.delta.values), units) if units != "pct" else 100 * np.std(R.delta.values))}
        out.update(round8_checks(v, m, originals, R, units, rng))
        print("   promotional words planted %+.3f [%+.3f, %+.3f] %s (%d of 32 in the vocabulary); slope of the rewrite's score on"
              " the original's %.3f" % (out["promo_planted"]["est_units"], out["promo_planted"]["lo_units"],
              out["promo_planted"]["hi_units"], units, out["promo_planted"]["words_in_vocabulary"], out["slope_rewrite_on_original"]))
        res["outcomes"][y] = out
        a_ = out["groups"]["all"]
        print("%-15s train %5d alpha %7.3g | mean change %+.4f [%+.4f, %+.4f] = %+.2f [%+.2f, %+.2f] %s | appealing %+.2f,"
              " concise %+.2f | positive %.2f | out of sample corr %.3f [%.3f, %.3f] (n %d)"
              % (y, len(tr), m.alpha_, a_["est"], a_["lo"], a_["hi"], a_["est_units"], a_["lo_units"], a_["hi_units"], units,
                 out["groups"]["appealing"]["est_units"], out["groups"]["concise"]["est_units"], a_["share_positive"],
                 out["oos"]["est"], out["oos"]["lo"], out["oos"]["hi"], out["oos"]["n"]))
        print("   planted top ten words: %+.3f [%+.3f, %+.3f] %s | planted bottom ten: %+.3f [%+.3f, %+.3f] | scale: sd of scores"
              " across originals %.4f, sd of pair changes %.4f, mean |change| %.4f" % (
                  out["planted_top"]["est_units"], out["planted_top"]["lo_units"], out["planted_top"]["hi_units"], units,
                  out["planted_bottom"]["est_units"], out["planted_bottom"]["lo_units"], out["planted_bottom"]["hi_units"],
                  out["scale"]["sd_score_originals"], out["scale"]["sd_change_pairs"], out["scale"]["mean_abs_change"]))
        assert out["planted_top"]["lo"] > 0 and out["planted_bottom"]["hi"] < 0, ("the planted check failed", y)
        print("   coverage: %d of %d originals with the outcome in a training cell; top words: %s | bottom: %s"
              % (out["oos"]["n"], len(originals), ", ".join(out["top_words"]), ", ".join(out["bottom_words"])))
        sh = R.sort_values("delta")
        for tag, part in (("lowest", sh.head(3)), ("highest", sh.tail(3))):
            for _, x in part.iterrows():
                print("   %s %s %s|%s change %+.4f | original: %s | rewrite: %s" % (
                    tag, x["item"], x.variant, x.model, x.delta, originals[x["item"]][:110].replace("\n", " "),
                    [t for i, va, mo, t in rewrites if i == x["item"] and va == x.variant][0][:110].replace("\n", " ")))
    # round 8: the same reference points with the title in every text (training listings, originals and rewrites)
    res["title_variant"] = {"note": "the title prepended to the training text, the original and the rewrite", "outcomes": {}}
    for y, label, units in [("log_price", "closing bid, log", "pct")] + REF_OUTCOMES:
        tr = (d[(d.period == "P0") & d.log_price.notna() & ~d["item"].isin(set(X))] if y == "log_price"
              else d[(d.period == "P0") & d[y].notna() & ~d["item"].isin(set(X))])
        v, m, _ = fit_text_model(tr.titletext, tr[y], tr.subcat_q)
        R = reference_deltas(v, m, originals_t, rewrites_t)
        out = {"label": label, "units": units, "n_train": int(len(tr)), "alpha": float(m.alpha_), "groups": {}}
        for g, h in (("all", R), ("appealing", R[R.variant == "V1_enhance"]), ("concise", R[R.variant == "V2_concise"])):
            b = boot_mean(h.delta.values, rng)
            out["groups"][g] = dict(b, est_units=float(to_units(b["est"], units)), lo_units=float(to_units(b["lo"], units)),
                                    hi_units=float(to_units(b["hi"], units)))
        out.update(round8_checks(v, m, originals_t, R, units, rng))
        res["title_variant"]["outcomes"][y] = out
        g_ = out["groups"]
        print("with the title: %-15s all %+.3f [%+.3f, %+.3f] %s | appealing %+.3f [%+.3f, %+.3f] | concise %+.3f [%+.3f, %+.3f] |"
              " promotional words planted %+.3f | slope %.3f" % (y, g_["all"]["est_units"], g_["all"]["lo_units"],
              g_["all"]["hi_units"], units, g_["appealing"]["est_units"], g_["appealing"]["lo_units"], g_["appealing"]["hi_units"],
              g_["concise"]["est_units"], g_["concise"]["lo_units"], g_["concise"]["hi_units"], out["promo_planted"]["est_units"],
              out["slope_rewrite_on_original"]))
    json.dump(res, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def reference_real(panel, feats, scores, inputs, out_json):
    """Session 12, review round 8 (the Opus referee's fix 3): the outcome models scored on real assistant-era listings,
    which carry heavier polish than our rewrites. Each model (the reference design, fit on the pre-ChatGPT listings
    outside the 600 originals; with and without the title in the text) predicts every assistant-era listing's outcome
    from its own text; the assistant-era design (seller and subcategory-by-month fixed effects, photos, reserve, and the
    capture lag for page views; clustered by seller) is run on the prediction (the expected gap: what buyers who kept the
    pre-ChatGPT relation would have done) and on the outcome itself (the realized gap, which must reproduce the paper's
    assistant-era design). Writes aggregates only."""
    from regs import ols_fe
    d = load(panel, feats, scores)
    d["fulltext"] = d.desc.fillna("") + "\n" + d.updates_text.fillna("")
    d["titletext"] = d.title.fillna("") + "\n" + d.fulltext
    X = {json.loads(l)["item"] for l in open(inputs)}
    p2 = d[d.period == "P2"].copy()
    res = {"source": "code/frozen_rewrites.py --reference-real (Session 12, review round 8)", "rule": d.attrs.get("ai_rule"),
           "variants": {}}
    for var, col in (("description", "fulltext"), ("title", "titletext")):
        res["variants"][var] = {}
        for y, label, units in [("log_price", "closing bid, log", "pct")] + REF_OUTCOMES:
            tr = d[(d.period == "P0") & d[y].notna() & ~d["item"].isin(X)]
            v, m, _ = fit_text_model(tr[col], tr[y], tr.subcat_q)
            h = p2[p2[y].notna()].copy()
            h["yhat"] = m.predict(v.transform(h[col]))
            xs = ["ai", "log_photos", "reserve_any"] + (["log_caplag"] if y == "log_views" else [])
            o_e, m_e = ols_fe(h, "yhat", xs, fes=["agency", "subcat_m"], cluster="agency")
            o_r, m_r = ols_fe(h, y, xs, fes=["agency", "subcat_m"], cluster="agency")
            row = {"label": label, "units": units, "n": int(m_e["n"]), "clusters": int(m_e["n_clusters"]), "n_train": int(len(tr))}
            for k, o in (("expected", o_e), ("realized", o_r)):
                b, se = float(o.loc["ai", "coef"]), float(o.loc["ai", "se"])
                row[k] = {"b": b, "se": se, "est_units": float(to_units(b, units)), "lo_units": float(to_units(b - 1.96 * se, units)),
                          "hi_units": float(to_units(b + 1.96 * se, units))}
            res["variants"][var][y] = row
            print("%-11s %-15s n %d (%d sellers) | expected from text %+.3f [%+.3f, %+.3f] %s | realized %+.3f [%+.3f, %+.3f]" % (
                var, y, row["n"], row["clusters"], row["expected"]["est_units"], row["expected"]["lo_units"],
                row["expected"]["hi_units"], units, row["realized"]["est_units"], row["realized"]["lo_units"], row["realized"]["hi_units"]))
    json.dump(res, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def main(panel, feats, scores, inputs, synth, out_dir=None):
    d = load(panel, feats, scores)
    d = d[d.log_price.notna()].copy()
    d["fulltext"] = d.desc.fillna("") + "\n" + d.updates_text.fillna("")
    X = {json.loads(l)["item"]: json.loads(l) for l in open(inputs)}
    S = [json.loads(l) for l in open(synth) if l.strip()]
    p0 = d[(d.period == "P0") & (~d["item"].isin(set(X)))].copy()
    r0 = p0.log_price - p0.subcat_q.map(p0.groupby("subcat_q").log_price.mean())
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    m = RidgeCV(alphas=np.logspace(-2, 4, 14)).fit(v.fit_transform(p0.fulltext), r0.values)
    print("frozen P0 model fit on %d listings (the 600 rewritten originals excluded); alpha %.3g" % (len(p0), m.alpha_))
    orig_s = {k: float(m.predict(v.transform([x["original"]]))[0]) for k, x in X.items()}
    rows = []
    for r in S:
        rows.append({"item": r["item"], "gen": "%s|%s" % (r["variant"], r["model"]),
                     "s_orig": orig_s[r["item"]], "s_ai": float(m.predict(v.transform([r["ai_text"]]))[0])})
    R = pd.DataFrame(rows)
    R["delta"] = R.s_ai - R.s_orig
    rng = np.random.default_rng(5)
    print("%-36s %5s %9s %9s %18s %8s" % ("generator", "n", "mean s0", "mean dS", "95% CI (paired)", "share>0"))
    for g, h in list(R.groupby("gen")) + [("ALL", R)]:
        bs = [h.delta.sample(len(h), replace=True, random_state=int(rng.integers(0, 1e9))).mean() for _ in range(2000)]
        print("%-36s %5d %9.3f %+9.3f   [%+.3f, %+.3f] %8.3f" % (g, len(h), h.s_orig.mean(), h.delta.mean(),
              np.quantile(bs, .025), np.quantile(bs, .975), (h.delta > 0).mean()))
    print("in percent of price, ALL: %+.1f%%" % (100 * (np.exp(R.delta.mean()) - 1)))
    if out_dir is None:
        return
    assert abs(R.delta.mean() - 0.172) < 0.0015, ("does not reproduce job 817664's +0.172", R.delta.mean())
    cell_mean = p0.groupby("subcat_q").log_price.mean()
    meta = d.set_index("item")
    R["variant"] = [g.split("|")[0] for g in R.gen]
    R["model"] = [g.split("|")[1] for g in R.gen]
    R["has_price"] = [bool(meta.log_price.get(i) == meta.log_price.get(i)) if i in meta.index else False for i in R.item]
    R["r"] = [float(meta.log_price[i] - cell_mean[meta.subcat_q[i]])
              if (i in meta.index and meta.log_price[i] == meta.log_price[i] and meta.subcat_q[i] in cell_mean.index)
              else np.nan for i in R.item]
    R["cat1"] = [meta.cat1[i] if i in meta.index else None for i in R.item]
    R["has_defect_strict"] = [float(meta.has_defect_strict[i]) if i in meta.index else np.nan for i in R.item]
    R["words"] = [len(X[i]["original"].split()) for i in R.item]
    print("coverage: %d pairs; %d originals in the panel; %d priced; %d with a quality value r"
          % (len(R), sum(i in meta.index for i in R.item), int(R.has_price.sum()), int(R.r.notna().sum())))
    Q = R[R.r.notna()].copy()
    cuts = np.quantile(Q.r.values, [1 / 3, 2 / 3])
    rng2 = np.random.default_rng(12)
    res = {"source": "code/frozen_rewrites.py --quality (Session 12); frozen model as in job 817664",
           "mean_lift_all": float(R.delta.mean()), "n_pairs": int(len(R)), "n_priced": int(R.has_price.sum()),
           "n_quality": int(len(Q)), "r_cuts": [float(c) for c in cuts], "groups": {}}
    for name, h in [("all", Q), ("appealing", Q[Q.variant == "V1_enhance"]), ("concise", Q[Q.variant == "V2_concise"])]:
        res["groups"][name] = with_intervals(h, cuts, rng2)
        res["groups"][name]["perm_p_slope"] = permutation_p(h, rng2)
        g = res["groups"][name]
        print("%-9s n %3d | slope of lift on r %+.3f [%+.3f, %+.3f] perm p %.3f | corr orig %.3f ai %.3f | gap top-bottom"
              " third %.3f -> %.3f (kept %.2f [%.2f, %.2f]) | sd ratio %.2f | lift low/mid/high %+.3f %+.3f %+.3f"
              % (name, g["n"], g["slope_lift_on_r"]["est"], g["slope_lift_on_r"]["lo"], g["slope_lift_on_r"]["hi"],
                 g["perm_p_slope"], g["corr_orig_r"]["est"], g["corr_ai_r"]["est"], g["gap_orig"]["est"],
                 g["gap_ai"]["est"], g["gap_kept"]["est"], g["gap_kept"]["lo"], g["gap_kept"]["hi"],
                 g["sd_ratio"]["est"], g["lift_low"]["est"], g["lift_mid"]["est"], g["lift_high"]["est"]))
    show = Q.sort_values("delta")
    print("lowest lifts (item, instruction|model, r, s_orig, s_ai):")
    for _, x in show.head(5).iterrows():
        print("  %s %s r %+.2f %+.3f -> %+.3f" % (x["item"], x.gen, x.r, x.s_orig, x.s_ai))
    print("highest lifts:")
    for _, x in show.tail(5).iterrows():
        print("  %s %s r %+.2f %+.3f -> %+.3f" % (x["item"], x.gen, x.r, x.s_orig, x.s_ai))
    # Session 12, review round 2 (the marketing referee): a version of the test that can fail on the real model. (1)
    # Planted: append the model's own ten most price-raising single words to the originals of the bottom third of quality
    # only; if rewriting had done that, the statistics must show the gap shrinking. (2) Coverage: of the words and
    # two-word phrases a rewrite adds (present in it, absent from its original), the share the model has a weight for.
    vocab = v.get_feature_names_out()
    coef = m.coef_
    uni = [(coef[j], vocab[j]) for j in range(len(vocab)) if " " not in vocab[j]]
    top = [w for _, w in sorted(uni, reverse=True)[:10]]
    lo3 = Q.r.values <= cuts[0]
    planted = Q.copy()
    planted["s_ai"] = [float(m.predict(v.transform([X[i]["original"] + " " + " ".join(top)]))[0]) if low else so
                       for i, low, so in zip(planted["item"], lo3, planted.s_orig)]
    res["planted"] = {"top_words": top, "groups": {"all": with_intervals(planted, cuts, rng2)}}
    pg = res["planted"]["groups"]["all"]
    print("planted (the ten most price-raising words appended to the bottom third's originals: %s): gap kept %.2f [%.2f, %.2f];"
          " correlation change %+.3f" % (", ".join(top), pg["gap_kept"]["est"], pg["gap_kept"]["lo"], pg["gap_kept"]["hi"],
                                          pg["corr_diff"]["est"]))
    an = v.build_analyzer()
    cov, n_added = [], 0
    for r in S:
        added = set(an(r["ai_text"])) - set(an(X[r["item"]]["original"]))
        if added:
            n_added += len(added)
            cov.append(sum(1 for t in added if t in v.vocabulary_) / len(added))
    res["coverage"] = {"pairs_with_added_terms": len(cov), "added_terms": n_added, "mean_share_with_weight": float(np.mean(cov)),
                       "median_share_with_weight": float(np.median(cov))}
    print("coverage: in %d pairs, %d added words and phrases; mean share the model has a weight for %.3f (median %.3f)" % (
        len(cov), n_added, np.mean(cov), np.median(cov)))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "quality.json"), "w"), indent=1)
    cols = ["item", "variant", "model", "s_orig", "s_ai", "delta", "has_price", "r", "cat1", "has_defect_strict", "words"]
    rows = [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in rec.items()}
            for rec in R[cols].to_dict("records")]
    json.dump(rows, open(os.path.join(out_dir, "pair_scores.json"), "w"))
    print("wrote", out_dir)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        selftest_mixed()
        selftest_reference()
        sys.exit(0)
    if sys.argv[1] == "--trimmed":
        trimmed(*sys.argv[2:5])             # PAIR_SCORES QUALITY_JSON OUT_JSON (local; no listing text)
        sys.exit(0)
    if sys.argv[1] == "--mixed":
        mixed(*sys.argv[2:5])               # PAIR_SCORES QUALITY_JSON OUT_JSON (local; no listing text)
        sys.exit(0)
    a = sys.argv[1:]
    if "--reference-real" in a:
        k = a.index("--reference-real")
        reference_real(*(a[:k] + a[k + 2:]), out_json=a[k + 1])   # PANEL FEATURES AISCORES INPUTS
        sys.exit(0)
    if "--reference" in a:
        k = a.index("--reference")
        reference(*(a[:k] + a[k + 2:]), out_json=a[k + 1])
        sys.exit(0)
    out = None
    if "--quality" in a:
        k = a.index("--quality")
        out = a[k + 1]
        a = a[:k] + a[k + 2:]
    main(*a, out_dir=out)
