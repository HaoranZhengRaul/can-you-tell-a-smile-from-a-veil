#!/usr/bin/env python3
"""Session 8: stress tests of the AI-vs-human difference in buyer questions (tool era).

1 distribution of questions before the close, by AI flag (quantiles, top listings)
2 functional forms: count, log(1+count), count capped at 10, any question, questions per 1,000 views
3 tight design: agency x start-date batches containing both AI and human listings (batch FE)
4 vehicles vs other goods
5 placebo: pre-ChatGPT listings whose text merely LOOKS AI-like (top 5% of detector score)
6 selection test: are an agency's human-written tool-era listings longer than its own earlier ones?
Session 10 (PI, 2026-10-29, "buyer questions by kind"): --sample writes the inputs of a blind coding of what buyers
ask, AI written vs not (tool era, questions asked before the close): up to N questions per group, at most 2 per
listing, in a shuffled order and under neutral ids, so neither a file's order nor its ids reveal the group. Three
files: facts_input.json (each listing's title, category, form table and item-specific text, for a fact extractor),
questions_blind.json (question id, listing id, question text, for the coder, who never sees the text or the group)
and key.json (ids -> item, agency, AI flag; kept apart from the coders). Raw text: these files stay on Mercury and
in the session scratchpad, never in git.
Usage: python3 analysis_questions.py PANEL FEATURES AISCORES
       python3 analysis_questions.py --sample PANEL FEATURES AISCORES PAGES_JSONL OUT_DIR [N_PER_GROUP] [SEED]
       python3 analysis_questions.py --kinds CODES_JSON KEY_JSON MY_CODES_JSON OUT_JSON   (local; aggregates only)
       python3 analysis_questions.py --ai-mentions PANEL FEATURES AISCORES PAGES_JSONL   (every question before the
       close that names AI, by period and flag; the matches are printed whole so each can be read as text)
       python3 analysis_questions.py --answers PANEL FEATURES AISCORES PAGES_JSONL [OUT_JSON]   (PI, 2026-10-30, "the
       answer should be analyzed too": the agency's answer to each question asked before the close: answered at all,
       answered before the close, hours to the answer, words, promotional and marker words in it; by period and, in
       the tool era, AI written against other listings within agency. OUT_JSON holds counts and means only)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))   # the same code, run locally
from analysis_core import load  # noqa: E402
from regs import ols_fe  # noqa: E402


def line(name, o, m, k="ai"):
    if k in o.index:
        return "  %-34s %s %8.4f (se %.4f, t %5.2f) n=%d clusters=%s mean=%.3f" % (
            name, k, o.loc[k, "coef"], o.loc[k, "se"], o.loc[k, "t"], m["n"], m["n_clusters"], m["y_mean"])
    return "  %-34s (coefficient absorbed) n=%d" % (name, m["n"])


def main(panel, feats, scores):
    d = load(panel, feats, scores)
    d["nq"] = d.n_q_before_end
    d["log_nq"] = np.log1p(d.nq)
    d["nq_cap"] = d.nq.clip(upper=10)
    d["q_per_kview"] = 1000.0 * d.nq / d.page_views.clip(lower=1)
    d["batch"] = d.agency + "|" + d.start_local_dt.dt.strftime("%Y-%m-%d")
    p2 = d[d.period == "P2"].copy()
    print("1. Questions before close, tool era:")
    for a in (0, 1):
        s = p2.loc[p2.ai == a, "nq"]
        print("  ai=%d n=%d mean=%.3f quantiles(50,75,90,95,99,max)=%s share>=1 %.3f" % (
            a, len(s), s.mean(), [float(s.quantile(q)) for q in (.5, .75, .9, .95, .99, 1.0)], (s >= 1).mean()))
    top = p2.sort_values("nq", ascending=False).head(10)
    for _, r in top.iterrows():
        print("   top: %s ai=%d nq=%d views=%s bids=%s $%s %s" % (r["item"], r.ai, r.nq, r.page_views, r.n_bids, r.final_bid, (r.title or "")[:60]))
    print("\n2. Functional forms (agency FE + subcategory x month FE; controls: photos, reserve, opening bid)")
    xs = ["ai", "log_photos", "reserve_any", "log_start"]
    for y in ["nq", "log_nq", "nq_cap", "any_q", "q_per_kview"]:
        o, m = ols_fe(p2, y, xs, fes=["agency", "subcat_m"], cluster="agency")
        print(line(y, o, m))
    o, m = ols_fe(p2, "nq", xs + ["log_w"], fes=["agency", "subcat_m"], cluster="agency")
    print(line("nq | + log words (descriptive)", o, m))
    o, m = ols_fe(p2, "nq", xs + ["log_facts"], fes=["agency", "subcat_m"], cluster="agency")
    print(line("nq | + log facts", o, m))
    print("\n3. Same agency, same start day (batches with both AI and human listings)")
    b = p2.groupby("batch").ai.agg(["mean", "size"])
    mixed = b[(b["mean"] > 0) & (b["mean"] < 1)].index
    bb = p2[p2.batch.isin(mixed)]
    print("  mixed batches %d, listings %d (AI %d), agencies %d" % (len(mixed), len(bb), int(bb.ai.sum()), bb.agency.nunique()))
    for y in ["nq", "log_nq", "any_q", "sold", "log_price", "log_bids", "log_views"]:
        o, m = ols_fe(bb, y, ["ai", "log_photos", "reserve_any", "log_start"], fes=["batch", "subcat"], cluster="agency")
        print(line(y + " (batch FE)", o, m))
    print("\n4. Vehicles (structured year+make) vs other goods")
    for lab, g in [("vehicles", p2[p2.veh]), ("other goods", p2[~p2.veh])]:
        for y in ["nq", "any_q", "log_price", "sold"]:
            o, m = ols_fe(g, y, xs, fes=["agency", "subcat_m"], cluster="agency")
            print(line("%s: %s" % (lab, y), o, m))
    print("\n5. Placebo: pre-ChatGPT listings whose text looks AI-like (top 5% of detector score within P0)")
    p0 = d[d.period == "P0"].copy()
    cut = p0.ai_score.quantile(0.95)
    p0["ai"] = (p0.ai_score >= cut).astype(float)
    print("  P0 listings %d, 'AI-looking' %d (score >= %.3f)" % (len(p0), int(p0.ai.sum()), cut))
    for y in ["nq", "any_q", "log_price", "sold", "log_w", "s_n_puff"]:
        o, m = ols_fe(p0, y, xs, fes=["agency", "subcat_m"], cluster="agency")
        print(line("P0 placebo: " + y, o, m))
    print("\n6. Selection: agency's human-written tool-era listings vs its own earlier listings (log item-specific words)")
    hum = d[(d.ai == 0)].copy()
    hum["tool_era"] = (hum.period == "P2").astype(float)
    users = set(p2.loc[p2.ai == 1, "agency"])
    for lab, g in [("agencies that used AI", hum[hum.agency.isin(users)]), ("agencies that never did", hum[~hum.agency.isin(users)])]:
        o, m = ols_fe(g, "log_w", ["tool_era"], fes=["agency", "subcat"], cluster="agency")
        print(line(lab, o, m, k="tool_era"))


def sample_main(panel, feats, scores, pages, out_dir, n_per=150, seed=10):
    import json
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_panel import qa_time
    from datetime import datetime
    n_per, seed = int(n_per), int(seed)
    d = load(panel, feats, scores)
    p2 = d[d.period == "P2"].copy()
    p2["item"] = p2["item"].astype(str)
    lst = p2.set_index("item")
    Q = {}
    n_q, n_before = 0, 0
    for l in open(pages):
        r = json.loads(l)
        it = str(r.get("item"))
        if r.get("page_kind") != "listing" or it not in lst.index or not r.get("qa"):
            continue
        end = r.get("end_local")
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S") if end else None
        for q in r["qa"]:
            n_q += 1
            t, txt = qa_time(q.get("q_time")), (q.get("q") or "").strip()
            if t is None or end_dt is None or t > end_dt or not txt:
                continue
            n_before += 1
            Q.setdefault(it, []).append(txt)
    print("tool-era listings %d; with a question before the close %d (AI written %d); questions read %d, before the close %d" % (
        len(p2), len(Q), int(sum(lst.loc[i, "ai"] == 1 for i in Q)), n_q, n_before))
    rng = np.random.default_rng(seed)
    picked = []
    for g in (1, 0):
        items = sorted(i for i in Q if int(lst.loc[i, "ai"]) == g)
        rng.shuffle(items)
        got = 0
        for i in items:
            qs = list(Q[i])
            rng.shuffle(qs)
            for txt in qs[:2]:
                if got < n_per:
                    picked.append((i, g, txt))
                    got += 1
            if got >= n_per:
                break
        print("group AI=%d: %d questions from %d listings (%d listings available)" % (
            g, got, len(set(i for i, gg, _ in picked if gg == g)), len(items)))
    order = rng.permutation(len(picked))
    items = sorted(set(i for i, _, _ in picked))
    lorder = rng.permutation(len(items))
    lid = {items[k]: "L%03d" % (n + 1) for n, k in enumerate(lorder)}
    def val(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return x.item() if hasattr(x, "item") else x
    facts = []
    for i in sorted(items, key=lambda i: lid[i]):
        x = lst.loc[i]
        facts.append({"lid": lid[i], "title": str(x.title), "category": str(x.subcat),
                      "form": {k: val(x[k]) for k in ("f_year", "f_make", "f_model", "f_miles", "f_starts", "f_drivable", "f_title", "f_inspected") if k in x.index},
                      "description": str(x.spec)})
    qs_out, key_q = [], {}
    for n, k in enumerate(order):
        i, g, txt = picked[k]
        qid = "Q%03d" % (n + 1)
        qs_out.append({"qid": qid, "lid": lid[i], "question": txt})
        key_q[qid] = {"item": i, "ai": int(g), "agency": str(lst.loc[i, "agency"])}
    key_l = {lid[i]: {"item": i, "ai": int(lst.loc[i, "ai"]), "agency": str(lst.loc[i, "agency"])} for i in items}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(facts, open(os.path.join(out_dir, "facts_input.json"), "w"), indent=1)
    json.dump(qs_out, open(os.path.join(out_dir, "questions_blind.json"), "w"), indent=1)
    json.dump({"rule": d.attrs.get("ai_rule"), "seed": seed, "n_per_group": n_per, "questions": key_q, "listings": key_l},
              open(os.path.join(out_dir, "key.json"), "w"), indent=1)
    print("wrote %d listings and %d questions to %s (key apart)" % (len(facts), len(qs_out), out_dir))


# what a question asks about, from the coder's free-text attribute label (228 labels in the 300 coded questions):
# ordered rules, the first that matches wins; every label's topic is printed so the mapping can be read
TOPICS = [
    ("Photos", r"photo|\bpics?\b|\bpictures?\b"),          # not "pictured"
    ("Sale and pickup", r"reserve|opening bid|starting price|relist|reset the auction|\bbid\b|bidding|auction|pick ?up|"
                        r"loading|\bload\b|shipping|freight|palletiz|payment|contact|viewing|appointment|title transfer|"
                        r"splitting|partial|fraud|removal"),
    ("What comes with it", r"includ|\bkeys?\b|fobs?|accessor|contents|software|charger|power cords|quantity|pieces|"
                           r"\btitle\b"),
    ("Hours, miles and history", r"hours?\b|hour meter|mileage|\bmiles\b|history|maintenance|\bservice\b|serviced|"
                                 r"prior use|\bage\b|tank year|test date|recall|registration|ownership|time parked|"
                                 r"last run|water pump|original price|certif"),
    ("Condition and function", r"\bcondition\b|work|\bruns?\b|running|(?<!wheel )drive\b|drives|drivab|starts?\b|starting|"
                               r"moves|defect|issue|problem|rust|leak|crack|\brot\b|damage|frame|check engine|"
                               r"warning light|inspect|pass\b|scrap|repair|lift|bearing|missing|function|paint|holes|"
                               r"flatness|tire|new or used"),
    ("Specifications and features", r"fuel|wheel drive|drivetrain|awd|4x4|dimension|length|width|height|size|gvw|"
                                    r"capacity|engine|turbo|horsepower|transmission|brake|air condition|heat|window|radio|"
                                    r"volt|phase|cpu|memory|model|make|brand|\bvin\b|serial|material|type|bed|deck|"
                                    r"door|fork|glass|scanner|electronic|electric|suspension|lights|spotlight|siren|pintle|"
                                    r"hitch|strainer|control|compatib|air line"),
]


def topic_of(label):
    import re
    for name, rx in TOPICS:
        if re.search(rx, label or "", re.I):
            return name
    return "Other"


def kinds_main(codes_json, key_json, my_json, out_json, n_boot=2000, seed=29):
    """The blind codes against the key: each kind's share among AI written and other listings' questions (Wilson 95%),
    the AI-minus-other difference with a bootstrap that resamples listings within each group (questions of one listing
    are not independent), the same among questions about the item (logistics and other left out), and the agreement
    with the main loop's own codes."""
    import json
    KINDS = ["already_stated", "verify_claim", "never_mentioned", "logistics", "other"]
    C = {r["qid"]: r for r in json.load(open(codes_json)) if r.get("qid", "").startswith("Q")}
    K = json.load(open(key_json))["questions"]
    assert set(C) == set(K), ("codes and key disagree", len(C), len(K), sorted(set(K) - set(C))[:5])
    bad = [q for q, r in C.items() if r.get("kind") not in KINDS]
    assert not bad, ("unknown kinds", bad[:5])
    z = 1.959964
    def wil(k, n):
        p, d = k / n, 1 + z * z / n
        c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return max(c - h, 0.0), min(c + h, 1.0)
    rng = np.random.default_rng(int(seed))
    OUT = {"n": len(C), "groups": {}, "diff": {}, "diff_item_questions": {}}
    by = {g: [q for q in C if K[q]["ai"] == g] for g in (1, 0)}
    lst = {g: sorted(set(K[q]["item"] for q in by[g])) for g in (1, 0)}
    qs_of = {g: {i: [q for q in by[g] if K[q]["item"] == i] for i in lst[g]} for g in (1, 0)}
    for g in (1, 0):
        n = len(by[g])
        OUT["groups"][str(g)] = {"n_questions": n, "n_listings": len(lst[g])}
        for k in KINDS:
            m = sum(1 for q in by[g] if C[q]["kind"] == k)
            lo, hi = wil(m, n)
            OUT["groups"][str(g)][k] = {"k": m, "share": m / n, "lo": lo, "hi": hi}
    def share(g, k, items, among_item=False):
        qs = [q for i in items for q in qs_of[g][i]]
        if among_item:
            qs = [q for q in qs if C[q]["kind"] not in ("logistics", "other")]
        return (sum(1 for q in qs if C[q]["kind"] == k) / len(qs)) if qs else float("nan")
    for tag, among in (("diff", False), ("diff_item_questions", True)):
        for k in (KINDS if not among else ["already_stated", "verify_claim", "never_mentioned"]):
            d0 = share(1, k, lst[1], among) - share(0, k, lst[0], among)
            bs = []
            for _ in range(int(n_boot)):
                a = list(rng.choice(lst[1], len(lst[1])))
                b = list(rng.choice(lst[0], len(lst[0])))
                bs.append(share(1, k, a, among) - share(0, k, b, among))
            bs = np.array([x for x in bs if x == x])
            OUT[tag][k] = {"diff": d0, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    print("questions coded %d (AI written %d on %d listings; other %d on %d listings)" % (
        len(C), len(by[1]), len(lst[1]), len(by[0]), len(lst[0])))
    for k in KINDS:
        a, b, d = OUT["groups"]["1"][k], OUT["groups"]["0"][k], OUT["diff"][k]
        print("  %-16s AI written %5.1f%% [%4.1f, %4.1f] | other %5.1f%% [%4.1f, %4.1f] | difference %+5.1f pp [%+5.1f, %+5.1f]" % (
            k, 100 * a["share"], 100 * a["lo"], 100 * a["hi"], 100 * b["share"], 100 * b["lo"], 100 * b["hi"],
            100 * d["diff"], 100 * d["lo"], 100 * d["hi"]))
    print("  among questions about the item (logistics and other left out):")
    for k, d in OUT["diff_item_questions"].items():
        print("  %-16s AI minus other %+5.1f pp [%+5.1f, %+5.1f]" % (k, 100 * d["diff"], 100 * d["lo"], 100 * d["hi"]))
    # topics: what each question asks about, by group (Wilson 95%; difference with the same listing bootstrap)
    names = [t for t, _ in TOPICS] + ["Other"]
    for q in C:
        C[q]["topic"] = topic_of(C[q].get("attribute"))
    labels = {}
    for q in C:
        labels.setdefault((C[q].get("attribute") or "").strip().lower(), C[q]["topic"])
    print("label -> topic (every distinct label):")
    for t in names:
        print("  %-28s %s" % (t, "; ".join(sorted(l for l, tt in labels.items() if tt == t))))
    OUT["topics"] = {}
    for t in names:
        row = {}
        for g in (1, 0):
            n = len(by[g])
            m = sum(1 for q in by[g] if C[q]["topic"] == t)
            lo, hi = wil(m, n)
            row[str(g)] = {"k": m, "share": m / n, "lo": lo, "hi": hi}
        def tshare(g, items):
            qs = [q for i in items for q in qs_of[g][i]]
            return (sum(1 for q in qs if C[q]["topic"] == t) / len(qs)) if qs else float("nan")
        d0 = tshare(1, lst[1]) - tshare(0, lst[0])
        bs = np.array([tshare(1, list(rng.choice(lst[1], len(lst[1])))) - tshare(0, list(rng.choice(lst[0], len(lst[0]))))
                       for _ in range(int(n_boot))])
        row["diff"] = {"diff": d0, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        OUT["topics"][t] = row
        print("  %-28s AI written %5.1f%% | other %5.1f%% | difference %+5.1f pp [%+5.1f, %+5.1f]" % (
            t, 100 * row["1"]["share"], 100 * row["0"]["share"], 100 * d0, 100 * row["diff"]["lo"], 100 * row["diff"]["hi"]))
    M = json.load(open(my_json))["codes"]
    common = sorted(q for q in M if q in C)
    agree = sum(1 for q in common if M[q]["kind"] == C[q]["kind"]) / len(common)
    OUT["agreement_with_main_loop"] = {"n": len(common), "agree": agree,
                                       "confusions": sorted("%s->%s" % (M[q]["kind"], C[q]["kind"]) for q in common if M[q]["kind"] != C[q]["kind"])}
    print("agreement with my codes on %d questions: %.2f; disagreements (mine->coder): %s" % (
        len(common), agree, OUT["agreement_with_main_loop"]["confusions"]))
    json.dump(OUT, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def ai_mentions_main(panel, feats, scores, pages, out_json=None):
    """Session 10: three of 150 sampled questions on AI written listings name the AI text ("lovely AI description,
    but what are the idle hours"); count every such question before the close, by period and flag."""
    import json
    import re
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_panel import qa_time
    from datetime import datetime
    # "robot" left the pattern after the first run (877829): its three matches were a robotic arm (twice) and a
    # robotic pool cleaner, and none of the four questions that name the AI text used it
    rx = re.compile(r"(?i)(\bA\.?I\b|\bchat ?gpt\b|\bchatbot\b|artificial intelligence|\bgenerated\b)")
    d = load(panel, feats, scores)
    d["item"] = d["item"].astype(str)
    lst = d.set_index("item")
    n_q = {}
    hits = []
    for l in open(pages):
        r = json.loads(l)
        it = str(r.get("item"))
        if r.get("page_kind") != "listing" or it not in lst.index or not r.get("qa"):
            continue
        end = r.get("end_local")
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S") if end else None
        x = lst.loc[it]
        g = (x.period, int(x.ai))
        for q in r["qa"]:
            t, txt = qa_time(q.get("q_time")), (q.get("q") or "").strip()
            if t is None or end_dt is None or t > end_dt or not txt:
                continue
            n_q[g] = n_q.get(g, 0) + 1
            if rx.search(txt):
                hits.append((g, it, " ".join(txt.split())[:300]))
    print("questions before the close, by (period, AI flag):", dict(sorted(n_q.items())))
    by = {}
    for g, _, _ in hits:
        by[g] = by.get(g, 0) + 1
    print("of which match the AI pattern:", dict(sorted(by.items())))
    for g, it, t in sorted(hits):
        print("  %s ai=%d item %s: %s" % (g[0], g[1], it, t))
    if out_json:      # counts only: no question text, no item number
        json.dump({"pattern": rx.pattern, "questions": {"%s|%d" % g: n for g, n in sorted(n_q.items())},
                   "matches": {"%s|%d" % g: n for g, n in sorted(by.items())},
                   "listings_matched": {"%s|%d" % g: len(set(i for gg, i, _ in hits if gg == g)) for g in by}},
                  open(out_json, "w"), indent=1)
        print("wrote", out_json)


def answer_record(q, end_dt, qa_time, features):
    """A question asked before the close -> what its answer shows; None for any other question. Missing is not
    measured: an answer without a time has no delay and no before-the-close value (None), never 0."""
    t_q, txt = qa_time(q.get("q_time")), (q.get("q") or "").strip()
    if t_q is None or end_dt is None or t_q > end_dt or not txt:
        return None
    a = (q.get("a") or "").strip()
    t_a = qa_time(q.get("a_time")) if a else None
    rec = {"answered": 1.0 if a else 0.0, "a_time_known": (1.0 if t_a is not None else 0.0) if a else None,
           "before_close": (float(t_a <= end_dt) if t_a is not None else None) if a else 0.0,
           "delay_h": (t_a - t_q).total_seconds() / 3600.0 if (a and t_a is not None) else None,
           "words": None, "n_mark": None, "n_puff": None}
    if a:
        x = features(a, None)
        rec.update(words=float(x["w"]), n_mark=float(x["n_mark"]), n_puff=float(x["n_puff"]))
    return rec


def answers_selftest():
    from datetime import datetime
    def qt(s):
        return datetime.strptime(s, "%m/%d/%Y %I:%M %p") if s else None
    def feats(a, sp):
        return {"w": len(a.split()), "n_mark": int("versatile" in a), "n_puff": int("great" in a)}
    end = datetime(2025, 6, 10, 12, 0)
    r = answer_record({"q": "hours?", "q_time": "06/01/2025 10:00 AM", "a": "1,200 hours, great shape",
                       "a_time": "06/01/2025 12:00 PM"}, end, qt, feats)
    assert (r["answered"], r["before_close"], r["delay_h"], r["words"], r["n_puff"]) == (1.0, 1.0, 2.0, 4.0, 1.0), r
    assert answer_record({"q": "late?", "q_time": "06/11/2025 10:00 AM", "a": "x"}, end, qt, feats) is None
    assert answer_record({"q": "no time", "q_time": None, "a": "x"}, end, qt, feats) is None
    r = answer_record({"q": "keys?", "q_time": "06/01/2025 10:00 AM", "a": None}, end, qt, feats)
    assert (r["answered"], r["before_close"], r["delay_h"], r["words"]) == (0.0, 0.0, None, None), r
    r = answer_record({"q": "rust?", "q_time": "06/01/2025 10:00 AM", "a": "some", "a_time": None}, end, qt, feats)
    assert (r["answered"], r["a_time_known"], r["before_close"], r["delay_h"]) == (1.0, 0.0, None, None), r
    r = answer_record({"q": "title?", "q_time": "06/01/2025 10:00 AM", "a": "yes", "a_time": "06/12/2025 09:00 AM"}, end, qt, feats)
    assert r["before_close"] == 0.0 and r["delay_h"] > 0, r


def answers_main(panel, feats_path, scores, pages, out_json=None):
    import json
    import os
    import re
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_panel import qa_time
    from textfeat import features
    from datetime import datetime
    answers_selftest()
    d = load(panel, feats_path, scores)
    d["item"] = d["item"].astype(str)
    lst = d.set_index("item")
    assert lst.index.is_unique, "one row per listing expected"
    rows, seen = [], set()
    for l in open(pages):
        r = json.loads(l)
        it = str(r.get("item"))
        if r.get("page_kind") != "listing" or it not in lst.index or not r.get("qa") or it in seen:
            continue
        seen.add(it)
        end = r.get("end_local")
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S") if end else None
        x = lst.loc[it]
        for q in r["qa"]:
            rec = answer_record(q, end_dt, qa_time, features)
            if rec is None:
                continue
            rec.update(item=it, agency=x.agency, period=x.period, ai=int(x.ai), subcat_m=x.subcat_m,
                       a_text=" ".join((q.get("a") or "").split()))
            rows.append(rec)
    Q = pd.DataFrame(rows)
    Q["mark2"] = np.where(Q.answered == 1, (Q.n_mark >= 2).astype(float), np.nan)
    Q["log_delay"] = np.where(Q.delay_h >= 0, np.log1p(Q.delay_h), np.nan)
    Q["log_words"] = np.log1p(Q.words)
    neg = int((Q.delay_h < 0).sum())
    print("questions before the close %d on %d listings; answered %d; answers without a time %d; answer before the question %d" % (
        len(Q), Q.item.nunique(), int(Q.answered.sum()), int((Q.a_time_known == 0).sum()), neg))
    def stats(g):
        a = g[g.answered == 1]
        dl = a.delay_h[a.delay_h >= 0]
        return {"questions": int(len(g)), "listings": int(g.item.nunique()), "answered": float(g.answered.mean()),
                "answered_before_close": float(g.before_close.dropna().mean()) if g.before_close.notna().any() else None,
                "delay_h_median": float(dl.median()) if len(dl) else None,
                "delay_h_p75": float(dl.quantile(0.75)) if len(dl) else None,
                "within_24h": float((dl <= 24).mean()) if len(dl) else None,
                "words_median": float(a.words.median()) if len(a) else None, "words_mean": float(a.words.mean()) if len(a) else None,
                "puff_mean": float(a.n_puff.mean()) if len(a) else None, "mark2_share": float(a.mark2.mean()) if len(a) else None}
    OUT_ = {"by_period": {p: stats(g) for p, g in Q.groupby("period")},
            "tool_era_by_ai": {str(a): stats(g) for a, g in Q[Q.period == "P2"].groupby("ai")},
            "answer_before_question": neg, "answers_without_time": int((Q.a_time_known == 0).sum())}
    print("\nby period (questions before the close):")
    for p, v in OUT_["by_period"].items():
        print("  %s %s" % (p, v))
    print("tool era, AI written (1) against other listings (0):")
    for a, v in OUT_["tool_era_by_ai"].items():
        print("  ai=%s %s" % (a, v))
    Q2 = Q[Q.period == "P2"].copy()
    OUT_["gaps"] = {}
    print("\nwithin-agency gaps, tool era (question level; agency-clustered):")
    for y, sub in (("answered", Q2), ("before_close", Q2[Q2.before_close.notna()]), ("log_delay", Q2[Q2.log_delay.notna()]),
                   ("log_words", Q2[Q2.answered == 1]), ("mark2", Q2[Q2.answered == 1])):
        for fes in (["agency"], ["agency", "subcat_m"]):
            o, m = ols_fe(sub, y, ["ai"], fes=fes, cluster="agency")
            k = "%s|%s" % (y, "+".join(fes))
            if "ai" in o.index:
                OUT_["gaps"][k] = {"b": float(o.loc["ai", "coef"]), "se": float(o.loc["ai", "se"]), "n": int(m["n"]),
                                   "clusters": int(m["n_clusters"]), "mean": float(m["y_mean"])}
            print(line("%s, FE %s" % (y, "+".join(fes)), o, m))
    red = lambda t: re.sub(r"\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", "[phone]", re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", t))
    a2 = Q2[Q2.answered == 1].copy()
    print("\nanswers to read (tool era; e-mails and phone numbers masked):")
    for label, part in (("shortest", a2.nsmallest(6, "words")), ("longest", a2.nlargest(4, "words")),
                        ("slowest", a2[a2.delay_h >= 0].nlargest(4, "delay_h")), ("fastest", a2[a2.delay_h >= 0].nsmallest(4, "delay_h"))):
        for _, x in part.iterrows():
            print("  %s ai=%d %.1f h %d words: %s" % (label, x.ai, x.delay_h if x.delay_h == x.delay_h else -1, x.words, red(x.a_text)[:220]))
    print("  answers with 2+ marker words (tool era, all):")
    for _, x in a2[a2.mark2 == 1].iterrows():
        print("    ai=%d: %s" % (x.ai, red(x.a_text)[:260]))
    if out_json:
        json.dump(OUT_, open(out_json, "w"), indent=1)
        print("wrote", out_json)


if __name__ == "__main__":
    if sys.argv[1] == "--answers":
        answers_main(*sys.argv[2:7])
    elif sys.argv[1] == "--ai-mentions":
        ai_mentions_main(*sys.argv[2:7])
    elif sys.argv[1] == "--kinds":
        kinds_main(*sys.argv[2:6])
    elif sys.argv[1] == "--sample":
        sample_main(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
