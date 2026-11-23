#!/usr/bin/env python3
"""Session 8: what an AI rewrite does to a fixed human listing (controlled experiment on text).

For each pre-ChatGPT original x and its synthetic rewrite r (code/sample_for_synthetic.py inputs,
six generators x two instructions), count:
  numbers kept      share of distinct digit tokens of x that appear in r (commas removed)
  defects kept      share of distinct defect-statement matches of x (DEFECT regex) also matched in r
  added positive    positive-condition matches in r that x does not contain ('runs', 'operational',
                    'good condition', ...) - a claim about this item's condition the clerk never made
  added promo       promotional adjectives (PUFF list) in r minus in x
  added markers     LLM-marker words (MARK list) in r minus in x
  words ratio       words(r) / words(x)
Session 10 (PI, 2026-10-29, "rewrite fidelity"): --fidelity aggregates a hand coding of all 600 pairs (four Claude
agents, blind to generator and instruction; codebook in the Session 10 WORKLOG): per pair the original's facts and
their fate (kept, dropped, changed and how), claims added about this item that nothing supports, generic praise
added, and the fate of each defect statement. It checks each record's arithmetic, reports every rate overall and by
generator x instruction with Wilson 95% intervals, and the agreement with the main loop's own codes of 30 pairs.
Writes aggregates only (no listing text).
Session 12 (PI, 2026-11-18; the advisor's C4): Session 10's per-pair codes and codebook were lost with its temporary
scratchpad, so --batches lays the 600 pairs out again for blind coding: a random order (seed 12) with new ids P001 to
P600, six batches of 100 holding only the id, title, category, original and rewrite (no model, no instruction), the
key kept apart, and a sample of 120 "appealing" pairs (60 Opus, 30 Sonnet, 30 Haiku) for the PI's own blind check.
All three outputs hold listing text and go to a scratch folder outside this repository.
Usage: python3 rewrite_audit.py INPUTS.jsonl OUT1.jsonl [OUT2.jsonl ...]
       python3 rewrite_audit.py --fidelity CODES_DIR KEY_JSON MY_CODES_JSON OUT_JSON
       python3 rewrite_audit.py --batches INPUTS.jsonl SYNTH.jsonl OUT_DIR
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from textfeat import DEFECT, MARK, POS_COND, PUFF, QUALITY  # noqa: E402

DIG = re.compile(r"\d[\d,]*(?:\.\d+)?")
W = re.compile(r"[a-z]+(?:'[a-z]+)?")


def nums(t):
    return set(x.replace(",", "").rstrip(".") for x in DIG.findall(t))


def stems(rx, t):
    return set(re.sub(r"\s+", " ", m.group(0).lower()) for m in rx.finditer(t))


def main(inp, outs):
    X = {json.loads(l)["item"]: json.loads(l) for l in open(inp)}
    agg = defaultdict(lambda: defaultdict(list))
    examples = []
    for fn in outs:
        for l in open(fn):
            if not l.strip():
                continue
            r = json.loads(l)
            x = X[r["item"]]
            orig = (x["title"] or "") + "\n" + (x["original"] or "")
            new = r["ai_text"] or ""
            key = (r.get("variant"), r.get("model"))
            no, nn = nums(orig), nums(new)
            do, dn = stems(DEFECT, orig), stems(DEFECT, new)
            po, pn = stems(POS_COND, orig), stems(POS_COND, new)
            wo, wn = W.findall(orig.lower()), W.findall(new.lower())
            a = agg[key]
            if no:
                a["numbers_kept"].append(len(no & nn) / len(no))
            if do:
                a["defects_kept"].append(len(do & dn) / len(do))
            qo, qn = stems(QUALITY, orig), stems(QUALITY, new)
            added_pos = qn - qo
            a["added_positive_any"].append(1.0 if added_pos else 0.0)
            a["added_promo"].append(len(set(wn) & PUFF) - len(set(wo) & PUFF))
            a["added_markers"].append(len(set(wn) & MARK) - len(set(wo) & MARK))
            a["words_ratio"].append(len(wn) / max(1, len(W.findall((x["original"] or "").lower()))))
            a["new_numbers_any"].append(1.0 if (nn - no) else 0.0)
            a["orig_has_quality"].append(1.0 if qo else 0.0)
            if added_pos and len(examples) < 14 and do:
                examples.append((r["item"], key, sorted(do), sorted(added_pos), x["original"][:200].replace("\n", " / "), new[:300].replace("\n", " / ")))
    print("%-34s %5s %8s %8s %9s %8s %8s %8s %8s" % ("generator", "n", "num_kept", "def_kept", "add_pos%", "add_prom", "add_mark", "w_ratio", "new_num%"))
    allk = defaultdict(list)
    for key in sorted(agg, key=str):
        a = agg[key]
        def m(v):
            return sum(a[v]) / len(a[v]) if a[v] else float("nan")
        print("%-34s %5d %8.3f %8.3f %9.3f %8.2f %8.2f %8.2f %8.3f" % (str(key)[:34], len(a["words_ratio"]), m("numbers_kept"), m("defects_kept"),
              100 * m("added_positive_any") / 100, m("added_promo"), m("added_markers"), m("words_ratio"), m("new_numbers_any")))
        for v in a:
            allk[v] += a[v]
    print("ALL: " + " ".join("%s=%.3f" % (v, sum(allk[v]) / len(allk[v])) for v in sorted(allk)))
    print("\nExamples where the rewrite adds a positive condition claim to an item the clerk described with a defect:")
    for e in examples:
        print(" ", e[0], e[1], "| defects:", e[2], "| added:", e[3])
        print("     ORIG:", e[4])
        print("     AI  :", e[5])


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# a specific claim that is only a reliability or quality word ("dependable", "sturdy") is where coders' lines differ
# (the main loop coded P539 and P565 generic, the agents specific): the strict flag leaves those out
RELIAB = re.compile(r"(?i)\b(dependab|reliab|sturdy|durab|quality|well[- ]?kept|solid|trusted|robust)\w*")


def fidelity(codes_dir, key_json, my_json, out_json):
    import glob
    import os
    key = json.load(open(key_json))
    C = {}
    for fn in sorted(glob.glob(os.path.join(codes_dir, "batch_*_codes.json"))):
        for r in json.load(open(fn)):
            if r.get("pid") in key:
                assert r["pid"] not in C, ("coded twice", r["pid"])
                C[r["pid"]] = r
    missing = sorted(set(key) - set(C))
    print("pairs coded: %d of %d; missing: %s" % (len(C), len(key), missing[:10]))
    bad = []
    for p, r in C.items():
        if r["kept"] + r["dropped"] + r["changed"] != r["n_orig_facts"]:
            bad.append((p, "facts"))
        if r["defects_kept"] + r["defects_softened"] + r["defects_dropped"] != r["orig_defects"]:
            bad.append((p, "defects"))
        if r["changed"] != len(r.get("changed_list") or []) or r["dropped"] != len(r.get("dropped_list") or []):
            bad.append((p, "lists"))
    print("records whose own arithmetic does not add up: %d %s" % (len(bad), bad[:12]))
    soft = lambda r: sum(1 for c in (r.get("changed_list") or []) if c.get("how") == "softened")
    flags = {"any_drop": lambda r: r["dropped"] > 0, "any_change": lambda r: r["changed"] > 0,
             "any_softened": lambda r: soft(r) > 0, "any_added_specific": lambda r: r["added_specific"] > 0,
             "any_added_specific_strict": lambda r: any(not RELIAB.search(x) for x in (r.get("added_specific_list") or [])),
             "any_added_generic": lambda r: r["added_generic"] > 0,
             "any_defect_dropped_or_softened": lambda r: (r["defects_dropped"] + r["defects_softened"]) > 0}
    cells = defaultdict(list)
    for p, r in C.items():
        cells[(key[p]["model"], key[p]["variant"])].append(r)
        cells[("all", "all")].append(r)
    OUT = {"n_coded": len(C), "n_key": len(key), "arithmetic_errors": len(bad), "cells": {}}
    print("%-34s %4s " % ("generator | instruction", "n") + " ".join("%-26s" % f for f in flags))
    for c in sorted(cells, key=lambda c: (c != ("all", "all"), c)):
        R = cells[c]
        row, o = [], {"n": len(R)}
        for f, fn in flags.items():
            k = sum(1 for r in R if fn(r))
            lo, hi = wilson(k, len(R))
            o[f] = {"k": k, "share": k / len(R), "lo": lo, "hi": hi}
            row.append("%5.1f%% [%4.1f, %5.1f]       " % (100 * k / len(R), 100 * lo, 100 * hi))
        facts = sum(r["n_orig_facts"] for r in R)
        o["facts"] = {"n": facts, "kept": sum(r["kept"] for r in R) / facts, "dropped": sum(r["dropped"] for r in R) / facts,
                      "changed": sum(r["changed"] for r in R) / facts}
        dfx = sum(r["orig_defects"] for r in R)
        o["defects"] = {"n": dfx, "dropped": (sum(r["defects_dropped"] for r in R) / dfx) if dfx else None,
                        "softened": (sum(r["defects_softened"] for r in R) / dfx) if dfx else None}
        o["mean_added_generic"] = sum(r["added_generic"] for r in R) / len(R)
        o["mean_added_specific"] = sum(r["added_specific"] for r in R) / len(R)
        OUT["cells"]["%s|%s" % c] = o
        print("%-34s %4d " % ("%s | %s" % c, len(R)) + "".join(row))
        print("      facts %d: kept %.3f dropped %.3f changed %.3f | defects %d: dropped %s softened %s | added per pair: specific %.2f generic %.2f" % (
            facts, o["facts"]["kept"], o["facts"]["dropped"], o["facts"]["changed"], dfx,
            "%.3f" % o["defects"]["dropped"] if dfx else "n/a", "%.3f" % o["defects"]["softened"] if dfx else "n/a",
            o["mean_added_specific"], o["mean_added_generic"]))
    M = json.load(open(my_json))["codes"]
    agree = {}
    for f, mf in (("any_drop", "any_drop"), ("any_change", "any_change"), ("any_added_specific", "any_added_specific"),
                  ("any_added_generic", "added_generic")):
        a = [1 if (M[p][mf] > 0) else 0 for p in sorted(M) if p in C]
        b = [1 if flags[f](C[p]) else 0 for p in sorted(M) if p in C]
        agree[f] = {"n": len(a), "agree": sum(x == y for x, y in zip(a, b)) / len(a), "kappa": kappa(a, b), "mine": sum(a), "agents": sum(b)}
        print("agreement with my codes, %-20s n=%d agree %.2f kappa %s (mine %d, agents %d)" % (
            f, len(a), agree[f]["agree"], ("%.2f" % agree[f]["kappa"]) if agree[f]["kappa"] == agree[f]["kappa"] else "n/a", sum(a), sum(b)))
    OUT["agreement_with_main_loop"] = agree
    json.dump(OUT, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def batches(inputs, synth, out_dir, seed=12, size=100):
    """Blind batches for coding, the key, and the PI's sample (see the module docstring)."""
    import os
    import random
    X = {str(json.loads(l)["item"]): json.loads(l) for l in open(inputs) if l.strip()}
    S = [json.loads(l) for l in open(synth) if l.strip()]
    assert len(S) == 600 and len({str(r["item"]) for r in S}) == 600 and all(str(r["item"]) in X for r in S), "expected 600 pairs, one per original"
    rnd = random.Random(seed)
    rnd.shuffle(S)
    key, pairs = {}, []
    for k, r in enumerate(S, 1):
        pid = "P%03d" % k
        x = X[str(r["item"])]
        key[pid] = {"item": str(r["item"]), "model": r["model"], "variant": r["variant"]}
        pairs.append({"pid": pid, "title": x["title"], "category": x["category"], "original": x["original"], "rewrite": r["ai_text"]})
    os.makedirs(out_dir, exist_ok=True)
    for b in range(0, len(pairs), size):
        json.dump(pairs[b:b + size], open(os.path.join(out_dir, "batch_%d_pairs.json" % (b // size + 1)), "w"), indent=1)
    json.dump(key, open(os.path.join(out_dir, "KEY_pid_to_item_model_variant.json"), "w"), indent=1)
    want = {"claude-opus-5-5": 60, "claude-sonnet-5-5": 30, "claude-haiku-5-5": 30}
    pi = []
    for m, n in want.items():
        pool = [p for p in pairs if key[p["pid"]]["variant"] == "V1_enhance" and key[p["pid"]]["model"] == m]
        pi += rnd.sample(pool, n)
    rnd.shuffle(pi)
    json.dump(pi, open(os.path.join(out_dir, "pi_check_pairs.json"), "w"), indent=1)
    print("wrote %d batches of %d, the key (%d pairs) and the PI's sample (%d appealing pairs: %s) to %s"
          % ((len(pairs) + size - 1) // size, size, len(key), len(pi), want, out_dir))


def pi_extra(r1_dir, pair_scores, done_answers, out_json, seed=19):
    """Session 12 (PI, 2026-11-19): after 4 pairs of the first page (all "appealing", none with a stated problem), the
    PI asked for 20 more, chosen for variety and with stated problems among them. The draw reads only what was fixed
    before any coding: the instruction and model (the key) and whether the original states a defect under the paper's
    strict pattern on its text less boilerplate (pair_scores.json has_defect_strict, from the panel's features); it never
    reads the coders' codes,
    so the agreement it yields is not selected on them. Quotas by cell (appealing: Opus 6, Sonnet 4, Haiku 4; concise:
    Opus 3, Sonnet 3), each with a fixed number of originals that state a defect (4, 3, 2, 2, 2: 13 of 20), random within
    strata (seed 19), the answered pairs left out, the 20 shuffled. The flag exists for the 473 originals that drew a bid
    (it is None for the other 127, which is not "no defect"), so the draw is among those 473; the strict pattern on the
    raw text instead would count boilerplate terms of sale (13 of the 473 differ, all so). Writes the pairs (listing
    text: session scratchpad only) and prints the draw by stratum (counts and ids only)."""
    import glob
    import os
    import random
    key = json.load(open(os.path.join(r1_dir, "KEY_pid_to_item_model_variant.json")))
    pairs = {q["pid"]: q for f in sorted(glob.glob(os.path.join(r1_dir, "batch_*_pairs.json"))) for q in json.load(open(f))}
    assert len(pairs) == 600 and set(pairs) == set(key), "expected the 600 pairs of the key"
    PS = {str(q["item"]): q for q in json.load(open(pair_scores))}
    assert len(PS) == 600
    flag = {k: PS[key[k]["item"]]["has_defect_strict"] for k in key}
    assert all(v in (0, 1, None) for v in flag.values()) and sum(v is not None for v in flag.values()) == 473
    done = set(json.load(open(done_answers))["answers"])
    assert done and done <= set(key), done
    cells = [("V1_enhance", "claude-opus-5-5", 6, 4), ("V1_enhance", "claude-sonnet-5-5", 4, 3),
             ("V1_enhance", "claude-haiku-5-5", 4, 2), ("V2_concise", "claude-opus-5-5", 3, 2),
             ("V2_concise", "claude-sonnet-5-5", 3, 2)]
    rnd = random.Random(seed)
    out = []
    for v, m, n, n_def in cells:
        pool = [k for k in sorted(key) if key[k]["variant"] == v and key[k]["model"] == m and k not in done and flag[k] is not None]
        dz = [k for k in pool if flag[k] == 1]
        nd = [k for k in pool if flag[k] == 0]
        assert len(dz) >= n_def and len(nd) >= n - n_def, (v, m, len(dz), len(nd))
        pick = rnd.sample(dz, n_def) + rnd.sample(nd, n - n_def)
        veh = sum(PS[key[k]["item"]]["cat1"] == "Automotive" for k in pick)
        print("%-10s %-18s %d pairs: %d state a defect (pool %d), %d vehicles: %s" % (v, m, n, n_def, len(dz), veh, " ".join(sorted(pick))))
        out += pick
    assert len(out) == len(set(out)) == 20 and not (set(out) & done)
    rnd.shuffle(out)
    json.dump([pairs[k] for k in out], open(out_json, "w"), indent=1)
    print("wrote %d pairs (%d with a stated defect) to %s; page order: %s" % (
        len(out), sum(flag[k] == 1 for k in out), out_json, " ".join(out)))


TYPES = ("condition", "specification", "suitability", "other")


def _share(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "share": k / n if n else None, "lo": lo, "hi": hi}


def _diff(a, b, rng, draws=2000):
    """Difference in shares a - b (lists of 0/1) with a bootstrap 95% interval and a permutation p (labels shuffled)."""
    import random as _r
    d = sum(a) / len(a) - sum(b) / len(b)
    pool, na = a + b, len(a)
    bs, hits = [], 0
    for _ in range(draws):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        bs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
        sh = pool[:]
        rng.shuffle(sh)
        hits += abs(sum(sh[:na]) / na - sum(sh[na:]) / (len(sh) - na)) >= abs(d) - 1e-12
    bs.sort()
    return {"diff": d, "lo": bs[int(0.025 * draws)], "hi": bs[int(0.975 * draws) - 1], "perm_p": hits / draws,
            "n_a": len(a), "n_b": len(b)}


def flags_of(r):
    """Per pair: which kinds of added claim the coders found (see the coders' brief in the Session 12 WORKLOG)."""
    cl = r.get("added_claims") or []
    f = {t: int(any(c.get("type") == t for c in cl)) for t in TYPES}
    f["any_claim"] = int(len(cl) > 0)
    f["unit_claim"] = int(f["condition"] or f["specification"])
    # Session 10's strict count left out bare reliability and quality words ("dependable", "sturdy"): the same rule
    # on the claims about the unit, to set the two codings side by side
    f["unit_strict"] = int(any(c.get("type") in ("condition", "specification") and not RELIAB.search(c.get("text") or "")
                               for c in cl))
    f["contradicts"] = int(any(c.get("contradicts") is True for c in cl))
    f["generic"] = int((r.get("added_generic") or 0) > 0)
    f["any_drop"] = int(r["dropped"] > 0)
    f["any_change"] = int(r["changed"] > 0)
    f["defect_dropped_or_softened"] = int((r["defects_dropped"] + r["defects_softened"]) > 0)
    return f


def fidelity_s12(r1_dir, inputs, pair_scores, quality_json, out_json, seed=12):
    """Session 12: aggregate the six blind coders' codes of the 600 pairs (code/rewrite_audit.py --batches) with each
    pair's instruction and model (the key), the original's category, length and a coded defect, and the item's quality
    (code/frozen_rewrites.py --quality). Writes counts and shares only (no listing text). Every input is required."""
    import glob
    import os
    import random
    key = json.load(open(os.path.join(r1_dir, "KEY_pid_to_item_model_variant.json")))
    C, models = {}, {}
    files = sorted(glob.glob(os.path.join(r1_dir, "batch_*_codes.json")))
    assert len(files) == 6, ("expected six coded batches", files)
    for fn in files:
        rows = json.load(open(fn))
        head = rows[0]
        assert "pid" not in head and "model" in head, (fn, "first element must be the coder's header")
        models[os.path.basename(fn)] = head["model"]
        for r in rows[1:]:
            assert r["pid"] in key and r["pid"] not in C, (fn, r.get("pid"))
            C[r["pid"]] = r
    assert len(C) == 600, len(C)
    bad = []
    for p, r in C.items():
        if r["kept"] + r["dropped"] + r["changed"] != r["n_orig_facts"]:
            bad.append((p, "facts"))
        if r["defects_kept"] + r["defects_softened"] + r["defects_dropped"] != r["orig_defects"]:
            bad.append((p, "defects"))
        if len(r.get("added_claims") or []) != r["added_specific"] or len(r.get("added_specific_list") or []) != r["added_specific"]:
            bad.append((p, "claims"))
        if any(c.get("type") not in TYPES for c in (r.get("added_claims") or [])):
            bad.append((p, "type"))
    X = {str(json.loads(l)["item"]): json.loads(l) for l in open(inputs) if l.strip()}
    PS = {str(q["item"]): q for q in json.load(open(pair_scores))}
    cuts = json.load(open(quality_json))["r_cuts"]
    meta = {}
    words = sorted(len(X[key[p]["item"]]["original"].split()) for p in C)
    med = words[len(words) // 2]
    for p in C:
        it = key[p]["item"]
        cat = X[it]["category"]
        r = PS[it]["r"]
        meta[p] = {"variant": key[p]["variant"], "model": key[p]["model"],
                   "vehicle": int(cat.startswith("Automotive") and not cat.startswith("Automotive > Parts")),
                   "defect": int(C[p]["orig_defects"] > 0), "short": int(len(X[it]["original"].split()) <= med),
                   "third": None if r is None else ("low" if r <= cuts[0] else ("high" if r > cuts[1] else "mid"))}
    F = {p: flags_of(C[p]) for p in C}
    keys = ["any_claim", "unit_claim", "unit_strict", "condition", "specification", "suitability", "contradicts", "generic",
            "any_drop", "any_change", "defect_dropped_or_softened"]

    def block(ps):
        return {k: _share(sum(F[p][k] for p in ps), len(ps)) for k in keys}
    out = {"source": "code/rewrite_audit.py --fidelity-s12 (Session 12): six blind coders, Claude Opus 5.5",
           "coder_models": models, "n_coded": len(C), "arithmetic_errors": len(bad), "arithmetic_error_list": bad[:20],
           "median_original_words": med}
    facts = sum(C[p]["n_orig_facts"] for p in C)
    out["facts"] = {"n": facts, "kept": sum(C[p]["kept"] for p in C), "dropped": sum(C[p]["dropped"] for p in C),
                    "changed": sum(C[p]["changed"] for p in C)}
    out["defects"] = {"n": sum(C[p]["orig_defects"] for p in C), "kept": sum(C[p]["defects_kept"] for p in C),
                      "softened": sum(C[p]["defects_softened"] for p in C), "dropped": sum(C[p]["defects_dropped"] for p in C)}
    cells = {}
    for p in C:
        cells.setdefault("%s|%s" % (meta[p]["model"], meta[p]["variant"]), []).append(p)
    out["cells"] = {c: block(ps) for c, ps in sorted(cells.items())}
    app = [p for p in C if meta[p]["variant"] == "V1_enhance"]
    con = [p for p in C if meta[p]["variant"] == "V2_concise"]
    out["all"] = block(list(C))
    out["appealing"] = block(app)
    out["concise"] = block(con)
    rng = random.Random(seed)
    sp = {}
    for name, f in (("defect", lambda p: meta[p]["defect"] == 1), ("short", lambda p: meta[p]["short"] == 1),
                    ("vehicle", lambda p: meta[p]["vehicle"] == 1)):
        a = [p for p in app if f(p)]
        b = [p for p in app if not f(p)]
        sp[name] = {"yes": block(a), "no": block(b),
                    "diff_any": _diff([F[p]["any_claim"] for p in a], [F[p]["any_claim"] for p in b], rng),
                    "diff_unit": _diff([F[p]["unit_claim"] for p in a], [F[p]["unit_claim"] for p in b], rng),
                    "diff_condition": _diff([F[p]["condition"] for p in a], [F[p]["condition"] for p in b], rng)}
    lo = [p for p in app if meta[p]["third"] == "low"]
    hi = [p for p in app if meta[p]["third"] == "high"]
    mid = [p for p in app if meta[p]["third"] == "mid"]
    sp["quality"] = {"low": block(lo), "mid": block(mid), "high": block(hi),
                     "diff_any": _diff([F[p]["any_claim"] for p in lo], [F[p]["any_claim"] for p in hi], rng),
                     "diff_unit": _diff([F[p]["unit_claim"] for p in lo], [F[p]["unit_claim"] for p in hi], rng),
                     "diff_condition": _diff([F[p]["condition"] for p in lo], [F[p]["condition"] for p in hi], rng),
                     "n_unpriced": sum(meta[p]["third"] is None for p in app)}
    out["splits_appealing"] = sp
    # review round 2: the two Opus "appealing" batches (inputs rows 0 to 99 and 100 to 199), the design's only replicate
    row = {str(json.loads(l)["item"]): k for k, l in enumerate(l for l in open(inputs) if l.strip())}
    for nm, lo_r, hi_r in (("first", 0, 100), ("second", 100, 200)):
        ps = [p for p in C if lo_r <= row[key[p]["item"]] < hi_r]
        assert len(ps) == 100 and all(meta[p]["model"] == "claude-opus-5-5" and meta[p]["variant"] == "V1_enhance" for p in ps)
        out.setdefault("opus_appealing_batches", {})[nm] = block(ps)
    cl = [c for p in app for c in (C[p].get("added_claims") or [])]
    out["claims_appealing"] = {t: sum(c.get("type") == t for c in cl) for t in TYPES}
    out["claims_appealing"]["total"] = len(cl)
    out["claims_appealing"]["contradicts"] = sum(c.get("contradicts") is True for c in cl)
    clc = [c for p in con for c in (C[p].get("added_claims") or [])]
    out["claims_concise"] = {t: sum(c.get("type") == t for c in clc) for t in TYPES}
    out["claims_concise"]["total"] = len(clc)
    json.dump(out, open(out_json, "w"), indent=1)
    A, Cn = out["appealing"], out["concise"]
    print("coded %d pairs by %s; arithmetic errors %d %s" % (len(C), sorted(set(models.values())), len(bad), bad[:6]))
    print("facts %d: kept %.4f dropped %d changed %d | defects %d: dropped %d softened %d" % (
        facts, out["facts"]["kept"] / facts, out["facts"]["dropped"], out["facts"]["changed"], out["defects"]["n"],
        out["defects"]["dropped"], out["defects"]["softened"]))
    for nm, B in (("appealing", A), ("concise", Cn), ("all", out["all"])):
        print("%-9s n %d | any claim %d (%.1f%%) | condition or spec %d (%.1f%%) | condition %d | spec %d | suitability %d"
              " | contradicts %d | generic %d" % (nm, B["any_claim"]["n"], B["any_claim"]["k"], 100 * B["any_claim"]["share"],
              B["unit_claim"]["k"], 100 * B["unit_claim"]["share"], B["condition"]["k"], B["specification"]["k"],
              B["suitability"]["k"], B["contradicts"]["k"], B["generic"]["k"]))
    for c, B in out["cells"].items():
        print("  %-34s n %d any %.1f%% unit %.1f%% condition %.1f%%" % (c, B["any_claim"]["n"], 100 * B["any_claim"]["share"],
              100 * B["unit_claim"]["share"], 100 * B["condition"]["share"]))
    for nm in ("defect", "short", "vehicle"):
        s = sp[nm]
        print("  appealing by %-7s yes %d: unit %.1f%% any %.1f%% | no %d: unit %.1f%% any %.1f%% | diff unit %+.3f [%+.3f, %+.3f] p %.3f"
              % (nm, s["yes"]["any_claim"]["n"], 100 * s["yes"]["unit_claim"]["share"], 100 * s["yes"]["any_claim"]["share"],
                 s["no"]["any_claim"]["n"], 100 * s["no"]["unit_claim"]["share"], 100 * s["no"]["any_claim"]["share"],
                 s["diff_unit"]["diff"], s["diff_unit"]["lo"], s["diff_unit"]["hi"], s["diff_unit"]["perm_p"]))
    q = sp["quality"]
    print("  appealing by quality third: low %d unit %.1f%% any %.1f%% | mid %d unit %.1f%% | high %d unit %.1f%% any %.1f%%"
          " | diff unit (low-high) %+.3f [%+.3f, %+.3f] p %.3f; unpriced %d" % (
              q["low"]["any_claim"]["n"], 100 * q["low"]["unit_claim"]["share"], 100 * q["low"]["any_claim"]["share"],
              q["mid"]["any_claim"]["n"], 100 * q["mid"]["unit_claim"]["share"], q["high"]["any_claim"]["n"],
              100 * q["high"]["unit_claim"]["share"], 100 * q["high"]["any_claim"]["share"], q["diff_unit"]["diff"],
              q["diff_unit"]["lo"], q["diff_unit"]["hi"], q["diff_unit"]["perm_p"], q["n_unpriced"]))
    print("claims (appealing): %s" % out["claims_appealing"])
    print("claims (concise): %s" % out["claims_concise"])
    print("wrote", out_json)


def agreement_s12(r1_dir, second_codes, out_json):
    """Session 12 (review round 1): a second coder (Claude Sonnet 5.5) coded 100 of the pairs blind; agreement with the
    first coding on each kind of flag (share of pairs on which the two agree; Cohen's kappa). Counts only."""
    import glob
    import os
    A = {r["pid"]: r for f in sorted(glob.glob(os.path.join(r1_dir, "batch_*_codes.json"))) for r in json.load(open(f))[1:]}
    B = json.load(open(second_codes))
    head, rows = B[0], B[1:]
    assert "pid" not in head and len(rows) == 100 and all(r["pid"] in A for r in rows), "expected 100 coded pairs and a header"
    keys = ["any_claim", "unit_claim", "condition", "suitability", "generic", "any_drop", "defect_dropped_or_softened"]
    out = {"source": "code/rewrite_audit.py --agreement-s12", "second_coder_model": head["model"], "n": len(rows), "flags": {}}
    for k in keys:
        a = [flags_of(A[r["pid"]])[k] for r in rows]
        b = [flags_of(r)[k] for r in rows]
        out["flags"][k] = {"agree": sum(x == y for x, y in zip(a, b)) / len(a), "kappa": kappa(a, b), "first": sum(a), "second": sum(b)}
        print("%-28s agree %.2f kappa %s (first %d, second %d)" % (k, out["flags"][k]["agree"],
              ("%.2f" % out["flags"][k]["kappa"]) if out["flags"][k]["kappa"] == out["flags"][k]["kappa"] else "n/a", sum(a), sum(b)))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def _pi_agree(pids, a, b):
    """Agreement of two 0/1 codings over the same pairs: count, Wilson 95% interval, Cohen's kappa (None when it is
    undefined), each side's count of 1s, and the ids coded 1 by one side only."""
    k = sum(x == y for x, y in zip(a, b))
    lo, hi = wilson(k, len(a))
    kp = kappa(a, b)
    return {"agree": k, "n": len(a), "lo": lo, "hi": hi, "kappa": None if kp != kp else kp, "pi": sum(a), "coders": sum(b),
            "pi_only": [p for p, x, y in zip(pids, a, b) if x and not y], "coders_only": [p for p, x, y in zip(pids, a, b) if y and not x]}


def selftest_pi_check():
    """The agreement can fail: a planted flip is found and named, and kappa matches a hand-computed table."""
    ids = ["P%d" % i for i in range(8)]
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    same = _pi_agree(ids, a, list(a))
    assert same["agree"] == 8 and same["kappa"] == 1 and not same["pi_only"] and not same["coders_only"]
    b = [1, 1, 1, 0, 0, 0, 0, 1]
    g = _pi_agree(ids, a, b)
    assert g["agree"] == 6 and g["pi_only"] == ["P3"] and g["coders_only"] == ["P7"] and abs(g["kappa"] - 0.5) < 1e-12, g
    print("selftest_pi_check: a planted disagreement is found and named; kappa 0.5 on the hand table")


def pi_check_s12(r1_dir, out_json, *answer_files):
    """Session 12 (PI, 2026-11-19): the PI's blind check on the page of label_app/build_label_app.py --pairs (each
    original beside its rewrite; three questions in the coders' terms; no model, instruction or code shown; answers
    downloaded before the codes were opened). Reads the downloaded answer files (a pair answered in two files must carry
    the same answers), keeps the pairs with all three questions answered, and sets each answer beside the six coders'
    flags: an unsupported claim (any_claim); a fact dropped or changed (any_drop or any_change); a stated problem
    softened or dropped (defect_dropped_or_softened). The page's third question also offers "no problem stated", which
    the PI used for 6 pairs and "no" for others without a stated problem, so it is scored as softened or not only.
    Writes counts and pair ids only (no listing text)."""
    import glob
    import os
    selftest_pi_check()
    key = json.load(open(os.path.join(r1_dir, "KEY_pid_to_item_model_variant.json")))
    C = {r["pid"]: r for f in sorted(glob.glob(os.path.join(r1_dir, "batch_*_codes.json"))) for r in json.load(open(f))[1:]}
    assert len(C) == 600 and set(C) == set(key)
    ans, files = {}, []
    for f in answer_files:
        d = json.load(open(f))
        files.append({"file": os.path.basename(f), "tool": d["tool"], "n_pairs": d["n_pairs"], "saved_at": d["saved_at"],
                      "answered": len(d["answers"])})
        for p, a in d["answers"].items():
            core = {k: a.get(k) for k in ("claim", "fact", "soft", "claim_text")}
            assert p not in ans or ans[p] == core, ("the same pair answered differently in two files", p)
            ans[p] = core
    full = {p: a for p, a in ans.items() if a["claim"] in ("yes", "no") and a["fact"] in ("yes", "no") and a["soft"] in ("yes", "no", "none")}
    assert full and set(full) <= set(key), "answers outside the key"
    pids = sorted(full)
    F = {p: flags_of(C[p]) for p in pids}
    pi = {"claim": [int(full[p]["claim"] == "yes") for p in pids], "fact": [int(full[p]["fact"] == "yes") for p in pids],
          "soft": [int(full[p]["soft"] == "yes") for p in pids]}
    co = {"claim": [F[p]["any_claim"] for p in pids], "fact": [int(F[p]["any_drop"] or F[p]["any_change"]) for p in pids],
          "soft": [F[p]["defect_dropped_or_softened"] for p in pids]}
    models = {}
    for p in pids:
        models[key[p]["model"]] = models.get(key[p]["model"], 0) + 1
    out = {"source": "code/rewrite_audit.py --pi-check-s12 (Session 12)", "answer_files": files, "n": len(pids), "pairs": pids,
           "variants": sorted({key[p]["variant"] for p in pids}), "models": models,
           "questions": {q: _pi_agree(pids, pi[q], co[q]) for q in ("claim", "fact", "soft")},
           "claim_by_model": {m: _pi_agree([p for p in pids if key[p]["model"] == m],
                                           [x for p, x in zip(pids, pi["claim"]) if key[p]["model"] == m],
                                           [x for p, x in zip(pids, co["claim"]) if key[p]["model"] == m]) for m in models},
           "coders_unit_claim_pairs": [p for p in pids if F[p]["unit_claim"]],
           "coders_stated_problem_pairs": [p for p in pids if C[p]["orig_defects"] > 0],
           "pi_claim_words_name_a_use": sum(1 for p in pids if full[p]["claim"] == "yes" and
                                            re.search(r"(?i)\b(uses?|utility)\b", full[p]["claim_text"] or "")),
           "pi_claims": sum(pi["claim"])}
    json.dump(out, open(out_json, "w"), indent=1)
    for q, g in out["questions"].items():
        print("%-5s agree %d of %d [%.2f, %.2f] kappa %s | PI %d, coders %d | PI only %s | coders only %s" % (
            q, g["agree"], g["n"], g["lo"], g["hi"], "n/a" if g["kappa"] is None else "%.2f" % g["kappa"], g["pi"], g["coders"],
            g["pi_only"], g["coders_only"]))
    print("models %s; coders' unit-claim pairs %d; coders' stated-problem pairs %d; PI claims %d, of which the PI's words name a use %d"
          % (models, len(out["coders_unit_claim_pairs"]), len(out["coders_stated_problem_pairs"]), out["pi_claims"],
             out["pi_claim_words_name_a_use"]))
    print("wrote", out_json)


NEGATED = re.compile(r"(?i)\b(no|not|without|never|nor|free of)\b")


def first_problem(text, skip_negated):
    """Where the first stated problem sits in a description (the paper's strict defect pattern, textfeat.DEFECT_STRICT):
    the words and the complete non-empty lines before it, the share of the description's words before it, and the
    matched words. With skip_negated, a match whose clause (from the last , . ; : ! ? or line break) carries a negation
    ("no major leaks", "no visible signs of excessive rust") is passed over. None when no match is left: not found is
    not a position."""
    from textfeat import DEFECT_STRICT
    t = text or ""
    m = None
    for x in DEFECT_STRICT.finditer(t):
        j = max(t.rfind(c, 0, x.start()) for c in ".;:!?\n,")
        if skip_negated and NEGATED.search(t[j + 1:x.start()]):
            continue
        m = x
        break
    if not m:
        return None
    before = t[:m.start()]
    w_all = len(t.split())
    return {"words_before": len(before.split()), "words": w_all, "share_before": len(before.split()) / w_all,
            "lines_before": sum(1 for line in before.split("\n")[:-1] if line.strip()), "match": m.group(0).lower()}


def selftest_burying():
    """The measure can fail: a problem in the second line is found early, the same problem pushed below a list is found
    late, and a text without one gives None, not a position."""
    o = first_problem("2004 Ford F-350 dump truck.\nDoes not start.", True)
    assert o["words_before"] == 5 and o["lines_before"] == 1 and o["match"] == "does not start", o
    r = first_problem("2004 Ford F-350 Dump Truck\n\nDetails:\n- Year: 2004\n- Make: Ford\n- Model: F-350\n"
                      "- Body: dump\n\nCondition notes:\n- The truck does not start.", True)
    assert r["words_before"] > o["words_before"] and r["lines_before"] == 7, r
    assert first_problem("A sturdy dump truck, ready for its next owner.", True) is None
    n = "Runs well, no major leaks. Rear seat torn."
    assert first_problem(n, False)["match"] == "leaks" and first_problem(n, True)["match"] == "torn", "negation not passed over"
    print("selftest_burying: an early problem, a pushed-down problem, no problem and a negated mention are told apart")


def burying(r1_dir, pair_scores, out_json, skip_negated, seed=23):
    """Session 12 (the PI's reading in the blind check, 2026-11-19: in a two-line original the problem is the second
    sentence; in the long, listed rewrite "you have to scroll"). For the pairs whose original states a problem, where
    the first stated problem sits in the original and in its rewrite. Selection: the panel's strict flag on the text
    less boilerplate (pair_scores has_defect_strict == 1; priced originals only, the flag is None for the others), and,
    as a check, the strict pattern on the raw original (which also counts boilerplate terms of sale). Each statistic is
    over the pairs where the pattern finds the problem in both texts; a rewrite in which it finds none is counted apart.
    Writes counts, medians, shares and pair ids only (no listing text); prints the extreme pairs for reading. Run with
    skip_negated "no" (burying_s12.json, the first computation) and "yes" (burying_s12b.json: negated mentions such
    as "no major leaks" are not stated problems; the main version)."""
    assert skip_negated in ("yes", "no"), skip_negated
    neg = skip_negated == "yes"
    import glob
    import os
    import random
    import statistics
    selftest_burying()
    key = json.load(open(os.path.join(r1_dir, "KEY_pid_to_item_model_variant.json")))
    pairs = {q["pid"]: q for f in sorted(glob.glob(os.path.join(r1_dir, "batch_*_pairs.json"))) for q in json.load(open(f))}
    assert len(pairs) == 600 and set(pairs) == set(key)
    PS = {str(q["item"]): q for q in json.load(open(pair_scores))}
    flag = {k: PS[key[k]["item"]]["has_defect_strict"] for k in key}
    sel = {"panel": [k for k in sorted(key) if flag[k] == 1],
           "raw": [k for k in sorted(key) if first_problem(pairs[k]["original"], neg) is not None]}
    rng = random.Random(seed)
    med = statistics.median
    out = {"source": "code/rewrite_audit.py --burying (Session 12)", "skip_negated": neg, "selections": {}}
    for sname, ids in sel.items():
        rows = []
        for k in ids:
            o, r = first_problem(pairs[k]["original"], neg), first_problem(pairs[k]["rewrite"], neg)
            rows.append((k, o, r))
        groups = {"all": rows, "appealing": [x for x in rows if key[x[0]]["variant"] == "V1_enhance"],
                  "concise": [x for x in rows if key[x[0]]["variant"] == "V2_concise"]}
        for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
            groups["appealing|" + m] = [x for x in rows if key[x[0]]["variant"] == "V1_enhance" and key[x[0]]["model"] == m]
        res = {}
        for gname, g in groups.items():
            both = [(k, o, r) for k, o, r in g if o is not None and r is not None]
            if not both:
                res[gname] = {"n": len(g), "found_in_both": 0}
                continue
            wo = [o["words_before"] for _, o, _ in both]
            wr = [r["words_before"] for _, _, r in both]
            d = [b - a for a, b in zip(wo, wr)]
            bs = sorted(med([d[rng.randrange(len(d))] for _ in d]) for _ in range(2000))
            share = lambda xs, c: sum(x <= c for x in xs) / len(xs)
            res[gname] = {
                "n": len(g), "orig_not_found": sum(o is None for _, o, _ in g),
                "rewrite_not_found": sum(o is not None and r is None for _, o, r in g), "found_in_both": len(both),
                "words_before_orig": {"median": med(wo), "q1": statistics.quantiles(wo, n=4)[0] if len(wo) > 1 else wo[0],
                                      "q3": statistics.quantiles(wo, n=4)[2] if len(wo) > 1 else wo[0]},
                "words_before_rewrite": {"median": med(wr), "q1": statistics.quantiles(wr, n=4)[0] if len(wr) > 1 else wr[0],
                                         "q3": statistics.quantiles(wr, n=4)[2] if len(wr) > 1 else wr[0]},
                "diff_median": med(d), "diff_lo": bs[49], "diff_hi": bs[1949],
                "later_in_rewrite": sum(x > 0 for x in d), "earlier_in_rewrite": sum(x < 0 for x in d),
                "within_25_orig": share(wo, 24), "within_25_rewrite": share(wr, 24),
                "within_50_orig": share(wo, 49), "within_50_rewrite": share(wr, 49),
                "lines_before_orig_median": med([o["lines_before"] for _, o, _ in both]),
                "lines_before_rewrite_median": med([r["lines_before"] for _, _, r in both]),
                "share_before_orig_median": med([o["share_before"] for _, o, _ in both]),
                "share_before_rewrite_median": med([r["share_before"] for _, _, r in both])}
        out["selections"][sname] = {"n_pairs": len(ids), "groups": res}
        g = res["all"]
        print("[%s] %d pairs; found in both %d; problem not found in the rewrite by the pattern %d; words before: original "
              "median %s (q1 %s, q3 %s), rewrite median %s (q1 %s, q3 %s); paired median difference %+s [%+s, %+s]; later %d, "
              "earlier %d; within 25 words %.0f%% -> %.0f%%; within 50 %.0f%% -> %.0f%%; lines before %s -> %s" % (
              sname, g["n"], g["found_in_both"], g["rewrite_not_found"], g["words_before_orig"]["median"],
              g["words_before_orig"]["q1"], g["words_before_orig"]["q3"], g["words_before_rewrite"]["median"],
              g["words_before_rewrite"]["q1"], g["words_before_rewrite"]["q3"], g["diff_median"], g["diff_lo"], g["diff_hi"],
              g["later_in_rewrite"], g["earlier_in_rewrite"], 100 * g["within_25_orig"], 100 * g["within_25_rewrite"],
              100 * g["within_50_orig"], 100 * g["within_50_rewrite"], g["lines_before_orig_median"], g["lines_before_rewrite_median"]))
        for gname in res:
            if gname != "all" and res[gname].get("found_in_both"):
                x = res[gname]
                print("   %-28s n %3d both %3d: words before %s -> %s (diff %+s [%+s, %+s]); within 25 %.0f%% -> %.0f%%; not found in rewrite %d" % (
                    gname, x["n"], x["found_in_both"], x["words_before_orig"]["median"], x["words_before_rewrite"]["median"],
                    x["diff_median"], x["diff_lo"], x["diff_hi"], 100 * x["within_25_orig"], 100 * x["within_25_rewrite"], x["rewrite_not_found"]))
        if sname == "panel":
            both = sorted([(r["words_before"] - o["words_before"], k, o, r) for k, o, r in rows if o and r], key=lambda z: z[0])
            out["extremes"] = {"most_later": [z[1] for z in both[-5:]], "most_earlier": [z[1] for z in both[:5]],
                               "rewrite_not_found": [k for k, o, r in rows if o and r is None]}
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def _paired(a, b, rng, draws=2000):
    """Paired difference in shares b - a over the same items (lists of 0/1), with a bootstrap 95% interval over items and
    the discordant counts (a only, b only)."""
    n = len(a)
    d = (sum(b) - sum(a)) / n
    bs = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        bs.append((sum(b[i] for i in idx) - sum(a[i] for i in idx)) / n)
    bs.sort()
    return {"diff": d, "lo": bs[int(0.025 * draws)], "hi": bs[int(0.975 * draws) - 1], "n": n,
            "a_only": sum(1 for x, y in zip(a, b) if x and not y), "b_only": sum(1 for x, y in zip(a, b) if y and not x)}


def arm_s12(r1_dir, arm_dir, synth, out_json, seed=13):
    """Session 12 (review round 1; both outside referees): the "appealing" and "concise" instructions differ in their aim
    and in the sentence "Use only the information in the title and the original description". A third arm gives Claude
    Opus 5.5 the "appealing" instruction plus that sentence, on the same 100 listings it rewrote under "appealing" (rows 0
    to 99 of the inputs), coded blind by a seventh coder with the same brief. Paired by item. Counts only."""
    import glob
    import os
    import random
    key = json.load(open(os.path.join(r1_dir, "KEY_pid_to_item_model_variant.json")))
    A = {}
    for f in sorted(glob.glob(os.path.join(r1_dir, "batch_*_codes.json"))):
        for r in json.load(open(f))[1:]:
            A[key[r["pid"]]["item"]] = (r, key[r["pid"]])
    ka = json.load(open(os.path.join(arm_dir, "KEY_arm.json")))
    rows = json.load(open(os.path.join(arm_dir, "arm_codes.json")))
    head, rows = rows[0], rows[1:]
    assert "pid" not in head and len(rows) == 100, "expected the coder's header and 100 pairs"
    B = {ka[r["pid"]]["item"]: r for r in rows}
    items = sorted(B)
    assert all(A[i][1]["model"] == "claude-opus-5-5" and A[i][1]["variant"] == "V1_enhance" for i in items), "not the Opus appealing items"
    bad = [r["pid"] for r in rows if r["kept"] + r["dropped"] + r["changed"] != r["n_orig_facts"]
           or r["defects_kept"] + r["defects_softened"] + r["defects_dropped"] != r["orig_defects"]
           or len(r.get("added_claims") or []) != r["added_specific"]]
    rng = random.Random(seed)
    keys = ["any_claim", "unit_claim", "condition", "specification", "suitability", "contradicts", "generic", "any_drop",
            "defect_dropped_or_softened"]
    out = {"source": "code/rewrite_audit.py --arm-s12 (Session 12)", "coder_model": head["model"], "n": len(items),
           "arithmetic_errors": len(bad), "flags": {}}
    for k in keys:
        a = [flags_of(A[i][0])[k] for i in items]
        b = [flags_of(B[i])[k] for i in items]
        out["flags"][k] = {"appealing": _share(sum(a), len(a)), "appealing_only_info": _share(sum(b), len(b)), "paired": _paired(a, b, rng)}
        e = out["flags"][k]
        print("%-28s appealing %3d | plus 'use only' %3d | difference %+.2f [%+.2f, %+.2f] (lost %d, new %d)" % (
            k, sum(a), sum(b), e["paired"]["diff"], e["paired"]["lo"], e["paired"]["hi"], e["paired"]["a_only"], e["paired"]["b_only"]))
    fa = sum(B[i]["kept"] for i in items) / sum(B[i]["n_orig_facts"] for i in items)
    out["facts_kept_only_info"] = fa
    out["claims_only_info"] = {t: sum(1 for i in items for c in (B[i].get("added_claims") or []) if c.get("type") == t) for t in TYPES}
    # how much each rewrite reorganizes: median ratio of rewrite words to original words, same 100 items
    Xi = {x["item"]: x for x in json.load(open(os.path.join(arm_dir, "arm_inputs.json")))}
    Ro = {r["item"]: r["rewrite"] for r in json.load(open(os.path.join(arm_dir, "arm_rewrites.json")))[1:]}
    Ya = {str(json.loads(l)["item"]): json.loads(l)["ai_text"] for l in open(synth) if l.strip()}
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2
    wc = lambda t: max(1, len(t.split()))
    out["length_ratio_appealing"] = med([wc(Ya[i]) / wc(Xi[i]["original"]) for i in items])
    out["length_ratio_only_info"] = med([wc(Ro[i]) / wc(Xi[i]["original"]) for i in items])
    out["identical_to_original"] = sum(1 for i in items if " ".join(Ro[i].split()) == " ".join(Xi[i]["original"].split()))
    print("facts kept under 'use only': %.4f; arithmetic errors %d; claims by type %s" % (fa, len(bad), out["claims_only_info"]))
    print("median length ratio: appealing %.2f, plus 'use only' %.2f; identical to the original: %d" % (
        out["length_ratio_appealing"], out["length_ratio_only_info"], out["identical_to_original"]))
    json.dump(out, open(out_json, "w"), indent=1)
    print("wrote", out_json)


def selftest_s12():
    """The split statistic finds a planted gap and none where there is none."""
    import random
    rng = random.Random(3)
    a = [1] * 30 + [0] * 70
    b = [1] * 10 + [0] * 90
    d = _diff(a, b, rng, 1000)
    assert abs(d["diff"] - 0.20) < 1e-9 and d["lo"] > 0.05 and d["perm_p"] < 0.01, d
    same = _diff([1] * 20 + [0] * 80, [1] * 20 + [0] * 80, rng, 1000)
    assert abs(same["diff"]) < 1e-9 and same["lo"] < 0 < same["hi"] and same["perm_p"] > 0.5, same
    r = {"n_orig_facts": 3, "kept": 3, "dropped": 0, "changed": 0, "orig_defects": 1, "defects_kept": 1, "defects_softened": 0,
         "defects_dropped": 0, "added_specific": 2, "added_generic": 1,
         "added_claims": [{"text": "ready to tow", "type": "condition", "contradicts": True}, {"text": "ideal for farms", "type": "suitability", "contradicts": False}]}
    f = flags_of(r)
    assert f["condition"] == 1 and f["suitability"] == 1 and f["specification"] == 0 and f["unit_claim"] == 1 and f["contradicts"] == 1, f
    pr = _paired([1] * 50 + [0] * 50, [1] * 10 + [0] * 90, rng, 1000)
    assert abs(pr["diff"] + 0.40) < 1e-9 and pr["hi"] < -0.25 and pr["a_only"] == 40 and pr["b_only"] == 0, pr
    same_p = _paired([1, 0] * 50, [1, 0] * 50, rng, 1000)
    assert same_p["diff"] == 0 and same_p["lo"] == 0 == same_p["hi"], same_p
    print("selftest: a 20-point planted gap is found (p %.3f); equal groups give %+.2f (p %.2f); flags read the claim types" % (
        d["perm_p"], same["diff"], same["perm_p"]))


if __name__ == "__main__":
    if sys.argv[1] == "--burying":
        burying(*sys.argv[2:6])             # R1_DIR PAIR_SCORES OUT_JSON SKIP_NEGATED(yes|no) (local; counts only)
        sys.exit(0)
    if sys.argv[1] == "--pi-check-s12":
        pi_check_s12(sys.argv[2], sys.argv[3], *sys.argv[4:])   # R1_DIR OUT_JSON ANSWERS.json [ANSWERS.json ...] (counts only)
        sys.exit(0)
    if sys.argv[1] == "--pi-extra":
        pi_extra(*sys.argv[2:6])            # R1_DIR PAIR_SCORES DONE_ANSWERS OUT_JSON (local; the output holds listing text)
        sys.exit(0)
    if sys.argv[1] == "--batches":
        batches(*sys.argv[2:5])
        sys.exit(0)
    if sys.argv[1] == "--selftest-s12":
        selftest_s12()
        sys.exit(0)
    if sys.argv[1] == "--arm-s12":
        selftest_s12()
        arm_s12(*sys.argv[2:6])          # R1_DIR ARM_DIR SYNTH_600 OUT_JSON
        sys.exit(0)
    if sys.argv[1] == "--agreement-s12":
        agreement_s12(*sys.argv[2:5])
        sys.exit(0)
    if sys.argv[1] == "--fidelity-s12":
        selftest_s12()
        fidelity_s12(*sys.argv[2:7])
        sys.exit(0)
    if sys.argv[1] == "--fidelity":
        fidelity(*sys.argv[2:6])
    else:
        main(sys.argv[1], sys.argv[2:])
