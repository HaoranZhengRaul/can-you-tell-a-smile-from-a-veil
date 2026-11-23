#!/usr/bin/env python3
"""Session 9: two standard open-weight AI-text detectors, Binoculars (Hans et al. 2024) and Fast-DetectGPT (Bao
et al. 2024), put through the paper's detector checks, with word-level views of a few example listings.

Both use the authors' published default pair, tiiuae/falcon-7b and tiiuae/falcon-7b-instruct in bfloat16
(Binoculars: observer and performer; Fast-DetectGPT: sampling and scoring model), so one forward pass of each model
per text gives both scores. The formulas follow the authors' code, read on 2026-10-26 (Binoculars
binoculars/metrics.py and detector.py; fast-detect-gpt scripts/fast_detect_gpt.py, get_sampling_discrepancy_analytic,
and scripts/local_infer.py); `selftest` checks this file against direct ports of those functions.
  Binoculars B = mean surprisal of each next token under the performer / mean cross-entropy of the performer
    against the observer's next-token distribution; AI when B < 0.8536 (their low-false-positive threshold).
  Fast-DetectGPT C = (sum_t log p(x_t) - sum_t mu_t) / sqrt(sum_t var_t), p from the scoring model, mu and var over
    the sampling model's distribution; P(AI) from their normal fits for this pair; AI when P(AI) > 0.5.
Metrics are computed in float32 from the bfloat16 models' logits; texts are cut at 512 tokens, as Binoculars does.
Each detector is also run at a threshold set to flag 1% of pre-ChatGPT listings, like our classifier.

Texts: every listing with a description, the 600 controlled rewrites and their originals, the 100 blind-read texts.
Outputs (OUT_DIR): scores.parquet (one row per text), blind_scores.csv, summary.json (coverage; flag shares by period
and month in the analysis sample; recall on the rewrites by instruction and generator; share of rewrites scored more
AI-like than their own original; rank correlation with our classifier), and two files with raw text that stay out of
git: tokens_examples.json (per-token contributions for the example listings) and top_records.json (the most AI-like
pre-ChatGPT listings under each detector, to be read).
Usage: python3 std_detectors.py RESULTS_DIR OUT_DIR MODELS_DIR [--limit N]      (GPU job)
       python3 std_detectors.py --selftest                                      (no models needed)
"""
import json
import math
import os
import sys

BINO_LOW_FPR = 0.8536432310785527   # binoculars/detector.py, "optimized for low-fpr", AI when below
BINO_ACCURACY = 0.9015310749276843  # binoculars/detector.py, "optimized for f1-score"
FDG_PAIR = {"mu0": -0.0707, "sigma0": 0.9520, "mu1": 2.9306, "sigma1": 1.9039}  # local_infer.py, falcon-7b_falcon-7b-instruct
MAX_TOKENS = 512
EXAMPLE_ITEMS = ["75374770", "76648142", "21346726"]  # the paper's field examples (bottles, hybrid bus, 2019 pickup);
# item numbers are text in the panel, and the blind rows have none, so items stay text throughout
EXAMPLE_REWRITE_SNIPPET = "great running vehicle, but has a few oil leaks"


def norm_pdf(x, mu, s):
    return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def fdg_prob(c):
    """P(AI) of local_infer.compute_prob_norm for the falcon pair (equal priors)."""
    p1 = norm_pdf(c, FDG_PAIR["mu1"], FDG_PAIR["sigma1"])
    p0 = norm_pdf(c, FDG_PAIR["mu0"], FDG_PAIR["sigma0"])
    return p1 / (p0 + p1)


def scores_from_logits(obs_logits, perf_logits, ids, pad_id, torch):
    """One text: obs_logits and perf_logits [n, V], ids [n]. Returns both scores and per-token pieces.

    Binoculars (metrics.py): ppl = mean over positions 0..n-2 of the performer's cross-entropy of the next token;
    x_ppl = mean over every non-pad position of -sum_v softmax(observer)_v log softmax(performer)_v.
    Fast-DetectGPT (analytic): scoring = performer, sampling (reference) = observer, positions 0..n-2."""
    o = obs_logits.float()
    p = perf_logits.float()
    lp_perf = torch.log_softmax(p, dim=-1)
    lab = ids[1:]
    ce = -lp_perf[:-1].gather(-1, lab.unsqueeze(-1)).squeeze(-1)          # surprisal of each next token
    p_obs = torch.softmax(o, dim=-1)
    xent = -(p_obs * lp_perf).sum(-1)                                      # expected surprisal, every position
    mask = (ids != pad_id).float()
    ppl = ce.mean()
    x_ppl = (xent * mask).sum() / mask.sum()
    ll = -ce
    mu = (p_obs[:-1] * lp_perf[:-1]).sum(-1)
    var = (p_obs[:-1] * lp_perf[:-1] ** 2).sum(-1) - mu ** 2
    sd = var.sum().sqrt()
    crit = (ll.sum() - mu.sum()) / sd
    return {"binoculars": float(ppl / x_ppl), "ppl": float(ppl), "x_ppl": float(x_ppl), "fdg_crit": float(crit),
            # token t (t >= 1): Fast-DetectGPT share of the criterion, and Binoculars surprisal minus expected surprisal
            "tok_fdg": ((ll - mu) / sd).tolist(), "tok_bino": (ce - xent[:-1]).tolist()}


def selftest():
    import torch
    torch.manual_seed(0)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    softmax_fn = torch.nn.Softmax(dim=-1)

    # direct ports of the authors' functions (batch of one, no padding)
    def a_perplexity(input_ids, attn, logits):
        sl, slab, sm = logits[..., :-1, :].contiguous(), input_ids[..., 1:].contiguous(), attn[..., 1:].contiguous()
        return ((ce_loss_fn(sl.transpose(1, 2), slab) * sm).sum(1) / sm.sum(1)).float().numpy()

    def a_entropy(p_logits, q_logits, input_ids, pad_token_id):
        V, T = p_logits.shape[-1], q_logits.shape[-2]
        p_proba = softmax_fn(p_logits).view(-1, V)
        ce_ = ce_loss_fn(input=q_logits.view(-1, V), target=p_proba).view(-1, T)
        pm = (input_ids != pad_token_id).type(torch.uint8)
        return (((ce_ * pm).sum(1) / pm.sum(1)).float().numpy())

    def a_fdg(logits_ref, logits_score, labels):
        labels = labels.unsqueeze(-1)
        lps = torch.log_softmax(logits_score, dim=-1)
        pr = torch.softmax(logits_ref, dim=-1)
        ll = lps.gather(dim=-1, index=labels).squeeze(-1)
        m = (pr * lps).sum(dim=-1)
        v = (pr * torch.square(lps)).sum(dim=-1) - torch.square(m)
        return ((ll.sum(dim=-1) - m.sum(dim=-1)) / v.sum(dim=-1).sqrt()).mean().item()

    for n, V in ((12, 50), (3, 7), (40, 300)):
        ids = torch.randint(1, V, (n,))
        obs, perf = torch.randn(n, V) * 3, torch.randn(n, V) * 3
        r = scores_from_logits(obs, perf, ids, 0, torch)
        want_b = float(a_perplexity(ids[None], torch.ones(1, n), perf[None])[0] / a_entropy(obs[None], perf[None], ids[None], 0)[0])
        want_f = a_fdg(obs[None, :-1], perf[None, :-1], ids[None, 1:])
        assert abs(r["binoculars"] - want_b) < 1e-5, (r["binoculars"], want_b)
        assert abs(r["fdg_crit"] - want_f) < 1e-4, (r["fdg_crit"], want_f)
        assert abs(sum(r["tok_fdg"]) - r["fdg_crit"]) < 1e-4, "token pieces must add up to the criterion"
        assert len(r["tok_fdg"]) == len(r["tok_bino"]) == n - 1
        # a failing world: the roles of the two models swapped, or the next-token shift dropped, give other numbers
        swapped = scores_from_logits(perf, obs, ids, 0, torch)
        assert abs(swapped["binoculars"] - want_b) > 1e-4 and abs(swapped["fdg_crit"] - want_f) > 1e-4
        noshift = float(ce_loss_fn(perf, ids).mean() / a_entropy(obs[None], perf[None], ids[None], 0)[0])
        assert abs(noshift - want_b) > 1e-4
    # padding: a pad id inside the text leaves Binoculars' expected-surprisal average over the other positions
    ids = torch.tensor([5, 0, 7, 9])
    obs, perf = torch.randn(4, 12), torch.randn(4, 12)
    r = scores_from_logits(obs, perf, ids, 0, torch)
    assert abs(r["x_ppl"] - float(a_entropy(obs[None], perf[None], ids[None], 0)[0])) < 1e-5
    # the published probability: equal densities give one half; the AI side of the fit gives more
    c_mid = None
    for x in [i / 100 for i in range(-100, 400)]:
        if fdg_prob(x) >= 0.5:
            c_mid = x
            break
    assert c_mid is not None and 0.5 < c_mid < 2.5 and fdg_prob(FDG_PAIR["mu1"]) > 0.9 and fdg_prob(FDG_PAIR["mu0"]) < 0.2
    print("selftest passed: Binoculars and Fast-DetectGPT match ports of the authors' functions on random logits "
          "(3 sizes, padding); token pieces add up; swapped models and a dropped shift fail; P(AI) crosses 0.5 at %.2f" % c_mid)


def main(res, out_dir, models_dir, limit=None):
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
    import detector
    from analysis_core import load
    P = res.rstrip("/") + "/"
    os.makedirs(out_dir, exist_ok=True)
    panel, feats, scores = P + "panel_s8/panel.parquet", P + "panel_s8/features.parquet", P + "detector_s8/ai_scores.parquet"

    # ---- texts, each normalized as our classifier normalizes it
    df = pd.read_parquet(panel)
    df = df[df.desc.notna() & df.start_local_dt.notna()][["item", "desc", "start_local_dt"]].copy()
    rows = [{"kind": "listing", "item": str(r.item), "text": detector.norm(r.desc)} for r in df.itertuples()]
    S = [json.loads(l) for l in open(P + "synth_s8/synthetic_600.jsonl") if l.strip()]
    X = {}
    for l in open(P + "synth_s8/inputs_600.jsonl"):
        if l.strip():
            x = json.loads(l)
            X[x["item"]] = x
    rows += [{"kind": "rewrite", "item": str(s["item"]), "variant": s["variant"], "model": s["model"],
              "text": detector.norm(s["ai_text"])} for s in S]
    rows += [{"kind": "original", "item": str(k), "text": detector.norm(x["original"])} for k, x in X.items()]
    B = [json.loads(l) for l in open(P + "blind_s8/texts.jsonl") if l.strip()]
    rows += [{"kind": "blind", "bid": b["id"], "text": detector.norm(b["description"])} for b in B]
    R = pd.DataFrame(rows)
    if limit:
        R = pd.concat([R[R.kind == k].head(limit) for k in R.kind.unique()], ignore_index=True)
    print("texts:", R.kind.value_counts().to_dict(), flush=True)

    # ---- models (local copies; compute nodes have no internet)
    tok = AutoTokenizer.from_pretrained(os.path.join(models_dir, "falcon-7b"))
    tok2 = AutoTokenizer.from_pretrained(os.path.join(models_dir, "falcon-7b-instruct"))
    assert tok.get_vocab() == tok2.get_vocab(), "the two tokenizers must agree (Binoculars asserts this)"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = "cuda"
    obs_m = AutoModelForCausalLM.from_pretrained(os.path.join(models_dir, "falcon-7b"), torch_dtype=torch.bfloat16).to(dev).eval()
    perf_m = AutoModelForCausalLM.from_pretrained(os.path.join(models_dir, "falcon-7b-instruct"), torch_dtype=torch.bfloat16).to(dev).eval()

    ex_items = set(EXAMPLE_ITEMS)
    ex_rewrite_items = {str(k) for k, x in X.items() if EXAMPLE_REWRITE_SNIPPET in (x.get("original") or "")}
    out, tokens = [], []
    with torch.inference_mode():
        for i, r in enumerate(R.itertuples()):
            rec = {"status": "scored"}
            if not r.text:
                rec["status"] = "empty"
            else:
                enc = tok(r.text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, return_token_type_ids=False,
                          return_offsets_mapping=True)
                offs = enc.pop("offset_mapping")[0].tolist()
                ids = enc["input_ids"][0]
                if len(ids) < 3:
                    rec["status"] = "too_short"
                else:
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    o = obs_m(**enc).logits[0]
                    p = perf_m(**enc).logits[0]
                    sc = scores_from_logits(o, p, ids.to(dev), pad_id, torch)
                    rec.update({k: sc[k] for k in ("binoculars", "ppl", "x_ppl", "fdg_crit")})
                    rec["fdg_prob"] = fdg_prob(sc["fdg_crit"])
                    rec["n_tokens"] = int(len(ids))
                    rec["truncated"] = bool(len(ids) >= MAX_TOKENS)
                    is_ex = (r.kind == "listing" and r.item in ex_items) or (r.kind in ("rewrite", "original") and r.item in ex_rewrite_items)
                    if is_ex:
                        tokens.append({"kind": r.kind, "item": r.item, "text": r.text, "spans": offs[1:],
                                       "tok_fdg": sc["tok_fdg"], "tok_bino": sc["tok_bino"],
                                       "binoculars": sc["binoculars"], "fdg_crit": sc["fdg_crit"]})
            out.append(rec)
            if i % 1000 == 0:
                print("  scored %d of %d" % (i, len(R)), flush=True)
    R = pd.concat([R.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    R["bino_ai"] = R.binoculars < BINO_LOW_FPR
    R["fdg_ai"] = R.fdg_prob > 0.5
    R.drop(columns=["text"]).to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)

    # ---- the checks, in the analysis sample (periods by start date, as in the paper)
    d = load(panel, feats, scores)
    L = R[(R.kind == "listing") & (R.status == "scored")][["item", "binoculars", "fdg_crit", "fdg_prob", "bino_ai", "fdg_ai"]]
    d["item"] = d["item"].astype(str)
    d = d.merge(L, on="item", how="left")
    p0 = d[(d.period == "P0") & d.binoculars.notna()]
    cut = {"bino": float(p0.binoculars.quantile(0.01)), "fdg": float(p0.fdg_crit.quantile(0.99))}  # flag 1% before ChatGPT
    d["bino_cal"] = d.binoculars <= cut["bino"]
    d["fdg_cal"] = d.fdg_crit >= cut["fdg"]
    Rw = R[(R.kind == "rewrite") & (R.status == "scored")].copy()
    Rw["bino_cal"], Rw["fdg_cal"] = Rw.binoculars <= cut["bino"], Rw.fdg_crit >= cut["fdg"]
    Ro = R[(R.kind == "original") & (R.status == "scored")].set_index("item")
    summ = {"coverage": {k: R[R.kind == k].status.value_counts().to_dict() for k in R.kind.unique()},
            "analysis_sample_scored": {per: int(g.binoculars.notna().sum()) for per, g in d.groupby("period")},
            "truncated": int(R.truncated.fillna(False).sum()), "cut_1pct_pre_chatgpt": cut,
            "thresholds": {"binoculars_low_fpr": BINO_LOW_FPR, "binoculars_accuracy": BINO_ACCURACY, "fdg_prob": 0.5}}
    rules = {"bino_ai": "Binoculars, published threshold", "fdg_ai": "Fast-DetectGPT, published rule",
             "bino_cal": "Binoculars, 1% before ChatGPT", "fdg_cal": "Fast-DetectGPT, 1% before ChatGPT"}
    dd = d[d.binoculars.notna()].copy()
    dd["m"] = dd.start_local_dt.dt.to_period("M").astype(str)
    for col in rules:
        s = {"label": rules[col],
             "share_by_period": {per: float(g[col].mean()) for per, g in dd.groupby("period")},
             "n_by_period": {per: int(len(g)) for per, g in dd.groupby("period")},
             "monthly": {m: {"n": int(len(g)), "ai": float(g[col].mean())} for m, g in dd.groupby("m")},
             "recall_rewrites": {"all": float(Rw[col].mean()), "n": int(len(Rw)),
                                 **{"%s|%s" % k: float(g[col].mean()) for k, g in Rw.groupby(["variant", "model"])}},
             "agree_with_main_rule_tool_era": float((dd.loc[dd.period == "P2", col] == (dd.loc[dd.period == "P2", "ai"] > 0)).mean())}
        summ[col] = s
    both = Rw.set_index("item").join(Ro[["binoculars", "fdg_crit"]], rsuffix="_orig", how="inner")
    summ["rewrite_more_ai_like_than_original"] = {"binoculars": float((both.binoculars < both.binoculars_orig).mean()),
                                                  "fdg": float((both.fdg_crit > both.fdg_crit_orig).mean()), "pairs": int(len(both))}
    t2 = dd[dd.period == "P2"]
    summ["spearman_with_our_classifier_tool_era"] = {"binoculars": float(spearmanr(-t2.binoculars, t2.ai_score).correlation),
                                                     "fdg": float(spearmanr(t2.fdg_crit, t2.ai_score).correlation), "n": int(len(t2))}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    key = pd.read_csv(P + "blind_s8/KEY_do_not_open.csv")
    Rb = R[R.kind == "blind"][["bid", "status", "binoculars", "fdg_crit", "fdg_prob", "bino_ai", "fdg_ai"]].merge(
        key[["bid", "stratum"]], on="bid", how="left")
    Rb["bino_cal"], Rb["fdg_cal"] = Rb.binoculars <= cut["bino"], Rb.fdg_crit >= cut["fdg"]
    Rb.to_csv(os.path.join(out_dir, "blind_scores.csv"), index=False)
    # raw text, to be read and kept out of git: the example token views and the most AI-like pre-ChatGPT listings
    json.dump(tokens, open(os.path.join(out_dir, "tokens_examples.json"), "w"), ensure_ascii=False)
    txt = dict(zip(df.item.astype(str), df.desc))
    top = {"binoculars": [{"item": str(r.item), "score": float(r.binoculars), "text": txt.get(str(r.item), "")[:600]}
                          for r in p0.nsmallest(10, "binoculars").itertuples()],
           "fdg": [{"item": str(r.item), "score": float(r.fdg_crit), "text": txt.get(str(r.item), "")[:600]}
                   for r in p0.nlargest(10, "fdg_crit").itertuples()]}
    json.dump(top, open(os.path.join(out_dir, "top_records.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote", out_dir, json.dumps({k: summ[k]["share_by_period"] for k in rules}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest":
        selftest()
    else:
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        selftest()
        main(a[0], a[1], a[2], lim)
