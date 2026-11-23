#!/usr/bin/env python3
"""Session 9: a Texas market price for each of our vehicles (PI, 2026-10-28, on the coverage counts: "same price of the
half of listings. That's good. Let's grab them. Grab them, save them."). For each listing with a year, make and model:
the Texas DMV sale prices of the same make, model and model year in the months around the listing's start month.

Reads, read-only: the listing panel and raw_data/dmv_texas/Texas DMV Transactions/<year>/<month>.csv, loading only
TRANSDATE, VIN, SALES_PRICE, ODOMETER_READING and the NHTSA make, model and model year (pandas usecols: owner, address
and dealer columns are never loaded). Keys, months and the alias are those of dmv_coverage.py (imported). A sale is
dated by TRANSDATE (SOLD_DATE is empty in these files). Dropped: prices below FLOOR (nominal transfers such as $1 or
$10) or above CAP (entry errors), and every sale of one of our own VINs (it may be the auctioned vehicle's own resale,
which is an outcome, not a benchmark).
Writes OUT/benchmark.parquet, one row per listing with a key and a start date: for windows of 3 and 6 months around the
start month, exact key and alias, the number of sales, the 25th, 50th and 75th percentile price and the median odometer
of the sales that report one; and OUT/summary.json: rows read, what the floor, the cap and the VIN rule dropped,
coverage, and whether the benchmark tracks our closing bids (correlation of log median and log closing bid among sold
vehicles with at least 10 sales, the median ratio, and the five listings furthest above and below).
Nothing else from the DMV files is written: no owner field, no other vehicle's VIN, no single sale.
Usage: python3 dmv_benchmark.py --selftest
       python3 dmv_benchmark.py PANEL_PARQUET DMV_DIR OUT_DIR [--workers N] [--files N]
"""
import glob
import json
import os
import sys
from datetime import date
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmv_coverage import COLS, TOOL_START, VALID, alias_of, key, month_index  # noqa: E402

FLOOR, CAP = 100.0, 2000000.0
WINDOWS = (3, 6)
STATS = ("n", "p25", "med", "p75", "odo")


def scan(job):
    """One month file -> its row counts and, for our keys (exact) and alias keys, {key: {month index: [[price, odometer],
    ...]}} of the sales kept."""
    path, keys, akeys, vins = job
    import pandas as pd
    d = pd.read_csv(path, usecols=COLS, dtype=str)
    n = len(d)
    price = pd.to_numeric(d["SALES_PRICE"], errors="coerce")
    odo = pd.to_numeric(d["ODOMETER_READING"], errors="coerce")
    when = pd.to_datetime(d["TRANSDATE"], format="%Y-%m-%d", errors="coerce")
    pos = price > 0
    low, high = pos & (price < FLOOR), pos & (price > CAP)
    mine = d["VIN"].fillna("").str.strip().str.upper().isin(vins)
    keep = pos & ~low & ~high & when.notna() & ~mine
    s = pd.DataFrame({"p": price[keep], "o": odo[keep], "m": (when.dt.year * 12 + when.dt.month - 1)[keep]})
    s["k"] = [key(a, b, c) for a, b, c in zip(d["nhtsa_make"][keep], d["nhtsa_model"][keep], d["nhtsa_model_year"][keep])]
    s = s[s["k"].notna()]
    s["a"] = [alias_of(x) for x in s["k"]]
    s["m"] = s["m"].astype(int)
    out = {}
    for tag, col, want in (("ex", "k", keys), ("al", "a", akeys)):
        g = s[s[col].isin(want)]
        out[tag] = {k: {int(m): gm[["p", "o"]].to_numpy(float) for m, gm in gk.groupby("m")} for k, gk in g.groupby(col)}
    counts = {"rows": n, "priced": int(pos.sum()), "below_floor": int(low.sum()), "above_cap": int(high.sum()),
              "our_vins_dropped": int((pos & mine).sum()), "kept_any_key": int(keep.sum())}
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path), counts, out


def merge(into, part):
    for k, ms in part.items():
        t = into.setdefault(k, {})
        for m, a in ms.items():
            t.setdefault(m, []).append(a)


def window(months, m0, w):
    """Sales within w calendar months of month m0 -> (n, p25, median, p75, median odometer of those that report one)."""
    arrs = [a for m, lst in months.items() if abs(m - m0) <= w for a in lst]
    if not arrs:
        return 0, None, None, None, None
    a = np.concatenate(arrs)
    q = np.percentile(a[:, 0], [25, 50, 75])
    o = a[:, 1][a[:, 1] > 0]
    return int(len(a)), float(q[0]), float(q[1]), float(q[2]), (float(np.median(o)) if len(o) else None)


def bench(listings, ex, al):
    """listings: [(item, key, start date)] -> rows of benchmark columns."""
    rows = []
    for item, k, start in listings:
        if k is None or start is None:
            continue
        m0 = month_index(start)
        r = {"item": item, "key": k, "alias_key": alias_of(k), "start_month": "%04d-%02d" % (start.year, start.month)}
        for tag, src, kk in (("ex", ex, k), ("al", al, alias_of(k))):
            for w in WINDOWS:
                for name, v in zip(STATS, window(src.get(kk, {}), m0, w)):
                    r["%s_%s%d" % (name, tag, w)] = v
        rows.append(r)
    return rows


def selftest():
    import tempfile
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        for y in ("2024",):
            os.makedirs(os.path.join(td, y))
        own = "1FTFW1EF5EKE12345"
        june = pd.DataFrame({
            "TRANSDATE": ["2024-06-03"] * 7 + ["bad"], "TRANSCD": ["TITLE"] * 8,
            "VIN": ["A" * 17, "B" * 17, "C" * 17, "D" * 17, own, "E" * 17, "F" * 17, "G" * 17],
            "OWNER_NAME1": ["A PERSON"] * 8, "OWNER_ADDRESS1": ["1 MAIN ST"] * 8,
            "SALES_PRICE": ["10000", "20000", "30000", "1", "99999", "3000000", "8000", "5000"], "SOLD_DATE": [None] * 8,
            "ODOMETER_READING": ["50000", "0", "70000", None, "1", "1", "120000", "1"],
            "nhtsa_make": ["FORD"] * 8, "nhtsa_model": ["F-150"] * 6 + ["F-350", "F-150"],
            "nhtsa_model_year": ["2017"] * 6 + ["2008", "2017"]})
        sept = pd.DataFrame({"TRANSDATE": ["2024-09-10"], "TRANSCD": ["TITLE"], "VIN": ["H" * 17], "OWNER_NAME1": ["B"],
                             "OWNER_ADDRESS1": ["2 MAIN ST"], "SALES_PRICE": ["40000"], "SOLD_DATE": [None],
                             "ODOMETER_READING": ["30000"], "nhtsa_make": ["FORD"], "nhtsa_model": ["F150"],
                             "nhtsa_model_year": ["2017"]})
        june.to_csv(os.path.join(td, "2024", "06.csv"), index=False)
        sept.to_csv(os.path.join(td, "2024", "09.csv"), index=False)
        keys = {"ford|f150|2017", "ford|f350sd|2008"}
        akeys = {alias_of(k) for k in keys}
        ex, al, tot = {}, {}, {}
        for f in sorted(glob.glob(os.path.join(td, "*", "*.csv"))):
            name, c, out = scan((f, keys, akeys, {own}))
            merge(ex, out["ex"])
            merge(al, out["al"])
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
    # the $1 transfer, the $3 million entry, our own VIN's sale and the bad date are dropped
    assert tot == {"rows": 9, "priced": 9, "below_floor": 1, "above_cap": 1, "our_vins_dropped": 1, "kept_any_key": 5}, tot
    L = [("a", "ford|f150|2017", date(2024, 9, 15)),     # June and September inside 3 months
         ("b", "ford|f350sd|2008", date(2024, 12, 1)),   # alias only; June is 6 months before: inside 6 only
         ("c", "ford|f150|2017", date(2025, 3, 1)),      # September 6 months before: inside 6 only; June 9: outside
         ("d", "ford|f150|2017", date(2024, 2, 1)),      # June 4 months after: inside 6 only; September 7: outside
         ("e", None, date(2024, 6, 1))]
    R = {r["item"]: r for r in bench(L, ex, al)}
    a, b, c, d_ = R["a"], R["b"], R["c"], R["d"]
    assert "e" not in R
    assert (a["n_ex3"], a["med_ex3"], a["p25_ex3"], a["p75_ex3"]) == (4, 25000.0, 17500.0, 32500.0), a
    assert a["odo_ex3"] == 50000.0, a                          # odometer 0 does not count; 30,000, 50,000, 70,000
    assert (b["n_ex6"], b["n_al3"], b["n_al6"], b["med_al6"]) == (0, 0, 1, 8000.0), b
    assert (c["n_ex3"], c["n_ex6"], c["med_ex6"]) == (0, 1, 40000.0), c
    assert (d_["n_ex3"], d_["n_ex6"], d_["med_ex6"]) == (0, 3, 20000.0), d_
    print("selftest: PASS")


def main(argv):
    panel, dmv_dir, out = argv[:3]
    opt = dict(zip(argv[3::2], argv[4::2]))
    workers, nfiles = int(opt.get("--workers", 8)), int(opt.get("--files", 0))
    import pandas as pd
    P = pd.read_parquet(panel, columns=["item", "f_vin", "f_year", "f_make", "f_model", "start_local_dt", "n_bids", "final_bid", "title"])
    P["start"] = pd.to_datetime(P["start_local_dt"], errors="coerce").dt.date
    L, vins = [], set()
    for item, vin, yr, mk, md, s in P[["item", "f_vin", "f_year", "f_make", "f_model", "start"]].itertuples(index=False):
        v = ("" if vin is None else str(vin)).strip().upper()
        if VALID.fullmatch(v):
            vins.add(v)
        L.append((str(item), key(mk, md, yr), (s if s == s else None)))
    keys = {k for _, k, _ in L if k}
    akeys = {alias_of(k) for k in keys}
    files = sorted(glob.glob(os.path.join(dmv_dir, "*", "*.csv")))[: (nfiles or None)]
    print("our keys %d (alias %d), our VINs %d, DMV files %d" % (len(keys), len(akeys), len(vins), len(files)))
    ex, al, tot, per_file = {}, {}, {}, []
    with Pool(workers) as pool:
        for name, c, part in pool.imap_unordered(scan, [(f, keys, akeys, vins) for f in files]):
            merge(ex, part["ex"])
            merge(al, part["al"])
            per_file.append(dict(c, file=name))
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
            print("  %s: %d rows, %d kept" % (name, c["rows"], c["kept_any_key"]))
    B = pd.DataFrame(bench(L, ex, al))
    os.makedirs(out, exist_ok=True)
    B.to_parquet(os.path.join(out, "benchmark.parquet"), index=False)
    # does the benchmark track what our vehicles sold for? (sold = at least one bid; a zero-bid page echoes the ask)
    J = B.merge(P.assign(item=P["item"].astype(str))[["item", "n_bids", "final_bid", "title", "start"]], on="item", how="left")
    J["tool"] = J["start"] >= TOOL_START
    sold = J[(J["n_bids"] >= 1) & (J["final_bid"] > 0) & (J["n_ex3"] >= 10)].copy()
    sold["gap"] = np.log(sold["final_bid"]) - np.log(sold["med_ex3"])
    sold = sold.sort_values("gap")
    show = lambda x: [{"item": r.item, "title": r.title, "start": str(r.start), "final_bid": float(r.final_bid),
                       "median_tx": r.med_ex3, "n_tx": int(r.n_ex3)} for r in x.itertuples()]
    summ = {"generated_by": "code/dmv_benchmark.py", "floor": FLOOR, "cap": CAP, "files_read": len(files), "totals": tot,
            "listings_with_key_and_start": int(len(B)), "tool_era_listings": int(J["tool"].sum())}
    for tag in ("ex", "al"):
        for w in WINDOWS:
            for t in (1, 10, 30):
                summ["%s%d_at_least_%d" % (tag, w, t)] = int((B["n_%s%d" % (tag, w)] >= t).sum())
                summ["tool_era_%s%d_at_least_%d" % (tag, w, t)] = int(((J["n_%s%d" % (tag, w)] >= t) & J["tool"]).sum())
    summ["check_sold_with_10_sales_3m"] = {
        "n": int(len(sold)), "n_tool_era": int(sold["tool"].sum()),
        "corr_log_final_bid_log_median": float(np.corrcoef(np.log(sold["final_bid"]), np.log(sold["med_ex3"]))[0, 1]) if len(sold) > 2 else None,
        "median_ratio_final_bid_to_median": float(np.exp(sold["gap"].median())) if len(sold) else None,
        "median_ratio_tool_era": float(np.exp(sold.loc[sold["tool"], "gap"].median())) if sold["tool"].any() else None,
        "furthest_below": show(sold.head(5)), "furthest_above": show(sold.tail(5))}
    summ["files"] = sorted(per_file, key=lambda x: x["file"])
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "files"}, indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        selftest()
        main(sys.argv[1:])
