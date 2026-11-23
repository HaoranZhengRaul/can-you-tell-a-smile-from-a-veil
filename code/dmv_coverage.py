#!/usr/bin/env python3
"""Session 9: how many of our vehicles have Texas DMV sale prices for the same make, model and model year near the
listing's start? Counts only (PI, 2026-10-28: "read only the vehicle and price columns, save counts only, no owner
fields").

Reads, read-only: the listing panel (item, f_vin, f_year, f_make, f_model, start date) and
raw_data/dmv_texas/Texas DMV Transactions/<year>/<month>.csv, loading only TRANSDATE, VIN, SALES_PRICE,
ODOMETER_READING, nhtsa_make, nhtsa_model, nhtsa_model_year (pandas usecols: the owner, address and dealer columns
are never loaded into memory). SOLD_DATE is empty in these files, so a sale is dated by TRANSDATE.
Keys: make|model|model year, letters and digits only; an upper-bound alias also drops a trailing "sd"/"superduty".
Writes OUT/coverage.json: per file the rows read and priced; for our listings the number with at least 1, 10 and 30
priced Texas sales of the same key within 3 and 6 calendar months of the listing's start month (exact and alias), the
same for tool-era listings, and how many of our VINs appear in the transactions before or after the listing started.
Usage: python3 dmv_coverage.py --selftest
       python3 dmv_coverage.py PANEL_PARQUET DMV_DIR OUT_DIR [--workers N] [--files N]
"""
import glob
import json
import os
import re
import sys
from datetime import date
from multiprocessing import Pool

COLS = ["TRANSDATE", "VIN", "SALES_PRICE", "ODOMETER_READING", "nhtsa_make", "nhtsa_model", "nhtsa_model_year"]
TOOL_START = date(2025, 2, 1)
VALID = re.compile(r"[A-HJ-NPR-Z0-9]{17}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", ("" if s is None else str(s)).lower())


def drop_sd(model):
    return re.sub(r"(superduty|sd)$", "", model)


def key(make, model, year):
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return None
    mk, md = norm(make), norm(model)
    return None if not mk or not md or y < 1900 else "%s|%s|%d" % (mk, md, y)


def alias_of(k):
    a = k.split("|")
    return "%s|%s|%s" % (a[0], drop_sd(a[1]), a[2])


def month_index(d):
    return d.year * 12 + d.month - 1


def scan(job):
    """One month file -> rows read, rows priced, {key: {month index: n}} (exact and alias) for our keys, and
    {our VIN: [month indexes]}."""
    path, keys, akeys, vins = job
    import pandas as pd
    d = pd.read_csv(path, usecols=COLS, dtype=str)
    n = len(d)
    price = pd.to_numeric(d["SALES_PRICE"], errors="coerce")
    d = d[price > 0]
    when = pd.to_datetime(d["TRANSDATE"], format="%Y-%m-%d", errors="coerce")
    d = d.assign(m=when.dt.year * 12 + when.dt.month - 1)[when.notna()]
    k = [key(a, b, c) for a, b, c in zip(d["nhtsa_make"], d["nhtsa_model"], d["nhtsa_model_year"])]
    ex, al = {}, {}
    for kk, m in zip(k, d["m"].astype(int)):
        if kk is None:
            continue
        if kk in keys:
            ex.setdefault(kk, {}).setdefault(m, 0)
            ex[kk][m] += 1
        ak = alias_of(kk)
        if ak in akeys:
            al.setdefault(ak, {}).setdefault(m, 0)
            al[ak][m] += 1
    vv = {}
    for v, m in zip(d["VIN"].fillna("").str.strip().str.upper(), d["m"].astype(int)):
        if v in vins:
            vv.setdefault(v, []).append(int(m))
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path), n, int(len(d)), ex, al, vv


def merge(into, part):
    for k, ms in part.items():
        t = into.setdefault(k, {})
        for m, c in ms.items():
            t[m] = t.get(m, 0) + c


def near_count(months, m0, w):
    return sum(c for m, c in months.items() if abs(m - m0) <= w)


def coverage(listings, ex, al, vin_months):
    """listings: [(item, key, start date, vin)] -> counts."""
    c = {"listings_with_year_make_model": 0, "tool_era_listings": 0}
    for tag in ("exact", "alias"):
        for w in (3, 6):
            for t in (1, 10, 30):
                c["%s_within_%dm_at_least_%d" % (tag, w, t)] = 0
                c["tool_era_%s_within_%dm_at_least_%d" % (tag, w, t)] = 0
    c.update({"our_vins": 0, "vins_in_dmv": 0, "vins_with_sale_before_start": 0, "vins_with_sale_after_start": 0})
    for item, k, start, vin in listings:
        if vin:
            c["our_vins"] += 1
            ms = vin_months.get(vin, [])
            if ms:
                c["vins_in_dmv"] += 1
                if start is not None:
                    m0 = month_index(start)
                    c["vins_with_sale_before_start"] += any(m < m0 for m in ms)
                    c["vins_with_sale_after_start"] += any(m >= m0 for m in ms)
        if k is None or start is None:
            continue
        c["listings_with_year_make_model"] += 1
        tool = start >= TOOL_START
        c["tool_era_listings"] += tool
        m0 = month_index(start)
        for tag, months in (("exact", ex.get(k, {})), ("alias", al.get(alias_of(k), {}))):
            for w in (3, 6):
                n = near_count(months, m0, w)
                for t in (1, 10, 30):
                    if n >= t:
                        c["%s_within_%dm_at_least_%d" % (tag, w, t)] += 1
                        c["tool_era_%s_within_%dm_at_least_%d" % (tag, w, t)] += tool
    return c


def selftest():
    import tempfile
    import pandas as pd
    assert key("FORD", "F-150", "2017") == "ford|f150|2017" and alias_of("ford|f350sd|2008") == "ford|f350|2008"
    assert month_index(date(2025, 1, 31)) - month_index(date(2024, 12, 1)) == 1
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "2024"))
        p = os.path.join(td, "2024", "06.csv")
        pd.DataFrame({"TRANSDATE": ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "bad"],
                      "TRANSCD": ["TITLE"] * 5, "VIN": ["1FTFW1EF5EKE12345", None, "2FMDK3GC5ABA12345", "1FTFW1EF5EKE12345", "y"],
                      "OWNER_NAME1": ["A PERSON"] * 5, "OWNER_ADDRESS1": ["1 MAIN ST"] * 5,
                      "SALES_PRICE": ["25000", "0", "18000", "21000", "9000"], "SOLD_DATE": [None] * 5,
                      "ODOMETER_READING": [None] * 5, "nhtsa_make": ["FORD", "FORD", "FORD", "FORD", "FORD"],
                      "nhtsa_model": ["F-150", "F-150", "F-350", "Explorer", "F-150"],
                      "nhtsa_model_year": ["2017", "2017", "2008", "2017", "2017"]}).to_csv(p, index=False)
        name, n, priced, ex, al, vv = scan((p, {"ford|f150|2017"}, {"ford|f350|2008", "ford|f150|2017"},
                                            {"1FTFW1EF5EKE12345", "2FMDK3GC5ABA12345"}))
    m = month_index(date(2024, 6, 1))
    assert name == "2024/06.csv" and n == 5 and priced == 3, (n, priced)            # price 0 and the bad date drop out
    assert ex == {"ford|f150|2017": {m: 1}} and al == {"ford|f150|2017": {m: 1}, "ford|f350|2008": {m: 1}}, (ex, al)
    assert vv == {"1FTFW1EF5EKE12345": [m, m], "2FMDK3GC5ABA12345": [m]}, vv
    L = [("a", "ford|f150|2017", date(2024, 9, 1), None),                       # 3 months after the sale: inside 3 m
         ("b", "ford|f350sd|2008", date(2024, 12, 15), None),                   # alias only; 6 months after: inside 6 m only
         ("c", "ford|f150|2017", date(2025, 3, 1), "1FTFW1EF5EKE12345"),        # 9 months after: outside; VIN sold before
         ("d", "ford|f150|2017", date(2024, 2, 1), None),                      # 4 months before the sale: inside 6 m only
         ("e", "ford|f350sd|2008", date(2024, 6, 15), "2FMDK3GC5ABA12345")]     # VIN sold in its start month: not "before"
    c = coverage(L, ex, al, vv)
    assert c["exact_within_3m_at_least_1"] == 1 and c["exact_within_6m_at_least_1"] == 2, c
    assert c["alias_within_3m_at_least_1"] == 2 and c["alias_within_6m_at_least_1"] == 4, c
    assert c["exact_within_6m_at_least_10"] == 0 and c["tool_era_listings"] == 1 and c["tool_era_exact_within_6m_at_least_1"] == 0, c
    assert c["our_vins"] == 2 and c["vins_in_dmv"] == 2 and c["vins_with_sale_before_start"] == 1 and c["vins_with_sale_after_start"] == 1, c
    print("selftest: PASS")


def main(argv):
    panel, dmv_dir, out = argv[:3]
    opt = dict(zip(argv[3::2], argv[4::2]))
    workers, nfiles = int(opt.get("--workers", 8)), int(opt.get("--files", 0))
    import pandas as pd
    P = pd.read_parquet(panel, columns=["item", "f_vin", "f_year", "f_make", "f_model", "start_local_dt"])
    P["start"] = pd.to_datetime(P["start_local_dt"], errors="coerce").dt.date
    L = []
    for item, vin, yr, mk, md, s in P[["item", "f_vin", "f_year", "f_make", "f_model", "start"]].itertuples(index=False):
        v = ("" if vin is None else str(vin)).strip().upper()
        L.append((str(item), key(mk, md, yr), (s if s == s else None), v if VALID.fullmatch(v) else None))
    keys = {k for _, k, _, _ in L if k}
    akeys = {alias_of(k) for k in keys}
    vins = {v for _, _, _, v in L if v}
    files = sorted(glob.glob(os.path.join(dmv_dir, "*", "*.csv")))[: (nfiles or None)]
    print("our keys %d (alias %d), our VINs %d, DMV files %d" % (len(keys), len(akeys), len(vins), len(files)))
    ex, al, vm, per_file = {}, {}, {}, []
    with Pool(workers) as pool:
        for name, n, priced, e, a, v in pool.imap_unordered(scan, [(f, keys, akeys, vins) for f in files]):
            per_file.append({"file": name, "rows": n, "priced": priced})
            merge(ex, e)
            merge(al, a)
            for k, ms in v.items():
                vm.setdefault(k, []).extend(ms)
            print("  %s: %d rows, %d priced" % (name, n, priced))
    c = coverage(L, ex, al, vm)
    c.update({"generated_by": "code/dmv_coverage.py", "files": sorted(per_file, key=lambda x: x["file"]),
              "rows_read": sum(x["rows"] for x in per_file), "rows_priced": sum(x["priced"] for x in per_file),
              "keys_with_any_sale": len(ex), "our_keys": len(keys)})
    os.makedirs(out, exist_ok=True)
    json.dump(c, open(os.path.join(out, "coverage.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in c.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        selftest()
        main(sys.argv[1:])
