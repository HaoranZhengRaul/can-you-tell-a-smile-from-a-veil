#!/usr/bin/env python3
"""Session 9: by capture month, how many archived listing pages carry the code of Municibid's AI writing assistant, for
the small line graph of the button (PI, 2026-10-28: "a line graph ... a very small one", on the months of Figure 1).
A page carries it when its bundled seller-editor code holds the assistant's description box, `improvedDescriptionText`
(first seen in January 2025); `aiDescriptionContent` marks the later revision (first seen in January 2026). The page
code is the platform's code on the capture day, so pages are counted by capture month, never by listing start.
Reads, read-only: the listing panel (item, capture_ts) and each banked page (analysis/muni_data/pages/<first 3
digits>/<item>.html.gz). Writes OUT_JSON: per capture month the listing pages, the pages with each marker and the
pages missing from the store (never counted as pages without the box); the first page with the box and the last page
captured before it; the pages without the box captured from February 2025 on (items, to read).
Usage: python3 button_dating.py --selftest
       python3 button_dating.py PANEL_PARQUET PAGES_DIR OUT_JSON [--workers N]
"""
import gzip
import json
import os
import sys
from multiprocessing import Pool

BOX, LATER = "improvedDescriptionText", "aiDescriptionContent"
FULL_FROM = "2025-02"


def look(job):
    """(item, capture timestamp, pages dir) -> (item, timestamp, status, has box, has later revision)."""
    item, ts, pages = job
    path = os.path.join(pages, item[:3], item + ".html.gz")
    if not os.path.exists(path):
        return item, ts, "missing", None, None
    try:
        t = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    except (OSError, EOFError):
        return item, ts, "unreadable", None, None
    return item, ts, "ok", BOX in t, LATER in t


def tally(results):
    """[(item, ts, status, box, later)] -> per-month counts, first page with the box, last page before it, and pages
    without the box from FULL_FROM on."""
    months, firsts, gaps = {}, [], []
    for item, ts, st, box, later in results:
        if not ts or len(ts) < 6:
            continue
        m = ts[:4] + "-" + ts[4:6]
        c = months.setdefault(m, {"pages": 0, "missing": 0, "box": 0, "later": 0})
        if st != "ok":
            c["missing"] += 1
            continue
        c["pages"] += 1
        c["box"] += int(box)
        c["later"] += int(later)
        firsts.append((ts, item, box))
        if m >= FULL_FROM and not box:
            gaps.append({"item": item, "capture_ts": ts})
    firsts.sort()
    first = next(((ts, it) for ts, it, b in firsts if b), None)
    before = max(((ts, it) for ts, it, b in firsts if first and ts < first[0]), default=None)
    return {"months": dict(sorted(months.items())),
            "first_with_box": {"capture_ts": first[0], "item": first[1]} if first else None,
            "last_before_first": {"capture_ts": before[0], "item": before[1]} if before else None,
            "without_box_from_%s" % FULL_FROM: sorted(gaps, key=lambda g: g["capture_ts"])}


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # every real page says "Item Description"; a decoy id that only starts like the box must not count
        pages = {"61000000": "<h1>Item Description</h1><script>var improvedDescriptionTitle;</script>",
                 "71559451": "<h1>Item Description</h1> editor code, no box",
                 "69668899": "<h1>Item Description</h1><script>const aiDescription = $('#improvedDescriptionText')</script>",
                 "70000001": "<h1>Item Description</h1><script>$('#improvedDescriptionText'); var aiDescriptionContent = 1;</script>",
                 "70000004": "<h1>Item Description</h1> a page from the first full month whose code lacks the box",
                 "70000002": "<h1>Item Description</h1> a later page whose code lacks the box"}
        for it, body in pages.items():
            os.makedirs(os.path.join(td, it[:3]), exist_ok=True)
            with gzip.open(os.path.join(td, it[:3], it + ".html.gz"), "wt", encoding="utf-8") as f:
                f.write(body)
        jobs = [("61000000", "20241215100000", td), ("71559451", "20250128235059", td), ("69668899", "20250129184626", td),
                ("70000001", "20260131120000", td), ("70000004", "20250203090000", td), ("70000002", "20250310080000", td),
                ("79999999", "20250311080000", td), ("70000003", None, td)]
        R = tally([look(j) for j in jobs])
    M = R["months"]
    assert M["2025-01"] == {"pages": 2, "missing": 0, "box": 1, "later": 0}, M
    assert M["2025-03"] == {"pages": 1, "missing": 1, "box": 0, "later": 0}, M      # the missing page is not a "no box" page
    assert M["2026-01"] == {"pages": 1, "missing": 0, "box": 1, "later": 1}, M
    assert R["first_with_box"] == {"capture_ts": "20250129184626", "item": "69668899"}, R
    assert R["last_before_first"] == {"capture_ts": "20250128235059", "item": "71559451"}, R
    assert M["2024-12"] == {"pages": 1, "missing": 0, "box": 0, "later": 0}, M      # the decoy id is not the box
    assert R["without_box_from_2025-02"] == [{"item": "70000004", "capture_ts": "20250203090000"},
                                             {"item": "70000002", "capture_ts": "20250310080000"}], R
    print("selftest: PASS")


def main(argv):
    panel, pages, out = argv[:3]
    opt = dict(zip(argv[3::2], argv[4::2]))
    import pandas as pd
    P = pd.read_parquet(panel, columns=["item", "capture_ts"])
    jobs = [(str(i), (None if ts is None or ts != ts else str(ts)), pages) for i, ts in P.itertuples(index=False)]
    print("listing pages in the panel: %d; with a capture time: %d" % (len(jobs), sum(1 for j in jobs if j[1])))
    with Pool(int(opt.get("--workers", 8))) as pool:
        res = list(pool.imap_unordered(look, jobs, chunksize=64))
    R = tally(res)
    R.update({"generated_by": "code/button_dating.py", "markers": {"box": BOX, "later": LATER},
              "panel_pages": len(jobs), "status": {s: sum(1 for r in res if r[2] == s) for s in ("ok", "missing", "unreadable")}})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(R, open(out, "w"), indent=1)
    for m, c in R["months"].items():
        if m >= "2024-06":
            print("  %s pages %4d  box %4d  later %4d  missing %d" % (m, c["pages"], c["box"], c["later"], c["missing"]))
    print("first with box:", R["first_with_box"], "| last before:", R["last_before_first"], "| status:", R["status"],
          "| without box from 2025-02:", len(R["without_box_from_2025-02"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        selftest()
        main(sys.argv[1:])
