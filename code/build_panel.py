#!/usr/bin/env python3
"""Session 8: listing-level panel from the s8p2 parse.

One row per listing page. Nothing is imputed: a value that cannot be read is NaN and its
`*_status` column says why. Text columns are kept verbatim (base description, seller updates).

Usage: python3 build_panel.py PAGES.jsonl OUT_DIR
Writes OUT_DIR/panel.parquet and OUT_DIR/panel_report.txt (coverage of every column).
"""
import json
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}


def yn(v):
    if v is None:
        return np.nan
    t = v.strip().lower().rstrip(".")
    if t in YES or t.startswith("yes"):
        return 1.0
    if t in NO or t.startswith("no"):
        return 0.0
    return np.nan


def num(v):
    if v is None:
        return np.nan
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", v)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def qa_time(s):
    if not s:
        return None
    for fmt in ("%m/%d/%Y %I:%M %p", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def main(pages, out_dir):
    rows = []
    for line in open(pages):
        r = json.loads(line)
        if r.get("page_kind") != "listing":
            continue
        st = r.get("status") or {}
        f = r.get("fields") or {}
        cats = r.get("categories") or r.get("categories_meta") or []
        end = r.get("end_local")
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S") if end else None
        qa = r.get("qa") or []
        q_before = 0
        q_after = 0
        q_untimed = 0
        first_q = None
        for q in qa:
            t = qa_time(q.get("q_time"))
            if t is None or end_dt is None:
                q_untimed += 1
                continue
            if t <= end_dt:
                q_before += 1
            else:
                q_after += 1
            first_q = t if first_q is None or t < first_q else first_q
        upd = r.get("desc_updates") or []
        cap = r.get("capture_ts")
        cap_dt = datetime.strptime(cap[:14], "%Y%m%d%H%M%S") if cap and len(cap) >= 14 else None
        row = {
            "item": r["item"],
            "layout": r.get("layout"),
            "auction_state": r.get("auction_state"),
            "capture_ts": cap,
            "capture_lag_days": ((cap_dt - end_dt).total_seconds() / 86400.0) if (cap_dt and end_dt) else np.nan,
            "start_local": r.get("start_local"),
            "end_local": end,
            "agency": (r.get("agency_slug") or "").lower() or None,
            "agency_name": r.get("agency_name") or r.get("agency_name_data_div"),
            "state": r.get("loc_state"),
            "country": r.get("loc_country"),
            "cat1": cats[0] if len(cats) > 0 else None,
            "cat2": cats[1] if len(cats) > 1 else None,
            "cat3": cats[2] if len(cats) > 2 else None,
            "title": r.get("title"),
            "final_bid": r.get("final_bid"),
            "final_bid_status": st.get("final_bid"),
            "starting_bid": r.get("starting_bid"),
            "current_price": r.get("current_price"),
            "buy_now_price": r.get("buy_now_price"),
            "buy_now_shown": r.get("buy_now_shown"),
            "reserve": r.get("reserve"),
            "buyer_fee_pct": r.get("buyer_fee_pct"),
            "n_bids": r.get("n_bids"),
            "extended": r.get("extended"),
            "page_views": r.get("page_views"),
            "high_bidder_id": r.get("high_bidder_id"),
            "high_bidder_is_email": r.get("high_bidder_is_email"),
            "n_photos": r.get("n_photos"),
            "n_photos_status": st.get("n_photos"),
            "n_questions": r.get("n_questions"),
            "n_answered": r.get("n_answered"),
            "n_askers": r.get("n_askers"),
            "n_q_before_end": q_before,
            "n_q_after_end": q_after,
            "n_q_untimed": q_untimed,
            "first_q_time": first_q.strftime("%Y-%m-%dT%H:%M:%S") if first_q else None,
            "qa_status": st.get("qa"),
            "desc": r.get("desc"),
            "desc_status": st.get("desc"),
            "n_updates": len(upd),
            "updates_text": "\n".join(u.get("text") or "" for u in upd) if upd else None,
            "first_update_added": upd[0].get("added") if upd else None,
            "payment": r.get("payment"),
            "terms": r.get("terms"),
            "og_description": r.get("og_description"),
            "markup_p": (r.get("desc_markup") or {}).get("p"),
            "markup_br": (r.get("desc_markup") or {}).get("br"),
            "markup_bold": (r.get("desc_markup") or {}).get("bold"),
            "markup_li": (r.get("desc_markup") or {}).get("li"),
            "markup_h": (r.get("desc_markup") or {}).get("h"),
            "markup_hr": (r.get("desc_markup") or {}).get("hr"),
            # structured table: vehicles / equipment
            "f_year": num(f.get("Year")),
            "f_make": (f.get("Make") or "").strip() or None,
            "f_model": (f.get("Model") or "").strip() or None,
            "f_trim": (f.get("Trim Level") or "").strip() or None,
            "f_body": (f.get("Body Style") or "").strip() or None,
            "f_engine": (f.get("Engine Type") or "").strip() or None,
            "f_driveline": (f.get("Driveline") or "").strip() or None,
            "f_miles": num(f.get("Miles")),
            "f_hours": num(f.get("Number of Hours")),
            "f_vin": (f.get("VIN") or "").strip() or None,
            "f_starts": yn(f.get("Does it start")),
            "f_drivable": yn(f.get("Is it drivable off the lot")),
            "f_inspected": yn(f.get("Is it currently inspected")),
            "f_title": yn(f.get("Vehicle has a transferable title")),
            "f_voted": yn(f.get("Will Item results be voted on")),
            "f_ship": yn(f.get("Will you ship this item")),
            "f_custom_id": (f.get("Custom ID") or "").strip() or None,
            "f_inspection_info": (f.get("In person item inspection information") or "").strip() or None,
            "fields_status": st.get("fields"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("start_local", "end_local", "first_q_time"):
        df[c + "_dt"] = pd.to_datetime(df[c], errors="coerce")
    df["end_ym"] = df["end_local_dt"].dt.to_period("M").astype(str)
    df["start_year"] = df["start_local_dt"].dt.year
    df["end_year"] = df["end_local_dt"].dt.year
    df.to_parquet(out_dir + "/panel.parquet", index=False)
    with open(out_dir + "/panel_report.txt", "w") as fo:
        fo.write("rows %d\n" % len(df))
        for c in df.columns:
            nn = df[c].notna().sum()
            fo.write("%-22s nonnull=%6d (%.1f%%)\n" % (c, nn, 100.0 * nn / max(1, len(df))))
        for c in ("auction_state", "final_bid_status", "qa_status", "desc_status", "fields_status", "reserve", "layout", "n_photos_status"):
            fo.write("\n%s %s\n" % (c, df[c].value_counts(dropna=False).to_dict()))
    print("panel rows", len(df), "->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
