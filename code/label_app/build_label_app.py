#!/usr/bin/env python3
"""Session 9: build the blind labelling app (one self-contained HTML file) for the human blind read.

Input: the 100 blind-read texts of Session 8 (Mercury results/blind_s8/texts.jsonl: one JSON object per line with
"id" (B000..B099), "title" and "description"; sha256 73e9a2c8...). The builder never reads the key. It
  1. scrubs contact details (e-mail addresses, phone numbers with separators, web links, a name after "contact",
     "call", "ask for" and similar), and writes every change, plus strings that may be phone numbers but were left
     alone, to a private report that a person reads before the app goes out;
  2. gives each listing a new random id, so nothing in the hosted file links back to the blind ids;
  3. adds two attention checks that say what to answer;
  4. writes OUT/index.html (the file to host) and PRIVATE/ (id map, scrub report, manifest). PRIVATE must not be
     hosted or committed; OUT holds listing text, so it stays out of git too (put it under output/ or the scratchpad).

Usage:
  python3 build_label_app.py --texts TEXTS.jsonl --out OUT_DIR --private PRIVATE_DIR --consent CONSENT.html
        [--submit-url URL] [--completion-url URL] [--completion-code CODE] [--expect-sha256 HEX] [--seed N]
  python3 build_label_app.py --demo --out OUT_DIR --private PRIVATE_DIR     (six invented listings, for testing)
  python3 build_label_app.py --selftest
  python3 build_label_app.py --pairs PAIRS.json --out OUT_DIR --tag TAG

Session 12 (PI, 2026-11-18: a blind check of 120 "appealing" rewrite pairs by the PI): --pairs builds a second page,
OUT_DIR/index.html, from the PI's sample written by `code/rewrite_audit.py --batches` (pid, title, category, original,
rewrite; no model, no instruction, no key). Each pair shows the original beside the rewrite with three questions in
the coders' own terms (a claim about the item the original does not support, with the words; a fact dropped or
changed; a stated problem softened). Answers stay in the browser and download as one JSON file; contact details are
scrubbed as above. The page holds listing text, so OUT_DIR goes under output/ (ignored by git) and is never hosted.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from datetime import datetime, timezone

APP_VERSION = "label-app-1"
HERE = os.path.dirname(os.path.abspath(__file__))

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+", re.I)
# ten-digit numbers written with a separator or parentheses (a bare ten-digit run may be a serial number: review only)
PHONE = re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}"
                   r"(?:\s*(?:x|ext\.?|extension)\s*\d{1,5})?(?![\w-])", re.I)
REVIEW = [("bare ten-digit number", re.compile(r"(?<![\w-])\d{10}(?![\w-])")),
          ("seven-digit number with a dash", re.compile(r"(?<![\w-])\d{3}-\d{4}(?![\w-])"))]
CUE = re.compile(r"\b(?:[Cc]ontact|[Cc]all|[Aa]sk\s+for|[Ee]-?mail|[Tt]ext|[Rr]each\s+out\s+to|[Rr]each)\s*:?\s+"
                 r"((?:(?:Mr|Mrs|Ms|Dr|Sgt|Officer|Lt|Capt|Det|Cpl|Deputy|Chief)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.)?"
                 r"(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?)")
# 2. a name written right before a contact detail, with no cue word ("Derek Platt at [email]", "Don Kelly ([email])")
BEFORE_CONTACT = re.compile(r"\b((?:(?:Mr|Mrs|Ms|Dr|Sgt|Officer|Lt|Capt|Det|Cpl|Deputy|Chief)\.?\s+)?[A-Z][a-z]+"
                            r"(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?)"
                            r"(?=\s*(?:at|@|-|:|,)?\s*\(?\[(?:email|phone)\]\)?)")
# a capitalized word after a cue that names an office, a time or a role, not a person
NOT_A_NAME = set("""Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July
August September October November December Mon Tue Wed Thu Fri Sat Sun The This That Our Your Us Me Them For City
County Town Village Township State Public Department Dept Office Fleet Purchasing Procurement Facilities Facility
Transportation Police Fire School District Maintenance Customer Municibid Buyer Buyers Seller Bidder Bidders Staff
Shop Garage Yard Main Front Admin Administration Finance Utilities Utility Water Parks Recreation Sheriff Library
Housing Transit Airport Port Services Service Support Sales Auction Auctions Questions Inspection Pickup Removal Any
All If Before After Between During Please Additional More With Contact Call Email Text Phone Tel Telephone Cell Mobile
Fax Desk Dispatch Ext Number Or And At Reach Hours Main Direct""".split())


def scrub(text):
    """Return (scrubbed text, list of changes, list of review flags)."""
    if not text:
        return text, [], []
    changes = []

    def sub(rx, repl, s, kind):
        def f(m):
            changes.append({"kind": kind, "found": m.group(0)})
            return repl
        return rx.sub(f, s)
    s = sub(EMAIL, "[email]", text, "email")
    s = sub(URL, "[link]", s, "link")
    s = sub(PHONE, "[phone]", s, "phone")

    def name_f(m):
        name = m.group(1)
        first = re.sub(r"^(?:Mr|Mrs|Ms|Dr)\.?\s+", "", name).split()[0]
        if first in NOT_A_NAME:
            return m.group(0)
        changes.append({"kind": "name", "found": name})
        return m.group(0)[: m.start(1) - m.start(0)] + "[name]"
    s = CUE.sub(name_f, s)

    def before_f(m):
        name = m.group(1)
        first = re.sub(r"^(?:Mr|Mrs|Ms|Dr|Sgt|Officer|Lt|Capt|Det|Cpl|Deputy|Chief)\.?\s+", "", name).split()[0]
        if first in NOT_A_NAME or name in NOT_A_NAME:
            return m.group(0)
        changes.append({"kind": "name", "found": name})
        return "[name]"
    s = BEFORE_CONTACT.sub(before_f, s)
    review = [{"kind": k, "found": m.group(0)} for k, rx in REVIEW for m in rx.finditer(s)]
    return s, changes, review


def attention_checks():
    return [
        {"title": "Attention check",
         "description": "This is not a real listing. To show that you are reading carefully, please choose \"Not sure\" for this one.",
         "check": True, "expected": "U"},
        {"title": "Attention check",
         "description": "This is not a real listing. Please choose \"A person wrote it\" for this one.",
         "check": True, "expected": "H"},
    ]


def demo_texts():
    """Six invented listings (not from the data) to test the app."""
    t = [("2012 Chevrolet Silverado 2500", "Runs and drives. Check engine light on. 148,211 miles. Some rust on rockers. Sold as is."),
         ("Office chairs (lot of 12)", "Assorted office chairs from the library renovation. Some stains. Must remove within 5 business days."),
         ("2016 Ford F-150 XL", "This dependable pickup is ready for its next owner. Key Features:\n- Powerful V6 engine\n- Spacious cab\nSold as-is."),
         ("Snow plow blade", "Western 8 ft blade, came off truck 14. Hydraulics leak. Contact Jane Doe at 555-201-3344 to inspect."),
         ("Riding mower", "Starts with jump. Deck is bent. Call (217) 555-0142 ext 9 Monday through Friday."),
         ("School desks", "Durable and versatile student desks, ideal for any classroom or workshop. Visit www.example.org/desks for photos.")]
    return [{"id": "D%03d" % i, "title": a, "description": b} for i, (a, b) in enumerate(t)]


def read_texts(path, expect_sha=None):
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if expect_sha and not sha.startswith(expect_sha.lower()):
        raise SystemExit("texts sha256 %s does not start with the expected %s" % (sha, expect_sha))
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in the texts file"
    for r in rows:
        assert set(r) >= {"id", "title", "description"}, r.keys()
    return rows, sha


def build(rows, sha, out_dir, private_dir, consent_html, submit_url="", completion_url="", completion_code="",
          seed=None, n_pre=None, est_minutes=None, demo=False):
    rng = random.Random(seed if seed is not None else random.SystemRandom().randrange(1 << 62))
    items, idmap, report = [], [], []
    used = set()

    def new_id():
        while True:
            x = "L%06x" % rng.randrange(1 << 24)
            if x not in used:
                used.add(x)
                return x
    for r in rows:
        t, ch_t, rv_t = scrub(r.get("title") or "")
        d, ch_d, rv_d = scrub(r.get("description") or "")
        aid = new_id()
        items.append({"id": aid, "title": t, "description": d})
        idmap.append((aid, r["id"]))
        if ch_t or ch_d or rv_t or rv_d:
            report.append({"blind_id": r["id"], "app_id": aid, "changes": ch_t + ch_d, "review": rv_t + rv_d})
    for c in attention_checks():
        c = dict(c, id=new_id())
        items.append(c)
        idmap.append((c["id"], "CHECK_" + c["expected"]))
    rng.shuffle(items)  # the order inside the file says nothing about the blind order
    items_json = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    item_set_hash = hashlib.sha256(items_json.encode("utf-8")).hexdigest()[:12]
    n_real = len(rows)
    n_pre = (25 if n_real == 100 else n_real // 4) if n_pre is None else n_pre
    config = {"appVersion": APP_VERSION, "itemSetHash": item_set_hash, "salt": item_set_hash, "submitUrl": submit_url,
              "completionUrl": completion_url, "completionCode": completion_code, "consentHtml": consent_html,
              "nReal": n_real, "nPre": n_pre, "nPost": n_real - n_pre,
              "estMinutes": est_minutes if est_minutes is not None else max(5, round(n_real * 12 / 60 / 5) * 5)}

    def js(o):
        # JSON is valid JavaScript; escaping "</" keeps listing text from closing the script element
        s = json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    assert tpl.count("/*CONFIG_JSON*/{}") == 1 and tpl.count("/*ITEMS_JSON*/[]") == 1, "template placeholders missing"
    html = tpl.replace("/*CONFIG_JSON*/{}", js(config)).replace("/*ITEMS_JSON*/[]", js(items))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)
    assert os.path.realpath(private_dir) != os.path.realpath(out_dir), "PRIVATE must not be the hosted folder"
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(html)
    with open(os.path.join(private_dir, "id_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["app_id", "blind_id"])
        w.writerows(idmap)
    json.dump(report, open(os.path.join(private_dir, "scrub_report.json"), "w"), indent=1, ensure_ascii=False)
    manifest = {"app_version": APP_VERSION, "item_set_hash": item_set_hash, "texts_sha256": sha, "n_listings": n_real,
                "n_attention_checks": len(items) - n_real, "listings_changed_by_scrub": sum(1 for x in report if x["changes"]),
                "listings_with_review_flags": sum(1 for x in report if x["review"]), "demo": demo,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": {k: v for k, v in config.items() if k != "consentHtml"}}
    json.dump(manifest, open(os.path.join(private_dir, "manifest.json"), "w"), indent=1)
    return html, items, idmap, report, manifest


def selftest():
    # scrubber: each case fails if a pattern is missing or too greedy
    cases = [
        ("Contact John Smith at 555-123-4567 or jsmith@city.gov.", "Contact [name] at [phone] or [email]."),
        ("Call (225) 389-3000 ext 12 for an appointment", "Call [phone] for an appointment"),
        ("Visit www.example.com/auction for photos.", "Visit [link] for photos."),
        ("See https://x.gov/a?b=1 now", "See [link] now"),
        ("Ask for Mrs. Jones in the shop.", "Ask for [name] in the shop."),
        ("Call Monday through Friday.", "Call Monday through Friday."),
        ("Contact Public Works to inspect.", "Contact Public Works to inspect."),
        ("VIN 1FTFW1ET5DFC10312, 2013 Ram, 125,000 miles, $7,618, part 4500-1234, 12x10 ft.",
         "VIN 1FTFW1ET5DFC10312, 2013 Ram, 125,000 miles, $7,618, part 4500-1234, 12x10 ft."),
        ("Serial 1234567890 on the plate.", "Serial 1234567890 on the plate."),
        ("2013 Ram Pick up, Bad Motor, Engine knocking", "2013 Ram Pick up, Bad Motor, Engine knocking"),
        ("Contact Sgt. Jason Smith 555-123-4567.", "Contact [name] [phone]."),
        ("Please direct questions to Dana Pratt at dp@x.org or Brian Ross at br@x.org",
         "Please direct questions to [name] at [email] or [name] at [email]"),
        ("contact Bob Lee (bob@x.org) 555-111-2222 or Don Kell (don@x.org) for a visit",
         "contact [name] ([email]) [phone] or [name] ([email]) for a visit"),
        ("Call Public Works at 555-111-2222.", "Call Public Works at [phone]."),
        ("Phone: 555-111-2222, Office (225) 555-1212", "Phone: [phone], Office [phone]"),
        ("water pump replacement @ 77,221 miles", "water pump replacement @ 77,221 miles"),
    ]
    for src, want in cases:
        got = scrub(src)[0]
        assert got == want, "scrub(%r) = %r, want %r" % (src, got, want)
    assert scrub("Serial 1234567890 on the plate.")[2][0]["kind"] == "bare ten-digit number"
    # builder: ids re-keyed, no blind id or script break in the hosted file, checks present, data parse back
    with tempfile.TemporaryDirectory() as td:
        rows = [{"id": "B007", "title": "Truck", "description": "Runs. Contact Bob Lee at 555-111-2222."},
                {"id": "B042", "title": "Desk", "description": "Bad top. </script><script>alert(1)</script>"},
                {"id": "B099", "title": "Chair", "description": "Line one\nLine two"}]
        html, items, idmap, report, man = build(rows, "x" * 64, os.path.join(td, "out"), os.path.join(td, "priv"),
                                                "<p>Consent</p>", seed=1, n_pre=1)
        for b in ("B007", "B042", "B099"):
            assert b not in html, "a blind id leaked into the hosted file"
        assert "</script><script>alert(1)" not in html, "listing text can close the script element"
        assert "Bob Lee" not in html and "555-111-2222" not in html, "contact details leaked"
        m = re.search(r"^const ITEMS = (.*);$", html, re.M)  # the data sit on one line (json escapes newlines)
        data = json.loads(m.group(1).replace("<\\/", "</"))
        assert len(data) == 5 and sum(1 for x in data if x.get("check")) == 2, "items or attention checks missing"
        assert len({x["id"] for x in data}) == 5 and all(re.fullmatch(r"L[0-9a-f]{6}", x["id"]) for x in data)
        assert {b for _, b in idmap} == {"B007", "B042", "B099", "CHECK_U", "CHECK_H"}
        assert man["listings_changed_by_scrub"] == 1 and report[0]["blind_id"] == "B007"
        assert any(x["description"] == "Line one\nLine two" for x in data), "line breaks must survive"
        try:
            build(rows, "x" * 64, os.path.join(td, "same"), os.path.join(td, "same"), "<p>C</p>", seed=1)
            raise AssertionError("PRIVATE equal to OUT must be refused")
        except AssertionError as e:
            if "must be refused" in str(e):
                raise
    print("selftest passed: scrubber (16 cases, review flag) and builder (re-keying, escaping, checks, privacy)")


PAIRS_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rewrite check</title>
<style>body{font:15px/1.5 -apple-system,Helvetica,Arial,sans-serif;margin:0 auto;max-width:1180px;padding:12px 18px;color:#1d1d1f}
h1{font-size:19px;margin:6px 0}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.prog{color:#555}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0}.box{border:1px solid #ccc;border-radius:6px;padding:10px;
white-space:pre-wrap;background:#fafafa;max-height:46vh;overflow:auto}.lab{font-weight:600;margin-bottom:4px}
.q{margin:8px 0;padding:8px;border:1px solid #e4e4e4;border-radius:6px}.q b{display:block;margin-bottom:4px}
button{font-size:14px;padding:5px 12px;margin-right:6px;border-radius:5px;border:1px solid #999;background:#fff;cursor:pointer}
button.on{background:#1d4ed8;color:#fff;border-color:#1d4ed8}textarea{width:100%;min-height:44px;font:14px Helvetica,Arial}
details{background:#f4f6fb;border:1px solid #d9def0;border-radius:6px;padding:6px 10px;margin:8px 0}.small{color:#666;font-size:12px}
@media(max-width:800px){.cols{grid-template-columns:1fr}}</style></head><body>
<h1>Rewrite check: does the rewrite stay true to the seller's text?</h1>
<details><summary><b>How to judge</b> (the same rules the coders follow)</summary>
<p><b>Unsupported claim:</b> the rewrite says something about <i>this item</i> that nothing in the title or the original
description supports: "ready to tow", "runs great" when the original only says it ran last year, "reliable",
"well maintained", "ideal for snow removal", or a specification the original never gives ("4WD", "seats 7"). A claim is
supported if the original states it or it follows directly (the title "2015 Ford F-150" supports "Ford pickup").
Restating a stated fact more formally is not a claim. Generic sales talk with nothing checkable ("great opportunity",
"don't miss out") does not count here.</p><p><b>Dropped or changed fact:</b> a fact in the original (a number, a
specification, a condition, a defect, what is included, a term of sale) is missing from the rewrite or says something
different. Same meaning in other words or formats ("6000.00" and "6,000") is kept.</p><p><b>Softened problem:</b> a
problem the original states (does not start, leaks, rust, missing parts, for parts only) is made to sound milder or
less certain.</p></details>
<div class="bar"><span class="prog" id="prog"></span><button onclick="go(-1)">Back</button><button id="next" onclick="go(1)">Next</button>
<button onclick="dl()">Download my answers</button><span class="small" id="saved"></span></div>
<div id="card"></div>
<script>
const PAIRS = __PAIRS__;
const KEY = "__KEY__";
let st = JSON.parse(localStorage.getItem(KEY) || '{"i":0,"a":{}}');
let t0 = Date.now();
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function save(){localStorage.setItem(KEY, JSON.stringify(st));document.getElementById("saved").textContent="saved "+new Date().toLocaleTimeString();}
function ans(){const p=PAIRS[st.i];return st.a[p.pid]||(st.a[p.pid]={});}
function pick(q,v){const a=ans();a[q]=v;save();draw();}
function txt(v){ans().claim_text=v;save();}
function done(a){return a.claim&&a.fact&&a.soft;}
function btn(q,v,l){const a=ans();return '<button class="'+(a[q]===v?"on":"")+'" onclick="pick(\''+q+'\',\''+v+'\')">'+l+'</button>';}
function draw(){const p=PAIRS[st.i],a=ans();const n=Object.values(st.a).filter(done).length;
document.getElementById("prog").textContent="Pair "+(st.i+1)+" of "+PAIRS.length+" ("+n+" answered)";
document.getElementById("next").disabled=!done(a);
document.getElementById("card").innerHTML='<div class="small">'+p.pid+' &middot; '+esc(p.category)+'</div><div><b>'+esc(p.title)+'</b></div>'+
'<div class="cols"><div><div class="lab">Original (the seller\'s text)</div><div class="box">'+esc(p.original)+'</div></div>'+
'<div><div class="lab">Rewrite</div><div class="box">'+esc(p.rewrite)+'</div></div></div>'+
'<div class="q"><b>1. Does the rewrite claim something about this item that the original does not support?</b>'+btn("claim","yes","Yes")+btn("claim","no","No")+
(a.claim==="yes"?'<div>Which claim(s)? Paste or type the words:</div><textarea oninput="txt(this.value)">'+esc(a.claim_text||"")+'</textarea>':"")+'</div>'+
'<div class="q"><b>2. Does the rewrite drop or change a fact the original states?</b>'+btn("fact","yes","Yes")+btn("fact","no","No")+'</div>'+
'<div class="q"><b>3. Does the rewrite soften a problem the original states?</b>'+btn("soft","yes","Yes")+btn("soft","no","No")+btn("soft","none","No problem stated")+'</div>'+
(n===PAIRS.length?'<p><b>All '+n+' pairs answered.</b> Please press "Download my answers" and send the file.</p>':"");}
function go(d){const a=ans();a.ms=(a.ms||0)+(Date.now()-t0);t0=Date.now();st.i=Math.max(0,Math.min(PAIRS.length-1,st.i+d));save();draw();window.scrollTo(0,0);}
function dl(){const out={tool:"__KEY__",n_pairs:PAIRS.length,saved_at:new Date().toISOString(),answers:st.a};
const b=new Blob([JSON.stringify(out,null,1)],{type:"application/json"});const u=URL.createObjectURL(b);const e=document.createElement("a");
e.href=u;e.download="__FILE__";document.body.appendChild(e);e.click();e.remove();}
draw();
</script></body></html>
"""


def build_pairs(pairs_json, out_dir, tag):
    """The PI's rewrite check (see the module docstring): one self-contained page, contact details scrubbed. TAG names
    the page's save slot in the browser (rewrite_check_TAG) and its download; each page needs its own, or a second page
    would open on the first page's answers. The first page (tag s12) keeps its download name, rewrite_check_answers.json."""
    assert re.fullmatch(r"s12[a-z]?", tag), "tag: s12 for the first page, s12b, s12c, ... after it"
    slot = "rewrite_check_" + tag
    fname = "rewrite_check_answers.json" if tag == "s12" else "rewrite_check_answers_%s.json" % tag
    P = json.load(open(pairs_json, encoding="utf-8"))
    assert P and all(set(p) == {"pid", "title", "category", "original", "rewrite"} for p in P), "unexpected pair fields"
    rows, n_changed = [], 0
    for p in P:
        o, co, _ = scrub(p["original"])
        r, cr, _ = scrub(p["rewrite"])
        n_changed += bool(co or cr)
        rows.append({"pid": p["pid"], "title": p["title"], "category": p["category"], "original": o, "rewrite": r})
    html = PAIRS_PAGE.replace("__PAIRS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__KEY__", slot).replace("__FILE__", fname)
    assert "__PAIRS__" not in html and "__KEY__" not in html and "__FILE__" not in html and html.count("<script>") == 1
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(html)
    print("wrote %s: %d pairs, %d with contact details scrubbed; save slot %s, download %s" % (
        os.path.join(out_dir, "index.html"), len(rows), n_changed, slot, fname))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs")
    ap.add_argument("--tag", help="with --pairs: s12 for the first page, s12b, s12c, ... after it")
    ap.add_argument("--texts")
    ap.add_argument("--out")
    ap.add_argument("--private")
    ap.add_argument("--consent", help="HTML file with the consent text approved for the study")
    ap.add_argument("--submit-url", default="")
    ap.add_argument("--completion-url", default="")
    ap.add_argument("--completion-code", default="")
    ap.add_argument("--expect-sha256", default="")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    if a.pairs:
        if not (a.out and a.tag):
            ap.error("--pairs needs --out and --tag")
        build_pairs(a.pairs, a.out, a.tag)
        return
    if not (a.out and a.private):
        ap.error("--out and --private are required")
    if a.demo:
        rows, sha = demo_texts(), "demo"
        consent = ("<p class=\"note\"><b>Demo build.</b> Invented listings; the consent text approved for the study "
                   "replaces this paragraph in the real build.</p>")
        n_pre = 2
    else:
        if not (a.texts and a.consent):
            ap.error("--texts and --consent are required (use --demo to test)")
        rows, sha = read_texts(a.texts, a.expect_sha256)
        consent = open(a.consent, encoding="utf-8").read()
        n_pre = None
    html, items, idmap, report, man = build(rows, sha, a.out, a.private, consent, a.submit_url, a.completion_url,
                                            a.completion_code, a.seed, n_pre=n_pre, demo=a.demo)
    print("wrote %s (%d listings + %d attention checks, item set %s)" % (os.path.join(a.out, "index.html"), man["n_listings"],
          man["n_attention_checks"], man["item_set_hash"]))
    print("private files in %s: id_map.csv, scrub_report.json (%d listings changed, %d with review flags), manifest.json" % (
        a.private, man["listings_changed_by_scrub"], man["listings_with_review_flags"]))


if __name__ == "__main__":
    main()
