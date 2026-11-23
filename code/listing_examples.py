#!/usr/bin/env python3
"""Session 9: the example listings drawn as pages in the paper's Figure 2 (PI, 2026-10-28: "an actual template of how
the actual listing looks, clearly human with a typo vs clearly AI").

Reads, read-only: the listing panel, text features and AI scores (results/panel_s8, results/detector_s8) and each
example's banked page (analysis/muni_data/pages/<first 3 digits>/<item>.html.gz). Writes OUT_JSON with, per item, the
page's own breadcrumb, title and header line (listing number, place, seller's public name), its description as blocks
(heading, paragraph, bullet; bold runs kept), the start date, the classifier score and threshold, the marker words the
text uses (the textfeat list) and the misspellings named on the command line. Checks that stop the job: a named
misspelling must occur in the description and must not be a dictionary word; the marker words found must equal the
panel's count (features f_n_mark); the description must carry no phone number or e-mail address.
Usage: python3 listing_examples.py --selftest
       python3 listing_examples.py PANEL FEATURES SCORES DETECTOR_META PAGES_DIR DICT OUT_JSON ITEM[:typo,typo] ...
"""
import gzip
import json
import os
import re
import sys
from html import unescape
from html.parser import HTMLParser

# the textfeat marker list (code/textfeat.py MARK), repeated so this script runs without textfeat's imports
MARK = {"versatile", "spacious", "ample", "withstand", "efficiently", "lasting", "ensures", "ensuring",
        "enhance", "enhanced", "efficient", "longevity", "dependable", "powerful", "optimal", "smoothly",
        "convenient", "durable", "tackle", "reliability", "convenience", "durability", "providing",
        "additionally", "seamlessly", "robust", "boasts", "elevate", "impressive", "elegant", "crafted",
        "essential", "effortlessly", "exceptional", "sleek", "ideal", "perfect", "sturdy"}
WORD = re.compile(r"[a-z]+")
PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|email\s*protected", re.I)
BLOCK = {"p": "p", "li": "li", "h1": "h", "h2": "h", "h3": "h", "h4": "h", "h5": "h", "h6": "h", "small": "note"}


class DescParser(HTMLParser):
    """Description markup -> [{"kind": p|li|h|note, "runs": [[text, bold], ...]}]; text outside a block becomes a
    paragraph; <br> starts a new block of the same kind."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.kind, self.bold, self.cur = [], None, 0, []

    def flush(self):
        runs = [[re.sub(r"\s+", " ", t), b] for t, b in self.cur if t]
        if runs and "".join(t for t, _ in runs).strip():
            runs[0][0] = runs[0][0].lstrip()
            runs[-1][0] = runs[-1][0].rstrip()
            self.blocks.append({"kind": self.kind or "p", "runs": runs})
        self.cur = []

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK:
            self.flush()
            self.kind = BLOCK[tag]
        elif tag in ("b", "strong"):
            self.bold += 1
        elif tag == "br":
            k = self.kind
            self.flush()
            self.kind = k

    def handle_endtag(self, tag):
        if tag in BLOCK:
            self.flush()
            self.kind = None
        elif tag in ("b", "strong"):
            self.bold = max(0, self.bold - 1)

    def handle_data(self, data):
        self.cur.append((data.replace("\xa0", " "), self.bold > 0))


def page_parts(html):
    """The banked page -> breadcrumb, title, header line and description blocks, as the page shows them."""
    crumbs = re.search(r'id="breadcrumb-item".*?</ul>', html, re.S)
    links = [unescape(re.sub(r"<[^>]+>|\s+", " ", x)).strip() for x in re.findall(r"<a[^>]*>(.*?)</a>", crumbs.group(0), re.S)] if crumbs else []
    links = [x for x in links if x and x not in ("Home", "Browse")]
    # the page title is the h1 just before the "Listing Number" line (other h1s, such as "Missed out on this deal?",
    # sit in the same kind of div)
    title = re.search(r"<h1>((?:(?!</?h1>).)*?)</h1>\s*<h2>\s*<b>\s*Listing Number:", html, re.S)
    head = re.search(r"<h2>\s*<b>\s*Listing Number:\s*</b>(.*?)</h2>", html, re.S)
    line = None
    if head:
        h = re.sub(r"<a[^>]*btn[^>]*>.*?</a>", "", head.group(1), flags=re.S)       # "Seller's Other Auctions" button
        h = re.sub(r"<a[^>]*>\s*\d+\s*</a>", "", h)                                  # its count
        line = "Listing Number: " + unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))).strip().rstrip("|").strip()
        line = re.sub(r"\s+\|", " |", line)
    a = html.find("Item Description")
    a = html.find("</h1>", a) + 5 if a >= 0 else -1
    b = min([i for i in (html.find("<!-- Seller Term Conditions -->", a), html.find('<div class="container-fluid clearfix">', a)) if i > 0] or [-1])
    if a < 5 or b < 0:
        return None
    p = DescParser()
    p.feed(html[a:b])
    p.flush()
    blocks = [x for x in p.blocks if not (x["kind"] == "note" and "Added:" in "".join(t for t, _ in x["runs"]))]
    return {"crumbs": links, "title": unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title.group(1)))).strip() if title else None,
            "header": line, "blocks": blocks}


def check(parts, typos, words_dict, n_mark_panel):
    """The page's description, the named misspellings and the panel's marker count -> marker words found; raises if
    a check fails."""
    text = " ".join("".join(t for t, _ in b["runs"]) for b in parts["blocks"])
    low = text.lower()
    for t in typos:
        if not re.search(r"\b%s\b" % re.escape(t.lower()), low):
            raise ValueError("misspelling %r not in the description" % t)
        if t.lower() in words_dict:
            raise ValueError("%r is a dictionary word" % t)
    marks = sorted(set(WORD.findall(low)) & MARK)
    if n_mark_panel is not None and len(marks) != int(n_mark_panel):
        raise ValueError("marker words on the page %s differ from the panel count %s" % (marks, n_mark_panel))
    if PHONE.search(text) or EMAIL.search(text):
        raise ValueError("a phone number or e-mail address is in the description")
    return marks


def selftest():
    # decoys from real pages: an escaped ampersand in the breadcrumb; an h1 in a col-12 div before the title's h1
    page = ('<div id="breadcrumb-item"><ul><li><a href="/">Home</a></li><li><a>Browse</a></li><li><a>Police &amp; Fire</a></li>'
            '<li><a>Dump Trucks</a></li><li>1989 International <em>(1)</em></li></ul></div>'
            '<div class="col-12"><h1>Missed out on this deal?</h1></div>'
            '<div class="col-12 page-title-item"> <h1> 1989 International </h1> <h2> <b>Listing Number:</b> 1 | susquehanna, PA US | '
            '<strong><a href="#">Harmony Township</a></strong> <a href="#">0</a> <a class="btn">Seller&#39;s Other Auctions</a></h2>'
            '<h1>Item Description </h1> <h1>Big Truck</h1> <p>milage 78908</p><p>&nbsp;</p>'
            '<ul><li><b>Spacious Bed:</b> ample room, robust frame</li><li>dt466<br>eng</li></ul>'
            '<hr><small><b> -Added: 12/15/2021 9:31:04 AM</b></small></div>'
            '<div class="container-fluid clearfix"><h1>Additional information</h1><p>Call 570 727-3004</p></div>')
    P = page_parts(page)
    assert P["crumbs"] == ["Police & Fire", "Dump Trucks"] and P["title"] == "1989 International", P
    assert P["header"] == "Listing Number: 1 | susquehanna, PA US | Harmony Township", P["header"]
    kinds = [b["kind"] for b in P["blocks"]]
    assert kinds == ["h", "p", "li", "li", "li"], kinds                      # empty paragraph and update stamp dropped
    assert P["blocks"][2]["runs"] == [["Spacious Bed:", True], [" ample room, robust frame", False]], P["blocks"][2]
    assert [b["runs"][0][0] for b in P["blocks"][3:]] == ["dt466", "eng"]
    words = {"mileage", "truck", "big", "room", "frame"}
    assert check(P, ["milage"], words, 3) == ["ample", "robust", "spacious"]
    # "room" is in the text and in the dictionary; "hydralics" is a misspelling the text does not carry
    for bad, why in ((["room"], "a correct word named as a typo"), (["hydralics"], "a typo not in the text")):
        try:
            check(P, bad, words, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("check passed with " + why)
    try:
        check(P, ["milage"], words, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("check passed with a marker count that disagrees with the panel")
    Q = dict(P, blocks=P["blocks"] + [{"kind": "p", "runs": [["Call Gene at 215-836-7606 x 1305", False]]}])
    try:
        check(Q, ["milage"], words, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("check passed with a phone number in the description")
    print("selftest: PASS")


def main(argv):
    panel, feats, scores, meta, pages, dict_path, out = argv[:7]
    import pandas as pd
    want = [a.split(":")[0] for a in argv[7:]]
    typos = {a.split(":")[0]: [t for t in (a.split(":")[1].split(",") if ":" in a else []) if t] for a in argv[7:]}
    P = pd.read_parquet(panel, columns=["item", "start_local_dt", "end_local_dt", "agency", "agency_name", "state", "cat1", "cat2", "title"])
    F = pd.read_parquet(feats, columns=["item", "f_w", "f_n_mark"])
    S = pd.read_parquet(scores, columns=["item", "ai_score", "ai_flag"])
    d = P.merge(F, on="item").merge(S, on="item", how="left")
    d = d[d["item"].astype(str).isin(want)]
    d = d.set_index(d["item"].astype(str))
    assert sorted(d.index) == sorted(want), (sorted(d.index), want)          # one panel row per example item
    tau = json.load(open(meta))["tau"]
    words = set(w.strip().lower() for w in open(dict_path, encoding="utf-8", errors="ignore") if w.strip().isalpha())
    res = []
    for i in want:
        r = d.loc[i]
        html = gzip.open(os.path.join(pages, i[:3], i + ".html.gz"), "rt", encoding="utf-8", errors="replace").read()
        parts = page_parts(html)
        marks = check(parts, typos[i], words, r.f_n_mark)
        res.append(dict(parts, item=i, start=str(r.start_local_dt)[:10], end=str(r.end_local_dt)[:10], words=int(r.f_w),
                        score=float(r.ai_score), clf_flag=int(r.ai_flag), tau=float(tau), marker_words=marks,
                        n_mark=int(r.f_n_mark), typos=typos[i], agency=r.agency))
        print("%s | %s | %s | start %s | words %d | classifier %.3f (flag %d) | marker words %s | typos %s | blocks %d"
              % (i, parts["title"], parts["header"], res[-1]["start"], r.f_w, r.ai_score, r.ai_flag, marks, typos[i], len(parts["blocks"])))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"generated_by": "code/listing_examples.py", "listings": res}, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        selftest()
        main(sys.argv[1:])
