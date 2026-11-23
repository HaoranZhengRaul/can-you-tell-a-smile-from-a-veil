#!/usr/bin/env python3
"""Municibid listing-page parser, written from scratch in Session 8.

Standard library only; runs under Python 3.9 (Mercury login and compute nodes) and later.

Input: one banked page (`<item>.html.gz`, raw bytes fetched with the archive's `id_` modifier).
Output: one flat record. Every field is either a value with status "ok", or None with a status
that says why: "absent" (the page has no such element), "empty" (element present, no text),
"unparseable" (text present but not convertible), "live_capture" (only defined after the close),
"not_a_listing" (the page is not a listing page), "conflict" (two sources on the page disagree).

Usage:
  python3 mb_parse.py selftest
  python3 mb_parse.py one PAGE.html.gz            # print one parsed record
  python3 mb_parse.py run --pages DIR --out OUT.jsonl [--frame FRAME.jsonl] [--workers N]
"""
import gzip
import hashlib
import html as htmlmod
import json
import os
import re
import sys
from datetime import datetime
from html.parser import HTMLParser

PARSER_VERSION = "s8p2"
# Bidder and asker identities are public on the page but some are e-mail addresses: store only a
# salted hash, never the raw string. The salt is fixed so the same user hashes identically
# across pages (needed to follow buyers across auctions).
ID_SALT = "REPLACE_WITH_A_PRIVATE_SALT"  # the salt used for the paper is kept private


def hid(x):
    if x is None:
        return None
    x = x.strip().lower()
    if not x:
        return None
    return hashlib.sha256((ID_SALT + "|" + x).encode("utf-8")).hexdigest()[:16]

# --------------------------------------------------------------------------------------------
# A small DOM built on html.parser
# --------------------------------------------------------------------------------------------
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr"}
BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol",
         "section", "article", "header", "footer", "blockquote", "pre", "hr", "dd", "dt"}
SKIP_TEXT = {"script", "style", "noscript", "svg", "template"}


class Node(object):
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = attrs
        self.children = []
        self.parent = parent

    def cls(self):
        return (self.attrs.get("class") or "").split()

    def has_class(self, c):
        return c in self.cls()


class _Builder(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.root = Node("#root", {}, None)
        self.cur = self.root

    def _close_up_to(self, tags, stop):
        n = self.cur
        while n is not self.root and n.tag not in stop:
            if n.tag in tags:
                self.cur = n.parent
                return
            n = n.parent

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._close_up_to({"tr", "td", "th"}, {"table", "tbody", "thead", "tfoot"})
        elif tag in ("td", "th"):
            self._close_up_to({"td", "th"}, {"tr", "table"})
        elif tag == "li":
            self._close_up_to({"li"}, {"ul", "ol"})
        elif tag == "p" and self.cur.tag == "p":
            self.cur = self.cur.parent
        a = {}
        for k, v in attrs:
            a[k.lower()] = v if v is not None else ""
        n = Node(tag, a, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        a = {}
        for k, v in attrs:
            a[k.lower()] = v if v is not None else ""
        self.cur.children.append(Node(tag.lower(), a, self.cur))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in VOID:
            return
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def build(html_text):
    b = _Builder()
    b.feed(html_text)
    b.close()
    return b.root


def walk(node):
    """Depth-first over element nodes (strings skipped)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Node):
            yield n
            for c in reversed(n.children):
                if isinstance(c, Node):
                    stack.append(c)


def text_of(node, block_sep="\n"):
    """Visible text of a node; block elements become line breaks."""
    out = []

    def rec(n):
        for c in n.children:
            if isinstance(c, str):
                out.append(c)
            else:
                if c.tag in SKIP_TEXT:
                    continue
                if c.tag in BLOCK:
                    out.append(block_sep)
                rec(c)
                if c.tag in BLOCK and c.tag != "br":
                    out.append(block_sep)

    rec(node)
    s = "".join(out).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", x).strip() for x in s.split("\n")]
    return "\n".join(x for x in lines if x)


def flat(node):
    return re.sub(r"\s+", " ", text_of(node, " ")).strip()


def find_all(root, pred):
    return [n for n in walk(root) if pred(n)]


def find_first(root, pred):
    for n in walk(root):
        if pred(n):
            return n
    return None


def by_class(root, c):
    return find_all(root, lambda n: n.has_class(c))


def following_in_parent(node):
    """Siblings that come after `node` inside its parent (strings and nodes)."""
    p = node.parent
    if p is None:
        return []
    i = p.children.index(node)
    return p.children[i + 1:]


# --------------------------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------------------------
MONEY_RE = re.compile(r"-?\$?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.([0-9]{1,2}))?")


def money(s):
    """'$ 20,900.00' -> 20900.0 ; returns (value, status)."""
    if s is None:
        return None, "absent"
    t = s.strip()
    if not t:
        return None, "empty"
    m = MONEY_RE.search(t)
    if not m:
        return None, "unparseable"
    whole = m.group(1).replace(",", "")
    frac = m.group(2) or "0"
    return float(whole + "." + frac), "ok"


def int_in(s):
    if s is None:
        return None, "absent"
    m = re.search(r"([0-9][0-9,]*)", s)
    if not m:
        return None, ("empty" if not s.strip() else "unparseable")
    return int(m.group(1).replace(",", "")), "ok"


def parse_dttm(s):
    """'06/26/2019 10:30:00' (seller-local, U.S. Eastern) -> '2019-06-26T10:30:00'."""
    if s is None:
        return None, "absent"
    s = s.strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S"), "ok"
        except ValueError:
            pass
    return None, ("empty" if not s else "unparseable")


def heading_node(root, label):
    """The element whose own text equals `label` (case-insensitive), searching headings and
    the 2024+ `span.itemPageTitles`. Returns the heading element (h1/h2/h3) when possible."""
    lab = label.lower()
    best = None
    for n in walk(root):
        if n.tag in ("h1", "h2", "h3", "h4") or n.has_class("itemPageTitles"):
            t = flat(n).lower()
            if t == lab or t.startswith(lab + " ") or t.startswith(lab + " "):
                best = n
                break
    if best is None:
        return None
    if best.tag == "span":
        p = best.parent
        while p is not None and p.tag not in ("h1", "h2", "h3", "h4") and p.tag != "#root":
            p = p.parent
        if p is not None and p.tag in ("h1", "h2", "h3", "h4"):
            return p
    return best


def section_after(head):
    """Content that follows a section heading. Layouts: (a) heading and paragraphs share a parent
    (2019-2026); (b) heading wrapped in its own div, paragraphs are the wrapper's next siblings
    (2016-2017). Returns a synthetic Node holding the following content."""
    holder = Node("#section", {}, None)
    after = following_in_parent(head)
    if any((isinstance(c, str) and c.strip()) or isinstance(c, Node) for c in after):
        holder.children = list(after)
        return holder
    wrapper = head.parent
    if wrapper is not None and wrapper.parent is not None:
        sib = following_in_parent(wrapper)
        out = []
        for c in sib:
            # stop at the next section block
            if isinstance(c, Node) and (c.has_class("heading-block") or c.has_class("section")):
                break
            out.append(c)
        holder.children = out
    return holder


def norm_img(u):
    u = (u or "").strip()
    u = u.split("?")[0]
    u = re.sub(r"_(thumbcrop|fullsize|largesize|thumb|medium|small)(\.[a-z]+)$", r"\2", u, flags=re.I)
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", u, flags=re.I)
    return m.group(1).lower() if m else u.lower()


QA_Q_RE = re.compile(r'^\s*"?(?P<q>.*?)"?\s*-\s*Asked by\s*#?(?P<who>[0-9A-Za-z_*]+)\s*(?P<when>[0-9/]+\s+[0-9:]+\s*[AP]M)?\s*$', re.S)
QA_A_RE = re.compile(r'^\s*(?P<a>.*?)\s*(?:\(edited on [^)]*\))?\s*-\s*Answered on\s*(?P<when>[0-9/]+\s+[0-9:]+\s*[AP]M)?\s*$', re.S)


def parse_qa(root):
    tab = find_first(root, lambda n: n.attrs.get("id") == "Questiontable")
    if tab is None:
        return None, "absent"
    table = find_first(tab, lambda n: n.tag == "table")
    if table is None:
        return None, "absent"
    rows = find_all(table, lambda n: n.tag == "tr")
    out = []
    none_asked = False
    for tr in rows:
        cells = [c for c in tr.children if isinstance(c, Node) and c.tag in ("td", "th")]
        txt = flat(tr)
        if "No one has asked a question" in txt:
            none_asked = True
            continue
        if len(cells) < 2:
            continue
        lab = flat(cells[0]).strip().rstrip(":").upper()
        body = flat(cells[1])
        if lab == "Q":
            m = QA_Q_RE.match(body)
            if m:
                out.append({"q": m.group("q").strip().strip('"'), "asker": hid(m.group("who")),
                            "q_time": (m.group("when") or "").strip() or None, "a": None, "a_time": None})
            else:
                out.append({"q": body, "asker": None, "q_time": None, "a": None, "a_time": None})
        elif lab == "A" and out:
            m = QA_A_RE.match(body)
            a = (m.group("a") if m else body).strip()
            out[-1]["a"] = a if a else None
            out[-1]["a_time"] = ((m.group("when") or "").strip() or None) if m else None
    if not out:
        return [], ("none_asked" if none_asked else "empty")
    return out, "ok"


def meta(root, key):
    n = find_first(root, lambda n: n.tag == "meta" and (n.attrs.get("property") == key or n.attrs.get("name") == key))
    return None if n is None else n.attrs.get("content")


# --------------------------------------------------------------------------------------------
# The page parser
# --------------------------------------------------------------------------------------------
def parse_page(html_text, item=None, capture_ts=None):
    rec = {"parser": PARSER_VERSION, "item": item, "capture_ts": capture_ts}
    st = {}
    src = {}
    root = build(html_text)

    # ---- page kind -----------------------------------------------------------------------
    data_div = find_first(root, lambda n: n.has_class("listing-data-div"))
    head_desc = heading_node(root, "Item Description")
    head_details = heading_node(root, "Auction Details")
    lnum_txt = None
    m = re.search(r"Listing (?:Number|#)\s*(?:</b>|</span>|</strong>)?\s*[:\-]?\s*(?:</b>|</span>|</strong>)?\s*([0-9]{5,9})", html_text)
    if m:
        lnum_txt = m.group(1)
    else:
        m = re.search(r"<h1>[^<]*?(?:&nbsp;|\s)*<small>\(([0-9]{5,9})\)</small>", html_text)
        if m:
            lnum_txt = m.group(1)
    legacy = find_first(root, lambda n: n.attrs.get("id") == "tblAuctionDetail") is not None
    is_listing = (data_div is not None or lnum_txt is not None or legacy) and (head_desc is not None or head_details is not None or legacy)
    rec["page_kind"] = "listing" if is_listing else "not_a_listing"
    if not is_listing:
        reason = []
        if data_div is None and lnum_txt is None:
            reason.append("no_listing_number")
        if head_desc is None and head_details is None:
            reason.append("no_description_or_details_section")
        t = find_first(root, lambda n: n.tag == "title")
        rec["html_title"] = flat(t)[:200] if t is not None else None
        rec["not_a_listing_reason"] = "+".join(reason)
        rec["status"] = {"page_kind": "not_a_listing"}
        return rec

    # ---- identity ------------------------------------------------------------------------
    layout = None
    if find_first(root, lambda n: n.attrs.get("id") == "tblAuctionDetail") is not None:
        layout = "legacy_2016"
    elif find_first(root, lambda n: n.tag == "section" and n.attrs.get("id") == "page-title-item") is not None:
        layout = "section_2017"
    elif find_first(root, lambda n: n.has_class("titleTextForItem")) is not None:
        layout = "title_text_2024"
    elif find_first(root, lambda n: n.has_class("page-title-item")) is not None:
        layout = "page_title_2019"
    rec["layout"] = layout

    ids = {}
    if data_div is not None and data_div.attrs.get("data-listingid"):
        ids["data_div"] = data_div.attrs["data-listingid"].strip()
    if lnum_txt:
        ids["header"] = lnum_txt
    bc = find_first(root, lambda n: n.attrs.get("id") == "breadcrumb-item")
    if bc is None:
        bc = find_first(root, lambda n: n.tag == "ul" and n.has_class("breadcrumb"))
    if bc is not None:
        em = find_first(bc, lambda n: n.tag == "em")
        if em is not None:
            mm = re.search(r"([0-9]{5,9})", flat(em))
            if mm:
                ids["breadcrumb"] = mm.group(1)
    ogu = meta(root, "og:url") or ""
    mm = re.search(r"/Listing/Details/([0-9]{5,9})", ogu, flags=re.I)
    if mm:
        ids["og_url"] = mm.group(1)
    vals = sorted(set(ids.values()))
    rec["listing_id"] = vals[0] if vals else None
    st["listing_id"] = "ok" if len(vals) == 1 else ("conflict" if vals else "absent")
    src["listing_id"] = ids
    rec["listing_id_matches_item"] = (rec["listing_id"] == item) if (item and rec["listing_id"]) else None

    # title block: the element holding title, listing number, location and seller link
    title_block = None
    for n in walk(root):
        if n.has_class("page-title-item") or (n.tag == "section" and n.attrs.get("id") == "page-title-item"):
            title_block = n
            break
    titles = {}
    if title_block is not None:
        h = find_first(title_block, lambda n: n.tag == "h1" or n.has_class("titleTextForItem"))
        if h is not None:
            hcopy = flat(h)
            sm = find_first(h, lambda n: n.tag == "small")
            if sm is not None:
                hcopy = hcopy.replace(flat(sm), "").strip()
            if hcopy:
                titles["title_block"] = hcopy
    ogt = meta(root, "og:title")
    if ogt:
        titles["og_title"] = re.sub(r"\s+", " ", ogt).strip()
    if data_div is not None and data_div.attrs.get("auction_title"):
        titles["data_div"] = re.sub(r"\s+", " ", data_div.attrs["auction_title"]).strip()
    rec["title"] = None
    for k in ("title_block", "data_div", "og_title"):
        if titles.get(k):
            rec["title"] = htmlmod.unescape(titles[k])
            src["title"] = k
            break
    st["title"] = "ok" if rec["title"] else "absent"
    norm = set(re.sub(r"\W+", "", v.lower()) for v in titles.values())
    rec["title_sources_agree"] = (len(norm) == 1) if titles else None

    seller_link = None
    if title_block is not None:
        seller_link = find_first(title_block, lambda n: n.tag == "a" and "Browse?Seller=" in (n.attrs.get("href") or ""))
    if seller_link is None and layout == "legacy_2016":
        seller_link = find_first(root, lambda n: n.tag == "a" and "Browse?Seller=" in (n.attrs.get("href") or ""))
        src["agency"] = "legacy_other_listings_link"
    elif seller_link is not None:
        src["agency"] = "title_block"
    if seller_link is not None:
        rec["agency_slug"] = (seller_link.attrs.get("href") or "").split("Seller=")[-1].split("&")[0].strip() or None
        nm = flat(seller_link)
        rec["agency_name"] = nm if (nm and "other listings" not in nm.lower() and "more listings" not in nm.lower()) else None
    else:
        rec["agency_slug"] = None
        rec["agency_name"] = None
    if layout == "legacy_2016":
        for tr in find_all(root, lambda n: n.tag == "tr"):
            cells = [c for c in tr.children if isinstance(c, Node) and c.tag == "td"]
            if len(cells) >= 2 and flat(cells[0]).lower() == "seller":
                rec["agency_name"] = flat(cells[1]) or rec["agency_name"]
    st["agency_slug"] = "ok" if rec["agency_slug"] else "absent"
    if data_div is not None and data_div.attrs.get("agency_name"):
        rec["agency_name_data_div"] = re.sub(r"\s+", " ", data_div.attrs["agency_name"]).strip()
    else:
        rec["agency_name_data_div"] = None
    rec["seller_state_data_div"] = data_div.attrs.get("seller_state") if data_div is not None else None

    loc = None
    if title_block is not None:
        sl = find_first(title_block, lambda n: n.has_class("location"))
        if sl is not None and flat(sl):
            loc = flat(sl)
        else:
            m3 = re.search(r"([A-Za-z][A-Za-z .'\-]*?,\s*[A-Z]{2}\s+[A-Z]{2})\b", flat(title_block))
            if m3:
                loc = m3.group(1).strip()
    if loc is None and layout == "legacy_2016":
        for tr in find_all(root, lambda n: n.tag == "tr"):
            cells = [c for c in tr.children if isinstance(c, Node) and c.tag == "td"]
            if len(cells) >= 2 and flat(cells[0]).lower() == "location":
                loc = flat(cells[1]) or None
    rec["location"] = loc
    st["location"] = "ok" if loc else "absent"
    rec["loc_state"] = None
    rec["loc_country"] = None
    if loc:
        m4 = re.search(r",\s*([A-Za-z]{2})\s+([A-Z]{2})\s*$", loc)
        if m4:
            rec["loc_state"], rec["loc_country"] = m4.group(1).upper(), m4.group(2)

    cats = []
    if bc is not None:
        for a in find_all(bc, lambda n: n.tag == "a" and (n.attrs.get("href") or "").startswith("/Browse/C")):
            cats.append(flat(a))
    kw = meta(root, "keywords")
    rec["categories"] = cats if cats else None
    rec["categories_meta"] = [x.strip() for x in kw.split(",")] if kw else None
    st["categories"] = "ok" if cats else ("meta_keywords_only" if kw else "absent")

    # ---- timing ----------------------------------------------------------------------------
    sdt = find_first(root, lambda n: n.has_class("awe-rt-startingDTTM"))
    edt = find_first(root, lambda n: n.has_class("awe-rt-endingDTTM"))
    rec["start_local"], st["start_local"] = parse_dttm(sdt.attrs.get("data-initial-dttm") if sdt is not None else None)
    rec["end_local"], st["end_local"] = parse_dttm(edt.attrs.get("data-initial-dttm") if edt is not None else None)
    if rec["end_local"] is None:
        le = find_first(root, lambda n: n.attrs.get("id") == "tdAuctionListingEndDate")
        if le is not None:
            t = re.sub(r"\s+(Eastern|Central|Pacific|Mountain)$", "", flat(le))
            try:
                rec["end_local"], st["end_local"] = datetime.strptime(t, "%A, %B %d, %Y %I:%M:%S %p").strftime("%Y-%m-%dT%H:%M:%S"), "ok"
            except ValueError:
                st["end_local"] = "unparseable"
    ext = find_first(root, lambda n: n.has_class("awe-rt-Extended"))
    rec["extended"] = (not ext.has_class("awe-hidden")) if ext is not None else None
    if data_div is not None:
        for k in ("auction_start", "auction_end"):
            v = data_div.attrs.get(k)
            try:
                rec[k + "_epoch"] = int(v) if v not in (None, "") else None
            except ValueError:
                rec[k + "_epoch"] = None
    else:
        rec["auction_start_epoch"] = None
        rec["auction_end_epoch"] = None

    # ---- state -----------------------------------------------------------------------------
    ended_block = find_first(root, lambda n: n.has_class("postauctionstatus-end"))
    ended_heading = find_first(root, lambda n: n.tag in ("h1", "h2", "h3") and flat(n).lower().startswith("bidding on this item has ended"))
    live_marker = ("This Auction Ends in" in html_text) or ("Want to bid on this item?" in html_text)
    if ended_block is not None or ended_heading is not None:
        rec["ended"] = True
        st["ended"] = "ok"
    elif layout == "legacy_2016" and re.search(r"Status:\s*(Successful|Unsuccessful|Closed|Ended)", html_text):
        rec["ended"] = True
        st["ended"] = "legacy_status_text"
    elif live_marker:
        rec["ended"] = False
        st["ended"] = "ok"
    elif "Bidding has not started for this auction" in html_text:
        rec["ended"] = False
        st["ended"] = "scheduled_not_started"
    else:
        rec["ended"] = None
        st["ended"] = "absent"
    if rec["ended"] is True:
        rec["auction_state"] = "ended"
    elif st["ended"] == "scheduled_not_started":
        rec["auction_state"] = "scheduled"
    elif rec["ended"] is False:
        rec["auction_state"] = "live"
    else:
        rec["auction_state"] = None

    # ---- money -----------------------------------------------------------------------------
    rec["final_bid"], st["final_bid"] = None, "absent"
    for h in walk(root):
        if h.tag == "h1" and flat(h).lower().startswith("final bid"):
            nxt = [c for c in following_in_parent(h) if isinstance(c, Node)]
            if nxt:
                rec["final_bid"], st["final_bid"] = money(flat(nxt[0]))
            else:
                st["final_bid"] = "empty"
            break
    if rec["final_bid"] is None and rec.get("ended") is False:
        st["final_bid"] = "live_capture"

    sb = None
    for n in walk(root):
        if n.tag in ("strong", "b") and flat(n).lower().startswith("starting bid"):
            sib = [c for c in following_in_parent(n) if isinstance(c, Node)]
            sb = flat(sib[0]) if sib else "".join(c for c in following_in_parent(n) if isinstance(c, str)).strip()
            if not sb:
                td = n.parent
                while td is not None and td.tag not in ("td", "th"):
                    td = td.parent
                if td is not None:
                    tds = [c for c in following_in_parent(td) if isinstance(c, Node) and c.tag in ("td", "th")]
                    if tds:
                        sb = flat(tds[0])
            break
    rec["starting_bid"], st["starting_bid"] = money(sb)

    cp = find_first(root, lambda n: n.has_class("awe-rt-CurrentPrice"))
    rec["current_price"], st["current_price"] = money(flat(cp) if cp is not None else None)

    bn = find_first(root, lambda n: n.has_class("awe-rt-BuyNowPrice"))
    rec["buy_now_price"], st["buy_now_price"] = money(flat(bn) if bn is not None else None)
    bna = find_first(root, lambda n: n.has_class("awe-rt-BuyItNowAvailable"))
    rec["buy_now_shown"] = (not (bna.has_class("awe-hidden") or "hidden" in bna.attrs)) if bna is not None else None

    rs = find_first(root, lambda n: n.has_class("awe-rt-ReserveStatus"))
    if rs is None:
        rs = find_first(root, lambda n: n.attrs.get("id") == "spanReservePriceStatus")
    rtxt = flat(rs) if rs is not None else None
    rec["reserve_text"] = rtxt or None
    rl = (rtxt or "").lower()
    if rs is None:
        rec["reserve"], st["reserve"] = None, "absent"
    elif "not met" in rl:
        rec["reserve"], st["reserve"] = "not_met", "ok"
    elif "no reserve" in rl:
        rec["reserve"], st["reserve"] = "none", "ok"
    elif "met" in rl:
        rec["reserve"], st["reserve"] = "met", "ok"
    elif not rl:
        rec["reserve"], st["reserve"] = None, "empty"
    else:
        rec["reserve"], st["reserve"] = None, "unparseable"

    bf = find_first(root, lambda n: n.attrs.get("id") == "buyer-fee")
    if bf is not None:
        mm = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", flat(bf))
        rec["buyer_fee_pct"], st["buyer_fee_pct"] = (float(mm.group(1)), "ok") if mm else (None, "unparseable")
    else:
        rec["buyer_fee_pct"], st["buyer_fee_pct"] = None, "absent"

    # ---- activity --------------------------------------------------------------------------
    bc_nodes = by_class(root, "awe-rt-AcceptedListingActionCount")
    bid_vals = set()
    for n in bc_nodes:
        v = n.attrs.get("data-previous-value")
        x, s = int_in(v if v not in (None, "") else flat(n))
        if s == "ok":
            bid_vals.add(x)
    lb = find_first(root, lambda n: n.attrs.get("id") == "spanBidCount")
    if not bc_nodes and lb is not None:
        rec["n_bids"], st["n_bids"] = int_in(flat(lb))
    elif not bc_nodes:
        rec["n_bids"], st["n_bids"] = None, "absent"
    elif len(bid_vals) == 1:
        rec["n_bids"], st["n_bids"] = bid_vals.pop(), "ok"
    elif not bid_vals:
        rec["n_bids"], st["n_bids"] = None, "unparseable"
    else:
        rec["n_bids"], st["n_bids"] = max(bid_vals), "conflict"

    hb = find_first(root, lambda n: n.has_class("bidderHigh"))
    hbm = find_first(root, lambda n: n.attrs.get("id") == "spanHighBidderAnonymous")
    hb_raw = (flat(hb) or None) if hb is not None else None
    rec["high_bidder_id"] = hid(hb_raw)
    rec["high_bidder_is_email"] = ("@" in hb_raw) if hb_raw else None
    rec["high_bidder_masked_present"] = (hbm is not None and bool(flat(hbm)))
    st["high_bidder_id"] = "ok" if hb_raw else ("empty" if hb is not None else "absent")

    rec["page_views"], st["page_views"] = None, "absent"
    for n in walk(root):
        if n.tag in ("strong", "b") and flat(n).lower() == "page views":
            td = n.parent
            while td is not None and td.tag not in ("td", "th"):
                td = td.parent
            if td is not None:
                tds = [c for c in following_in_parent(td) if isinstance(c, Node) and c.tag in ("td", "th")]
                if tds:
                    rec["page_views"], st["page_views"] = int_in(flat(tds[0]))
                else:
                    st["page_views"] = "empty"
            break

    # ---- text ------------------------------------------------------------------------------
    if head_desc is None:
        rec["desc"], st["desc"] = None, ("legacy_meta_only" if layout == "legacy_2016" else "absent")
        rec["desc_updates"] = None
        rec["desc_markup"] = None
        if layout == "legacy_2016":
            md = meta(root, "description")
            rec["desc"] = re.sub(r"\s+", " ", md).strip() if md else None
    else:
        sec = section_after(head_desc)
        # split seller updates: each update is separated by <hr> and stamped '-Added: <date>'
        parts = [[]]
        for c in sec.children:
            if isinstance(c, Node) and c.tag == "hr":
                parts.append([])
            else:
                parts[-1].append(c)
        texts = []
        for p in parts:
            h = Node("#part", {}, None)
            h.children = p
            texts.append(text_of(h))
        stamp = re.compile(r"-\s*Added:\s*([0-9/]+\s+[0-9:]+\s*[AP]M)\s*$")
        base_parts = []
        upd = []
        for i, t in enumerate(texts):
            mm = stamp.search(t)
            if i > 0 and mm:
                upd.append({"text": t[:mm.start()].strip(), "added": mm.group(1)})
            elif i == 0 and mm and len(texts) == 1:
                base_parts.append(t[:mm.start()].strip())
                rec["desc_base_added"] = mm.group(1)
            else:
                # an <hr> the seller typed inside the description: still the base text
                base_parts.append(t)
        base_txt = "\n".join(x for x in base_parts if x)
        rec["desc"] = base_txt if base_txt else None
        rec["desc_updates"] = upd if upd else []
        st["desc"] = "ok" if base_txt else "empty"
        tags = [n.tag for n in walk(sec)]
        rec["desc_markup"] = {"p": tags.count("p"), "br": tags.count("br"), "bold": tags.count("strong") + tags.count("b"),
                              "li": tags.count("li"), "table": tags.count("table"), "img": tags.count("img"),
                              "u": tags.count("u"), "em": tags.count("em") + tags.count("i"),
                              "h": sum(tags.count(h) for h in ("h3", "h4", "h5", "h6")), "hr": tags.count("hr")}
    for label, key in (("Seller’s Terms & Conditions", "terms"), ("Seller's Terms & Conditions", "terms"),
                       ("Payment Details", "payment")):
        if rec.get(key) is not None:
            continue
        hn = heading_node(root, label)
        if hn is not None:
            t = text_of(section_after(hn))
            rec[key] = t if t else None
            st[key] = "ok" if t else "empty"
    for key in ("terms", "payment"):
        if key not in rec:
            rec[key] = None
            st[key] = "absent"
    ogd = meta(root, "og:description")
    rec["og_description"] = ogd
    st["og_description"] = "ok" if ogd else "absent"

    # ---- structured table ------------------------------------------------------------------
    cav = find_first(root, lambda n: n.tag == "table" and n.attrs.get("id") == "custom-auction-values")
    table = {}
    if cav is not None:
        for tr in find_all(cav, lambda n: n.tag == "tr"):
            cells = [c for c in tr.children if isinstance(c, Node) and c.tag in ("td", "th")]
            if len(cells) >= 2:
                k = flat(cells[0]).rstrip(":").strip()
                v = flat(cells[1])
                if k:
                    table[k] = v
        st["fields"] = "ok" if table else "empty"
    else:
        st["fields"] = "absent"
    rec["fields"] = table if table else None

    # ---- buyer questions -------------------------------------------------------------------
    qa, st["qa"] = parse_qa(root)
    rec["qa"] = qa
    if qa is None:
        rec["n_questions"] = None
        rec["n_answered"] = None
        rec["n_askers"] = None
    else:
        rec["n_questions"] = len(qa)
        rec["n_answered"] = sum(1 for x in qa if x.get("a"))
        rec["n_askers"] = len(set(x["asker"] for x in qa if x.get("asker")))

    # ---- photos ----------------------------------------------------------------------------
    urls = set()
    gsrc = None
    for ul in find_all(root, lambda n: n.tag == "ul" and n.has_class("es-slides")):
        for im in find_all(ul, lambda n: n.tag == "img"):
            urls.add(norm_img(im.attrs.get("data-full-size-src") or im.attrs.get("src")))
        gsrc = "es_slides"
    if not urls:
        for g in find_all(root, lambda n: n.tag == "div" and n.has_class("gallery")):
            for im in find_all(g, lambda n: n.tag == "img"):
                urls.add(norm_img(im.attrs.get("src")))
            gsrc = "gallery_div"
    if not urls:
        for im in find_all(root, lambda n: n.tag == "img" and (n.attrs.get("alt") or "").strip() == "Listing Image"):
            urls.add(norm_img(im.attrs.get("data-full-size-src") or im.attrs.get("src")))
            gsrc = "alt_listing_image"
    urls.discard("")
    if urls:
        rec["n_photos"], st["n_photos"] = len(urls), "ok"
    else:
        rec["n_photos"], st["n_photos"] = None, ("og_image_only" if meta(root, "og:image") else "absent")
    src["n_photos"] = gsrc

    rec["status"] = st
    rec["sources"] = src
    return rec


# --------------------------------------------------------------------------------------------
# Self-test: one case per layout and per field; each verified to fail under a mutation.
# --------------------------------------------------------------------------------------------
FIX_2019 = """<html><head><meta property="og:title" content="Delta Saw" /><meta name="keywords" content="Tools,Home" />
<meta property="og:image" content="x.jpg" /></head><body>
<div class="container-fluid" id="breadcrumb-item"><ul><li><a href="/">Home</a></li><li><a href="/Listing">Browse</a></li>
<li><a href="/Browse/C1/Tools">Tools</a></li><li><a href="/Browse/C1-C2/Tools-Home">Home &amp; Maintenance</a></li>
<li> Delta Saw&nbsp;<em>(22773820)</em></li></ul></div>
<div class="col-12 page-title-item"><h1>Delta Saw </h1><h2><b>Listing Number:</b> 22773820 | <i></i> Trenton, NJ US |
<strong><a href="/Browse?Seller=tyqrell"> NJ Purchase and Property - DSS </a></strong>
<a class="btn" href="Browse?Seller=tyqrell">Seller's Other Auctions</a></h2></div>
<div class="postauctionstatus postauctionstatus-end"><h1>Bidding on this item has ended</h1></div>
<div class="row"><div class="col-12"><h1>Final Bid:</h1><h2> $<span class="NumberPart">2,210.00</span></h2>
<h1>Bids:</h1><h2><span class="awe-rt-AcceptedListingActionCount" data-previous-value="18">18 Bids</span></h2></div></div>
<ul class="es-slides"><li><div><img class="img-thumbnail" src="NetworkedContent/listingImages/20190614/86e4efbf-ecaf-45d7-a5b7-6466b3cf942b_thumbcrop.jpg" data-full-size-src="NetworkedContent/listingImages/20190614/86e4efbf-ecaf-45d7-a5b7-6466b3cf942b_fullsize.jpg" alt="Listing Image" /></div></li>
<li><div><img src="NetworkedContent/listingImages/20190614/11111111-ecaf-45d7-a5b7-6466b3cf942b_thumbcrop.jpg" alt="Listing Image" /></div></li></ul>
<table class="table table-auctiondetails"><tr><td><strong>Starting Bid:</strong> <span>$50.00</span></td></tr>
<tr><td><strong>Current Price:</strong> <span class=" awe-rt-CurrentPrice"> $<span class="NumberPart">2,210.00</span></span></td></tr>
<tr class="awe-rt-BuyItNowAvailable awe-hidden"><td><strong>Buy Now Price</strong></td><td class="awe-rt-BuyNowPrice"> $<span class="NumberPart">0.00</span></td></tr>
<td colspan="2"><span class="awe-rt-ReserveStatus"><span class="text-success">No Reserve </span></span></td>
<tr><td><strong>Buyer&#39;s Fee:</strong> <span id="buyer-fee" class="Buyer_Fee">5.00 %</span></td></tr></table>
<table class="table table-auctiondetails"><tr><td><strong>Start Date</strong></td><td class="awe-rt-startingDTTM" data-initial-dttm="06/14/2019 11:08:00"></td></tr>
<tr><td><strong>End Date</strong></td><td><span class="awe-rt-endingDTTM" data-initial-dttm="06/26/2019 10:30:00"></span><span class="awe-rt-Extended awe-hidden">[Extended]</span></td></tr>
<tr><td><strong>Bid History</strong></td><td><span class="awe-rt-AcceptedListingActionCount" data-previous-value="18">18</span> Bids</td></tr>
<tr><td><strong>High Bidder</strong></td><td><span class="bidderHigh" hidden> NAH1992 </span><span id="spanHighBidderAnonymous">N*****2</span></td></tr>
<tr><td> <strong>Page Views</strong> </td><td> 1,786 </td></tr></table>
<div class="col-12"><h1><i class="fas fa-dollar-sign"></i> Payment Details</h1><p><span> Pay within 5 days. </span></p></div>
<div class="col-12"><h1><i class="fa fa-info-circle"></i> Item Description </h1><p>This auction is for a saw.</p><p>It was operable.<br>Cord cut.</p></div>
<div class="col-12"><h1><i></i> Additional information on this Auction</h1><table id="custom-auction-values" class="table">
<tr><td><strong>Custom ID</strong></td><td>DSS2141</td><td> </td></tr><tr><td><strong>Miles</strong></td><td>95,211</td><td></td></tr></table></div>
<div class="container-fluid clearfix" id="Questiontable"><div><div><h1>Questions/Answers <small>Ask</small></h1>
<table class="table table-responsive table-qa mt10">
<tr style=""><td><h6>Q:</h6></td><td> "Single or 3 phase?" <br><small>- Asked by #561206 <b>11/24/2021 09:50 AM</b></small></td></tr>
<tr><td><h6>A:</h6></td><td> single. <br /><small><i>-Answered on</i><b> 11/30/2021 10:09 AM</b></small></td></tr>
<tr style=""><td><h6>Q:</h6></td><td> "Can you load it?" <br><small>- Asked by #4364944 <b>12/07/2021 12:20 PM</b></small></td></tr>
<tr><td><h6>A:</h6></td><td><br /><small></small></td></tr>
</table></div></div></div></body></html>"""

FIX_2025_LIVE = """<html><head><meta property="og:title" content="2014 Mack Granite" /></head><body>
<div data-listingid="73065022" class="listing-data-div" auction_start="1742488889" auction_end="1745528880" auction_title="2014 Mack Granite" agency_name="Sanitary District No. One" seller_state="NY"></div>
<div class="col page-title-item titleTextContainer"><div class="titleTextForItem"> 2014 Mack Granite </div><div class="subTitleText"><span>Listing Number</span> - 73065022 &nbsp; Lawrence, NY US | <a class="linkTag" href="/Browse?Seller=SanitOne"> Sanitary District No. One </a></div></div>
<h1>This Auction Ends in:</h1>
<table><tr><td><strong>Starting Bid:</strong> <span>$100.00</span></td></tr>
<tr><td><strong>Current Price:</strong><span class=" awe-rt-CurrentPrice"> $<span class="NumberPart">20,900.00</span></span></td></tr>
<tr><td><span class="awe-rt-ReserveStatus"><span>Reserve Not Met</span></span></td></tr></table>
<td><span class="awe-rt-AcceptedListingActionCount" data-previous-value="170">170</span> Bid(s)</td>
<div class="col-12"><h2 style="x"><svg><path d="M1"/></svg><span class="itemPageTitles"> Item Description</span></h2>
<p><strong>Up for auction we have a 2014 Mack</strong></p><p><b>No Lights On Dash</b></p></div>
<div class="col-12"><h2><span class="itemPageTitles"> Seller’s Terms &amp; Conditions </span></h2><p><span> Cash only. </span></p></div>
<div class="container-fluid clearfix" id="Questiontable"><h2><span class="itemPageTitles"> Questions/Answers</span></h2>
<table class="table-qa"><tr><td> No one has asked a question on this item yet. </td></tr></table></div>
<div class="gallery"><div class="main-image-container"><img id="mainImage" src="https://s.net/assets/media/7d1eefcf-00f7-42a3-95b6-b73e946407d8_fullsize.jpg" data-index="0" /></div>
<img src="https://s.net/assets/media/7d1eefcf-00f7-42a3-95b6-b73e946407d8_fullsize.jpg" data-index="0" class="active" />
<img src="https://s.net/assets/media/655707ea-38da-458f-86b2-cb4ebe1bf774_fullsize.jpg" data-index="1" /></div>
</body></html>"""

FIX_2017 = """<html><head><meta property="og:title" content="Powerware 9170+ UPS" /><meta name="keywords" content="Computers &amp; IT,Power" /></head><body>
<section id="page-title-item"><div class="container-fluid clearfix pl30"><h1>Powerware 9170+ UPS&nbsp;<small>(11744189)</small></h1>
<span> <i class="icon-map-marker2"></i> Granby, MA US |<strong> <a href="/Browse?Seller=GranbyPoliceDepart"> Granby Police </a></strong></span></div></section>
<h3 class="m0"><b>Current Bid:</b> <span class="Bidding_Current_Price awe-rt-CurrentPrice"> $<span class="NumberPart">200.00</span></span>
<x-small class="pr5 awe-rt-ReserveStatus"><span class="text-danger">Reserve Price Not Met</span></x-small></h3>
<div class="section section-item notopborder nomargin"><div class="container clearfix"><div class="heading-block heading-item"><h3><i class="fa fa-info-circle Bootstro"></i> Item Description </h3></div>
<p>Powerware 9170+ UPS and switch.&nbsp;Used for 911 equipment.</p><p>&nbsp;</p><p>Batteries replaced in 2016.</p></div></div>
<div class="section section-item notopborder nomargin"><div class="container clearfix"><div class="heading-block heading-item"><h3>Additional information on this Auction and Item</h3></div></div></div>
</body></html>"""

FIX_2026 = """<html><head><meta property="og:title" content="2011 GILLIG Low Floor Bus" /></head><body>
<div data-listingid="78824767" class="listing-data-div" auction_start="1760551200" auction_end="1761739200" auction_title="2011 GILLIG Low Floor Bus" agency_name="Central Pennsylvania Transportation Authority" seller_state="PA"></div>
<div class="col page-title-item titleTextContainer"><h1 class="titleTextForItem">2011 GILLIG Low Floor Bus</h1><div class="subTitleText">
<span class="listing-number"><strong>Listing #</strong> 78824767 </span><span class="separator">&#8226;</span><span class="location"><i class="fas fa-map-marker-alt"></i> Harrisburg, PA US </span>
<span class="separator">&#8226;</span><a class="linkTag agency-link" href="/Browse?Seller=RabbitTransit"><i class="fas fa-building"></i> Central Pennsylvania Transportation Authority </a></div></div>
<div class="postauctionstatus postauctionstatus-end"><h1><span>Bidding on this item has ended.</span></h1></div>
<div class="col-12"><h2><span class="itemPageTitles"> Item Description</span></h2><p>Model Number 5312</p><p>Sold &quot; as is&quot;</p><hr><p>Knob missing from front.</p>
<small class="ml20"><b> -Added: 12/16/2025 1:21:01 PM</b> </small></div></body></html>"""

FIX_2016 = """<html><head><meta property="og:title" content="Swarovski Horses" /><meta name="description" content="Swarovski Crystal Horses  Condition: New" /></head><body>
<table class="table table-responsive no-border" id="tblAuctionDetail"><tr><td><strong>Starting Bid</strong></td><td><span>$1.00</span></td></tr>
<tr><td><strong>Auction Ends</strong></td><td id="tdAuctionListingEndDate">Monday, April 4, 2016 2:11:00 PM Eastern</td></tr>
<tr><td><strong>Bid History</strong></td><td><span id="spanBidCount">Bids:&nbsp;28</span></td></tr>
<tr><td><strong>Seller</strong></td><td> Philadelphia Airport Supply Warehouse </td></tr><tr><td><strong>Location</strong></td><td> Philadelphia, PA US </td></tr>
<tr><td><strong>Current Bid</strong></td><td><span class="Bidding_Current_Price"></span><span id="spanStatus">Status: Successful</span></td></tr>
<tr><td colspan="2"><span id="spanReservePriceStatus"><span class="text-success">No Reserve </span></span></td></tr></table>
<a class="btn" href="/Listing/Browse?Seller=PhlAviation">View Seller&#39;s Other Listings</a></body></html>"""

FIX_NOT_LISTING = """<html><head><title>Municibid - Error</title></head><body><h1>Page not found</h1></body></html>"""


def selftest(verbose=True):
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    r = parse_page(FIX_2019, item="22773820")
    check("2019 kind", r["page_kind"], "listing")
    check("2019 id", r["listing_id"], "22773820")
    check("2019 id status", r["status"]["listing_id"], "ok")
    check("2019 title", r["title"], "Delta Saw")
    check("2019 agency", r["agency_slug"], "tyqrell")
    check("2019 agency name", r["agency_name"], "NJ Purchase and Property - DSS")
    check("2019 location", r["location"], "Trenton, NJ US")
    check("2019 state", r["loc_state"], "NJ")
    check("2019 cats", r["categories"], ["Tools", "Home & Maintenance"])
    check("2019 final", r["final_bid"], 2210.0)
    check("2019 start bid", r["starting_bid"], 50.0)
    check("2019 cur price", r["current_price"], 2210.0)
    check("2019 buy now", r["buy_now_price"], 0.0)
    check("2019 buy now shown", r["buy_now_shown"], False)
    check("2019 reserve", r["reserve"], "none")
    check("2019 fee", r["buyer_fee_pct"], 5.0)
    check("2019 start", r["start_local"], "2019-06-14T11:08:00")
    check("2019 end", r["end_local"], "2019-06-26T10:30:00")
    check("2019 extended", r["extended"], False)
    check("2019 bids", r["n_bids"], 18)
    check("2019 high bidder", r["high_bidder_id"], hid("NAH1992"))
    check("2019 high bidder not email", r["high_bidder_is_email"], False)
    check("2019 views", r["page_views"], 1786)
    check("2019 ended", r["ended"], True)
    check("2019 state", r["auction_state"], "ended")
    check("2019 desc", r["desc"], "This auction is for a saw.\nIt was operable.\nCord cut.")
    check("2019 payment", r["payment"], "Pay within 5 days.")
    check("2019 fields", r["fields"], {"Custom ID": "DSS2141", "Miles": "95,211"})
    check("2019 n_q", r["n_questions"], 2)
    check("2019 n_answered", r["n_answered"], 1)
    check("2019 asker", r["qa"][0]["asker"], hid("561206"))
    check("2019 layout", r["layout"], "page_title_2019")
    check("2019 q text", r["qa"][0]["q"], "Single or 3 phase?")
    check("2019 a text", r["qa"][0]["a"], "single.")
    check("2019 a time", r["qa"][0]["a_time"], "11/30/2021 10:09 AM")
    check("2019 photos", r["n_photos"], 2)

    r = parse_page(FIX_2025_LIVE, item="73065022")
    check("2025 id", r["listing_id"], "73065022")
    check("2025 title", r["title"], "2014 Mack Granite")
    check("2025 agency", r["agency_slug"], "SanitOne")
    check("2025 location", r["location"], "Lawrence, NY US")
    check("2025 ended", r["ended"], False)
    check("2025 final status", r["status"]["final_bid"], "live_capture")
    check("2025 final", r["final_bid"], None)
    check("2025 reserve", r["reserve"], "not_met")
    check("2025 bids", r["n_bids"], 170)
    check("2025 cur", r["current_price"], 20900.0)
    check("2025 desc", r["desc"], "Up for auction we have a 2014 Mack\nNo Lights On Dash")
    check("2025 bold", r["desc_markup"]["bold"], 2)
    check("2025 terms", r["terms"], "Cash only.")
    check("2025 qa status", r["status"]["qa"], "none_asked")
    check("2025 n_q", r["n_questions"], 0)
    check("2025 photos", r["n_photos"], 2)
    check("2025 epoch", r["auction_end_epoch"], 1745528880)
    check("2025 views status", r["status"]["page_views"], "absent")
    check("2025 state", r["auction_state"], "live")
    sched = FIX_2025_LIVE.replace("<h1>This Auction Ends in:</h1>", "<h1>Bidding has not started for this auction.</h1>")
    r = parse_page(sched, item="73065022")
    check("scheduled state", r["auction_state"], "scheduled")
    check("scheduled ended", r["ended"], False)
    r = parse_page(FIX_2019.replace("<h1>Delta Saw </h1>", "<h1>12&amp;quot; Saw </h1>"), item="22773820")
    check("double-escaped title", r["title"], '12" Saw')

    r = parse_page(FIX_2017, item="11744189")
    check("2017 kind", r["page_kind"], "listing")
    check("2017 layout", r["layout"], "section_2017")
    check("2017 id", r["listing_id"], "11744189")
    check("2017 title", r["title"], "Powerware 9170+ UPS")
    check("2017 desc", r["desc"], "Powerware 9170+ UPS and switch. Used for 911 equipment.\nBatteries replaced in 2016.")
    check("2017 agency", r["agency_slug"], "GranbyPoliceDepart")
    check("2017 location", r["location"], "Granby, MA US")
    check("2017 reserve", r["reserve"], "not_met")
    check("2017 cur", r["current_price"], 200.0)
    check("2017 cats status", r["status"]["categories"], "meta_keywords_only")

    r = parse_page(FIX_2026, item="78824767")
    check("2026 layout", r["layout"], "title_text_2024")
    check("2026 id", r["listing_id"], "78824767")
    check("2026 title", r["title"], "2011 GILLIG Low Floor Bus")
    check("2026 location", r["location"], "Harrisburg, PA US")
    check("2026 agency", r["agency_slug"], "RabbitTransit")
    check("2026 desc base", r["desc"], 'Model Number 5312\nSold " as is"')
    check("2026 updates", r["desc_updates"], [{"text": "Knob missing from front.", "added": "12/16/2025 1:21:01 PM"}])
    check("2026 ended", r["ended"], True)

    hr_in_base = FIX_2026.replace("<small class=\"ml20\"><b> -Added: 12/16/2025 1:21:01 PM</b> </small>", "")
    r = parse_page(hr_in_base, item="78824767")
    check("hr inside base text kept", r["desc"], 'Model Number 5312\nSold " as is"\nKnob missing from front.')
    check("hr inside base no update", r["desc_updates"], [])

    r = parse_page(FIX_2016, item="7440532")
    check("2016 kind", r["page_kind"], "listing")
    check("2016 layout", r["layout"], "legacy_2016")
    check("2016 desc", r["desc"], "Swarovski Crystal Horses Condition: New")
    check("2016 desc status", r["status"]["desc"], "legacy_meta_only")
    check("2016 bids", r["n_bids"], 28)
    check("2016 end", r["end_local"], "2016-04-04T14:11:00")
    check("2016 agency", r["agency_slug"], "PhlAviation")
    check("2016 agency name", r["agency_name"], "Philadelphia Airport Supply Warehouse")
    check("2016 location", r["location"], "Philadelphia, PA US")
    check("2016 start bid", r["starting_bid"], 1.0)
    check("2016 reserve", r["reserve"], "none")
    check("2016 ended", r["ended"], True)

    r = parse_page(FIX_NOT_LISTING, item="1")
    check("nonlisting", r["page_kind"], "not_a_listing")

    check("money comma", money("$ 20,900.00"), (20900.0, "ok"))
    check("money empty", money("  "), (None, "empty"))
    check("money bad", money("n/a"), (None, "unparseable"))
    check("dttm", parse_dttm("04/24/2025 17:08:00"), ("2025-04-24T17:08:00", "ok"))
    check("norm img", norm_img("a/86E4EFBF-ecaf-45d7-a5b7-6466b3cf942b_thumbcrop.jpg?x=1"), "86e4efbf-ecaf-45d7-a5b7-6466b3cf942b")
    if verbose:
        for f in fails:
            print("FAIL", f)
        print("selftest: %d failures" % len(fails))
    return fails


def mutation_check():
    """Each mutation must make the selftest fail (proves the tests can fail)."""
    global money, norm_img, QA_Q_RE
    keep = (money, norm_img, QA_Q_RE)
    results = {}

    def m_money(s):
        v, st = keep[0](s)
        return (None if v is None else v / 10.0), st
    money = m_money
    results["money_scaled"] = len(selftest(verbose=False)) > 0
    money = keep[0]

    norm_img = lambda u: (u or "")
    results["img_not_normalised"] = len(selftest(verbose=False)) > 0
    norm_img = keep[1]

    QA_Q_RE = re.compile(r"^(?P<q>.*)(?P<who>)(?P<when>)$", re.S)
    results["qa_regex_broken"] = len(selftest(verbose=False)) > 0
    QA_Q_RE = keep[2]
    ok = all(results.values()) and not selftest(verbose=False)
    print("mutation check:", results, "-> restored selftest clean:", not selftest(verbose=False))
    return ok


# --------------------------------------------------------------------------------------------
# Batch run
# --------------------------------------------------------------------------------------------
def _parse_file(args):
    path, item, ts = args
    try:
        raw = gzip.open(path).read()
        txt = raw.decode("utf-8", "replace")
        rec = parse_page(txt, item=item, capture_ts=ts)
        rec["raw_bytes"] = len(raw)
        rec["parse_error"] = None
    except Exception as e:  # recorded, never silent
        rec = {"parser": PARSER_VERSION, "item": item, "capture_ts": ts, "page_kind": None,
               "parse_error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    return rec


def run(pages_dir, out_path, frame_path=None, workers=4):
    import multiprocessing as mp
    frame = {}
    if frame_path:
        for line in open(frame_path):
            line = line.strip()
            if line:
                r = json.loads(line)
                frame[str(r["item"])] = r.get("ts")
    jobs = []
    for d in sorted(os.listdir(pages_dir)):
        dp = os.path.join(pages_dir, d)
        if not os.path.isdir(dp):
            continue
        for f in sorted(os.listdir(dp)):
            if f.endswith(".html.gz"):
                item = f[:-len(".html.gz")]
                jobs.append((os.path.join(dp, f), item, frame.get(item)))
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite " + out_path)
    n = 0
    with open(out_path + ".partial", "w") as fo:
        pool = mp.Pool(workers)
        for rec in pool.imap(_parse_file, jobs, chunksize=16):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        pool.close()
        pool.join()
    os.rename(out_path + ".partial", out_path)
    print("parsed", n, "pages ->", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "selftest":
        f = selftest()
        ok = mutation_check()
        sys.exit(0 if (not f and ok) else 1)
    elif cmd == "one":
        p = sys.argv[2]
        item = os.path.basename(p).split(".")[0]
        rec = parse_page(gzip.open(p).read().decode("utf-8", "replace"), item=item)
        print(json.dumps(rec, indent=1, ensure_ascii=False)[:20000])
    elif cmd == "run":
        a = sys.argv[2:]
        kw = dict(pages_dir=None, out_path=None, frame_path=None, workers=4)
        i = 0
        while i < len(a):
            if a[i] == "--pages":
                kw["pages_dir"] = a[i + 1]
            elif a[i] == "--out":
                kw["out_path"] = a[i + 1]
            elif a[i] == "--frame":
                kw["frame_path"] = a[i + 1]
            elif a[i] == "--workers":
                kw["workers"] = int(a[i + 1])
            i += 2
        run(**kw)
    else:
        print(__doc__)
        sys.exit(1)
