#!/usr/bin/env python3
"""Session 8 text features for Municibid descriptions.

Every feature is computed on two strings per listing:
  full  = base description + seller updates (what a buyer could read at the close);
  spec  = `full` minus boilerplate sentences (a normalised sentence of >= 6 words that appears in
          >= 3 distinct listings anywhere in the corpus: pasted disclaimers, payment/pickup terms).
Feature families (all counts are of DISTINCT matches unless stated):
  size      words, sentences, characters, boilerplate share of words
  facts     numbers; numbers with a unit; condition statements (positive / defect); identifiers
  generic   LLM-marker words (the 2025 excess list, provisional until validated); promotional
            adjectives; second-person address
  polish    misspellings per 100 words (not in /usr/share/dict/linux.words, rare in the corpus,
            one edit from a dictionary word), ALL-CAPS word share, label lines ("Condition: ...")

Usage: python3 textfeat.py PANEL.parquet WORDLIST OUT.parquet
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
NUM = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,]*(?:\.\d+)?")
UNIT = re.compile(r"(?i)\b\d[\d,]*(?:\.\d+)?\s*(?:-|\s)?\s*(miles?|mi|hrs?|hours?|gal(?:lons?)?|lbs?|pounds?|ft|feet|foot|'|\"|inch(?:es)?|in\.?|cc|hp|kw|watts?|w|v|volts?|amps?|psi|gpm|yds?|yards?|tons?|mph|cyl(?:inders?)?|l|liters?|litre|mm|cm|kva|btu|rpm|gb|tb|ghz|seats?|passengers?|axles?|speeds?|cu\.?\s*ft|sq\.?\s*ft|acres?|x)\b")
YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
IDENT = re.compile(r"(?i)\b(vin|serial|s/n|sn|model|mod|part\s*#|asset|unit\s*#|tag)\b")
POS_COND = re.compile(r"(?i)\b(runs?|running|starts?|drives?|driveable|drivable|works?|working|operational|operable|functional|functions|good condition|great condition|excellent condition|fair condition|new tires|new battery|recently (?:serviced|replaced|rebuilt)|rebuilt|replaced|serviced|inspected|clean title|low miles)\b")
DEFECT = re.compile(r"(?i)\b(does not|doesn'?t|won'?t|will not|not working|not run(?:ning)?|no start|needs?|requires?|repairs?|broken|damaged?|cracked?|cracks|leaks?|leaking|rust(?:ed|y)?|rot(?:ted)?|missing|inoperable|non[- ]?operational|for parts|parts only|blown|bad|worn|dents?|dented|scratch(?:es|ed)?|check engine|no title|no keys?|unknown|overheats?|seized|knock(?:ing|s)?|smoke|smokes|tear|torn|stains?|faded|corrosion|frame rot|salvage)\b")
# strict defect statements: excludes need/needs/require(s)/requires/unknown, which LLM text uses in
# generic sales lines ("Whether you need to clear snow ...") - found by reading AI-flagged listings
DEFECT_STRICT = re.compile(r"(?i)\b(does not (?:start|run|work|drive|turn on|function|operate)|doesn'?t (?:start|run|work|drive)|won'?t (?:start|run|work|drive)|will not (?:start|run|work|drive|turn)|not (?:working|running|operational|functional)|non[- ]?(?:operational|working|functional)|no start|inoperable|broken|damaged|cracked|cracks|leaks?|leaking|rust(?:ed|y)?|rot(?:ted)?|missing|for parts|parts only|blown|worn|dents?|dented|scratch(?:es|ed)?|check engine|no title|no keys?|overheats?|seized|locked up|knock(?:ing|s)?|smokes?|torn|tears?|stains?|stained|faded|corrosion|salvage|needs? (?:repair|repairs|work|a new|new|replacement|replacing|to be (?:towed|repaired|replaced))|requires? (?:repair|repairs|work|replacement))\b")
# evaluative claims about THIS item's quality/condition that a buyer cannot check from the text
QUALITY = re.compile(r"(?i)\b(runs? (?:great|well|good|smooth(?:ly)?|strong|excellent|perfect(?:ly)?|like new)|"
                     r"(?:in|is in|remains in) (?:good|great|excellent|very good|solid|working|fine) (?:condition|shape|working order|running order|order)|"
                     r"well[- ](?:maintained|kept|cared[- ]for)|(?:ready|ready to go) (?:to|for) (?:work|go|use|service|its next)|"
                     r"(?:reliable|dependable|trusty|solid) (?:engine|vehicle|truck|machine|unit|performance|choice|option|workhorse)|"
                     r"low[- ](?:mileage|miles|hours|hour)|fully (?:functional|operational|operable)|"
                     r"operates (?:smoothly|well|as (?:it )?should|perfectly)|shifts (?:smoothly|seamlessly)|"
                     r"(?:great|excellent|good) (?:value|deal|opportunity|investment)|like new|mint condition|"
                     r"no (?:known )?(?:issues|problems)|in (?:great|excellent) shape)\b")
# provisional LLM-marker list (2025 excess vocabulary, base share <= 0.12% of 2019-22 listings)
MARK = {"versatile", "spacious", "ample", "withstand", "efficiently", "lasting", "ensures", "ensuring",
        "enhance", "enhanced", "efficient", "longevity", "dependable", "powerful", "optimal", "smoothly",
        "convenient", "durable", "tackle", "reliability", "convenience", "durability", "providing",
        "additionally", "seamlessly", "robust", "boasts", "elevate", "impressive", "elegant", "crafted",
        "essential", "effortlessly", "exceptional", "sleek", "ideal", "perfect", "sturdy"}
PUFF = {"excellent", "great", "perfect", "ideal", "amazing", "beautiful", "nice", "awesome", "pristine",
        "superb", "outstanding", "fantastic", "impressive", "exceptional", "premium", "reliable",
        "durable", "sturdy", "versatile", "powerful", "efficient", "robust", "dependable", "quality",
        "spacious", "comfortable", "convenient", "elegant", "sleek", "stylish", "wonderful", "terrific"}
LABEL_LINE = re.compile(r"(?m)^\s*(?:[-*•]\s*)?[A-Z][A-Za-z0-9 /&()'\-]{1,40}:\s+\S")


def norm_sent(s):
    s = s.lower()
    s = re.sub(r"[0-9]+", "#", s)
    s = re.sub(r"[^a-z# ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_sents(t):
    return [x for x in SENT_SPLIT.split(t) if x.strip()]


def boilerplate_set(texts, min_words=6, min_listings=3):
    cnt = Counter()
    for t in texts:
        cnt.update(set(n for n in (norm_sent(x) for x in split_sents(t)) if len(n.split()) >= min_words))
    return set(s for s, c in cnt.items() if c >= min_listings)


class Speller(object):
    """Misspelling = alphabetic lowercase token of >= 4 letters, not in the word list, seen in at most
    `max_df` listings, and within one deletion of a dictionary word (covers one insertion, deletion,
    transposition or substitution in most cases)."""

    def __init__(self, wordlist_path, corpus_df, max_df=3):
        words = set()
        for w in open(wordlist_path, encoding="utf-8", errors="ignore"):
            w = w.strip().lower()
            if w.isalpha():
                words.add(w)
        self.words = words
        self.corpus_df = corpus_df
        self.max_df = max_df
        self.del1 = {}
        for w in words:
            if 3 <= len(w) <= 18:
                for i in range(len(w)):
                    self.del1.setdefault(w[:i] + w[i + 1:], 1)
        self.cache = {}

    def is_typo(self, w):
        if w in self.cache:
            return self.cache[w]
        r = False
        if len(w) >= 4 and w.isalpha() and w not in self.words and self.corpus_df.get(w, 0) <= self.max_df:
            if w in self.del1:            # w = dictionary word with one letter removed
                r = True
            else:
                for i in range(len(w)):   # w = dictionary word with one letter added/changed
                    d = w[:i] + w[i + 1:]
                    if d in self.words or d in self.del1:
                        r = True
                        break
        self.cache[w] = r
        return r


def features(text, speller):
    toks = TOK.findall(text)
    low = [t.lower() for t in toks]
    nw = len(toks)
    sents = split_sents(text)
    s = set(low)
    nums = set(NUM.findall(text))
    units = set(m.group(0).lower() for m in UNIT.finditer(text))
    pos = set(m.group(0).lower() for m in POS_COND.finditer(text))
    dfc = set(m.group(0).lower() for m in DEFECT.finditer(text))
    typos = [w for w in low if speller.is_typo(w)] if speller is not None else []
    caps = sum(1 for t in toks if len(t) >= 2 and t.isupper())
    return {
        "w": nw, "sent": len(sents), "chars": len(text),
        "n_num": len(nums), "n_unit": len(units), "n_year": len(set(YEAR.findall(text))),
        "n_ident": len(set(m.group(0).lower() for m in IDENT.finditer(text))),
        "n_pos": len(pos), "n_defect": len(dfc),
        "n_quality": len(set(m.group(0).lower() for m in QUALITY.finditer(text))),
        "n_defect_strict": len(set(m.group(0).lower() for m in DEFECT_STRICT.finditer(text))),
        "n_mark": len(s & MARK), "n_puff": len(s & PUFF),
        "n_you": low.count("you") + low.count("your"),
        "n_typo": len(typos), "typo_per100": (100.0 * len(typos) / nw) if nw else np.nan,
        "caps_share": (caps / nw) if nw else np.nan,
        "n_label": len(LABEL_LINE.findall(text)),
        "n_excl": text.count("!"),
        "emdash": text.count("—"),
        "curly": text.count("’") + text.count("“") + text.count("”"),
    }


def main(panel, wordlist, out):
    df = pd.read_parquet(panel)
    df = df[df.desc.notna()].copy()
    df["full"] = df.desc.fillna("") + np.where(df.updates_text.notna(), "\n" + df.updates_text.fillna(""), "")
    boiler = boilerplate_set(df.full.tolist())
    spec = []
    bshare = []
    for t in df.full:
        keep = [x for x in split_sents(t) if norm_sent(x) not in boiler]
        sp = "\n".join(keep)
        spec.append(sp)
        wt = len(TOK.findall(t))
        bshare.append(1.0 - len(TOK.findall(sp)) / wt if wt else np.nan)
    df["spec"] = spec
    df["boiler_share"] = bshare
    cdf = Counter()
    for t in df.spec:
        cdf.update(set(WORD.findall(t.lower())))
    sp = Speller(wordlist, cdf)
    F = [features(t, sp) for t in df.spec]
    G = [features(t, None) for t in df.full]
    f = pd.DataFrame(F, index=df.index).add_prefix("s_")
    g = pd.DataFrame(G, index=df.index)[["w", "chars", "n_num", "n_mark", "n_puff"]].add_prefix("f_")
    res = pd.concat([df[["item", "boiler_share", "spec"]], f, g], axis=1)
    res.to_parquet(out, index=False)
    print("features", res.shape, "boilerplate sentences", len(boiler), "->", out)
    print(res.describe().T.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
