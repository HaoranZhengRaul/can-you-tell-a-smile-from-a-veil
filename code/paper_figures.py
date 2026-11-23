#!/usr/bin/env python3
"""The paper's figures (Sessions 9 to 11), drawn from report_inputs/ only (aggregates; no raw listing text), except
fig_listings and fig_words, whose inputs carry raw listing text and so stay out of the repository.

Only figures that a version of the paper includes are drawn here (Session 11, PI, 2026-11-02: "I just need canon codes
not obsolete ones"). The file as Session 10 left it, with the figures the paper no longer uses (fig_inform,
fig_button_spark, the layout demo of fig_words), is archive/paper_figures_session10.py.

  fig_adoption  all new listings by month (top) and the listings flagged as AI written (bottom: by both tests, by the
                classifier only, by the AI marker words only; the month's 95% Wilson interval in listings); ChatGPT's
                release and the first archived page carrying the AI assistant, whose button is the third row of that
                date's label
  fig_button_share  half the text width: by capture month, the share of archived pages whose code carries the AI
                assistant (button_dating.json), the same button on top
                (both figures cut the button by crop_button from one file, the PI's screenshot of Municibid's help page,
                report_inputs/municibid_help_step3_screenshot.png, each time they are drawn; no other button file)
  fig_listings  two listing pages as buyers saw them, drawn from the banked page's own layout and words
                (code/listing_examples.py): a pre-ChatGPT listing with its misspelling underlined, an assistant-era AI
                written one with its AI marker words shaded; --bg puts the pages on a background (backdrop: one warm grey;
                tint: grey behind the human written page, light green behind the AI written one)
                (Session 12, --after REWRITE_JSON: the left listing's LLM rewrite drawn between the two, a page of
                the same listing marked as never posted)
  fig_text      form and facts per listing: real listings (AI written against human written, same agency, subcategory
                and month) beside the 600 controlled rewrites (the original against our LLM rewrite)
  fig_claims_quality (Session 12) (a) rewrites adding a claim the original does not support, by rewriting model and
                instruction (fidelity_s12.json); (b) the pricing model's lift by third of item quality (quality_s12.json)
  fig_buyers    buyer outcomes in their own units: assistant-era human written listings at their level, AI written
                listings moved by their gap against the same agency's other listings (95% interval) and, in the two
                price panels, the expected price before ChatGPT
  fig_texas     the closing price as a share of the Texas price: over time, every vehicle by group, and cumulative
  fig_questions what buyers ask before the close, AI written against human written listings (the blind coding)
  fig_price_gap if the human written listing closes at $1,000, what the AI written one closes at, and the expected
                price before ChatGPT given the same text
  fig_text_rank (appendix) how many times as much the fifth of listings the ranking model ranks highest sells for as
                the fifth it ranks lowest, by period
  fig_spelling  (appendix) the share of new listings with at least one misspelling, by start quarter
  fig_marker_words (appendix) each of the 38 AI marker words' share of listings, by period
  fig_test_distributions (appendix) the two tests by period: listings by their number of AI marker words; the
                classifier's score
  fig_words     (the main version's appendix) what each detector sees, word by word, in the paper's example listings:
                our classifier (phrase weight x tf-idf), Fast-DetectGPT and Binoculars (per-token pieces)
One input made once and kept for the record: texas_shares (texas_shares.json from job 887835's points).
Every input is required (PI, 2026-11-02: "absolutely no need to be fallback"): a missing file, a missing count or an
example without its detector tokens stops the run; no figure is skipped or drawn from an older input.
Colours: before ChatGPT light grey, chatbots only mid grey, the assistant era green; human written dark grey, AI written
green with white stripes; our LLM rewrites blue, their originals light grey; the expected price before ChatGPT teal.
Figure 1 keeps its own: all new listings purple; flagged by both tests dark green, by the classifier only yellow, by the
AI marker words only light blue, each striped; the AI assistant's date orange, ChatGPT's dark grey. fig_words shades words
green toward AI and grey toward human.
Usage: python3 paper_figures.py REPORT_INPUTS_DIR FIG_OUT_DIR PREVIEW_DIR
       python3 paper_figures.py --words WORD_VIEWS_JSON TOKENS_JSON FIG_OUT_DIR PREVIEW_DIR
       python3 paper_figures.py --listings LISTINGS_JSON LEFT_ITEM RIGHT_ITEM FIG_OUT_DIR PREVIEW_DIR [NAME]
               [--cut ITEM:i,j,...] [--bg backdrop|tint]
       python3 paper_figures.py --texas-shares POINTS_JSON OUT_JSON    (job 887835's points to texas_shares.json)
"""
import re
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.lines  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402

ORANGE = "#eb6834"  # the AI assistant's date line
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
GRAY = "#898781"  # neutral bars (human written, typical listing): 3.5:1 on white
PURPLE = "#9a8bd6"  # Figure 1, all new listings: 2.98:1 on white, named by its panel subtitle
# Figure 1's flagged listings in three parts (PI, 2026-10-28: three blues; red, orange and yellow; blue, yellow and
# green; then 2026-10-29: both tests green, "the other 2 is yellow and blue"; then "lighter blue and darker green"):
# both tests dark green, the classifier only yellow, the AI marker words only a lighter blue (blue ramp step 350; the
# light blue that marks AI marker words in Figure 2). In stack order from the base (green, yellow, blue) the set passes
# validate_palette.js on white (CVD 19.6, normal vision 31.2, lightness band), yellow 2.17:1 against white (relief:
# the legend and Table 1); grey levels 0.14, 0.43, 0.30, each part apart from its neighbours in print
FLAG_BOTH, FLAG_CLF, FLAG_LEX = "#137a3d", "#eda100", "#5598e7"
# the dates: the assistant orange again (no bar is orange now); ChatGPT a dark grey dash, since green is now a bar colour
TOOL_LINE, CHAT_LINE = ORANGE, INK2
# non-AI written and AI written listings, one pair of colours in every figure (PI, 2026-10-29: "the non written in
# black written in the dark green"): AI written is Figure 1's dark green, the part of its bars that both tests flag.
# validate_palette.js on white: CVD separation 26.5, normal vision 28.9; black fails the lightness band and the chroma
# floor, as an achromatic colour must. Tints of the pair: the page panels of Figure 2 and the word shading of the
# appendix (grey toward human, green toward AI)
# Human written (the not flagged group) a softer black from 2026-10-31 (PI: "make not flag ... not entirely black, maybe
# like a mellow version of black, so it's not that deep"; the black #222222 until then). #4a4946 is the lightest grey
# that keeps apart from the AI written green: validate_palette.js CVD separation 11.7 (deutan), normal vision 16.4;
# mid greys fail (#6b6965: 5.0 and 12.8, the green's own lightness). White values on it 9.0:1
NONAI, AIW = "#4a4946", FLAG_BOTH
# the expected price before ChatGPT (the form reader's price) in the buyer-outcomes and price-gap figures: a bluish
# teal (PI, 2026-10-31: "maybe like a bluish teal for that", in place of the orange #e8833a it was until then).
# validate_palette.js beside the not flagged black and the AI written green: CVD separation from the green 17.2
# (deutan; tritan 14.1), normal vision 18.1, 3.23:1 on white; values on it print in dark ink (5.4:1)
FORM = "#2a9cb3"
# Scheme B, the PI's choice from the colour-scheme gallery (Session 10, 2026-10-31: "the black, the green, and the gray
# ... is probably the best"; the pre-AI price filled orange, "I don't like the empty color"; the gallery's other schemes
# and the outlined price were cut in Session 11 as obsolete): before ChatGPT light grey, chatbots only mid grey, the
# assistant era's listings not flagged black and its AI written listings green with stripes; a whole-era green one hex
# off AIW so it is never striped. Figure 1 keeps its own colours (PI: "I did like my original color scheme for figure one")
P0_COL, P1_COL, P2_ERA = "#c9c7c1", "#8f8d88", "#137a3e"
# the assistant era's listings the detector does not flag are "human written" (PI, 2026-10-31: "the method is not
# perfect we are frank on that one could argue the AI writtne is not ground truth AI written as well so Human
# written"); both names are the detector's verdicts, defined so in the text. "Not flagged" until then
BASE_COL, BASE_TXT, BASE_LAB, BASE_LAB_FLAT = NONAI, "white", "Human-\nwritten", "Assistant era, human-written"
# the teal bar is the benchmark, the frozen ridge regression's reading of AI text, set apart from the observed bars behind
# a thin line and named under it (PI, 2026-10-31: "I like the benchmark ... ridge regression prediction ... I like that
# little line between"; of the name's place, "up top and lower ... lower is probably better"); "If priced as before AI"
# and then "Predicted from the text" until then
# named for what it is (PI, 2026-10-31: "Just say, expected price ... before ... ChatGPT ... we'll have to be
# deliberate about it"); "Benchmark (ridge reg.)" for one push before that
# one wording per figure, each exact for its bar (PI, 2026-10-31: "given the same text on the new figure, given an LLM
# rewrite on Figure 5"): here the human written price raised by the premium for an LLM rewrite of the same item
AI_LAB, FORM_LAB = "AI-\nwritten", "With the\npre-ChatGPT\npremium"
# the price-gap figure's teal bar: the frozen model's price for the AI written listings' own text
FORM_LAB_SAME = "Predicted\nby the\npricing\nmodel"
# three bars of one width in every panel's three slots: the two observed bars close together, the benchmark beyond a gap
# with a thin line in it
BAR_XS_APART, DIVIDER_X = (-0.05, 0.85, 2.05), 1.45
EXP_BASE_COL, EXP_BASE_LAB = P0_COL, "Original, before ChatGPT"
# the experiment's rewrites in a colour of their own (PI, 2026-10-31, of Figure 4: the rewrites "we created ... we know
# for a fact that's causal. The other one is we flag, so that's different"): "AI written" names only listings the
# detector flags, drawn green and striped; the rewrites we made are named as ours and drawn solid in this colour.
# validate_palette.js beside the originals' grey, the AI written green and the not flagged black: CVD separation from
# the green 22.7 (deutan), normal vision 24.1
REWRITE_COL, REWRITE_LAB = "#2a78d6", "Our LLM rewrite"
PERIOD_BAR_COLS = {"P0": P0_COL, "P1": P1_COL, "P2": P2_ERA}
Z = 1.959964
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
W = 5.5  # NeurIPS text width, inches

# Arial first: macOS ships Helvetica as one .ttc file and matplotlib reads only its regular face, so every "bold" in
# the figures printed regular until 2026-10-28 (pdffonts listed Helvetica alone); Arial has the same letter widths
# and a separate bold file
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "hatch.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# a solid green backing for a value printed on a hatched AI written bar, so the stripes do not cross the digits
SOLID_BACK = dict(boxstyle="square,pad=0.12", facecolor=FLAG_BOTH, edgecolor="none")


def hatch_ai(fig):
    """Print in black and white (PI, 2026-10-31: "make sure if it's printed, it's monotone, that people can read it"):
    every AI written bar (the dark green) gets a fine white diagonal hatch, so it stays apart from the black non-AI
    written bars in grey; legend patches too. In grey the two solid colours print at 13% and 33%."""
    for ax in fig.axes:
        arts = list(ax.patches)
        leg = ax.get_legend()
        if leg is not None:
            arts += list(leg.get_patches())
        for pt in arts:
            if matplotlib.colors.to_hex(pt.get_facecolor()) == AIW.lower():
                pt.set_hatch("////")
                pt.set_edgecolor("white")
                pt.set_linewidth(0)
    for leg in fig.legends:
        for pt in leg.get_patches():
            if matplotlib.colors.to_hex(pt.get_facecolor()) == AIW.lower():
                pt.set_hatch("////")
                pt.set_edgecolor("white")
                pt.set_linewidth(0)


def save(fig, name, out, prev, pdf_dpi=None):
    # pdf_dpi: the resolution at which matplotlib resamples any picture in the figure; left at the figure's 100 dpi it
    # shrank the 336-pixel button to 95 pixels in Figure 1's PDF until 2026-10-28 (pdfimages -list)
    if name not in ("fig_adoption", "fig_listings", "fig_words"):
        hatch_ai(fig)
    if name == "fig_adoption":
        # Figure 1 striped throughout (PI, 2026-10-31: "why is figure 1's yellow and green not in the stripe version, only
        # blue is? Let's make it consistent"): every flagged segment carries a white stripe, one direction per test, so
        # the three still part in grey: AI marker words only one diagonal, classifier only the other, both tests both.
        # The bars keep their own thin white outline (lw 0.25, set where they are drawn), which parts one segment from
        # the next; set to 0 here from 7247527 it ran the segments together (PI, 2026-10-31: "why did figure one's bar
        # disappear? Now you just blob of color altogether ... or space like the old one did")
        hatches = {FLAG_LEX.lower(): "////", FLAG_CLF.lower(): "\\\\\\\\", FLAG_BOTH.lower(): "xxxx"}
        for ax in fig.axes:
            leg_pts = list(ax.get_legend().get_patches()) if ax.get_legend() is not None else []
            for pt in list(ax.patches) + leg_pts:
                h_ = hatches.get(matplotlib.colors.to_hex(pt.get_facecolor()))
                if h_:
                    pt.set_hatch(h_)
                    pt.set_edgecolor("white")
                    if pt in leg_pts:
                        pt.set_linewidth(0)
    fig.savefig(os.path.join(out, name + ".pdf"), **({"dpi": pdf_dpi} if pdf_dpi else {}))
    fig.savefig(os.path.join(prev, name + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


def month_range(a, b):
    y, m, ks = int(a[:4]), int(a[5:7]), []
    while "%04d-%02d" % (y, m) <= b:
        ks.append("%04d-%02d" % (y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return ks


def over_bar(ax, xs, ys, bars=(), **kw):
    """A line in secondary ink, drawn again in white inside each bar patch in `bars` (clipped to it), so that it
    shows on a black or dark green bar as well as on the page."""
    ax.plot(xs, ys, color=INK2, **kw)
    for p in bars:
        ln, = ax.plot(xs, ys, color="white", **kw)
        ln.set_clip_path(p)


def whisker(ax, lo, hi, y, cap=0.09, bar=None):
    """95% interval: a thin line with end caps, drawn over the bar it belongs to (white inside the bar)."""
    kw = dict(lw=0.6, solid_capstyle="butt", zorder=4, bars=[bar] if bar is not None else [])
    over_bar(ax, [lo, hi], [y, y], **kw)
    for x in (lo, hi):
        over_bar(ax, [x, x], [y - cap, y + cap], **kw)


def wilson(k, n):
    """95% Wilson interval for a share of k in n."""
    p, d = k / n, 1 + Z * Z / n
    c, h = (p + Z * Z / (2 * n)) / d, Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(c - h, 0.0), c + h


BTN_W = 0.75  # inches (PI, 2026-10-28: "can be 50% smaller" than 1.25, so 0.625; 2026-10-31: "a little bit bigger
#   ... 20% bigger. Same position", so 0.75, anchored at the same top-right corner); kept at 600 ppi in the PDF (save)
# the button's one source (PI, 2026-11-02: "that will be our source file for cropping ... absolutely no need to be
# fallback"): the PI's screenshot of the help page (2026-10-28, 1406 x 104 px; moved from the Desktop into
# report_inputs/), cut to the button (336 x 72 px) by crop_button every time Figure 1 or the dating graph is drawn
BTN_SRC = "municibid_help_step3_screenshot.png"
BTN_W_SMALL = BTN_W   # inches: the button on top of the half-width dating graph, the same size as in Figure 1


def fig_adoption(R, PN, K, out, prev):
    """All new listings (top) and listings flagged as AI written (bottom: by both tests, the AI-text classifier and 2+
    AI marker words, by the classifier only and by the AI marker words only), with the 95% Wilson interval of the month's
    flagged share drawn in listings (explained in the caption, not the legend): chance in
    which listings were archived, not detector error. One slot per month from the first to the last month with at
    least 10 listings (a thinner month inside the range would leave an empty slot). Every label hangs from the top of
    the lower panel, whose limit is the smallest that clears the bars and bands beneath each label."""
    assert PN, "fig_adoption needs paper_numbers.json (monthly counts under all three rules)"
    M = PN["monthly"]
    full = sorted(k for k in M["union"] if M["union"][k]["n"] >= 10)
    keys = month_range(full[0], full[-1])
    global FIG1_MONTHS
    FIG1_MONTHS = keys
    print("adoption: %s to %s; months inside the range not drawn (fewer than 10 listings): %s" % (
        full[0], full[-1], [k for k in keys if k not in full]))
    C, CI = {}, {}
    for k in full:
        n = M["union"][k]["n"]
        assert n == M["lex"][k]["n"] == M["c1"][k]["n"], "month %s: the rules disagree on n" % k
        u, l, c = (round(M[r][k]["ai"] * n) for r in ("union", "lex", "c1"))
        b = l + c - u
        C[k] = (n, u, b, c - b, l - b)                # listings; flagged by either; by both; classifier only; words only
        assert min(C[k][2:]) >= 0 and sum(C[k][2:]) == u, ("month %s: the three parts do not add up" % k, C[k])
        lo, hi = wilson(u, n)
        CI[k] = (n * lo, n * hi)
    xs = {k: i for i, k in enumerate(keys)}
    H, top, bottom, ratio, gap = 2.35, 0.925, 0.09, 1.75, 0.40
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, H), sharex=True, gridspec_kw={"height_ratios": [1, ratio], "hspace": gap})
    fig.subplots_adjust(left=0.095, right=0.995, top=top, bottom=bottom)   # fills the text width, so it prints at 100%
    a2_h = H * (top - bottom) * ratio / (1 + ratio + gap)                   # the lower panel's height, inches
    years = sorted({k[:4] for k in keys})
    for a in (a1, a2):
        a.set_xlim(-0.9, len(keys) - 0.1)
        a.grid(axis="y", color=GRID, lw=0.5)
        a.set_axisbelow(True)
        a.set_xticks([(xs[min(k for k in keys if k[:4] == y)] + xs[max(k for k in keys if k[:4] == y)]) / 2 for y in years])
        a.set_xticklabels(years if a is a2 else [])
        a.tick_params(axis="x", which="major", length=0, pad=3)
        a.set_xticks([xs[y + "-01"] - 0.5 for y in years if y + "-01" in xs], minor=True)
        a.tick_params(axis="x", which="minor", length=3, width=0.6, color=MUTED)
    for k in full:
        a1.bar(xs[k], C[k][0], width=0.8, color=PURPLE, lw=0, zorder=3)
    a1.set_ylim(0, max(C[k][0] for k in full) * 1.08)
    # the interval behind each bar, as wide as the bar (PI, 2026-10-29: "why is the grey interval bars wider than the
    # other bar?"; until then each piece spanned the whole month, 1.0 against the bars' 0.8)
    for k in full:
        a2.fill_between([xs[k] - 0.4, xs[k] + 0.4], [CI[k][0]] * 2, [CI[k][1]] * 2, color=GRAY, alpha=0.28, lw=0, zorder=2)
    # both tests at the base of each bar and in the legend's third row (PI, 2026-10-29: "make the both test show on
    # the bar graph as the lower portion not top"); from the base: both tests, the classifier only, the AI marker words only
    for k in full:
        base = 0
        for v, col in zip((C[k][2], C[k][3], C[k][4]), (FLAG_BOTH, FLAG_CLF, FLAG_LEX)):
            if v:
                a2.bar(xs[k], v, bottom=base, width=0.8, color=col, edgecolor="white", lw=0.25, zorder=3)
                base += v
    x_chat, x_tool = xs["2022-11"] + 0.5, xs["2025-01"] + 0.5   # 30 Nov 2022 and 29 Jan 2025 sit at month ends
    chat_kw, tool_kw = dict(color=CHAT_LINE, lw=1.2, ls=(0, (3, 1.6))), dict(color=TOOL_LINE, lw=1.4)
    pk = max(full, key=lambda k: C[k][1] / C[k][0])
    # the two dates hang from the top of the lower panel, 2.4 months left of their lines, each with a short sloped
    # pointer to its line (PI, 2026-10-28: "a small tail pointer ... like the peak to the break lines and dont be
    # parral lines to the x axist ata small slope"); the assistant's label reads "Municibid adds" / the button /
    # "optional AI writing assistant" / "(January 2025)" (PI: the button "the second row, not the third row")
    gap_m = 2.4
    labs = [(x_chat - gap_m, "right", "ChatGPT released\n(November 2022)", (x_chat - 13, x_chat))]
    tool_top, tool_bot, tool_span = "Municibid adds", "optional AI writing assistant\n(January 2025)", (x_tool - 21, x_tool)
    frac = lambda t: (t.count("\n") + 1) * 6.4 * 1.15 / 72 / a2_h
    under = lambda a, b: max([CI[k][1] for k in full if a <= xs[k] <= b] + [0])
    # Municibid's own button (help page "Selling Tutorial", dated 17 Sep 2026; cut from the PI's screenshot, BTN_SRC);
    # its height counts toward the room its label needs. Until 2026-10-28 the zoom used the screen's 100 dpi where the
    # PDF counts 72 points per inch, so the button printed 1.39 times the width meant (its shape unchanged), and the
    # PDF kept it at 100 ppi (save's pdf_dpi)
    btn = crop_button(os.path.join(INP, BTN_SRC))
    print("adoption: button cut from %s, %d x %d px" % (BTN_SRC, btn.shape[1], btn.shape[0]))
    btn_w = BTN_W                                                            # inches on the page
    btn_gap = 0.015                                                          # share of the panel, above and below it
    btn_frac = btn_w * btn.shape[0] / btn.shape[1] / a2_h + 2 * btn_gap
    tool_frac = frac(tool_top) + btn_frac + frac(tool_bot)                  # the whole label, button included
    # the peak, pointed at by a short line rising at a small slope from the bar's top to the label's first row, the
    # label in the free space to the right (PI, 2026-10-28: "a small line pointing at the peak", up to three rows)
    pk_t = "Peak:\n%d of %d\n%s %s" % (C[pk][1], C[pk][0], MONTH_NAMES[int(pk[5:7]) - 1], pk[:4])
    pk_fs = 6.2                        # a touch under the other labels: the three rows fill the space to the right
    row = pk_fs * 1.15 / 72 / a2_h                                          # one text row, share of the panel
    y_p0 = C[pk][1] + 1.0                                                   # listings: just above the bar's top
    y_p1 = y_p0 + 4.0                                                       # the label's first row
    assert y_p0 > under(xs[pk] + 1, xs[pk] + 1) + 1.0, "the peak pointer would cross the next month's band"
    x_lead0, x_lab = xs[pk] + 0.42, xs[pk] + 1.95                           # the bar's right edge; clear of the next band
    print("adoption: peak %s, %d of %d flagged; pointer from %.1f to %.1f listings" % (pk, C[pk][1], C[pk][0], y_p0, y_p1))
    ytop = max([under(*span) / (0.985 - frac(t) - 0.03) for _, _, t, span in labs]
               + [under(*tool_span) / (0.985 - tool_frac - 0.03), 1.04 * max(CI[k][1] for k in full), y_p1 / (0.985 - 0.6 * row)])
    a2.set_ylim(0, ytop)
    assert y_p1 - 2.6 * row * ytop > under(xs[pk] + 2, xs[pk] + 8) + 2, "the peak label would sit on the bars beneath it"
    a2.plot([x_lead0, x_lab - 0.15], [y_p0, y_p1], color=INK2, lw=0.6, zorder=6, solid_capstyle="butt")
    t_pk = a2.text(x_lab, y_p1 + 0.5 * row * ytop, pk_t, ha="left", va="top", fontsize=pk_fs, color=INK, linespacing=1.15,
                   zorder=6)
    fig.canvas.draw()
    bb = t_pk.get_window_extent().transformed(a2.transData.inverted())
    print("adoption: peak label spans months %.2f to %.2f past the peak; the axis ends at %.2f" % (
        bb.x0 - xs[pk], bb.x1 - xs[pk], len(keys) - 0.1 - xs[pk]))
    assert bb.x1 <= len(keys) - 0.1 + 0.3, "the peak label runs past the plot"
    for x, ha, t, _ in labs:
        a2.text(x, 0.985 * ytop, t, ha=ha, va="top", fontsize=6.4, color=INK, linespacing=1.15, zorder=6)
    a2.text(x_tool - gap_m, 0.985 * ytop, tool_top, ha="right", va="top", fontsize=6.4, color=INK, linespacing=1.15, zorder=6)
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    y_btn = (0.985 - frac(tool_top) - btn_gap) * ytop
    # OffsetImage sizes an image in points (zoom x its pixels), so btn_w inches is a zoom of btn_w x 72 / pixels
    a2.add_artist(AnnotationBbox(OffsetImage(btn, zoom=btn_w * 72.0 / btn.shape[1]), (x_tool - gap_m, y_btn),
                                 box_alignment=(1, 1), frameon=False, zorder=6))
    t_bot = a2.text(x_tool - gap_m, (0.985 - frac(tool_top) - btn_frac) * ytop, tool_bot, ha="right", va="top", fontsize=6.4,
                    color=INK, linespacing=1.15, zorder=6)
    # each date label's pointer: from just past its first row's end down a small slope to its line
    y_a = (0.985 - 0.45 * frac("x")) * ytop
    for x_line in (x_chat, x_tool):
        a2.plot([x_line - gap_m + 0.3, x_line - 0.25], [y_a, y_a - 5.0], color=INK2, lw=0.6, zorder=6, solid_capstyle="butt")
    fig.canvas.draw()
    bb_bot = t_bot.get_window_extent().transformed(a2.transData.inverted())
    x_btn0 = x_tool - gap_m - btn_w / ((0.995 - 0.095) * W / (len(keys) - 0.1 + 0.9))
    print("adoption: assistant label from month %.1f (button from %.1f); the ChatGPT line is at %.1f" % (
        bb_bot.x0 - x_tool, x_btn0 - x_tool, x_chat - x_tool))
    assert min(bb_bot.x0, x_btn0) > x_chat + 1.0, "the assistant's label reaches the ChatGPT line"
    # the legend lists the parts top to bottom as they are stacked, so both tests is its third row, named alone (PI,
    # 2026-10-29: "just both test no ()"; the other two rows name the tests), between the 50 and 100 gridlines (PI,
    # 2026-10-28: "a little bit upper ... in between those two lines"); the 95% band is in the caption
    a2.set_yticks([0, 50, 100])
    assert ytop > 110, ytop
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FLAG_LEX, FLAG_CLF, FLAG_BOTH)]
    a2.legend(hs, ["Marker test only", "Classifier only", "Both tests"],
              loc="center left", bbox_to_anchor=(0.0, 75.0 / ytop), frameon=False, handlelength=1.0, handleheight=0.9,
              labelspacing=0.35, borderpad=0.2, fontsize=6.6)
    # no "per month" in the titles: the monthly axis says it (PI, 2026-10-28)
    a1.set_title("Total new listings", loc="left", fontsize=7.2, color=INK, fontweight="bold", pad=3)
    a2.set_title("Flagged listings", loc="left", fontsize=7.2, color=INK, fontweight="bold", pad=3)
    a1.set_ylabel("Listings")
    a2.set_ylabel("Listings")
    fig.align_ylabels([a1, a2])
    # the date lines: one line each from the lower panel's foot to the upper panel's top, through the gap between the
    # panels (PI, 2026-10-28: "connect the break / threshold line above n below")
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    y0, y1 = inv.transform(a2.transAxes.transform((0, 0)))[1], inv.transform(a1.transAxes.transform((0, 1)))[1]
    for x, kw in ((x_chat, chat_kw), (x_tool, tool_kw)):
        xf = inv.transform(a2.transData.transform((x, 0)))[0]
        assert abs(xf - inv.transform(a1.transData.transform((x, 0)))[0]) < 1e-9, "the panels' x axes disagree"
        fig.add_artist(Line2D([xf, xf], [y0, y1], transform=fig.transFigure, zorder=5, **kw))
    save(fig, "fig_adoption", out, prev, pdf_dpi=600)


ROWS_TEXT = [("Presentation", None), ("Words", "s_w"), ("“Label: text” lines", "s_n_label"),
             ("Persuasive content", None), ("Promotional words", "s_n_puff"), ("“you” / “your”", "s_n_you"),
             ("Any quality phrase", "has_quality"), ("Defect stated, item praised", "defect_and_praise"),
             ("Informative content", None), ("Numbers", "s_n_num"), ("Numbers with units", "s_n_unit"),
             ("Defect statements", "s_n_defect_strict")]
REWRITE_KEY = {"s_w": "w", "s_n_puff": "n_puff", "s_n_you": "n_you", "s_n_label": "n_label", "s_n_num": "n_num", "s_n_unit": "n_unit",
               "s_n_defect_strict": "n_defect_strict", "has_quality": "has_quality", "defect_and_praise": "defect_and_praise"}


PCT_ROWS = {"has_quality", "defect_and_praise"}


def fmt_level(v, pct):
    if pct:
        return "%.0f%%" % (100 * v)
    return "%.0f" % v if v >= 10 else ("%.1f" % v if v >= 1 else "%.2f" % v)


# what each group of rows measures, printed in its band (PI, 2026-10-31: "the facts and claim a form looks exactly the
# same. You want to like look a bit different")
GROUP_NOTES = {"Presentation": "how the listing reads", "Persuasive content": "promotional language",
               "Informative content": "what a buyer can check"}


def fig_text(R, PN, out, prev):
    """Levels, not ratios (PI, 2026-10-26). Field (left): the human mean and the human mean plus the gap within
    agency x subcategory x month (main rule), with 95% interval. Experiment (right): the originals' mean and the
    rewrites' mean, with the paired 95% interval. Each row has its own scale, set by its largest value or interval end;
    every value is printed. Session 10 (PI, 2026-10-31: "the words appear the both the Y scale"): the row labels on both
    panels, and each group (form, facts, claims) a tinted band across both panels with a line on what it measures."""
    assert PN, "fig_text needs paper_numbers.json (the controlled rewrites)"
    c, rw = R["rules"]["union"]["content"], PN["rewrites"]["all"]
    LBL = -0.74  # left edge of the row-label column, in axes widths
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(W, 3.15), sharey=True, gridspec_kw={"wspace": 0.78})
    fig.subplots_adjust(left=0.205, right=0.975, top=0.855, bottom=0.015)
    y, ticks, heads = 0.0, [], []
    for lab, key in ROWS_TEXT:
        if key is None:
            heads.append((y, lab))
            y -= 0.9
            continue
        pct, e, q = key in PCT_ROWS, c[key], rw[REWRITE_KEY[key]]
        hm = e["human_mean"]
        fld = (hm, hm + e["b"], hm + e["b"] - Z * e["se"], hm + e["b"] + Z * e["se"])
        exp_ = (q["base"], q["base"] + q["diff"], q["base"] + q["lo"], q["base"] + q["hi"])
        top = max(fld[0], fld[3], exp_[0], exp_[3])
        for ax, (h, a, lo, hi), bcol, acol in ((aL, fld, BASE_COL, AIW), (aR, exp_, EXP_BASE_COL, REWRITE_COL)):
            ax.barh(y + 0.2, h / top, height=0.36, color=bcol, lw=0, zorder=3)
            bar = ax.barh(y - 0.2, a / top, height=0.36, color=acol, lw=0, zorder=3).patches[0]
            whisker(ax, max(lo, 0) / top, hi / top, y - 0.2, bar=bar)
            ax.text(h / top + 0.03, y + 0.2, fmt_level(h, pct), ha="left", va="center", fontsize=6.0, color=INK2)
            ax.text(hi / top + 0.04, y - 0.2, fmt_level(a, pct), ha="left", va="center", fontsize=6.0, color=INK)
        ticks.append((y, lab))
        y -= 1.0
    for ax, title in ((aL, "Real listings: AI-written against human-written,\nseller and subcategory-by-month fixed effects"),
                      (aR, "Our experiment: 600 pre-ChatGPT listings, each\nagainst the LLM rewrite we made of it")):
        ax.set_xlim(0, 1.3)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=True, labelsize=6.6)
        ax.set_yticks([p for p, _ in ticks])
        ax.set_yticklabels([t for _, t in ticks])
        # left-aligned with the band (PI, 2026-10-31: "the title is leaking to the right side ... Make it left align")
        ax.text(LBL, 1.0, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, color=INK)
        for p, h in heads:
            # the band runs under the row labels too: from the label column's left edge (LBL, in axes widths) to the
            # panel's right edge, so a group header spans its whole panel
            ax.add_patch(plt.Rectangle((LBL, p - 0.36), 1 - LBL, 0.72, transform=ax.get_yaxis_transform(), color=BACKDROP,
                                       lw=0, zorder=0, clip_on=False))
            ax.text(LBL + 0.02, p, "%s: %s" % (h.upper(), GROUP_NOTES[h]), transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=6.3, color=INK, fontweight="bold", zorder=4, clip_on=False)
    aL.set_ylim(y + 0.45, 0.6)
    # one legend over each panel, since the two panels compare different things: the flagged listings against those not
    # flagged (left), the originals against our rewrites of them (right); each pair in time order
    for ax, items in ((aL, [(BASE_COL, "Human-written"), (AIW, "AI-written")]),
                      (aR, [(EXP_BASE_COL, EXP_BASE_LAB), (REWRITE_COL, REWRITE_LAB)])):
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c_) for c_, _ in items], [l_ for _, l_ in items], loc="lower left",
                  bbox_to_anchor=(LBL, 1.105), ncol=2, frameon=False, handlelength=1.0, handleheight=0.9,
                  columnspacing=1.2, borderaxespad=0, borderpad=0, fontsize=6.6)
    save(fig, "fig_text", out, prev)


BUYER_ROWS = [("Received a bid (%)", None, "pooled_sold", "share"), ("Bids per listing", None, "pooled_log_bids", "log1p"),
              ("Page views", None, "pooled_log_views", "log1p"), ("Buyer questions per listing", None, "pooled_n_q_before_end", "lin"),
              ("Vehicle price", "same make, model, age, mileage", "veh_pooled", "price"),
              ("Price, all goods", "same opening bid", "all_pooled_start", "price")]


def buyer_levels(e, kind, bench):
    """(non-AI written, AI written, 95% low, 95% high, the form reader's price or None) in the outcome's own units.
    Non-AI written: tool-era listings the rule does not flag (price_hedonics.py, ref_level): their mean (shares,
    questions), exp(mean of log(1 + y)) - 1 (bids, views), median closing price (prices). AI written: that level moved
    by the estimated gap."""
    assert "ref" in e, "hedonic JSON without the non-AI written levels: rerun price_hedonics.py (Session 9)"
    b, se, r = e["b"], e["se"], e["ref"]
    if kind in ("share", "lin"):
        s = 100 if kind == "share" else 1
        t = r["mean"] * s
        return t, t + b * s, t + (b - Z * se) * s, t + (b + Z * se) * s, None
    if kind == "log1p":
        g = math.exp(r["mean"])
        return g - 1, g * math.exp(b) - 1, g * math.exp(b - Z * se) - 1, g * math.exp(b + Z * se) - 1, None
    m = r["median_price"]
    return m, m * math.exp(b), m * math.exp(b - Z * se), m * math.exp(b + Z * se), m * math.exp(bench)


def fmt_buyer(v, kind, key):
    if kind == "share":
        return "%.1f%%" % v
    if kind == "price":
        return "\\${:,.0f}".format(v)
    return "{:,.0f}".format(v) if key == "pooled_log_views" else ("%.2f" % v if kind == "lin" else "%.1f" % v)


def fig_buyers(H, K, out, prev):
    """Six panels, three per row (PI, 2026-10-27): vertical bars from zero on each outcome's own scale, the value of
    every bar printed inside it on one line, the 95% interval on the AI written bar and a thin dashed line at the
    non-AI written level. Every panel spans three bar slots, so two-bar and three-bar panels draw bars of one width."""
    bench = K["benchmark_rewrites"]["b"]
    fig, axes = plt.subplots(2, 3, figsize=(W, 2.65), gridspec_kw={"hspace": 0.68, "wspace": 0.42})
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.10)
    for ax, (title, sub, key, kind) in zip(axes.flat, BUYER_ROWS):
        base, ai, lo, hi, form = buyer_levels(H["union"][key], kind, bench)
        # (level, bar colour, value colour, name): white values on the black and green bars, ink on the light orange
        bars = [(base, BASE_COL, BASE_TXT, BASE_LAB), (ai, AIW, "white", AI_LAB)] + (
            [(form, FORM, INK, FORM_LAB)] if form else [])
        xs = list(BAR_XS_APART) if form else [0.5, 1.5]
        top = max(hi, form or 0) * 1.12
        if form:
            ax.axvline(DIVIDER_X, color=MUTED, lw=0.6, zorder=2)
        pt = []
        for x, (v, col, tc, _) in zip(xs, bars):
            pt.append(ax.bar(x, v, width=0.8, color=col, lw=0, zorder=3).patches[0])
            ax.text(x, 0.05 * top, fmt_buyer(v, kind, key), ha="center", va="bottom", fontsize=6.8, color=tc,
                    fontweight="bold", zorder=6, bbox=SOLID_BACK if col == AIW else None)
        xa = xs[1]
        over_bar(ax, [xa, xa], [lo, hi], bars=[pt[1]], lw=0.7, zorder=4, solid_capstyle="butt")
        for yv in (lo, hi):
            over_bar(ax, [xa - 0.11, xa + 0.11], [yv, yv], bars=[pt[1]], lw=0.7, zorder=4, solid_capstyle="butt")
        # the non-AI written level, white where it runs inside a taller bar (not the black bar it tops)
        over_bar(ax, [-0.5, 2.5], [base, base], bars=pt[1:], lw=0.6, ls=(0, (3, 2)), zorder=5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, top)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for _, _, _, n in bars], fontsize=6.0)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=6.2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if kind == "price":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "\\${:,.0f}".format(v)))
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        # one title height for every panel, so titles in a row line up; a subtitle sits on the line beneath (PI,
        # 2026-10-31: "the buy question align with the vehicle price")
        ax.set_title(title, loc="left", fontsize=7.0, color=INK, pad=9)
        if sub:
            ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.0, color=MUTED, va="bottom", ha="left")
    save(fig, "fig_buyers", out, prev)


def texas_shares(points_json, out_json):
    """The Texas figure's inputs (Session 10; PI, 2026-10-31: "V1 to 3 ... tells a very good story"), from the points of
    job 887835 (price_hedonics.py VEH_TX_POINTS: each Texas-matched vehicle's log Texas median, log closing price, period,
    AI flag and start half-year, no item id): for each group in time order, the sorted shares (closing price over Texas
    median, rounded to 0.001) and, for each start half-year, the median share and its count. Nothing else is kept: no
    dates finer than the half-year, no Texas price, no closing price. The vehicles of January 2025 (before the assistant)
    keep their own point, so no vehicle is left out of the time line."""
    P = json.load(open(points_json))
    sh = np.exp(np.array(P["log_price"]) - np.array(P["log_tx"]))
    per, ai, half = np.array(P["period"]), np.array(P["ai"]), np.array(P["half"])
    groups = {"P0": per == "P0", "P1": per == "P1", "P2_not": (per == "P2") & (ai == 0), "P2_ai": (per == "P2") & (ai == 1)}
    assert sum(int(m.sum()) for m in groups.values()) == len(sh) == P["n"], "the groups do not cover every vehicle"
    line = []
    for h in sorted(set(half)):
        for g, m in (("pre", per != "P2"), ("not", groups["P2_not"]), ("ai", groups["P2_ai"])):
            sel = m & (half == h)
            if sel.sum():
                line.append({"half": str(h), "group": g, "n": int(sel.sum()), "median": round(float(np.median(sh[sel])), 4),
                             "january_only": bool(g == "pre" and h >= "2025H1")})
    assert sum(r["n"] for r in line) == len(sh), "the time line does not cover every vehicle"
    out = {"source": "results/hedonic_s10g/texas_points.json (job 887835)", "n": int(len(sh)),
           "shares": {g: sorted(round(float(x), 3) for x in sh[m]) for g, m in groups.items()}, "line": line}
    json.dump(out, open(out_json, "w"), indent=0)
    print("texas shares: %s; time-line points %d" % ({g: len(v) for g, v in out["shares"].items()}, len(line)))


def fig_texas(TS, K, out, prev):
    """Session 10, redrawn 2026-10-31 on the PI's choice (V1 to V3: "those three tells a lot better story"; scheme B;
    real dates, not half-year codes; the assistant era shaded green behind clear borders; a caveat for its small
    samples). The closing price as a share of the Texas median of the same make, model and model year, raw. (a) Over time:
    each point the median of the vehicles that started in that half-year, before the assistant one series, in the
    assistant era two (not flagged, AI written), each assistant-era point with its count. (b) Every vehicle by group in
    time order, the middle half as a box and the median as a line. (c) The cumulative shares by group."""
    import datetime as dt
    import matplotlib.dates as mdates
    S, L = TS["shares"], TS["line"]
    # in colour (PI, 2026-10-31: "the Texas seems like it's fine with orange ... why did the Texas thing becomes gray
    # again?"): before ChatGPT orange, chatbots only yellow; the assistant era as everywhere (not flagged black, AI written
    # green)
    T0, T1 = "#eb6834", "#d1a900"
    G = [("P0", "Before ChatGPT", T0), ("P1", "Chatbots only", T1), ("P2_not", "Assistant era, human-written", BASE_COL),
         ("P2_ai", "Assistant era, AI-written", AIW)]
    fig = plt.figure(figsize=(W, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.12, 1.0], hspace=0.62, wspace=0.34, left=0.1, right=0.985,
                          top=0.95, bottom=0.1)
    aa, ab, ac = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    pct = plt.FuncFormatter(lambda v, p: "%g%%" % (100 * v))
    when = lambda r: dt.date(2025, 1, 16) if r["january_only"] else dt.date(int(r["half"][:4]), 4 if r["half"][-1] == "1" else 10, 1)
    d0, d1 = dt.date(2019, 1, 1), dt.date(2026, 1, 1)
    chat, tool = dt.date(2022, 11, 30), dt.date(2025, 1, 29)
    for a_, b_, col, al, lab in ((d0, chat, T0, 0.09, "Before ChatGPT"), (chat, tool, T1, 0.10, "Chatbots only"),
                                 (tool, d1, P2_ERA, 0.13, "Optional\nAI assistant")):
        aa.axvspan(a_, b_, color=col, alpha=al, lw=0, zorder=0)
        aa.text(a_ + (b_ - a_) / 2, 2.08, lab, ha="center", va="top", fontsize=6.2, color=INK2)
    for x_ in (chat, tool):
        aa.axvline(x_, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
    lw = 1.3
    # the grey line is every vehicle of each half-year, not split, until the assistant arrives; it is named on the line
    # itself, under its 2020 to 2021 stretch (PI, 2026-10-31, of its old name "Before the assistant" printed at its end
    # beside the assistant era's two groups: "What is before assistant?")
    aa.text(dt.date(2021, 1, 1), 0.47, "All vehicles", fontsize=6.0, color=INK2, va="center", ha="center")
    for g, col, mk, lab in (("pre", INK2, "o", None), ("not", BASE_COL, "s", "Human-written"), ("ai", AIW, "o", "AI-written")):
        rows = [r for r in L if r["group"] == g]
        xs, ys = [when(r) for r in rows], [r["median"] for r in rows]
        aa.plot(xs, ys, color=col, lw=lw, zorder=3)
        for r, x_, y_ in zip(rows, xs, ys):
            fc = (T0 if x_ < chat else T1) if g == "pre" else col
            aa.plot([x_], [y_], marker=mk, ms=4.6, mfc=fc, mec=INK2 if g == "pre" else col, mew=0.7, zorder=4)
            # each count clear of the date lines, the series lines and the other labels (centred on its point, a count
            # crossed the assistant's date line or the line into its point): January 2025's ends left of the assistant's
            # line; a first-half count starts just right of that line, a second-half count just right of its point; the
            # not flagged counts sit under their points, the AI written counts over theirs
            if g == "pre" and r["january_only"]:
                aa.text(x_ - dt.timedelta(days=12), y_ + 0.075, "n = %d" % r["n"], fontsize=5.3, color=INK2, va="bottom", ha="right")
            elif g != "pre":
                below = g == "not"
                x_lab = (tool if x_ < dt.date(2025, 7, 1) else x_) + dt.timedelta(days=10)
                aa.text(x_lab, y_ - 0.12 if below else y_ + 0.09, "n = %d" % r["n"], fontsize=5.3, color=INK2,
                        va="top" if below else "bottom", ha="left")
        if lab:
            aa.text(xs[-1] + dt.timedelta(days=75), ys[-1], lab, fontsize=6.0, color=INK, va="center", ha="left")
    aa.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    aa.text(d0 + dt.timedelta(days=20), 1.03, "the Texas price", fontsize=5.6, color=MUTED, va="bottom")
    aa.set_xlim(d0, dt.date(2026, 9, 1))
    aa.set_ylim(0, 2.1)
    aa.yaxis.set_major_formatter(pct)
    aa.xaxis.set_major_locator(mdates.YearLocator())
    aa.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    aa.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))
    aa.tick_params(labelsize=6.2)
    aa.set_ylabel("Median closing price,\nshare of Texas price", fontsize=6.4)
    aa.grid(axis="y", color=GRID, lw=0.5)
    aa.set_axisbelow(True)
    aa.set_title("(a) Over time: the median of each half-year's vehicles", loc="left", fontsize=7.0, color=INK)
    # (b) the periods shaded and named over the columns as in panel (a), so the two assistant-era groups stand under
    # "Optional AI assistant" (PI, 2026-10-31: "you should say assistant era, human written, assistant era, AI written";
    # then "I do like this new zoning thing ... it's echoing the other ... panel"); the columns before the assistant are
    # "All vehicles", as the grey line of panel (a)
    X1 = 3.6   # the panel's right edge, a little past the last column so its median label fits
    for a_, b_, col, al, lab in ((-0.5, 0.5, T0, 0.09, "Before\nChatGPT"), (0.5, 1.5, T1, 0.10, "Chatbots\nonly"),
                                 (1.5, X1, P2_ERA, 0.13, "Optional\nAI assistant")):
        ab.axvspan(a_, b_, color=col, alpha=al, lw=0, zorder=0)
        # the period names over the plot, since the first axis leaves no empty room at its top
        ab.text((a_ + b_) / 2, 1.015, lab, transform=ab.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8, color=INK2)
    for x_ in (0.5, 1.5):
        ab.axvline(x_, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
    rng = np.random.default_rng(4)
    # the box over the dots as first drawn, in thinner lines on a thinner white edge so it hides fewer dots (PI,
    # 2026-10-31: "I like the original one with rewritten axes and better spacing, but the [median] remains on the
    # figure"; earlier "make the line a little bit thinner ... So it blocks less"); each column a little narrower and
    # left of centre, and the panel a little wider than (c), so its median prints to the right of its box inside its own
    # period (PI: the 82% "is looking into the other one, so give it more space"; "Give it better spacing for the 82%,
    # 72%")
    halo = lambda w: [patheffects.withStroke(linewidth=w + 0.9, foreground="white")]
    for k, (g, lab, col) in enumerate(G):
        v = np.array(S[g])
        c = k - 0.13
        xj = c + rng.uniform(-0.18, 0.18, len(v))
        ab.scatter(xj, v, s=3 if len(v) > 500 else 7, color=col, lw=0, alpha=0.5 if len(v) > 500 else 0.9, zorder=2)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ab.add_patch(plt.Rectangle((c - 0.16, q1), 0.32, q3 - q1, fill=False, edgecolor=INK, lw=0.5, zorder=3, path_effects=halo(0.5)))
        ab.plot([c - 0.16, c + 0.16], [md, md], color=INK, lw=1.0, zorder=4, solid_capstyle="butt", path_effects=halo(1.0))
        ab.text(k + 0.09, md, "%.0f%%" % (100 * md), fontsize=5.6, ha="left", va="center", color=INK, zorder=5)
    ab.set_yscale("log")
    # the first axis, 4% to 900% of the Texas price (PI, 2026-10-31: "I like the old y-axis ... Now it's all just clumped
    # together"; a longer axis that showed every vehicle was tried that day); the vehicles it leaves out are counted for
    # the caption
    ab.set_ylim(0.04, 9)
    off = {g: int(sum(1 for x in S[g] if not 0.04 <= x <= 9)) for g, _, _ in G}
    print("fig_texas (b): vehicles outside the 4%% to 900%% axis: %s, %d of %d" % (off, sum(off.values()), sum(len(S[g]) for g, _, _ in G)))
    ab.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8]))
    ab.yaxis.set_major_formatter(pct)
    ab.yaxis.set_minor_formatter(plt.NullFormatter())
    ab.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ab.set_xlim(-0.5, X1)
    ab.set_xticks([k - 0.13 for k in range(4)])
    ab.set_xticklabels(["%s\n(%d)" % (nm, len(S[g])) for (g, _, _), nm in
                        zip(G, ("All\nvehicles", "All\nvehicles", "Human-\nwritten", "AI-\nwritten"))])
    # the column labels at their own size (a plain tick_params(labelsize=...) here set both axes and overrode it)
    ab.tick_params(axis="x", length=0, labelsize=5.8)
    ab.tick_params(axis="y", labelsize=6.0)
    ab.set_ylabel("Closing price, share of Texas price", fontsize=6.4)
    ab.set_title("(b) Each vehicle, by group", loc="left", fontsize=7.0, color=INK, pad=19)
    for g, lab, col in G:
        v = np.array(S[g])
        y = np.arange(1, len(v) + 1) / len(v)
        # AI written dashed, as in the V3 the PI chose, so it stays apart from before ChatGPT in grey print (the two
        # print at about the same grey)
        # the legend names the groups without their counts, which panel (b) prints under each group: "human written"
        # made the longest name run into the curves
        ac.step(v, y, where="post", color=col, lw=lw, ls=(0, (3, 1.6)) if g == "P2_ai" else "-", label=lab)
    ac.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ac.set_xscale("log")
    ac.set_xlim(0.05, 8)
    ac.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.25, 0.5, 1, 2, 4]))
    ac.xaxis.set_major_formatter(pct)
    ac.xaxis.set_minor_formatter(plt.NullFormatter())
    ac.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "%.0f%%" % (100 * v)))
    ac.set_xlabel("Closing price, share of Texas price", fontsize=6.4)
    ac.set_ylabel("Share of vehicles at or below", fontsize=6.4)
    ac.tick_params(labelsize=6.0)
    ac.grid(color=GRID, lw=0.5)
    ac.set_axisbelow(True)
    ac.legend(frameon=False, loc="upper left", fontsize=5.4, handlelength=1.4, handletextpad=0.5, borderaxespad=0.1)
    ac.set_title("(c) The same, cumulative", loc="left", fontsize=7.0, color=INK, pad=19)
    save(fig, "fig_texas", out, prev)


# ---------------- fig_words: what each detector sees, word by word ----------------
def last_month_label(PN):
    """The last month with at least 10 new listings, as the adoption figure draws it (paper_macros figAdoptTo)."""
    m = PN["monthly"]["union"]
    last = max(k for k in m if m[k]["n"] >= 10)
    return "%s %s" % (MONTH_NAMES[int(last[5:7]) - 1], last[:4])


def fig_price_gap(PN, K, out, prev, name="fig_price_gap"):
    """The price figure's panel (b) as a figure of its own (PI, 2026-10-31: "the one that's interesting is the price of
    AI written listing ... can be ... by itself a graph ... we need N"; of the sketches, "A I like the best"; "the first
    one's better", half the text width). Within the same agency and subcategory, assistant-era listings only: if the
    human written listing closes at $1,000, the AI written one closes at $1,000 x e^r, and the frozen model's second
    version expects $1,000 x e^s from the AI written listings' own text (paper_constants frozen_field, job 817639; the
    sample size from that job's log, frozen_field.n). $1,000 is a scale: each bar is a percent gap in dollars per $1,000."""
    F = K["frozen_field"]["union"]
    (vs, ts), (vr, tr) = F["s"], F["r"]
    ss, sr, base = abs(vs / ts), abs(vr / tr), 1000.0
    act = [base * math.exp(x) for x in (vr, vr - Z * sr, vr + Z * sr)]
    exp_ = [base * math.exp(x) for x in (vs, vs - Z * ss, vs + Z * ss)]
    n = K["frozen_field"]["n"]["union"]
    fig, a = plt.subplots(figsize=(W * 0.56, 1.95))
    xs = list(BAR_XS_APART)
    bars = [(base, BASE_COL, BASE_TXT, BASE_LAB), (act[0], AIW, "white", AI_LAB), (exp_[0], FORM, INK, FORM_LAB_SAME)]
    top = 1500
    pts = []
    for x, (v, col, tc, _) in zip(xs, bars):
        pts.append(a.bar(x, v, width=0.8, color=col, lw=0, zorder=3).patches[0])
        a.text(x, 0.06 * top, "\\${:,.0f}".format(v), ha="center", va="bottom", fontsize=6.6, color=tc, fontweight="bold",
               zorder=6, bbox=SOLID_BACK if col == AIW else None)
    for k, (m, lo, hi) in ((1, act), (2, exp_)):
        x = xs[k]
        over_bar(a, [x, x], [lo, hi], bars=[pts[k]], lw=0.7, zorder=4, solid_capstyle="butt")
        for yv in (lo, hi):
            over_bar(a, [x - 0.11, x + 0.11], [yv, yv], bars=[pts[k]], lw=0.7, zorder=4, solid_capstyle="butt")
    over_bar(a, [-0.5, 2.5], [base, base], bars=pts[1:], lw=0.6, ls=(0, (3, 2)), zorder=5)
    a.axvline(DIVIDER_X, color=MUTED, lw=0.6, zorder=2)
    a.set_xlim(-0.5, 2.5)
    a.set_ylim(0, top)
    a.set_yticks([0, 500, 1000, 1500])
    a.set_xticks(xs)
    a.set_xticklabels([nm for _, _, _, nm in bars], fontsize=6.0)
    a.tick_params(axis="x", length=0, pad=2)
    a.tick_params(labelsize=6.2)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "\\${:,.0f}".format(v)))
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_title("If the human-written listing closes at \\$1,000", loc="left", fontsize=7.0, color=INK, pad=17)
    a.text(0, 1.03, "assistant era, February 2025 to %s; same seller and subcategory;\n%s listings from %s sellers"
           % (last_month_label(PN), "{:,}".format(n["listings"]), "{:,}".format(n["agencies"])),
           transform=a.transAxes, fontsize=5.6, color=MUTED, va="bottom")
    save(fig, name, out, prev)
    print("wrote %s: AI-written $%.0f [%.0f, %.0f]; expected $%.0f [%.0f, %.0f]" % tuple([name] + act + exp_))


def fig_text_rank(out, prev, name="fig_text_rank"):
    """The price figure's panel (a) redone for the appendix (PI, 2026-10-31: "expensive from cheap I liek it better now"):
    in each period, how many times as much the fifth of listings the ranking model ranks highest sells for as the fifth
    it ranks lowest, with a dashed line at x1, where the text would tell nothing. The spreads and their intervals come
    from inform_b200.json (200 agency-bootstrap draws) only; the older copy in paper_constants.json is not read."""
    sp = json.load(open(os.path.join(INP, "inform_b200.json")))["spreads"]
    S = {p: [sp["full|%s" % p][x] for x in ("mean", "lo", "hi")] for p in ("P0", "P1", "P2")}
    fig, a = plt.subplots(figsize=(W * 0.62, 2.2))
    for k, (p, lab, col) in enumerate(PERIOD_COLS):
        m, lo, hi = (math.exp(x) for x in S[p][:3])
        bar = a.bar(k, m, width=0.7, color=PERIOD_BAR_COLS[p], lw=0, zorder=3).patches[0]
        over_bar(a, [k, k], [lo, hi], bars=[bar], lw=0.7, zorder=4, solid_capstyle="butt")
        for yv in (lo, hi):
            over_bar(a, [k - 0.1, k + 0.1], [yv, yv], bars=[bar], lw=0.7, zorder=4, solid_capstyle="butt")
        a.text(k, 0.25, "\u00d7%.1f" % m, ha="center", va="bottom", fontsize=6.4, color="white" if p == "P2" else INK,
               fontweight="bold", zorder=6)
    a.axhline(1.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=5)
    a.text(2.42, 1.0, "\u00d71: the text\ntells nothing", fontsize=5.8, va="center", ha="left", color=INK)
    a.set_xlim(-0.5, 3.35)
    a.set_ylim(0, 4.8)
    a.set_xticks(range(3))
    a.set_xticklabels([lab.replace(" ", "\n", 1) for _, lab, _ in PERIOD_COLS], fontsize=6.0)
    a.tick_params(axis="x", length=0)
    a.tick_params(labelsize=6.2)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "\u00d7%g" % v))
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_title("How well the text tells expensive listings from cheap ones", loc="left", fontsize=7.0, color=INK, pad=16)
    a.text(0, 1.015, "the top fifth's price over the bottom fifth's, as the text ranks them", transform=a.transAxes,
           fontsize=5.8, color=MUTED, va="bottom")
    save(fig, name, out, prev)


TOKP = re.compile(r"(?u)\b[A-Za-z][A-Za-z']+\b")   # the classifier's token pattern (detector.C1)


def words_of(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def classifier_word_weights(ex):
    """Spread each phrase's contribution over the words that make it: a word gets its own weight (split over its
    occurrences) and half of each two-word phrase it starts or ends."""
    text, terms = ex["text"], ex["terms"]
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in TOKP.finditer(text)]
    uni, bi = {}, {}
    for t, _, _ in toks:
        uni[t] = uni.get(t, 0) + 1
    for i in range(len(toks) - 1):
        k = toks[i][0] + " " + toks[i + 1][0]
        bi[k] = bi.get(k, 0) + 1
    tw = [terms.get(t, 0.0) / uni[t] for t, _, _ in toks]
    for i in range(len(toks) - 1):
        k = toks[i][0] + " " + toks[i + 1][0]
        if k in terms:
            tw[i] += terms[k] / bi[k] / 2
            tw[i + 1] += terms[k] / bi[k] / 2
    return [sum(w for (t, a, b), w in zip(toks, tw) if a >= s0 and b <= e0) for _, s0, e0 in words_of(text)]


def token_word_weights(text, spans, vals):
    """Sum per-token pieces over the display word that holds each token's first character."""
    W_ = words_of(text)
    out = [0.0] * len(W_)
    for (a, b), v in zip(spans, vals):
        for i, (_, s0, e0) in enumerate(W_):
            if s0 <= a < e0 or (a < s0 < b):
                out[i] += v
                break
    return out


# the word shading: green toward AI, grey toward human (the pair's tints; until 2026-10-29 orange and blue, and blue now
# marks the AI marker words). A grey as light as the green at full weight (#666666, both shades luminance 0.262) is the
# green's twin to a deuteranope (validate_palette.js on the two full-weight shades: CVD 3.7); #444444 passes (8.1) at
# the cost of a darker human side (luminance 0.171 against 0.262; black text on it 4.2:1)
SHADE_HUMAN = "#444444"


def _shade(w, m):
    a = 0.0 if m <= 0 else min(1.0, abs(w) / m) * 0.75
    base = np.array(matplotlib.colors.to_rgb(AIW if w > 0 else SHADE_HUMAN))
    return tuple(1 - a * (1 - base))


def fig_words(cases, out, prev, name="fig_words", max_words=70):
    """cases: list of {"title", "text", "rows": [(label, word weights, AI-marker-word flags)]}."""
    fs, lh, gap = 6.6, 0.135, 0.03
    fig = plt.figure(figsize=(W, 30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 30)
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    # the shading key is in the LaTeX caption: as header lines here it ran past the text width, made the figure
    # 6.9 in wide and so printed it at 79% size
    y = 29.9
    x0, x1 = 1.12, W - 0.02
    for c in cases:
        ax.text(0, y, c["title"], fontsize=fs + 0.6, color=INK, fontweight="bold", va="top")
        y -= lh + 0.02
        W_ = words_of(c["text"])[:max_words]
        for label, ww, mk in c["rows"]:
            ww = ww[:len(W_)]
            m = max([abs(v) for v in ww] + [1e-12])
            ax.text(0, y, label, fontsize=fs - 0.4, color=MUTED, va="top")
            y_row = y
            x = x0
            for (word, _, _), v, is_mark in zip(W_, ww, mk):
                t = ax.text(x, y, word, fontsize=fs, color=INK, va="top", ha="left",
                            bbox=dict(boxstyle="square,pad=0.12", fc=_shade(v, m), ec="none"))
                bb = t.get_window_extent(renderer=rend).transformed(inv)
                if bb.x1 > x1 and x > x0:
                    t.remove()
                    x, y = x0, y - lh
                    t = ax.text(x, y, word, fontsize=fs, color=INK, va="top", ha="left",
                                bbox=dict(boxstyle="square,pad=0.12", fc=_shade(v, m), ec="none"))
                    bb = t.get_window_extent(renderer=rend).transformed(inv)
                if is_mark:
                    ax.plot([bb.x0, bb.x1], [bb.y0 - 0.012, bb.y0 - 0.012], color=INK, lw=0.6)
                x = bb.x1 + 0.035
            if len(words_of(c["text"])) > max_words:
                ax.text(x, y, "...", fontsize=fs, color=MUTED, va="top")
            # a row is at least as tall as its two-line label
            y = min(y - lh, y_row - 2 * lh) - gap
        y -= 0.06
    lo = y - 0.02
    ax.set_ylim(lo, 30)
    fig.set_size_inches(W, 30 - lo)
    save(fig, name, out, prev)


def build_word_cases(word_views, tokens):
    """Join our classifier's word views with the open detectors' token views, per example text."""
    marks = lambda ex: [bool(TOKP.fullmatch(re.sub(r"[^A-Za-z']", "", w.lower())) and re.sub(r"[^A-Za-z']", "", w.lower())
                             in set(ex["marker_words"])) for w, _, _ in words_of(ex["text"])]
    tk = {(t["kind"], str(t["item"]), t["text"]): t for t in tokens}   # item numbers: text in one file, numbers in the other
    titles = {"listing": "Real listing, item %d", "original": "Original of a controlled rewrite, item %d",
              "rewrite": "Its controlled rewrite, item %d"}
    cases = []
    for ex in word_views:
        rows = [("Our classifier\n(score %.2f)" % ex["score"], classifier_word_weights(ex), marks(ex))]
        # every example carries the standard detectors' row: an example without its tokens stops the run
        key = (ex["kind"], str(ex["item"]), ex["text"])
        assert key in tk, "no detector tokens for the %s of item %s" % (ex["kind"], ex["item"])
        t = tk[key]
        # both detectors sum the same per-token quantity with the same model pair (how much more predictable the
        # token was than Falcon-7B expected); Fast-DetectGPT scales the sum by its spread, Binoculars takes a ratio,
        # so one row shows both, with each detector's own score
        # green = more predictable than expected (toward AI), so the row is not labelled "surprise"
        rows.append(("Standard detectors\n(FDG %.2f, Bino. %.2f)" % (t["fdg_crit"], t["binoculars"]),
                     token_word_weights(ex["text"], t["spans"], t["tok_fdg"]), marks(ex)))
        cases.append({"title": titles[ex["kind"]] % ex["item"], "text": ex["text"], "rows": rows})
    return cases


TYPO_RED = "#c8372d"   # spell-checker red under a misspelling (named in the card's footer, so not color alone)
MARK_FILL = "#d3e4fb"  # light blue behind the AI marker words of the second test (blue = the AI marker words, as in Figure 1)
CARD_EDGE = "#cfcdc8"
# behind the pages (PI, 2026-10-28: "a color for the background, so just make it stand out a little bit more"):
# bg="backdrop" one warm light grey behind both; bg="tint" one panel per page, grey for the human written page and
# light green for the AI written one (2026-10-29: 12% of the AI written green on white, luminance 0.839 against the
# grey's 0.854; until then light blue)
BACKDROP = "#efeee9"
TINT_HUMAN, TINT_AI = "#efeee9", "#e3efe8"


def _flow(ax, rend, inv, runs, xl, xr, y, fs, lh, color=INK, marks=(), typos=(), hang=0.0):
    """Lay words out left to right from (xl, y), wrapping at xr; runs = [(text, bold)]. A word whose letters are in
    `marks` gets the marker shading; one in `typos` a red wavy underline. Returns the y below the last line."""
    sp = 0.278 * fs / 72                       # Helvetica's space is 0.278 em
    x, first = xl, True
    for text, bold in runs:
        for word in text.split():
            key = re.sub(r"[^a-z]", "", word.lower())
            kw = dict(fontsize=fs, color=color, va="top", ha="left", fontweight="bold" if bold else "normal", zorder=3)
            if key in marks:
                kw["bbox"] = dict(boxstyle="square,pad=0.08", fc=MARK_FILL, ec="none")
            t = ax.text(x, y, word, **kw)
            bb = t.get_window_extent(renderer=rend).transformed(inv)
            if bb.x1 > xr and not first:
                t.remove()
                x, y = xl + hang, y - lh
                t = ax.text(x, y, word, **kw)
                bb = t.get_window_extent(renderer=rend).transformed(inv)
            if key in typos:
                xs = np.linspace(bb.x0, bb.x1, 40)
                # the baseline is the box bottom plus the measured descent (the box hugs the glyphs); the wave sits
                # 0.6 of Arial's descent (0.212 em) below it: closer cut into the letters, lower met the next line
                base = bb.y0 + t._get_layout(rend)[2] / ax.figure.dpi
                yb = base - 0.6 * 0.212 * fs / 72
                ax.plot(xs, yb + 0.0025 * np.sin((xs - bb.x0) / 0.03 * 2 * np.pi), color=TYPO_RED, lw=0.5, zorder=4)
            x, first = bb.x1 + sp, False
    return y - lh


STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
          "CT": "Connecticut", "DE": "Delaware", "DC": "the District of Columbia", "FL": "Florida", "GA": "Georgia",
          "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
          "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
          "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
          "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
          "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
          "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
          "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# "township" before "town", "school district" before "district": the first kind found in the seller's name wins
SELLER_KINDS = (("school district", "a school district"), ("township", "a township"), ("borough", "a borough"),
                ("county", "a county"), ("city", "a city"), ("village", "a village"), ("town", "a town"),
                ("fire", "a fire department"), ("police", "a police department"), ("authority", "a public authority"),
                ("university", "a university"), ("college", "a college"))


def _month_words(iso):
    return "%s %s" % (MONTH_NAMES[int(iso[5:7]) - 1], iso[:4])


def _seller_words(header):
    """The page's 'Listing Number: N | town, ST US | seller' -> 'a township in Pennsylvania': the listing number, the
    town and the seller's name are dropped (PI, 2026-10-28: "anonymize a little bit"); the kind of seller and its state
    stay."""
    parts = [x.strip() for x in (header or "").split("|")]
    st = re.search(r"\b([A-Z]{2})\s+US\b", parts[1]) if len(parts) > 2 else None
    name = parts[2].lower() if len(parts) > 2 else ""
    kind = next((w for k, w in SELLER_KINDS if k in name), "a public seller")
    return kind + (" in " + STATES.get(st.group(1), st.group(1)) if st else "")


def _with_cuts(blocks, cut):
    """The description's blocks with those numbered in `cut` left out; each run of left-out blocks becomes one "[...]"
    line, indented like the first block it stands for."""
    out, inside = [], False
    for i, b in enumerate(blocks):
        if i in cut:
            if not inside:
                out.append({"kind": "gap", "indent": b["kind"] == "li", "runs": [["[…]", False]]})
            inside = True
        else:
            out.append(b)
            inside = False
    return out


def _redacted_seller(header):
    """'Listing Number: N | town, ST US | Harmony Township' -> ('Township', 'Pennsylvania'): the seller's kind and state;
    the name and town are drawn as a black bar (PI, 2026-11-21: "a redacted township, Pennsylvania")."""
    w = _seller_words(header)                       # e.g. "a township in Pennsylvania"
    kind, _, state = w.partition(" in ")
    return kind.split(" ", 1)[1].title(), state


def _mark(ax, x, y, ok, fs):
    """A check (green) or a cross (red) at (x, y); DejaVu Sans carries both glyphs."""
    ax.text(x, y, "✓" if ok else "✗", fontsize=fs + 1.2, color=AIW if ok else TYPO_RED, va="top", ha="left",
            fontfamily="DejaVu Sans", fontweight="bold", zorder=3)


def _listing_pages(pair, split, cuts, bg=None):
    """Session 13 redesign (PI, 2026-11-21): three cards in a row, each edged in its figure colour (human-written
    soft black, our LLM rewrite blue, AI-written green, as in Figure 4). The two real listings show the breadcrumb,
    title, seller (name redacted as a bar), start date and how the auction closed; the rewrite card, which is not a
    listing, is inset, dashed and lightly tinted, with a smaller header. Under each card, each test's reading with a
    check (flags) or a cross (does not). Returns the figure, its axes and the drawing's bottom."""
    gap, pad, top = 0.14, 0.085, 11.9
    widths = [(W - gap * (len(pair) - 1)) / len(pair)] * len(pair)
    fig = plt.figure(figsize=(W, 12))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 12)
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    L = lambda fs: fs / 72 * 1.22
    FS = 6.1                                        # body and footer size: one size, no small print
    ends, tops, boxes = [], [], []
    xs = [sum(widths[:c]) + gap * c for c in range(len(pair))]
    # headers first, so every card starts at the same height (the rewrite card a little lower: it is not a listing)
    spec = []
    for c, e in enumerate(pair):
        rw = "head" in e
        col = REWRITE_COL if rw else (NONAI if e["start"] < "2022-11-30" else AIW)
        x0, cw = xs[c], widths[c]
        if rw:                                      # inset: not a listing
            x0, cw = x0 + 0.05, cw - 0.10
        head = e["head"] if rw else ("Human-written, before ChatGPT" if col == NONAI else
                                     "AI-written, after Municibid's AI assistant")
        fs_h = 5.9 if rw else 6.6
        yh = _flow(ax, rend, inv, [(head, True)], x0, x0 + cw, top, fs_h, L(fs_h), color=col)
        spec.append((rw, col, x0, cw, yh))
    card_top = min(sp_[4] for sp_ in spec) - 0.03
    for c, e in enumerate(pair):
        rw, col, x0, cw, _ = spec[c]
        xl, xr = x0 + pad, x0 + cw - pad
        y = card_top - (0.06 if rw else 0.0)
        tops.append(y)
        y -= 0.07
        if not rw:
            y = _flow(ax, rend, inv, [("  ›  ".join(e["crumbs"]), False)], xl, xr, y, 5.6, L(5.6), color=MUTED)
        y = _flow(ax, rend, inv, [(e["title"], True)], xl, xr, y - 0.005, 7.4, L(7.4))
        if not rw:
            kind, state = _redacted_seller(e["header"])
            t = ax.text(xl, y, "Seller:", fontsize=FS, fontweight="bold", color=INK2, va="top", ha="left", zorder=3)
            bb = t.get_window_extent(renderer=rend).transformed(inv)
            bx, bh = bb.x1 + 0.035, FS / 72 * 0.78
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((bx, y - bh - 0.006), 0.36, bh, fc=INK, ec="none", zorder=3))
            ax.text(bx + 0.36 + 0.03, y, "%s, %s" % (kind, state), fontsize=FS, color=INK2, va="top", ha="left", zorder=3)
            y -= L(FS)
            price, bids = e["sale"]
            d = e["start"]
            when = "%d %s %s" % (int(d[8:10]), MONTH_NAMES[int(d[5:7]) - 1][:3], d[:4])
            y -= 0.035                                  # a break after the seller: the sale's facts as two bullets
            # the three numbers that matter underlined (PI, 2026-11-21): the start date, the closing price, the bids
            for segs in ([("Started ", False), (when, True)],
                         [("Closed at ", False), ("$" + "{:,}".format(price), True), (" after ", False), (str(bids), True), (" bids", False)]):
                ax.text(xl + 0.02, y, "•", fontsize=FS, color=col, va="top", zorder=3)
                x = xl + 0.1
                sp_ = 0.278 * FS / 72                     # Helvetica's space
                for text_, ul in segs:
                    if text_.startswith(" "):
                        x += sp_
                    t_ = ax.text(x, y, text_.strip(), fontsize=FS, color=INK2, va="top", ha="left", zorder=3)
                    bb = t_.get_window_extent(renderer=rend).transformed(inv)
                    if ul:
                        base = bb.y0 + t_._get_layout(rend)[2] / ax.figure.dpi
                        ax.plot([bb.x0, bb.x1], [base - 0.012, base - 0.012], color=INK2, lw=0.55, zorder=3)
                    x = bb.x1 + (sp_ if text_.endswith(" ") else 0.0)
                y -= L(FS)
            y -= 0.02
        ax.plot([xl, xr], [y + 0.01, y + 0.01], color=CARD_EDGE, lw=0.5, zorder=2)
        y = _flow(ax, rend, inv, [("Item Description", True)], xl, xr, y - 0.02, 6.5, L(6.5))
        marks, typos = set(e["marker_words"]), set(t_.lower() for t_ in e["typos"])
        for b in _with_cuts(e["blocks"], cuts.get(str(e["item"]), set()) if not rw else set()):
            runs = [(t_, bold) for t_, bold in b["runs"]]
            if b["kind"] == "gap":
                y = _flow(ax, rend, inv, runs, xl + (0.1 if b["indent"] else 0.0), xr, y, FS, L(FS), color=MUTED)
            elif b["kind"] == "h":
                y = _flow(ax, rend, inv, [(t_, True) for t_, _ in runs], xl, xr, y - 0.005, FS, L(FS), marks=marks, typos=typos)
            elif b["kind"] == "li":
                ax.text(xl + 0.025, y, "•", fontsize=FS, color=INK, va="top", zorder=3)
                y = _flow(ax, rend, inv, runs, xl + 0.1, xr, y, FS, L(FS), marks=marks, typos=typos)
            else:
                y = _flow(ax, rend, inv, runs, xl, xr, y, FS, L(FS), marks=marks, typos=typos)
            y -= 0.014
        ends.append(y)
        boxes.append((x0, cw, col, rw))
    bottom = min(ends) - pad + 0.02
    lo = bottom
    from matplotlib.patches import FancyBboxPatch
    for c, e in enumerate(pair):
        x0, cw, col, rw = boxes[c]
        ax.add_patch(FancyBboxPatch((x0, bottom), cw, tops[c] - bottom, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#f3f7fd" if rw else "white", ec=col, lw=1.0 if rw else 1.3,
                                    ls=(0, (3, 2)) if rw else "-", zorder=1))
        lines = []
        if not rw:
            lines.append(("Classifier %.2f" % e["score"], e["clf_flag"] == 1))
        n = e["n_mark"]
        lines.append(("%s AI marker word%s" % (n if n else "No", "" if n == 1 else "s"), n >= 2))
        y = bottom - 0.05
        for text, ok in lines:
            t = ax.text(x0 + 0.02, y, text, fontsize=FS, color=INK2, va="top", ha="left", zorder=3)
            bb = t.get_window_extent(renderer=rend).transformed(inv)
            _mark(ax, bb.x1 + 0.04, y + 0.01, ok, FS)
            y -= L(FS) + 0.01
        lo = min(lo, y)
    return fig, ax, lo - 0.02


def rewrite_page(left_entry, after):
    """Session 12 (PI, 2026-11-18: "present a before and after with real example"): the left listing's own text as
    rewritten by an LLM under the experiment's "appealing" prompt (one rewrite, used as written). Lines starting "- "
    become list items, the rest paragraphs; AI marker words are counted with the detector's own 38-word list
    (textfeat.MARK). Session 13: the card says in its header that it is our rewrite, by which model, as a control."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from textfeat import MARK
    assert str(after["item"]) == str(left_entry["item"]), ("the rewrite is not of the left listing", after["item"])
    blocks = []
    for line in after["rewrite"].split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            blocks.append({"kind": "li", "runs": [[line[2:], False]]})
        else:
            blocks.append({"kind": "p", "runs": [[line, False]]})
    words = set(re.findall(r"[a-z]+", after["rewrite"].lower()))
    marks = sorted(w for w in MARK if w in words)
    model = {"claude-haiku-5-5": "Claude Haiku 5.5"}[after["model"]]
    return dict(left_entry, blocks=blocks, marker_words=marks, n_mark=len(marks), typos=[],
                head="Our rewrite of the same listing by %s, as a control" % model)


def fig_listings(listings, left, right, out, prev, name="fig_listings", cuts=None, split=0.5, bg=None, after=None, sales=None):
    """Two banked listing pages side by side, the same width (PI, 2026-10-28: "same size"), drawn from each page's own
    breadcrumb, title and description markup (code/listing_examples.py; photos, bidding box and questions left out),
    with the listing number, town and seller's name removed: left a pre-ChatGPT listing flagged by neither test, its
    misspelling underlined in red; right a tool-era listing flagged by both tests, its AI marker words shaded, and the
    blocks in cuts[item] shown as "[...]". Under each page, what the two tests read (always on the full text).
    With `after` (Session 12), the left listing's LLM rewrite is drawn between the two (rewrite_page)."""
    assert _seller_words("Listing Number: 1 | susquehanna, PA US | Harmony Township") == "a township in Pennsylvania"
    assert _seller_words("Listing Number: 2 | Athol, MA US | Town of Athol - DPW") == "a town in Massachusetts"
    assert _redacted_seller("Listing Number: 1 | susquehanna, PA US | Harmony Township") == ("Township", "Pennsylvania")
    t = _with_cuts([{"kind": k, "runs": []} for k in ("h", "p", "li", "li", "li", "li")], {0, 2, 3})
    assert [b["kind"] for b in t] == ["gap", "p", "gap", "li", "li"] and [t[0]["indent"], t[2]["indent"]] == [False, True], t
    cuts = {str(k): set(v) for k, v in (cuts or {}).items()}
    by = {str(x["item"]): x for x in listings}
    pair = [by[str(left)], by[str(right)]]
    for e in pair:
        bad = cuts.get(str(e["item"]), set()) - set(range(len(e["blocks"])))
        assert not bad, ("no such description blocks", e["item"], sorted(bad))
        kept = " ".join("".join(t for t, _ in b["runs"]) for i, b in enumerate(e["blocks"]) if i not in cuts.get(str(e["item"]), set()))
        shown = set(re.findall(r"[a-z]+", kept.lower()))
        # a cut must not hide a AI marker word or a misspelling that the footer counts
        assert set(e["marker_words"]) <= shown and set(t.lower() for t in e["typos"]) <= shown, (e["item"], "cut hides a counted word")
    # each real listing's closing price and bids are required inputs (the panel read of Session 10, WORKLOG s10); a
    # missing one stops the run
    for e in pair:
        assert sales and str(e["item"]) in sales, ("no --sale for item", e["item"])
    pair = [dict(e, sale=sales[str(e["item"])]) for e in pair]
    if after is not None:
        pair = [pair[0], rewrite_page(pair[0], after), pair[1]]
    fig, ax, lo = _listing_pages(pair, split, cuts, bg)
    print("listing pages: left width share %.2f, height %.2f in, background %s" % (split, 12 - lo, bg))
    ax.set_ylim(lo, 12)
    fig.set_size_inches(W, 12 - lo)
    save(fig, name, out, prev)


def fig_button_share(BD, months, out, prev, name="fig_button_share", w=W / 2, h=1.15, line=INK2):
    """Half the text width, set beside its caption (PI, 2026-10-28: "a small, very small graph, take half of the
    column"; it replaces the word-sized line): by capture month, on Figure 1's months, the share of archived listing
    pages whose bundled code carries the AI writing assistant (code/button_dating.py, job 874348). Months with fewer
    than 10 captured pages are gaps, never zeros. The words in the graph use paper_macros.py's rule (every month with
    10 or more pages, from February 2025, floored), so they match the text. Saved at exactly w x h inches, so it prints
    at 100% at half the text width. From Session 10 (PI, 2026-10-28: "add like a break there and add a ... button on
    the top as well"): Figure 1's orange date line at 29 January 2025 (the end of January)
    with the assistant's button on top, right-aligned to it, and the data line in dark ink so the date line stands out."""
    M = BD["months"]
    assert all(M[m]["box"] == 0 for m in M if m < "2025-01"), "a page captured before 2025 carries the code"
    xs = {m: i for i, m in enumerate(months)}
    ok = [m for m in months if m in M and M[m]["pages"] >= 10]
    share = {m: 100.0 * M[m]["box"] / M[m]["pages"] for m in ok}
    post = [m for m in M if M[m]["pages"] >= 10 and m >= "2025-02"]
    lo = math.floor(100 * min(M[m]["box"] / M[m]["pages"] for m in post))
    hi = math.floor(100 * max(M[m]["box"] / M[m]["pages"] for m in post))
    runs, cur = [], []
    for m in months:
        if m in share:
            cur.append(m)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    years = sorted({m[:4] for m in months})
    # a year is labelled only with 6 or more of its months on the axis: at half width, 2026 (two months) ran into 2025
    named = [y for y in years if sum(1 for m in months if m[:4] == y) >= 6]
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig = plt.figure(figsize=(w, h))
        b_in, t_in = 0.195, 0.04                               # inches below (year labels) and above the plot
        ax = fig.add_axes([0.14, b_in / h, 0.845, 1 - (b_in + t_in) / h])   # 0.39 in at the left holds "100%"
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for r in runs:
            ax.plot([xs[m] for m in r], [share[m] for m in r], color=line, lw=1.1, zorder=3,
                    solid_joinstyle="round", solid_capstyle="round")
        x_tool = xs["2025-01"] + 0.5                          # 29 January 2025 sits at January's end, as in Figure 1
        ax.axvline(x_tool, color=TOOL_LINE, lw=1.2, zorder=2)
        ax.set_xlim(-0.9, len(months) - 0.1)
        ax.set_ylim(-6, 108)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xticks([(xs[min(k for k in months if k[:4] == y)] + xs[max(k for k in months if k[:4] == y)]) / 2 for y in named])
        ax.set_xticklabels(named)
        ax.tick_params(axis="x", which="major", length=0, pad=3)
        ax.set_xticks([xs[y + "-01"] - 0.5 for y in years if y + "-01" in xs], minor=True)
        ax.tick_params(axis="x", which="minor", length=3, width=0.6, color=MUTED)
        # no y title: the caption beside the graph names the share, and the words in it say what each stretch is
        # each regime named beside the stretch of line it describes
        ax.text(0, 9, "None of the pages captured before 2025", ha="left", va="bottom", fontsize=6.4, color=INK)
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        btn = crop_button(os.path.join(INP, BTN_SRC))
        ax.add_artist(AnnotationBbox(OffsetImage(btn, zoom=BTN_W_SMALL * 72.0 / btn.shape[1]), (x_tool - 0.7, 107),
                                     box_alignment=(1, 1), frameon=False, zorder=6))
        btn_units = BTN_W_SMALL * btn.shape[0] / btn.shape[1] / (h - b_in - t_in) * (108 + 6)   # its height in % points
        y_txt = 107 - btn_units - 4
        ax.text(x_tool - 0.7, y_txt, "%d%% to %d%% in every month\nfrom February 2025" % (lo, hi), ha="right", va="top",
                fontsize=6.4, color=INK, linespacing=1.15)
        assert y_txt - 2 * 6.4 * 1.15 / 72 / (h - b_in - t_in) * 114 > 22, "the words under the button reach the zero line's label"
        fig.savefig(os.path.join(out, name + ".pdf"), dpi=600)     # 600 dpi keeps the button's pixels (see save)
        fig.savefig(os.path.join(prev, name + ".png"), dpi=600)
        plt.close(fig)
    print("wrote %s: %d of %d months drawn; from February 2025 %d%% to %d%% (floored, all months with 10+ pages)" % (
        name, len(ok), len(months), lo, hi))


def fig_spelling(T, months, out, prev, name="fig_spelling"):
    """Session 10 (PI, 2026-10-30: misspellings are "interesting might worth a graph in appendix"; then "dont think we
    need AI writing listing the green line on that? just spelling will do"): by start quarter, on Figure 1's months,
    the share of all new listings with at least one misspelling (Figure 1's purple for all new listings; Wilson 95%
    band in grey), from describe_trends.py --monthly (job 884255; a missing misspelling count is never read as zero).
    The listings neither test flags give the same series within the gap printed here (the caption's number,
    paper_macros.py typoFlagGap). Date lines as in Figure 1."""
    xs = {m: i for i, m in enumerate(months)}
    q_of = lambda m: m[:4] + "Q%d" % ((int(m[5:7]) - 1) // 3 + 1)
    Q = {}
    for m in months:
        r, r0 = T["months"].get(m), T["months_noflag"].get(m)
        if not r:
            continue
        q = Q.setdefault(q_of(m), {"x": [], "k": 0, "n": 0, "k0": 0, "n0": 0})
        q["x"].append(xs[m])
        q["k"] += r["k_typo"]
        q["n"] += r["n_typo_known"]
        q["k0"] += r0["k_typo"] if r0 else 0
        q["n0"] += r0["n_typo_known"] if r0 else 0
    qs = [q for q in sorted(Q) if Q[q]["n"] >= 10]
    x = [sum(Q[q]["x"]) / len(Q[q]["x"]) for q in qs]
    y = [100.0 * Q[q]["k"] / Q[q]["n"] for q in qs]
    ci = [wilson(Q[q]["k"], Q[q]["n"]) for q in qs]
    gap = max(abs(100.0 * Q[q]["k"] / Q[q]["n"] - 100.0 * Q[q]["k0"] / Q[q]["n0"]) for q in qs)
    fig, ax = plt.subplots(figsize=(W, 1.6))
    ax.fill_between(x, [100 * a for a, _ in ci], [100 * b for _, b in ci], color=GRAY, alpha=0.28, lw=0, zorder=2)
    ax.plot(x, y, color=PURPLE, lw=1.4, zorder=4, marker="o", ms=2.8, solid_joinstyle="round")
    x_chat, x_tool = xs["2022-11"] + 0.5, xs["2025-01"] + 0.5
    ax.axvline(x_chat, color=CHAT_LINE, lw=1.0, ls=(0, (3, 1.6)), zorder=3)
    ax.axvline(x_tool, color=TOOL_LINE, lw=1.2, zorder=3)
    top = 36
    assert max(100 * b for _, b in ci) < top - 1, "a quarter's interval reaches the top"
    ax.text(x_chat - 0.8, top - 1, "ChatGPT released", ha="right", va="top", fontsize=6.4, color=INK)
    ax.text(x_tool - 0.8, top - 1, "AI assistant added", ha="right", va="top", fontsize=6.4, color=INK)
    ax.set_ylim(0, top)
    ax.set_yticks([0, 10, 20, 30])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d%%" % v))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    years = sorted({m[:4] for m in months})
    named = [yy for yy in years if sum(1 for m in months if m[:4] == yy) >= 6]
    ax.set_xlim(-0.9, len(months) - 0.1)
    ax.set_xticks([(xs[min(k for k in months if k[:4] == yy)] + xs[max(k for k in months if k[:4] == yy)]) / 2 for yy in named])
    ax.set_xticklabels(named)
    ax.tick_params(axis="x", which="major", length=0, pad=3)
    ax.set_xticks([xs[yy + "-01"] - 0.5 for yy in years if yy + "-01" in xs], minor=True)
    ax.tick_params(axis="x", which="minor", length=3, width=0.6, color=MUTED)
    save(fig, name, out, prev)
    print("wrote %s: %d quarters (%s to %s), all new listings %.1f%% in %s to %.1f%% in %s; listings neither test flags "
          "within %.2f points of them in every quarter" % (name, len(qs), qs[0], qs[-1], y[0], qs[0], y[-1], qs[-1], gap))


# the three periods in the AI-marker-word figures: greys before the assistant, the assistant's orange after it
# names: the PI's, 2026-10-30 ("Before ChatGPT · Chatbots Only · W/ Built-in AI assistant", then "Optional AI assistant":
# "built-in implies it's mandatory, but it's optional")
PERIOD_COLS = (("P0", "Before ChatGPT", P0_COL), ("P1", "Chatbots only", P1_COL), ("P2", "Optional AI assistant", P2_ERA))


def fig_marker_words(M, out, prev, name="fig_marker_words"):
    """Session 10 (PI, 2026-10-30: "for AI marker word show the statistical distribution, these are improtant tis the
    foundation of our arguments"): for each of the 38 AI marker words, the share of listings that use it in each period
    (describe_trends.py --markers, job 884642: the full text, as the rule reads it), words ordered by their use with the
    assistant; a line joins each word's share before ChatGPT and with the assistant."""
    # every word carries a count for every period, zeros included (a missing one stops the run)
    words = sorted(M["words"], key=lambda w: (-M["words"][w]["P2"], w))
    n = M["n"]
    sh = {w: {p: 100.0 * M["words"][w][p] / n[p] for p, _, _ in PERIOD_COLS} for w in words}
    fig, ax = plt.subplots(figsize=(W, 4.3))
    for i, w in enumerate(words):
        y = len(words) - 1 - i
        ax.plot([sh[w]["P0"], sh[w]["P2"]], [y, y], color=GRID, lw=1.2, zorder=1, solid_capstyle="butt")
        for p, lab, col in PERIOD_COLS:
            ax.plot([sh[w][p]], [y], "o", ms=3.4, color=col, zorder=3 if p != "P1" else 2, mec="white", mew=0.4)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words[::-1], fontsize=6.2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.7, len(words) - 0.3)
    top = max(sh[w]["P2"] for w in words)
    ax.set_xlim(0, math.ceil(top + 0.3))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g%%" % v))
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of listings that use the word", fontsize=6.8)
    hs = [matplotlib.lines.Line2D([], [], color=c, marker="o", ls="", ms=3.4) for _, _, c in PERIOD_COLS]
    ax.legend(hs, ["%s (%s listings)" % (lab, "{:,}".format(n[p])) for p, lab, _ in PERIOD_COLS], loc="lower right",
              frameon=False, fontsize=6.4, handletextpad=0.3, borderaxespad=0.3)
    save(fig, name, out, prev)
    print("wrote %s: %d words; with the assistant %.1f%% (%s) to %.2f%% (%s); before ChatGPT at most %.2f%% (%s)" % (
        name, len(words), sh[words[0]]["P2"], words[0], sh[words[-1]]["P2"], words[-1],
        max(sh[w]["P0"] for w in words), max(words, key=lambda w: sh[w]["P0"])))


def fig_test_distributions(M, tau, out, prev, name="fig_test_distributions"):
    """Session 10: the two tests by period. (a) listings with 1, 2, 3, 4 and 5 or more AI marker words (the rule flags 2 or
    more); (b) the classifier's score, in 20 bins, with the threshold. Shares of each period's listings."""
    n = M["n"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.85), gridspec_kw={"wspace": 0.28})
    ks, w_ = ["1", "2", "3", "4", "5"], 0.26
    for j, (p, lab, col) in enumerate(PERIOD_COLS):
        # pandas value_counts writes only the counts that occur, so an absent count is no listing; the counts must add
        # up to the period's listings, or the input is not the one this figure reads
        d = M["n_mark_dist"][p]
        assert sum(d.values()) == n[p], "AI-marker-word counts of %s add up to %d, not %d" % (p, sum(d.values()), n[p])
        a1.bar([i + (j - 1) * w_ for i in range(len(ks))], [100.0 * d.get(k, 0) / n[p] for k in ks], width=w_ * 0.92,
               color=col, lw=0, zorder=3, label=lab)
    a1.axvline(0.5, color=INK, lw=0.8, ls=(0, (3, 1.6)), zorder=4)
    a1.text(0.58, 0.97, "flagged: 2 or more", transform=a1.get_xaxis_transform(), ha="left", va="top", fontsize=6.2, color=INK)
    a1.set_xticks(range(len(ks)))
    a1.set_xticklabels(["1", "2", "3", "4", "5+"])
    a1.set_xlabel("AI marker words in the listing", fontsize=6.8)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g%%" % v))
    a1.grid(axis="y", color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    a1.set_title("(a) AI marker words per listing", loc="left", fontsize=7.0, color=INK, fontweight="bold", pad=3)
    edges = M["bins"]
    for p, lab, col in PERIOD_COLS:
        h = M["score_hist"][p]
        a2.step(edges, [100.0 * h[0] / n[p]] + [100.0 * x / n[p] for x in h], where="pre", color=col, lw=1.3, zorder=3)
    a2.axvline(tau, color=INK, lw=0.8, ls=(0, (3, 1.6)), zorder=4)
    a2.text(tau + 0.02, 0.97, "flagged: %.3f or more" % tau, transform=a2.get_xaxis_transform(), ha="left", va="top",
            fontsize=6.2, color=INK)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g%%" % v))
    a2.set_xlim(0, 1)
    a2.set_xlabel("Classifier score", fontsize=6.8)
    a2.grid(axis="y", color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    a2.set_title("(b) Classifier score (listings per bin of 0.05)", loc="left", fontsize=7.0, color=INK, fontweight="bold", pad=3)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PERIOD_COLS]
    fig.legend(hs, [lab for _, lab, _ in PERIOD_COLS], loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False,
               fontsize=6.6, handlelength=1.0, handleheight=0.9, columnspacing=1.6)
    save(fig, name, out, prev)
    two = {p: 100.0 * sum(v for k, v in M["n_mark_dist"][p].items() if int(k) >= 2) / n[p] for p, _, _ in PERIOD_COLS}
    print("wrote %s: 2+ AI marker words %s; classifier flag %s" % (name, {p: round(v, 2) for p, v in two.items()},
                                                               {p: round(100 * M["tau_share"][p], 2) for p in n}))


def fig_questions(QK, out, prev, name="fig_questions"):
    """Session 10 (PI, 2026-10-29: the question-and-answer table is "a great section to show how buyers respond ...
    lets make a major section with fig"): what buyers ask before the close, AI written against non-AI written
    tool-era listings, from the blind coding (questions_kinds.json; 150 questions a group). (a) the kind of question,
    (b) what it asks about; bars are shares of each group's questions with Wilson 95% intervals, black and green as in
    every figure."""
    kinds = [("never_mentioned", "Never mentioned in the listing"), ("verify_claim", "Asks to verify a stated fact"),
             ("logistics", "Sale and pickup"), ("already_stated", "Already stated in the listing"), ("other", "Other")]
    tops = sorted([t for t in QK["topics"] if t != "Other"],
                  key=lambda t: -(QK["topics"][t]["1"]["share"] + QK["topics"][t]["0"]["share"])) + ["Other"]
    g1, g0 = QK["groups"]["1"], QK["groups"]["0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.95})
    fig.subplots_adjust(left=0.215, right=0.985, top=0.80, bottom=0.12)
    h, off = 0.36, 0.19
    def panel(ax, rows, get, xmax, title):
        for i, (key, lab) in enumerate(rows):
            y = len(rows) - 1 - i
            for g, col, dy in (("0", BASE_COL, +off), ("1", AIW, -off)):     # non-AI on top, as listed in the legend
                v = get(key, g)
                bar = ax.barh(y + dy, 100 * v["share"], height=h, color=col, lw=0, zorder=3).patches[0]
                whisker(ax, 100 * v["lo"], 100 * v["hi"], y + dy, cap=0.08, bar=bar)
                ax.text(100 * v["hi"] + 0.012 * xmax, y + dy, "%.0f%%" % (100 * v["share"]), va="center", ha="left",
                        fontsize=6.0, color=INK2, zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=6.6)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "%d%%" % x))
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=7.2, color=INK, fontweight="bold", pad=4, x=-0.02)
    panel(a1, kinds, lambda k, g: QK["groups"][g][k], 80, "(a) What kind of question")
    panel(a2, [(t, t) for t in tops], lambda t, g: QK["topics"][t][g], 45, "(b) What it asks about")
    hs = [plt.Rectangle((0, 0), 1, 1, color=BASE_COL), plt.Rectangle((0, 0), 1, 1, color=AIW)]
    fig.legend(hs, [BASE_LAB_FLAT + " (%d questions on %d listings)" % (g0["n_questions"], g0["n_listings"]),
                    "AI-written (%d questions on %d listings)" % (g1["n_questions"], g1["n_listings"])],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, fontsize=6.6, handlelength=1.0,
               handleheight=0.9, columnspacing=1.6)
    save(fig, name, out, prev)


CLAIM_ANY = "#a9c8ef"   # a light tint of the rewrites' blue: any added claim, uses included


def fig_claims_quality(FS, QS, out, prev, name="fig_claims_quality"):
    """Session 12 (the advisor's C4): (a) the share of controlled rewrites that add a claim the original does not
    support, by rewriting model and instruction (fidelity_s12.json: six blind coders), split into claims about the
    unit's condition or features and any claim, uses included, with 95% Wilson intervals; (b) the pricing model's lift
    from the rewrite by third of item quality, by instruction (quality_s12.json, job 988391), in percent with
    pair-bootstrap 95% intervals."""
    cells = [("claude-opus-5-5|V1_enhance", "Opus, “appealing”"), ("claude-sonnet-5-5|V1_enhance", "Sonnet, “appealing”"),
             ("claude-haiku-5-5|V1_enhance", "Haiku, “appealing”"), ("claude-opus-5-5|V2_concise", "Opus, “concise”"),
             ("claude-sonnet-5-5|V2_concise", "Sonnet, “concise”")]
    for c, _ in cells:
        assert c in FS["cells"], ("fidelity_s12.json lacks", c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.72})
    ys = np.arange(len(cells))[::-1]
    h = 0.36
    for y, (c, lab) in zip(ys, cells):
        B = FS["cells"][c]
        for off, k, col in ((h / 2, "unit_claim", REWRITE_COL), (-h / 2, "any_claim", CLAIM_ANY)):
            e = B[k]
            bar = a.barh(y + off, 100 * e["share"], height=h, color=col, lw=0, zorder=3)[0]
            whisker(a, 100 * e["lo"], 100 * e["hi"], y + off, cap=0.07)
            a.text(100 * e["hi"] + 1.5, y + off, "%d of %d" % (e["k"], e["n"]), va="center", ha="left", fontsize=6.0, color=INK2)
    a.set_yticks(ys)
    a.set_yticklabels([lab for _, lab in cells])
    a.set_xlim(0, 124)                     # room for the count labels beyond the longest bar
    a.set_xticks([0, 20, 40, 60, 80, 100])
    a.set_xlabel("Share of rewrites (%)")
    a.grid(axis="x", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_title("(a) Rewrites adding a claim the\noriginal does not support", loc="left", fontsize=7.2, fontweight="bold", color=INK)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (REWRITE_COL, CLAIM_ANY)]
    a.legend(hs, ["About the unit's condition\nor features", "Any claim, uses included"], loc="lower right", frameon=False,
             fontsize=6.0, handlelength=1.0, borderaxespad=0.1)
    groups = [("appealing", "“Appealing”", REWRITE_COL), ("concise", "“Concise”", P1_COL)]
    thirds = [("lift_low", "Bottom third"), ("lift_mid", "Middle third"), ("lift_high", "Top third")]
    pct = lambda x: 100 * (math.exp(x) - 1)
    yb = np.arange(len(thirds))[::-1]
    for gi, (g, glab, col) in enumerate(groups):
        G = QS["groups"][g]
        for y, (k, _) in zip(yb, thirds):
            e = G[k]
            yy = y + (h / 2 if gi == 0 else -h / 2)
            b.barh(yy, pct(e["est"]), height=h, color=col, lw=0, zorder=3)
            whisker(b, pct(e["lo"]), pct(e["hi"]), yy, cap=0.07)
    b.set_yticks(yb)
    b.set_yticklabels([lab for _, lab in thirds])
    b.set_xlim(0, 40)
    b.set_xlabel("Rise in the predicted closing price (%)")
    b.grid(axis="x", color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.set_title("(b) Boost: how much the rewrite raised the\npredicted closing price, by item quality", loc="left", fontsize=7.2, fontweight="bold", color=INK)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups]
    b.legend(hs, [glab for _, glab, _ in groups], loc="lower right", frameon=False, fontsize=6.0, handlelength=1.0,
             borderaxespad=0.1)
    save(fig, name, out, prev)


def crop_button(src, pad=3):
    """The "Optimize Description" button cut from the PI's screenshot of the help page's step 3 (BTN_SRC), returned as
    pixels to the figure that draws it: the box of its blue-dominant pixels (blue at least 0.15 above red, mean below
    0.6; the sentence beside it is grey, so none of it qualifies), plus `pad` pixels of page for the anti-aliased edge.
    Refuses a screenshot that is missing, shows no button or shows blue in more than one place far apart."""
    a = plt.imread(src)
    rgb = a[:, :, :3]
    blue = ((rgb[:, :, 2] - rgb[:, :, 0]) > 0.15) & (rgb.mean(2) < 0.6)
    ys, xs = np.where(blue)
    assert len(xs) > 500, "no button found in %s" % src
    cols = np.unique(xs)
    assert cols.max() - cols.min() + 1 <= 1.05 * len(cols), "blue pixels in more than one place: %s" % src
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    return a[y0:y1, x0:x1]


def main(inp, out, prev):
    global INP
    INP = inp
    os.makedirs(out, exist_ok=True)
    os.makedirs(prev, exist_ok=True)
    # every input is required: a missing file stops the run, and no figure is skipped or drawn from an older input
    # (PI, 2026-11-02: "absolutely no need to be fallback")
    load = lambda f: json.load(open(os.path.join(inp, f)))
    R, K, PN = load("report_data.json"), load("paper_constants.json"), load("paper_numbers.json")
    # the pooled designs with AI = flagged in the assistant era (Session 8's any-period files are not read)
    H = {r: load("hedonic_%s_tool.json" % r) for r in ("union", "lex", "c1")}
    fig_adoption(R, PN, K, out, prev)
    fig_button_share(load("button_dating.json"), FIG1_MONTHS, out, prev)
    fig_text(R, PN, out, prev)
    # Session 12: what rewriting adds and whether it tracks the item's quality (both inputs required)
    fig_claims_quality(load("fidelity_s12.json"), load("quality_s12.json"), out, prev)
    fig_buyers(H, K, out, prev)
    # the Texas figure (PI, 2026-10-31: V1 to V3) from texas_shares.json, reduced from job 887835's points
    fig_texas(load("texas_shares.json"), K, out, prev)
    # the price figure's two panels apart (PI, 2026-10-31): panel (b) a figure of its own in the main text, panel (a)
    # redone for the appendix
    fig_price_gap(PN, K, out, prev)
    fig_text_rank(out, prev)
    fig_questions(load("questions_kinds.json"), out, prev)
    fig_spelling(load("trends_monthly.json"), FIG1_MONTHS, out, prev)
    fig_marker_words(load("markers_union.json"), out, prev)
    fig_test_distributions(load("markers_union.json"), K["detector"]["tau"], out, prev)


if __name__ == "__main__":
    if sys.argv[1] == "--words":
        # EXAMPLES (paper_numbers.py's examples.json; its "word_views") and TOKENS (std_detectors.py's
        # tokens_examples.json), both required
        wv = json.load(open(sys.argv[2]))["word_views"]
        tk = json.load(open(sys.argv[3]))
        fig_words(build_word_cases(wv, tk), sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "--texas-shares":
        texas_shares(sys.argv[2], sys.argv[3])                 # POINTS_JSON (job 887835) OUT_JSON (report_inputs)
    elif sys.argv[1] == "--listings":
        # LISTINGS_JSON (code/listing_examples.py; raw listing text, so read from the session scratchpad) LEFT RIGHT
        # FIG_OUT PREVIEW [NAME] [--cut ITEM:i,j,...] [--bg backdrop|tint]: description blocks i, j, ... of ITEM shown
        # as "[...]"; the pages on a background
        lj = json.load(open(sys.argv[2]))["listings"]
        rest, cuts, bg, after, sales = sys.argv[7:], {}, None, None, {}
        name = rest.pop(0) if rest and not rest[0].startswith("--") else "fig_listings"
        for flag, val in zip(rest[::2], rest[1::2]):
            assert flag in ("--cut", "--bg", "--after", "--sale"), flag
            if flag == "--sale":
                it_, pr_, bd_ = val.split(":")
                sales[it_] = (int(pr_), int(bd_))
                continue
            if flag == "--bg":
                bg = val
                continue
            if flag == "--after":
                after = json.load(open(val))      # the rewrite JSON (raw text; session scratchpad)
                continue
            it, idx = val.split(":")
            cuts[it] = {int(i) for i in idx.split(",") if i}
        fig_listings(lj, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], name=name, cuts=cuts, bg=bg, after=after,
                     sales=sales)
    else:
        main(*sys.argv[1:])
