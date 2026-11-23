# Can You Tell a Smile from a Veil? How Buyers Respond When AI Rewrites Product Descriptions

Haoran Zheng (University of Chicago)

Code and aggregate data for the paper. Paper: arXiv link to come.

## The study in brief

Municibid is an online auction site where public agencies sell surplus vehicles and equipment. In January 2025 its
listing form gained an AI assistant that rewrites a seller's description. The study has two parts.

1. **A rewriting experiment.** We drew 600 listings written before ChatGPT and had three AI models (Claude Opus 5.5,
   Sonnet 5.5 and Haiku 5.5) rewrite them under two prompts. LLM coders compared each rewrite with its original:
   which facts and defect statements it kept, where it placed them, and what claims it added that the original did
   not support. A pricing model and outcome models, fit on pre-ChatGPT listings, score each original and its rewrite
   and give the gap a buyer swayed by the wording would show.
2. **A market comparison.** From Internet Archive captures of the site we rebuilt the listings of 2019 to 2026. A
   validated detector marks the assistant era's listings as AI-written or human-written. We compare page views,
   buyer questions, bids, whether a listing received a bid and the closing price of AI-written listings with the same
   seller's human-written listings.

## What is here

```
code/            every script the paper's numbers, tables and figures come from
report_inputs/   the aggregates (JSON and one CSV) the paper's numbers, tables and figures are built from
```

`report_inputs/` holds counts, estimates, intervals and model outputs only. It holds no listing text and no bidder or
asker identities. From it alone, three scripts rebuild the paper's macros (every number in the text), its generated
tables and five of its six figures:

```bash
python3 code/paper_macros.py report_inputs numbers.tex
python3 code/paper_tables.py report_inputs tables/
python3 code/paper_figures.py report_inputs figures/ previews/
```

We checked that these commands reproduce the paper's `numbers.tex` and tables byte for byte and its figures pixel for
pixel. `paper_figures.py` also draws figures that the paper does not print. The paper's remaining figure, the example
listings, shows listing text and is drawn by the `--listings` mode of `paper_figures.py` from a file we do not host (see Data below).

These three scripts need numpy and matplotlib. We ran them with Python 3.11, numpy 1.26 and matplotlib 3.8.

## The pipeline

The upstream steps ran as batch jobs on a university cluster (Python 3.9, pandas 2.3, numpy 2.0, scipy 1.13,
scikit-learn 1.6; no statsmodels, so regressions use `regs.py`). Their paths point at that cluster's project folder,
and `job.sh` and `parse.sbatch` carry a placeholder Slurm account.

| step | scripts | what it does |
|---|---|---|
| parse | `mb_parse.py`, `parse.sbatch` | parses archived listing pages (standard library only; `python3 mb_parse.py selftest` runs its tests and a mutation check) |
| panel | `build_panel.py`, `textfeat.py` | one row per listing; description features after removing each seller's boilerplate |
| AI assistant's date | `button_dating.py` | dates the assistant from the first archived page whose code carries it |
| detector | `excess_vocab.py`, `detector.py`, `detector_rules.py`, `blind_sample.py`, `label_app/` | the AI marker words (excess vocabulary of 2025 listings over the pre-ChatGPT years), the classifier, the decision rules, and the blind samples and labelling app for the human check |
| standard detectors | `std_detectors.py`, `setup_detectors.sh` | Fast-DetectGPT and Binoculars, for comparison (GPU job) |
| rewriting experiment | `sample_for_synthetic.py`, `rewrite_audit.py`, `frozen_rewrites.py` | the sample of 600 originals; the coding batches, the coders' aggregates, agreement, where stated problems moved, the third arm; the pricing and outcome models held fixed and their scores of each original and rewrite |
| market comparison | `regs.py`, `analysis_core.py`, `describe_trends.py`, `adoption.py`, `analysis_questions.py`, `price_hedonics.py`, `informativeness.py`, `inform_summary.py`, `count_p0_s12.py` | fixed-effects regressions, trends, adoption and balance, buyer questions, prices with vehicle controls, whether descriptions rank prices |
| Texas check | `dmv_coverage.py`, `dmv_benchmark.py` | closing prices against Texas DMV sale prices of the same make, model and year (aggregates only) |
| paper inputs | `paper_numbers.py`, `listing_examples.py`, `report_data.py` | collect the cluster outputs into the aggregates of `report_inputs/` |
| paper exhibits | `paper_macros.py`, `paper_tables.py`, `paper_figures.py` | build the macros, tables and figures from `report_inputs/` |

Each script's header states its inputs, outputs and the checks it runs.

## The rewriting experiment

The 600 originals were drawn from listings started before 30 November 2022, in proportion to product categories,
with descriptions of 3 to 400 words. They were split at random into six batches of 100, and each batch was rewritten
once by one AI model under one prompt. A seventh batch used the "appealing" prompt plus the last sentence of
"concise".

- "appealing": *Improve this auction listing description. Make it clear, well-organized, professional and appealing
  to buyers. Use the listing title and the seller's original description.*
- "concise": *Rewrite this auction listing description so it is clear, complete and professional. Keep it concise and
  factual. Use only the information in the title and the original description.*

Six LLM coders, each a fresh session of Claude Opus 5.5 blind to the AI model and the prompt, coded one batch each.
For every fact and defect statement in the original they recorded whether the rewrite kept, dropped or changed it,
and they listed every claim the rewrite added that the original did not support, with its type and whether it
contradicted the original. Claude Sonnet 5.5 coded 100 pairs a second time, and a human coded 20 pairs blind.
`rewrite_audit.py` turns the codes into the paper's aggregates.

## Data

- **Hosted here:** the aggregates in `report_inputs/`, including the human blind-read labels (labels only).
- **Not hosted:** the archived listing pages (about 2.8 GB of Internet Archive captures), the listing-level panel
  (about 23 MB) and the 600 original and rewritten descriptions with their per-pair codes. These hold listing text,
  and some descriptions carry public employees' names and phone numbers. The pages are public Internet Archive
  captures and can be fetched again from the item numbers.
- **Not ours to share:** Texas DMV records; only aggregates computed from them are here.

Bidder and asker usernames were stored only as salted hashes. The salt used for the paper is not published, and
`mb_parse.py` carries a placeholder.

## License

The code is released under the MIT License (see `LICENSE`).
