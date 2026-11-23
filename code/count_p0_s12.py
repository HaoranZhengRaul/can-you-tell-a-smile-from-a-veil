import json, sys
sys.path.insert(0, "/project/dmv_project/projects/ai_pricing_kbb/code")
from analysis_core import load
d = load("results/panel_s8/panel.parquet", "results/panel_s8/features.parquet", "results/detector_s8/ai_scores.parquet")
X = {json.loads(l)["item"] for l in open("results/synth_s8/inputs_600.jsonl")}
p0 = d[d.period == "P0"]
print("P0 listings in the analysis sample:", len(p0))
print("of the 600 originals, in the analysis sample's P0:", int(p0["item"].isin(X).sum()))
print("P0 outside the 600:", int((~p0["item"].isin(X)).sum()))
for y in ("sold", "log_bids", "log_views", "n_q_before_end"):
    print(y, "non-null outside the 600:", int(p0.loc[~p0["item"].isin(X), y].notna().sum()))
