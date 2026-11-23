#!/bin/bash
# Session 8: parse every banked Municibid page with the from-scratch parser (code/mb_parse.py).
# Reads analysis/muni_data/pages (read-only); writes only results/s8_parse_<jobid>/.
#SBATCH --job-name=s8_parse
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:30:00
#SBATCH --output=/project/dmv_project/projects/ai_pricing_kbb/log/%x_%j.out
#SBATCH --error=/project/dmv_project/projects/ai_pricing_kbb/err/%x_%j.err
set -euo pipefail
P=/project/dmv_project/projects/ai_pricing_kbb
D=$P/analysis/muni_data
OUT=$P/results/s8_parse_$SLURM_JOB_ID
export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
mkdir -p "$OUT"
cp "$P/code/mb_parse.py" "$P/code/parse.sbatch" "$OUT/"
sha256sum "$OUT/mb_parse.py" "$OUT/parse.sbatch" "$D/frame.jsonl" > "$OUT/SHA256SUMS"
python3 -V
python3 -B "$OUT/mb_parse.py" selftest
python3 -B "$OUT/mb_parse.py" run --pages "$D/pages" --frame "$D/frame.jsonl" --out "$OUT/pages_s8p2.jsonl" --workers 8
sha256sum "$OUT/pages_s8p2.jsonl" >> "$OUT/SHA256SUMS"
wc -l "$OUT/pages_s8p2.jsonl"
rm -rf "$TMPDIR"
