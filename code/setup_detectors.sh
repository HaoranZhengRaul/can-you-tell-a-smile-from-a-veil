#!/bin/bash
# Session 9: the Python environment and model weights for code/std_detectors.py, in our project folder.
# PI, 2026-10-26: download to our workspace and delete the models after the detector runs (an explicit exception to
# the constitution's "no model weights on /project"); everything sits under one folder, removed in one command:
#   rm -rf /project/dmv_project/projects/ai_pricing_kbb/tmp_models
# Runs on a Mercury login node (compute nodes have no internet); detach it with nohup and read the log.
#   nohup bash code/setup_detectors.sh > log/s9_setup_detectors.log 2>&1 &
# The venv carries its own copies of the analysis packages (pinned to the versions in ~/.local), because a venv does
# not see the user site-packages. Weights: the authors' default pair for Binoculars and Fast-DetectGPT,
# tiiuae/falcon-7b and tiiuae/falcon-7b-instruct (Apache-2.0), safetensors only (14.4 GB each).
set -euo pipefail
ROOT="${DET_ROOT:-/project/dmv_project/projects/ai_pricing_kbb/tmp_models}"
case "$ROOT" in /project/dmv_project/projects/ai_pricing_kbb/*|/scratch/$USER/*) ;; *) echo "refusing ROOT=$ROOT"; exit 2;; esac
mkdir -p "$ROOT/tmp" "$ROOT/pip_cache" "$ROOT/hf_models" "$ROOT/hf_home"
echo "Temporary: model weights and a Python environment for code/std_detectors.py (Session 9). Delete this whole folder once the detector runs are done (PI, 2026-10-26)." > "$ROOT/README_DELETE_AFTER_USE.txt"
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/pip_cache" HF_HOME="$ROOT/hf_home" HF_HUB_DISABLE_TELEMETRY=1 ROOT
cd "$ROOT"
echo "start $(date -u +%FT%TZ) on $(hostname) pid $$; free where ROOT is: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/python -m pip install --quiet --upgrade pip
venv/bin/python -m pip install --quiet "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
venv/bin/python -m pip install --quiet "transformers==4.44.2" "huggingface_hub>=0.24,<1" "safetensors>=0.4" \
    "numpy==2.0.2" "pandas==2.3.3" "scipy==1.13.1" "scikit-learn==1.6.1" "pyarrow==21.0.0"
venv/bin/python -c "import torch, transformers, pandas, sklearn, pyarrow; print('ENV_OK torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)"
venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
root = os.environ["ROOT"]
for repo in ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"):
    dest = os.path.join(root, "hf_models", repo.split("/")[1])
    snapshot_download(repo, local_dir=dest, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    files = sorted(os.listdir(dest))
    size = sum(os.path.getsize(os.path.join(dest, f)) for f in files if os.path.isfile(os.path.join(dest, f)))
    print("DOWNLOADED", repo, "%.1f GB" % (size / 1e9), files)
PY
rm -rf "$ROOT/tmp"/* "$ROOT/pip_cache"   # the cache is not needed once the packages are installed
du -sh "$ROOT/hf_models"/* "$ROOT/venv"
echo "SETUP_DONE $(date -u +%FT%TZ)"
