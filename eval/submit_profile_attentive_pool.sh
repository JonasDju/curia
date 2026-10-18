#!/bin/bash
# Profiles where an attentive_pool head epoch spends its time (eval/profile_attentive_pool.py). Submit from the
# repo root:
#     sbatch eval/submit_profile_attentive_pool.sh
# Env vars (pass with --export=ALL,..., RWTH's sbatch uses --export=NONE): CONFIG (default
# eval/configs/profile-attentive-pool.yaml), OUT_DIR (default /hpcwork/va105917/curia/profile/<job id>; must not
# exist), TASK (attentive_pool or attentive_slice_pool, default attentive_pool), REPO_DIR, KNEENO_DIR.
#
# Seven runs of 26 head micro-batches (16 exams each at the default settings) take ~15-20 min after the ~2 min
# data extraction. Like the evaluation, it needs the model in the HF cache (`.venv/bin/hf download raidium/curia`
# on the login node) and runs with HF_HUB_OFFLINE=1.
##SBATCH --account=truhnlab
##SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --job-name=curia-b-profile
#SBATCH --output=slurm-%j-curia_profile_attentive_pool.out

set -euo pipefail

# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
CONFIG="${CONFIG:-eval/configs/profile-attentive-pool.yaml}"  # relative to REPO_DIR
OUT_DIR="${OUT_DIR:-/hpcwork/va105917/curia/profile/$SLURM_JOB_ID}"
TASK="${TASK:-attentive_pool}"
: "${TMP:?TMP (the node-local scratch directory of the job) is not set}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Profile config not found at $CONFIG (set CONFIG)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
[[ ! -e "$OUT_DIR" ]] || { echo "$OUT_DIR exists already (set OUT_DIR)" >&2; exit 1; }

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"

# The model must already be in the Hugging Face cache (see the header)
"$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
from huggingface_hub import snapshot_download
name = yaml.safe_load(open(sys.argv[1]))["model_name"]
try:
    snapshot_download(name, local_files_only=True)
except Exception as ex:
    sys.exit(f"Model {name!r} is not in the Hugging Face cache ({ex}); run `.venv/bin/hf download {name}` on the login node")
PY

# --- Data -> node-local scratch: the internal labeled data only (no test set is scored) ----------------------
srun "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     --pool-size 16

# --- Profile --------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
srun "$PYTHON" eval/profile_attentive_pool.py --config "$CONFIG" --out-dir "$OUT_DIR" --task "$TASK"

# Cleanup
rm -rf "$TMP/kneeno_data"
echo "Deleted KneeNo data"
