#!/bin/bash
# Standalone KneeNo classification evaluation of an official Curia checkpoint. Submit from the repo root:
#     sbatch eval/submit_eval_curia_kneeno.sh
# Pass env vars (CONFIG, REPO_DIR, KNEENO_DIR, SPLIT_LEVEL, SPLIT_REP) with --export: RWTH's sbatch submits with
# --export=NONE, so a `VAR=... sbatch ...` prefix never reaches the job, e.g. `sbatch --export=ALL,CONFIG=... eval/submit_eval_curia_kneeno.sh`.
# SPLIT_LEVEL / SPLIT_REP (default 1.0 / 0) pick the split manifest's train/val split, one job per (level, rep):
#     for level in 1.0 0.1 0.01; do for rep in 0 1 2; do
#         sbatch --export=ALL,SPLIT_LEVEL=$level,SPLIT_REP=$rep eval/submit_eval_curia_kneeno.sh
#     done; done
#
# Compute nodes are treated as offline, so the job runs with HF_HUB_OFFLINE=1. Download the model once on the
# login node first (after `.venv/bin/hf auth login` if the repo is gated):
#     .venv/bin/hf download raidium/curia
# The evaluation runs in a single process on one GPU (KneeNo evaluates on rank 0 only, more GPUs would not help).
# --cpus-per-task feeds the eval loader's eval.data.num_workers (31) workers. Curia encodes 512² slices (1,024
# patches each, 4x DINOv2 at 224²) and the attentive tasks re-encode every head epoch, hence the long time limit.
##SBATCH --account=truhnlab
##SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=curia-b-eval
#SBATCH --output=slurm-%j-curia_kneeno_eval.out

set -euo pipefail

# --- Locations (override through --export) ---------------------------------------------------------------
# Model, output directories and everything else about the evaluation are in the eval config.
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
CONFIG="${CONFIG:-eval/configs/eval-MI-curia-b.yaml}"  # relative to REPO_DIR
# The split of KneeNo's split manifest the heads train and validate on (the eval config's eval.split.level / rep).
# Defaults: the whole internal "train" split, rep 0. Each (level, rep) writes to its own output folder.
export SPLIT_LEVEL="${SPLIT_LEVEL:-1.0}"
export SPLIT_REP="${SPLIT_REP:-0}"
# The job's node-local scratch: the data is extracted there and the eval config's data_root points there.
: "${TMP:?TMP (the node-local scratch directory of the job) is not set}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Fail here rather than after the data extraction
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Eval config not found at $CONFIG (set CONFIG)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "The evaluation runs in a single process; submit with one task (got ${SLURM_NTASKS})" >&2
    exit 1
fi

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

# --- Data -> node-local scratch ($TMP/kneeno_data/{internal,external}, prepare_data.py's default): the labeled
# eval datasets (internal: train part + "internal" test set; external: "external" test set); each skipped if
# already populated
srun "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     --external-tar /hpcwork/p0021834/workspace_roman/jonas/uka_gold_nifti.tar --pool-size 16

# --- Evaluation -------------------------------------------------------------------------------------------
srun "$PYTHON" eval/eval_curia_kneeno.py --config "$CONFIG"

# Cleanup
rm -rf "$TMP/kneeno_data"
echo "Deleted KneeNo data"
