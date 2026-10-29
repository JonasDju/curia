"""Standalone KneeNo classification evaluation of an official Curia checkpoint.

The counterpart of lightly-train's ``cluster/eval_dinov2_kneeno.py`` + ``lightly-train eval_classification``. The
eval config (``--config``) is a YAML with exactly

    model_name:  the Hugging Face model to evaluate, e.g. raidium/curia (or a local directory)
    eval:        KneeNo's evaluation config (deep-merged over KneeNo's DEFAULT_EVAL_CONFIG)

Every task with a non-null ``eval.freq`` runs once and logs its test metrics (heads early-stopped on val). Environment variables
are expanded in every string value; an unset one is an error. Next to ``eval.logging.per_label_dir`` the script
writes the final metrics (``results.json``) and the expanded config (``params.yaml``, once the model, dataset and
split are loaded, before evaluating); it refuses to start if earlier results exist there. The evaluation runs in a
single process: KneeNo evaluates on rank 0 only.

Submit through ``eval/submit_eval_curia_kneeno.sh``; that also prepares the node-local data first. Run by hand:

    .venv/bin/python eval/eval_curia_kneeno.py --config eval/configs/eval-MI-curia-b.yaml
"""

from __future__ import annotations

import os

# Must be set before numpy is imported (OpenBLAS reads it once, at load time), hence above the other imports.
# numpy's bundled OpenBLAS otherwise starts one thread per core in every DataLoader worker -- torch only limits
# its own threads there -- so any BLAS work in the workers oversubscribes the cores. Workers inherit the
# environment, under fork and spawn alike.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Must be set before torch is imported. torch otherwise runs CPU ops in the main process on one OpenMP thread per
# core, on cores the eval DataLoader workers keep busy. Forced, not setdefault: the cluster's job environment
# already sets OMP_NUM_THREADS=32, which would otherwise win.
os.environ["OMP_NUM_THREADS"] = "1"

# Expandable segments let freed memory be reused for any size. setdefault, so an explicit setting in the job
# environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from kneeno import expand_env_vars  # noqa: E402
from kneeno.evaluation import ClassificationEvaluator, enabled_tasks, load_eval_config  # noqa: E402
from transformers import AutoModel  # noqa: E402

from kneeno_adapter import CuriaAdapter  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_KEYS = ("model_name", "eval")
# What os.path.expandvars expands when the variable is set, i.e. what is left of an unset one.
_ENV_VAR_PATTERN = re.compile(r"\$(\{\w+\}|[A-Za-z_]\w*)")


def load_config(path: Path) -> dict[str, Any]:
    """The eval config file with env vars expanded, in the file's key order (what ``params.yaml`` records)."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or set(raw) != set(CONFIG_KEYS):
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"Eval config '{path}' must have exactly the top-level keys {CONFIG_KEYS}, got {keys}.")
    if not isinstance(raw["model_name"], str) or not isinstance(raw["eval"], dict):
        raise ValueError(f"Eval config '{path}' needs 'model_name' as a string and 'eval' as a mapping.")
    config = expand_env_vars(raw)
    # expand_env_vars leaves unset variables as they are, which would create a literal '${VAR}' directory.
    unexpanded = sorted(set(_find_env_vars(config)))
    if unexpanded:
        raise ValueError(f"Eval config '{path}' uses environment variables that are not set: {unexpanded}.")
    return config


def _find_env_vars(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [var for value in obj.values() for var in _find_env_vars(value)]
    if isinstance(obj, (list, tuple)):
        return [var for value in obj for var in _find_env_vars(value)]
    if isinstance(obj, str):
        return [match.group(0) for match in _ENV_VAR_PATTERN.finditer(obj)]
    return []


def evaluate(config_path: Path) -> dict[str, float]:
    config = load_config(config_path)
    eval_cfg = load_eval_config(config["eval"])
    tasks = enabled_tasks(eval_cfg["freq"])
    if not tasks:
        raise ValueError(f"No task is enabled: every task in eval.freq is null or <= 0 ({eval_cfg['freq']}).")
    if eval_cfg.get("encoder", "target") != "target":
        raise ValueError(f"Curia has a single set of weights; eval.encoder must be 'target', got {eval_cfg['encoder']!r}.")

    # Everything that can fail before the (hours long) evaluation does so now.
    log_cfg = eval_cfg["logging"]
    if log_cfg["per_label_dir"] is None:
        raise ValueError("eval.logging.per_label_dir must be set: results.json and params.yaml are written next to it.")
    out_dir = Path(log_cfg["per_label_dir"]).resolve().parent
    results_path, params_path = out_dir / "results.json", out_dir / "params.yaml"
    existing = [
        str(path)
        for path in (results_path, params_path, log_cfg["per_label_dir"], log_cfg["tensorboard_dir"])
        if path is not None and Path(path).exists()
    ]
    if existing:
        raise ValueError(f"Results of an earlier evaluation exist: {existing}. Delete them or point eval.logging elsewhere.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading '{config['model_name']}', evaluating on {device} on tasks {tasks}.")
    model = AutoModel.from_pretrained(config["model_name"])
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    adapter = CuriaAdapter(model_name=config["model_name"], embed_dim=model.config.hidden_size)

    evaluator = ClassificationEvaluator(config=eval_cfg, adapter=adapter, device=device)
    # Written after everything that fails fast (the config, the model, and the dataset and split the evaluator builds):
    # a config they reject must not leave a params.yaml that makes the corrected rerun refuse. But before evaluating,
    # so a run that crashes still records its settings.
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path.write_text(yaml.safe_dump(config, sort_keys=False))
    logger.info(f"Wrote the evaluation config to '{params_path}'")
    try:
        metrics = evaluator.evaluate(model, tasks=tasks, epoch=0)
    finally:
        evaluator.cleanup()

    logger.info(f"Final metrics: {metrics}")
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info(f"Wrote metrics to '{results_path}'")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="eval config YAML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(args.config)


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
