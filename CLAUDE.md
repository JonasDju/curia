# CLAUDE.md

## What this repo is

A clone of **Raidium's open-source Curia repo** (official evaluation code for the Curia medical foundation model;
weights at `huggingface.co/raidium/curia`). It's a SOTA comparison in the thesis, like 3DINO, scored with the same
KneeNo classification evaluation as the thesis models. See `MA/CLAUDE.md` for how the repos fit together.

Upstream code (`trainer.py`, `modeling_dinov2.py`, `curia_image_processor.py`, `pre_processor.py`, `configs/`,
`simple_test.py`) is left untouched. Ours: `eval/` (KneeNo adapter, standalone eval launcher, sbatch, config),
`tests/`, the `kneeno` + `pytest` dependencies and the pytest settings in `pyproject.toml`.

**The model:** Curia-B on the Hub is a stock `transformers` `Dinov2Model` (no `auto_map`, so no remote model code):
ViT-B/16, 1 input channel, 512², SwiGLU FFN, no registers, `hidden_size` 768. `last_hidden_state` is post-layernorm,
index 0 is the cls token (what Curia's own `trainer.py` uses). Only the image processor is remote code
(`trust_remote_code=True`): the Hub's `curia_image_processor.py` equals the repo-root copy up to formatting (pinned by
a test). `pre_processor.py` is a different, unused variant (per-slice z-score, argmin slice axis); don't use it.

## Environment

- **`uv`**, Python **3.13**, `torch 2.9.0+cu128`, `transformers 4.57.1`. Run Python as **`.venv/bin/python`**.
- `kneeno` is an **editable path dependency** in `pyproject.toml` (`../KneeNo`), as in vjepa2, so `uv sync` keeps it.
- No usable GPU on the dev box: everything is verified on CPU.

## KneeNo evaluation

`eval/kneeno_adapter.py::CuriaAdapter(model_name, embed_dim)`: slice-wise, like lightly-train's 2D `DINOv2Adapter`.
- `has_cls_token = True`, `has_slice_tokens = True`: every task runs, the slice tasks included. Needs
  `eval.data.series_depth > 0`.
- `prepare_input`: `(1, D, H, W)` → `(H, W, D)` → Curia's processor → `(1, D, 512, 512)` float32. The processor
  resizes each slice (bicubic, antialias) and **z-scores the whole volume** with one mean/std (only its 2D path is
  per image). That is exactly Curia's own 3D benchmark path (`trainer.py::extract_features`), pinned by a test. It
  ignores `do_normalize`. Internal (uint8) and external (raw uint16/float) volumes are handled the same way, no
  `to_uint8`: Curia documents raw values as input, and the z-score makes intensity scale irrelevant. So the
  `source` argument KneeNo passes (`"internal"` / `"external"`, per exam) is ignored; `FakeLabeledDataset` still returns
  it as the third item element, as every evaluation dataset must.
- **Orientation:** Curia wants slices `(H, W)` in `PL` (axial), `IL` (coronal), `IP` (sagittal), read as axcodes
  (standard radiological display). KneeNo's labeled datasets (both) return LIP / PIL / SPL in `(D, H, W)` order, so the
  in-plane codes already match and nothing is reoriented. `prepare_input` raises if `orientation[1:]` is anything else
  (`None`, from test fakes, is allowed).
- `forward_features`: all `B * D` slices in one forward → `slice_cls` `(B, D, 768)`, `slice_patches`
  `(B, D, 1024, 768)`. 1,024 patches per slice (4x DINOv2 224²/14), so memory and time per slice are much higher than
  for the 2D DINOv2 baseline.

## Standalone evaluation

- `eval/eval_curia_kneeno.py --config <yaml>`: mirrors lightly-train's `cluster/eval_dinov2_kneeno.py` +
  `eval_classification` command in one file. Sets `OPENBLAS_NUM_THREADS=1`, forced `OMP_NUM_THREADS=1` and
  `expandable_segments` before importing numpy/torch.
  - Config: exactly `model_name` (Hub id or local dir) and `eval`; env vars expanded, **an unset one is an error**.
  - Tasks: those with a non-null `> 0` `eval.freq` on the merged config (a missing key keeps KneeNo's default).
    `eval.encoder` must be `target` (KneeNo's default): Curia has one set of weights.
  - Outputs in the parent of `eval.logging.per_label_dir` (required): `params.yaml` (expanded config, written once
    the model is loaded and the evaluator has built the dataset and split, before evaluating; since 2026-10-29, before
    that a bad model name or split left one behind that blocked the rerun) and `results.json`. Refuses if either,
    `per_label_dir` or `tensorboard_dir` exists; there is no overwrite flag.
  - One `evaluate(..., epoch=0)` call: each task logs its metrics once per test set (KneeNo's global-split protocol:
    heads train on the internal `train` split, early-stopped on val; `eval.verbose` logs the val metrics per head
    epoch, then the best head is scored on each test set), keys `{test_set}/{task}/{metric}`, plus each trained
    head's best-epoch `evaluation_val/{task}/val_auroc` and chosen `evaluation_val/{task}/lr`, and k-NN each k's
    `evaluation_val/knn/k{k}/val_auroc` and chosen `.../temperature`. CUDA if available, else CPU. The config sets
    `eval.data.abort_on_io_error: true`: an exam that fails to load raises instead of being replaced by another one.
- `eval/submit_eval_curia_kneeno.sh`: the 2D lightly-train eval sbatch without `PRETRAIN_JOB_ID`. 1 GPU, 1 task,
  32 CPUs, 24 h. Extracts the internal and the external data (`--external-tar /hpcwork/p0021834/workspace_roman/jonas/uka_gold_nifti.tar`,
  hard-coded) to `$TMP/kneeno_data`, runs the launcher, cleans up. Env vars (pass with `--export=ALL,...`, RWTH's
  sbatch uses `--export=NONE`): `CONFIG` (default `eval/configs/eval-MI-curia-b.yaml`), `REPO_DIR`, `KNEENO_DIR`,
  `SPLIT_LEVEL` / `SPLIT_REP` (default `1.0` / `0`, see below).
  - **Compute nodes are treated as offline:** the job sets `HF_HUB_OFFLINE=1` and checks the config's `model_name` is
    in the HF cache *before* extracting the data. Download once on the login node: `.venv/bin/hf download raidium/curia`.
- `eval/configs/eval-MI-curia-b.yaml`: the 2D DINOv2 eval config's `eval:` block (same heads, split, seed, data),
  `test_sets: [internal, external]` (`label_meta.external`: `/hpcwork/p0021834/workspace_roman/jonas/metadata/external_labeled.json`),
  outputs under `/hpcwork/va105917/curia/curia-b/eval/internal/`.
  - Train/val (since 2026-10-28): `split.manifest` (KneeNo's split manifest,
    `/hpcwork/p0021834/workspace_roman/jonas/metadata/internal_split_manifest.json`), `level: ${SPLIT_LEVEL}`,
    `rep: ${SPLIT_REP}` (since 2026-10-29), which the submit script exports (default `1.0` / `0`): one job per
    (level, rep) to report, `sbatch --export=ALL,SPLIT_LEVEL=0.05,SPLIT_REP=3 ...`. The output folders end in
    `level-${SPLIT_LEVEL}.rep-${SPLIT_REP}/`, so the runs don't collide (the launcher refuses existing results).
    Running the launcher without the submit script needs both set (an unset env var is an error). `split.val_fraction` and `data.subset` are gone (they raise). The tests write a
    manifest for the fakes with `kneeno.evaluation.splits.write_manifest_for`.
  - `attentive_pool` is off (`freq: null`): it is the weaker variant of `attentive_slice_pool` (same tokens, without
    knowing their slice) and takes hours.
  - Early stopping: `training.early_stop_patience_global: 0` (none) except `early_stop_patience: 5` on the two
    re-encoding heads, `attentive_pool` and `attentive_slice_pool` (the cross-repo rule, see `MA/CLAUDE.md`).
  - Sweeps (since 2026-10-23): `lr: [0.0001, ..., 0.02]` (8 values) on `linear`, `linear_pool` and
    `linear_slice_cls`, and `knn.temperature: [0.01, ..., 1.0]` (8 values). Every attentive task has `lr: 0.0005`.
    The two re-encoding heads train with batch 64 as 4 micro-batches of 16 (`grad_accum_steps: 4`),
    `max_epochs: 15` and `warmup_epochs: 3`.
  - `encoder_amp_dtype: bfloat16` (`amp_dtype` before), `head_amp_dtype: null`, `compile_attentive_heads: true`.
  - Memory-only settings (results unchanged): `extraction_batch_size: 8` instead of 32, `data.prefetch_factor: 8`,
    `grad_accum_steps: 1` on every head but the two re-encoding ones (`4` since 2026-10-23, with their `batch_size`
    raised to 64). At 512² a 32-exam batch is ~3 GiB of float32 input. An earlier run with 8 exams per batch, 31
    workers and prefetch 2 ran out of worker RAM in the cache pass. A later commit (`07e2ea3`) set 8 / 8 / 1 while
    this file still said 4 / 1 / 4, so 8 / 8 / 1 is unverified on the cluster. The other heads' batch sizes
    (`batch_size`) are kept for comparability. `extraction_batch_size` caps every encoder call,
    in the cache pass and in the re-encoding tasks; `grad_accum_steps` only sets the head's micro-batch
    (`batch_size / grad_accum_steps`). If a re-encoding head runs out of GPU memory, raise its `grad_accum_steps`.
  - `compile_encoder: true` (KneeNo's `torch.compile` of the encoder forward, rounding-only): the encoder took 57
    instead of 86 ms per exam in the profile (H100).
- Not run on the cluster yet.

## Testing

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests -q
```

- `tests/conftest.py`: `tiny_curia_dir`, a local tiny random `Dinov2Model` (1 channel, SwiGLU, 32², patch 8) plus the
  repo-root processor with `crop_size: 32`, loaded through the same `AutoModel` / `AutoImageProcessor` calls, offline.
  `FakeLabeledDataset` stands in for one labeled dataset split (internal `orientations`, uint8 or raw uint16
  volumes, `samples` with case ids); `fake_eval_dataset(test_sets)` wraps fakes in the `KneeNoEvalDataset` that
  `from_config` would build (the launcher tests monkeypatch `KneeNoEvalDataset.from_config` to return it).
- `tests/test_kneeno_adapter.py`: preprocessing (dtypes, whole-volume z-score, slice order, no transpose, orientation
  guard against both datasets' codes), `forward_features` vs. per-slice encoding, a full evaluator run with
  `num_workers=2`, and tests against the real `raidium/curia` (skipped unless it is in the local HF cache).
- `tests/test_eval_curia_kneeno.py`: the launcher end to end (all tasks, outputs, env var expansion), task selection,
  refusal on existing results, config errors, and the shipped config: it loads, enables every task but
  `attentive_pool`, and has the patience settings above (`5` on the re-encoding heads).
- `pytest` puts `eval/` on `sys.path` (`[tool.pytest.ini_options] pythonpath`): it is not a package, and the
  launcher imports `kneeno_adapter` as a sibling module.
