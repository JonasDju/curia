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
  - Outputs in the parent of `eval.logging.per_label_dir` (required): `params.yaml` (expanded config, written before
    the model is loaded) and `results.json`. Refuses if either, `per_label_dir` or `tensorboard_dir` exists; there is
    no overwrite flag.
  - One `evaluate(..., epoch=0)` call: each task logs its test metrics once (KneeNo's train/val/test protocol, heads
    early-stopped on val; `eval.verbose` logs the val metrics per head epoch), plus each trained head's best-epoch
    `<task>/val_auroc`. CUDA if available, else CPU. The config sets `eval.data.abort_on_io_error: true`: an exam
    that fails to load raises instead of being replaced by one of another split.
- `eval/submit_eval_curia_kneeno.sh`: the 2D lightly-train eval sbatch without `PRETRAIN_JOB_ID`. 1 GPU, 1 task,
  32 CPUs, 24 h. Extracts the internal data to `$TMP/kneeno_data`, runs the launcher, cleans up. Env vars (pass with
  `--export=ALL,...`, RWTH's sbatch uses `--export=NONE`): `CONFIG` (default `eval/configs/eval-MI-curia-b.yaml`),
  `REPO_DIR`, `KNEENO_DIR`.
  - **Compute nodes are treated as offline:** the job sets `HF_HUB_OFFLINE=1` and checks the config's `model_name` is
    in the HF cache *before* extracting the data. Download once on the login node: `.venv/bin/hf download raidium/curia`.
- `eval/configs/eval-MI-curia-b.yaml`: the 2D DINOv2 eval config's `eval:` block (same heads, split, seed, data),
  outputs under `/hpcwork/va105917/curia/curia-b/eval/internal/`, except `knn.batch_size: 8` instead of 32. That only
  sets the shared extraction pass's batch (results unchanged). At 512² a 32-exam batch is ~3 GiB of float32 input,
  and 31 workers × prefetch 2 could exceed the job's ~186 GB. The head batch sizes are kept for comparability; the
  re-encoding tasks still load 16 (`attentive_pool`) and 8 (`attentive_slice_pool`) exams per batch.
- Not run on the cluster yet.

## Testing

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests -q
```

- `tests/conftest.py`: `tiny_curia_dir`, a local tiny random `Dinov2Model` (1 channel, SwiGLU, 32², patch 8) plus the
  repo-root processor with `crop_size: 32`, loaded through the same `AutoModel` / `AutoImageProcessor` calls, offline.
  `FakeLabeledDataset` stands in for the labeled datasets (internal `orientations`, uint8 or raw uint16 volumes).
- `tests/test_kneeno_adapter.py`: preprocessing (dtypes, whole-volume z-score, slice order, no transpose, orientation
  guard against both datasets' codes), `forward_features` vs. per-slice encoding, a full evaluator run with
  `num_workers=2`, and tests against the real `raidium/curia` (skipped unless it is in the local HF cache).
- `tests/test_eval_curia_kneeno.py`: the launcher end to end (all tasks, outputs, env var expansion), task selection,
  refusal on existing results, config errors, and that the shipped config loads.
- `pytest` puts `eval/` on `sys.path` (`[tool.pytest.ini_options] pythonpath`): it is not a package, and the
  launcher imports `kneeno_adapter` as a sibling module.
