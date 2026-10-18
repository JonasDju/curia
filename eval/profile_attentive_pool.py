"""Where an ``attentive_pool`` head epoch spends its time: a short profiling run, not an evaluation.

The first full Curia evaluation (2026-10-18) took ~127 min per ``attentive_pool`` head epoch, of which only ~73 min
are the encoder at the cache pass's rate. This script trains the head for a few dozen micro-batches, the way
``ClassificationEvaluator`` does (same dataset, collate, loaders, head, loss, optimizer and encoder autocast), and
reports where the time goes. Nothing is evaluated or written except the profiling results.

Runs, each on its own train exams (disjoint, so every run reads uncached files like a real head epoch):

- ``kneeno``: KneeNo's own code path (``_encoder_source`` + ``_train_head_epoch``), unmodified and unsynchronized,
  timed from the outside. Its time per exam is the reference: the instrumented runs must reproduce it.
- the variants (``VARIANTS``), the same steps written out here, with a ``torch.cuda.synchronize()`` around each
  phase: ``data`` (waiting for the loader), ``h2d`` (host-to-device copy), ``encoder``, ``select`` (turning the
  encoder output into the head's input), ``head`` (forward, loss, backward) and ``optimizer``. ``baseline``
  changes nothing; every other variant changes one thing:
  - ``tf32``: the head's float32 matmuls on TF32 tensor cores (``set_float32_matmul_precision("high")``);
  - ``head_bf16``: the head's forward and loss under bfloat16 autocast;
  - ``free_prev``: the previous micro-batch's encoder output and head input are freed before the next encoder
    call (KneeNo keeps both alive until then, ~10 GB at 16 exams);
  - ``pin_memory``: the loader pins its batches, and the copy to the GPU is non-blocking;
  - ``all``: all of them.

Every run also records the CUDA allocator's peak memory and its ``num_alloc_retries`` (a retry frees the cached
blocks and synchronizes). Each variant's last ``--profile-steps`` micro-batches run under ``torch.profiler`` (not
counted in the timings) and give a per-kernel table and a Chrome trace. ``report.json`` holds every number, together
with analytic FLOP counts and the throughput they imply.

Submit through ``eval/submit_profile_attentive_pool.sh``. By hand:

    .venv/bin/python eval/profile_attentive_pool.py --config eval/configs/profile-attentive-pool.yaml --out-dir <dir>
"""

from __future__ import annotations

# First: sets OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and PYTORCH_CUDA_ALLOC_CONF before numpy and torch are
# imported, exactly as in the evaluation run that is being profiled.
import eval_curia_kneeno  # noqa: I001

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from kneeno.evaluation import ClassificationEvaluator  # noqa: E402
from kneeno.evaluation.features import extract_exams, feature  # noqa: E402
from kneeno.evaluation.loss import build_loss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModel  # noqa: E402

from kneeno_adapter import CuriaAdapter  # noqa: E402

logger = logging.getLogger(__name__)

TASKS = ("attentive_pool", "attentive_slice_pool")
VARIANTS = {
    "baseline": {},
    "tf32": {"tf32": True},
    "head_bf16": {"head_bf16": True},
    "free_prev": {"free_prev": True},
    "pin_memory": {"pin_memory": True},
    "all": {"tf32": True, "head_bf16": True, "free_prev": True, "pin_memory": True},
}
PHASES = ("data", "h2d", "encoder", "select", "head", "optimizer")


def profile(
    config_path: Path,
    out_dir: Path,
    task: str = "attentive_pool",
    variants: tuple[str, ...] = tuple(VARIANTS),
    warmup: int = 3,
    steps: int = 20,
    profile_steps: int = 3,
) -> dict[str, Any]:
    """Profile ``task``'s head training; writes ``report.json`` (and per variant a profiler table and trace) to
    ``out_dir``, which must not exist yet. ``warmup``, ``steps`` and ``profile_steps`` count head micro-batches.
    Returns the report."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    if steps < 1 or warmup < 0 or profile_steps < 0:
        raise ValueError("steps must be >= 1, warmup and profile_steps >= 0")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    config = eval_curia_kneeno.load_config(config_path)
    (out_dir / "params.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(config["model_name"])
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    adapter = CuriaAdapter(model_name=config["model_name"], embed_dim=model.config.hidden_size)
    evaluator = ClassificationEvaluator(config=config["eval"], adapter=adapter, device=device)

    cfg = evaluator.config[task]
    micro = cfg["batch_size"] // cfg["grad_accum_steps"]
    encoder_batch = min(evaluator.extraction_batch_size, micro)  # as in ClassificationEvaluator._encoder_source
    if micro % encoder_batch:
        raise ValueError(
            f"The head micro-batch ({micro} exams) must be a multiple of the encoder batch ({encoder_batch}) here"
        )

    # Disjoint exams per run, drawn from the train part like a head epoch's.
    runs = ("kneeno", *variants)
    per_run = (warmup + steps + profile_steps) * micro
    order = torch.randperm(len(evaluator.train_indices), generator=torch.Generator().manual_seed(evaluator.seed))
    pool = [evaluator.train_indices[i] for i in order.tolist()]
    if per_run * len(runs) > len(pool):
        raise ValueError(
            f"{len(runs)} runs of {per_run} exams need {per_run * len(runs)} train exams, there are {len(pool)}; "
            "lower --steps or run fewer variants"
        )
    chunks = {name: pool[i * per_run : (i + 1) * per_run] for i, name in enumerate(runs)}

    num_slices = evaluator.config["data"]["series_depth"]
    flops = _analytic_flops(model.config, adapter.processor.crop_size, evaluator.num_sequences, num_slices, cfg)
    report = {
        "task": task,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "amp_dtype": str(evaluator.amp_dtype),
        "num_workers": evaluator.num_workers,
        "head_micro_batch": micro,
        "encoder_batch": encoder_batch,
        "grad_accum_steps": cfg["grad_accum_steps"],
        "warmup": warmup,
        "steps": steps,
        "train_exams_full": len(evaluator.train_indices),
        "flops_per_exam": flops,
        "runs": {},
    }
    logger.info("Profiling %s: %s", task, json.dumps({k: v for k, v in report.items() if k != "runs"}))

    try:
        kneeno_indices = chunks["kneeno"][: (warmup + steps) * micro]  # KneeNo's loop has no profiling pass
        report["runs"]["kneeno"] = _run_kneeno(evaluator, model, task, kneeno_indices, warmup, micro)
        _finish_run(report, "kneeno")
        for name in variants:
            report["runs"][name] = _run_variant(
                evaluator,
                model,
                task,
                chunks[name],
                VARIANTS[name],
                warmup,
                steps,
                profile_steps,
                micro,
                encoder_batch,
                out_dir / name,
            )
            _finish_run(report, name)
    finally:
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        evaluator.cleanup()
    logger.info(
        "Summary (ms per exam; train pass = one pass over all %d train exams):\n%s",
        report["train_exams_full"],
        _summary_table(report),
    )
    logger.info("Wrote %s", out_dir / "report.json")
    return report


# -- runs -----------------------------------------------------------------------------------------------------


def _new_head(evaluator, task):
    """``task``'s head, optimizer and loss as ``ClassificationEvaluator`` builds them, from the task's seed."""
    evaluator._reseed(task)
    key, prepare, head = evaluator._head_spec(task)
    cfg = evaluator.config[task]
    head = head.to(evaluator.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return key, prepare, head, optimizer, build_loss(evaluator.training_cfg)


def _run_kneeno(evaluator, model, task, indices, warmup, micro):
    """KneeNo's own head-epoch code over ``indices``, timed from the first micro-batch after ``warmup``."""
    device = evaluator.device
    key, prepare, head, optimizer, loss_fn = _new_head(evaluator, task)
    memory = _MemoryStats(device)
    start = []

    def timed(batches):
        for i, item in enumerate(batches):
            if i == warmup:
                _sync(device)
                start.append(time.perf_counter())
            yield item

    train_indices = evaluator.train_indices
    evaluator.train_indices = list(indices)  # _train_head_epoch counts the epoch's exams from it
    try:
        source = evaluator._encoder_source(model, lambda out: prepare(feature(out, key)), micro)
        try:
            evaluator._train_head_epoch(
                head, optimizer, loss_fn, timed(source.train_batches(0)), evaluator.config[task]["batch_size"]
            )
            _sync(device)
            wall = time.perf_counter() - start[0]
        finally:
            source.close()
    finally:
        evaluator.train_indices = train_indices
    exams = len(indices) - warmup * micro
    return {"exams": exams, "wall_s": wall, "phases_s": None, **memory.read()}


def _run_variant(
    evaluator, model, task, indices, options, warmup, steps, profile_steps, micro, encoder_batch, profile_dir
):
    """The head-epoch loop written out, with every phase timed; see the module docstring for ``options``."""
    device = evaluator.device
    cuda = device.type == "cuda"
    key, prepare, head, optimizer, loss_fn = _new_head(evaluator, task)
    head.train()
    accum = evaluator.config[task]["grad_accum_steps"]
    pinned = bool(options.get("pin_memory")) and cuda
    free_prev = bool(options.get("free_prev"))
    loader = _loader(evaluator, indices, encoder_batch, pinned)
    timer = _PhaseTimer(device)
    memory = _MemoryStats(device)
    precision = torch.get_float32_matmul_precision()

    def head_autocast():
        if options.get("head_bf16") and cuda:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    iterator = iter(loader)
    out = feats = None
    with contextlib.ExitStack() as stack:
        if options.get("tf32"):
            torch.set_float32_matmul_precision("high")
            stack.callback(torch.set_float32_matmul_precision, precision)
        stack.callback(_shutdown_workers, iterator)
        optimizer.zero_grad()
        for step in range(warmup + steps + profile_steps):
            if step == warmup:
                timer.start()
            if step == warmup + steps:
                timer.stop()
                if profile_steps:
                    prof = stack.enter_context(_profiler(cuda))
            parts, label_parts = [], []
            while sum(part.shape[0] for part in parts) < micro:
                if free_prev:
                    out = feats = None
                with timer("data"):
                    batch, labels = next(iterator)
                with timer("h2d"):
                    batch = _to_device(batch, device, non_blocking=pinned)
                with timer("encoder"):
                    out = extract_exams(
                        model, evaluator.adapter, batch, device, evaluator.num_sequences, evaluator.amp_dtype
                    )
                with timer("select"):
                    parts.append(prepare(feature(out, key)))
                label_parts.append(labels)
                del batch
            # A micro-batch from one encoder batch is that batch itself, as in KneeNo's rebatch.
            feats = parts[0] if len(parts) == 1 else torch.cat(parts)
            labels = label_parts[0] if len(label_parts) == 1 else torch.cat(label_parts)
            del parts, label_parts
            with timer("head"):
                labels = labels.to(device)
                with head_autocast():
                    logits = head(feats)
                    loss = loss_fn(logits.float(), labels) / accum
                loss.backward()
                del logits, loss
            if (step + 1) % accum == 0:
                with timer("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
        timer.stop()  # without a profiling pass (a no-op otherwise: it stopped before the profiler started)
        _sync(device)
    if profile_steps:
        _write_profile(prof, profile_dir, cuda)
    return {"exams": steps * micro, "wall_s": timer.wall, "phases_s": dict(timer.totals), **memory.read()}


def _loader(evaluator, indices, batch_size, pin_memory):
    """The evaluator's loader over ``indices`` (unshuffled: the indices are already a random draw), optionally
    pinning its batches."""
    loader = evaluator._loader(list(indices), batch_size, shuffle=False)
    if not pin_memory:
        return loader
    kwargs = {}
    if loader.num_workers > 0:
        kwargs = {
            "multiprocessing_context": loader.multiprocessing_context,
            "persistent_workers": loader.persistent_workers,
            "prefetch_factor": loader.prefetch_factor,
        }
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        drop_last=False,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=True,
        **kwargs,
    )


def _shutdown_workers(iterator):
    """Stop the loader's workers now: the run stops mid-pass, with batches still prefetched."""
    shutdown = getattr(iterator, "_shutdown_workers", None)  # private torch API, as in KneeNo
    if shutdown is not None:
        shutdown()


def _to_device(batch, device, non_blocking):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device, non_blocking) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    return batch


# -- measuring ------------------------------------------------------------------------------------------------


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _PhaseTimer:
    """Wall time per phase, each phase between two ``synchronize`` calls; only counted between ``start`` and
    ``stop``, which also measure the total (``wall``)."""

    def __init__(self, device):
        self.device = device
        self.totals = defaultdict(float)
        self.active = False
        self.wall = 0.0
        self._started = None

    def start(self):
        _sync(self.device)
        self.active, self._started = True, time.perf_counter()

    def stop(self):
        if self.active:
            _sync(self.device)
            self.wall = time.perf_counter() - self._started
            self.active = False

    @contextlib.contextmanager
    def __call__(self, phase):
        _sync(self.device)
        begin = time.perf_counter()
        with torch.profiler.record_function(phase):
            yield
        _sync(self.device)
        if self.active:
            self.totals[phase] += time.perf_counter() - begin


class _MemoryStats:
    """CUDA allocator statistics of one run: peaks since the run started and allocation retries during it."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.device = device
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            self.retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
            self.ooms = torch.cuda.memory_stats(device).get("num_ooms", 0)

    def read(self):
        if not self.cuda:
            return {}
        stats = torch.cuda.memory_stats(self.device)
        gib = 2**30
        return {
            "peak_allocated_gib": stats.get("allocated_bytes.all.peak", 0) / gib,
            "peak_reserved_gib": stats.get("reserved_bytes.all.peak", 0) / gib,
            "alloc_retries": stats.get("num_alloc_retries", 0) - self.retries,
            "ooms": stats.get("num_ooms", 0) - self.ooms,
        }


def _profiler(cuda):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities)


def _write_profile(prof, profile_dir, cuda):
    profile_dir.mkdir(parents=True)
    sort_by = "self_device_time_total" if cuda else "self_cpu_time_total"
    (profile_dir / "kernels.txt").write_text(prof.key_averages().table(sort_by=sort_by, row_limit=40))
    prof.export_chrome_trace(str(profile_dir / "trace.json"))


def _analytic_flops(model_config, crop_size, num_sequences, num_slices, head_cfg):
    """FLOPs per exam of the encoder forward and of the head's key/value projections (forward + backward, the
    head's dominant matmuls), counted as 2 per multiply-add. Biases, norms and softmax are left out.

    The head count assumes no token self-attention blocks (``patch_sa_depth: 0``), and is None otherwise.
    """
    dim, layers, patch = model_config.hidden_size, model_config.num_hidden_layers, model_config.patch_size
    patches = (crop_size // patch) ** 2
    tokens = patches + 1  # + cls; Curia has no register tokens
    hidden = int(dim * model_config.mlp_ratio)
    if getattr(model_config, "use_swiglu_ffn", False):
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8  # transformers' Dinov2SwiGLUFFN
        ffn = 3 * dim * hidden  # weights_in (dim -> 2 * hidden) + weights_out (hidden -> dim)
    else:
        ffn = 2 * dim * hidden
    per_image = layers * (tokens * (4 * dim * dim + ffn) + 2 * tokens * tokens * dim)
    per_image += patches * dim * patch * patch * model_config.num_channels
    images = num_sequences * num_slices
    head_tokens = images * patches
    head = None
    if head_cfg["patch_sa_depth"] == 0:
        # kv: (tokens, dim) @ (dim, 2 * dim) per decoder layer; backward = weight grad + input grad.
        head = 3 * 2 * head_tokens * dim * 2 * dim * head_cfg["qa_depth"]
    return {"encoder": 2 * per_image * images, "head_train": head, "head_tokens": head_tokens}


# -- reporting ------------------------------------------------------------------------------------------------


def _finish_run(report, name):
    """Add per-exam times, projections and throughputs to run ``name`` and log it."""
    run = report["runs"][name]
    exams, flops = run["exams"], report["flops_per_exam"]
    run["wall_ms_per_exam"] = 1000 * run["wall_s"] / exams
    run["train_pass_min"] = run["wall_s"] / exams * report["train_exams_full"] / 60
    phases = run["phases_s"]
    if phases is not None:
        run["phases_ms_per_exam"] = {phase: 1000 * phases.get(phase, 0.0) / exams for phase in PHASES}
        run["encoder_tflops"] = flops["encoder"] * exams / phases["encoder"] / 1e12 if phases.get("encoder") else None
        run["head_tflops"] = (
            flops["head_train"] * exams / phases["head"] / 1e12
            if flops["head_train"] is not None and phases.get("head")
            else None
        )
    logger.info("Run %s: %s", name, json.dumps(run))


def _summary_table(report):
    columns = ("wall", *PHASES, "train pass min", "enc TFLOP/s", "head TFLOP/s", "peak alloc GiB", "retries")
    rows = []
    for name, run in report["runs"].items():
        phases = run.get("phases_ms_per_exam") or {}
        cells = [
            f"{run['wall_ms_per_exam']:.1f}",
            *(f"{phases[phase]:.1f}" if phase in phases else "-" for phase in PHASES),
            f"{run['train_pass_min']:.1f}",
            _fmt(run.get("encoder_tflops")),
            _fmt(run.get("head_tflops")),
            _fmt(run.get("peak_allocated_gib")),
            str(run.get("alloc_retries", "-")),
        ]
        rows.append((name, *cells))
    header = ("run", *columns)
    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header))]
    return "\n".join("  ".join(str(cell).rjust(width) for cell, width in zip(row, widths)) for row in (header, *rows))


def _fmt(value):
    return "-" if value is None else f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="eval config YAML (model_name + eval)")
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory; must not exist")
    parser.add_argument("--task", default="attentive_pool", choices=TASKS)
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {list(VARIANTS)}")
    parser.add_argument("--warmup", type=int, default=3, help="untimed micro-batches at the start of each run")
    parser.add_argument("--steps", type=int, default=20, help="timed micro-batches per run")
    parser.add_argument("--profile-steps", type=int, default=3, help="micro-batches under torch.profiler per variant")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    profile(
        args.config,
        args.out_dir,
        task=args.task,
        variants=tuple(v for v in args.variants.split(",") if v),
        warmup=args.warmup,
        steps=args.steps,
        profile_steps=args.profile_steps,
    )


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
