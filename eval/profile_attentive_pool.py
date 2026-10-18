"""Where an ``attentive_pool`` head epoch spends its time, and what each speed-up costs in numerical accuracy: a
short profiling run, not an evaluation.

The first full Curia evaluation (2026-10-18) took ~127 min per ``attentive_pool`` head epoch. The first profile
(job 5121209) split that into ~86 ms per exam for the encoder, ~104 ms for the head (almost all of it the float32
attention backward, PyTorch's memory-efficient kernel) and ~17 ms for the unpinned host-to-device copy. This script
trains the head for a few dozen micro-batches, the way ``ClassificationEvaluator`` does (same dataset, collate,
loaders, head, loss, optimizer and encoder autocast), and reports where the time goes, once per variant. Nothing is
evaluated or written except the profiling results.

Runs, each on its own train exams (disjoint, so every run reads uncached files like a real head epoch):

- ``kneeno``: KneeNo's own code path (``_encoder_source`` + ``_train_head_epoch``), unmodified and unsynchronized,
  timed from the outside. Its time per exam is the reference: the instrumented runs must reproduce it.
- the variants (``VARIANTS``), the same steps written out here, with a ``torch.cuda.synchronize()`` around each
  phase: ``data`` (waiting for the loader), ``h2d`` (host-to-device copy), ``encoder``, ``select`` (turning the
  encoder output into the head's input), ``head`` (forward, loss, backward) and ``optimizer``. ``baseline``
  changes nothing. The options the variants combine:
  - bit-identical results: ``pin_memory`` (the loader pins its batches, the copy to the GPU is non-blocking) and
    ``free_prev`` (the previous micro-batch's encoder output and head input are freed before the next encoder
    call; KneeNo keeps both alive until then, ~10 GB at 16 exams);
  - the same precision, different rounding: ``math_sdpa`` (the head's attention through PyTorch's math backend,
    ``softmax(q k^T) v`` as plain matmuls, instead of the memory-efficient kernel) and ``encoder_compile``
    (``torch.compile`` of the encoder, which fuses its element-wise ops; ``dynamic=False``, one graph per input
    shape);
  - lower precision: ``tf32`` (the head's float32 matmuls on TF32 tensor cores), ``head_autocast`` (the head's
    forward under bfloat16 autocast, as the encoder already runs: float32 weights, matmuls and attention in
    bfloat16, norms, softmax and loss in float32) and ``encoder_bf16`` (the whole encoder in bfloat16: weights
    and input cast, no autocast, so unlike under autocast the residual stream and the norms are bfloat16 too).
  ``exact`` combines every option of the first two groups, ``fastest`` the fastest of each kind.

Every run also records the CUDA allocator's peak memory and its ``num_alloc_retries`` (a retry frees the cached
blocks and synchronizes), and how long its warm-up took (worker start-up and, for a compiled encoder, its
compilation). Each variant's last ``--profile-steps`` micro-batches run under ``torch.profiler`` (not counted in the
timings) and give a per-kernel table and a Chrome trace.

**Numerics.** ``--numerics-exams`` exams (loaded in this process) are encoded once in float32 without autocast and
fed to a float64 copy of the freshly initialized head (one forward and backward): the reference. For every variant,
the same exams go through its encoder and its head, and the report gives the relative error
``||x - ref|| / ||ref||`` of the encoder output (``encoder_err``, against the float32 encoder), of the head's logits
and of its parameter gradients on the reference encoder output (``head_logits_err``, ``head_grad_err``: the head
alone, against float64), and of the logits of the variant's head on the variant's encoder output
(``end_to_end_err``). ``baseline``'s errors are those of the current setup; a variant only changes the numerics
beyond float rounding where its errors are clearly larger. They describe one step from the initial head, not the
accuracy of a trained head.

``report.json`` holds every number, together with analytic FLOP counts and the throughput they imply. A variant that
fails (an OOM, a compile error) records its error and the others still run.

Submit through ``eval/submit_profile_attentive_pool.sh``. By hand:

    .venv/bin/python eval/profile_attentive_pool.py --config eval/configs/profile-attentive-pool.yaml --out-dir <dir>
"""

from __future__ import annotations

# First: sets OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and PYTORCH_CUDA_ALLOC_CONF before numpy and torch are
# imported, exactly as in the evaluation run that is being profiled.
import eval_curia_kneeno  # noqa: I001

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
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
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModel  # noqa: E402

from kneeno_adapter import CuriaAdapter  # noqa: E402

logger = logging.getLogger(__name__)

TASKS = ("attentive_pool", "attentive_slice_pool")
VARIANTS = {
    "baseline": {},
    "pin_memory": {"pin_memory": True},
    "free_prev": {"free_prev": True},
    "math_sdpa": {"math_sdpa": True},
    "tf32": {"tf32": True},
    "head_autocast": {"head_autocast": True},
    "encoder_compile": {"encoder_compile": True},
    "encoder_bf16": {"encoder_bf16": True},
    "encoder_bf16_compile": {"encoder_bf16": True, "encoder_compile": True},
    "exact": {"pin_memory": True, "free_prev": True, "math_sdpa": True, "encoder_compile": True},
    "fastest": {
        "pin_memory": True,
        "free_prev": True,
        "head_autocast": True,
        "encoder_bf16": True,
        "encoder_compile": True,
    },
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
    numerics_exams: int = 2,
) -> dict[str, Any]:
    """Profile ``task``'s head training; writes ``report.json`` (and per variant a profiler table and trace) to
    ``out_dir``, which must not exist yet. ``warmup``, ``steps`` and ``profile_steps`` count head micro-batches;
    ``numerics_exams = 0`` skips the numerics. Returns the report."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    if steps < 1 or warmup < 0 or profile_steps < 0 or numerics_exams < 0:
        raise ValueError("steps must be >= 1, warmup, profile_steps and numerics_exams >= 0")
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
    encoders = _Encoders(model)

    cfg = evaluator.config[task]
    micro = cfg["batch_size"] // cfg["grad_accum_steps"]
    encoder_batch = min(evaluator.extraction_batch_size, micro)  # as in ClassificationEvaluator._encoder_source
    if micro % encoder_batch:
        raise ValueError(
            f"The head micro-batch ({micro} exams) must be a multiple of the encoder batch ({encoder_batch}) here"
        )

    # Disjoint exams per run, drawn from the train part like a head epoch's, then the numerics exams.
    runs = ("kneeno", *variants)
    per_run = (warmup + steps + profile_steps) * micro
    order = torch.randperm(len(evaluator.train_indices), generator=torch.Generator().manual_seed(evaluator.seed))
    pool = [evaluator.train_indices[i] for i in order.tolist()]
    needed = per_run * len(runs) + numerics_exams
    if needed > len(pool):
        raise ValueError(
            f"{len(runs)} runs of {per_run} exams and {numerics_exams} numerics exams need {needed} train exams, "
            f"there are {len(pool)}; lower --steps or run fewer variants"
        )
    chunks = {name: pool[i * per_run : (i + 1) * per_run] for i, name in enumerate(runs)}
    numerics_indices = pool[per_run * len(runs) : needed]

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
        "numerics_exams": numerics_exams,
        "train_exams_full": len(evaluator.train_indices),
        "flops_per_exam": flops,
        "runs": {},
    }
    logger.info("Profiling %s: %s", task, json.dumps({k: v for k, v in report.items() if k != "runs"}))

    try:
        reference = _numerics_reference(evaluator, encoders, task, numerics_indices) if numerics_exams else None
        kneeno_indices = chunks["kneeno"][: (warmup + steps) * micro]  # KneeNo's loop has no profiling pass
        _record(report, "kneeno", lambda: _run_kneeno(evaluator, model, task, kneeno_indices, warmup, micro))
        for name in variants:
            options = VARIANTS[name]

            def run(name=name, options=options):
                result = _run_variant(
                    evaluator,
                    encoders,
                    task,
                    chunks[name],
                    options,
                    warmup,
                    steps,
                    profile_steps,
                    micro,
                    encoder_batch,
                    out_dir / name,
                )
                if reference is not None:
                    result["numerics"] = _numerics(evaluator, encoders, task, options, reference)
                return result

            _record(report, name, run)
    finally:
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        evaluator.cleanup()
    logger.info(
        "Summary (ms per exam; train pass = one pass over all %d train exams; errors relative to float32 / float64, "
        "see the module docstring):\n%s",
        report["train_exams_full"],
        _summary_table(report),
    )
    logger.info("Wrote %s", out_dir / "report.json")
    return report


def _record(report, name, run):
    """``run()`` into ``report["runs"][name]``, or its error if it fails, so one broken variant leaves the others."""
    try:
        report["runs"][name] = run()
    except Exception as ex:  # noqa: BLE001 -- recorded, not swallowed: the report and the log say what failed
        logger.exception("Run %s failed", name)
        report["runs"][name] = {"error": f"{type(ex).__name__}: {ex}"}
        return
    _finish_run(report, name)


# -- what the variants change ---------------------------------------------------------------------------------


class _Encoders:
    """The encoder variants, each built on first use and then shared by every run that uses it (a compiled encoder
    then compiles once per input shape, not once per run)."""

    def __init__(self, model):
        self._models = {(False, False): model}

    def get(self, bf16, compiled):
        key = (bf16, compiled)
        if key not in self._models:
            if compiled:
                self._models[key] = torch.compile(self.get(bf16, False), dynamic=False)
            else:  # the bfloat16 copy (~0.2 GB for Curia-B)
                self._models[key] = copy.deepcopy(self._models[False, False]).to(torch.bfloat16)
        return self._models[key]


def _encode(evaluator, encoders, options, batch):
    """``extract_exams`` with the variant's encoder: the float32 weights under the evaluator's autocast (as in
    KneeNo), or with ``encoder_bf16`` the bfloat16 weights on a bfloat16 input without autocast. The output is
    float32 either way."""
    bf16 = bool(options.get("encoder_bf16"))
    model = encoders.get(bf16, bool(options.get("encoder_compile")))
    if bf16:
        batch = batch.to(torch.bfloat16)
    amp_dtype = None if bf16 else evaluator.amp_dtype
    return extract_exams(
        model, evaluator.adapter, batch, evaluator.device, evaluator.num_sequences, amp_dtype, note_precision=False
    )


@contextlib.contextmanager
def _matmul_precision(options):
    """TF32 for the float32 matmuls inside, with ``tf32``; forward and backward both."""
    if not options.get("tf32"):
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _head_forward_context(options, cuda):
    """The context of the head's forward: bfloat16 autocast (``head_autocast``) and the attention backend
    (``math_sdpa``; the backward of the math backend is the autograd graph built in the forward)."""
    stack = contextlib.ExitStack()
    if options.get("head_autocast") and cuda:
        stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    if options.get("math_sdpa"):
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
    return stack


def _head_step(head, loss_fn, feats, labels, options, cuda, scale=1.0):
    """Forward, loss and backward of one micro-batch, as in ``_train_head_epoch``; returns the logits."""
    with _matmul_precision(options):
        with _head_forward_context(options, cuda):
            logits = head(feats)
        # The loss in the labels' precision: under autocast, BCEWithLogitsLoss would run in float32 anyway.
        loss = loss_fn(logits.to(labels.dtype), labels) * scale
        loss.backward()
    return logits


# -- runs -----------------------------------------------------------------------------------------------------


def _new_head(evaluator, task):
    """``task``'s head, optimizer and loss as ``ClassificationEvaluator`` builds them, from the task's seed, and the
    feature key and ``prepare`` that turn an encoder output into the head's input."""
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
    began = time.perf_counter()
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
    return {"exams": exams, "wall_s": wall, "warmup_s": start[0] - began, "phases_s": None, **memory.read()}


def _run_variant(
    evaluator, encoders, task, indices, options, warmup, steps, profile_steps, micro, encoder_batch, profile_dir
):
    """The head-epoch loop written out, with every phase timed; see the module docstring for ``options``."""
    device = evaluator.device
    cuda = device.type == "cuda"
    key, prepare, head, optimizer, loss_fn = _new_head(evaluator, task)
    head.train()
    accum = evaluator.config[task]["grad_accum_steps"]
    pinned = bool(options.get("pin_memory")) and cuda
    free_prev = bool(options.get("free_prev"))
    timer = _PhaseTimer(device)
    memory = _MemoryStats(device)
    began = time.perf_counter()
    loader = _loader(evaluator, indices, encoder_batch, pinned)

    iterator = iter(loader)
    out = feats = None
    with contextlib.ExitStack() as stack:
        stack.callback(_shutdown_workers, iterator)
        optimizer.zero_grad()
        for step in range(warmup + steps + profile_steps):
            if step == warmup:
                timer.start()
                warmup_s = timer.started - began
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
                    out = _encode(evaluator, encoders, options, batch)
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
                _head_step(head, loss_fn, feats, labels, options, cuda, scale=1 / accum)
            if (step + 1) % accum == 0:
                with timer("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
        timer.stop()  # without a profiling pass (a no-op otherwise: it stopped before the profiler started)
        _sync(device)
    if profile_steps:
        _write_profile(prof, profile_dir, cuda)
    return {
        "exams": steps * micro,
        "wall_s": timer.wall,
        "warmup_s": warmup_s,
        "phases_s": dict(timer.totals),
        **memory.read(),
    }


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


# -- numerics -------------------------------------------------------------------------------------------------


def _numerics_reference(evaluator, encoders, task, indices):
    """The numerics exams, their float32 encoder output (float32 weights, no autocast, full-precision matmuls)
    and the float64 head's logits and gradients on it."""
    device = evaluator.device
    batch, labels = evaluator._collate_fn([evaluator.dataset[i] for i in indices])
    batch, labels = batch.to(device), labels.to(device)
    key, prepare, *_ = _new_head(evaluator, task)
    model = encoders.get(False, False)
    out = extract_exams(model, evaluator.adapter, batch, device, evaluator.num_sequences, None, note_precision=False)
    feats = prepare(feature(out, key))
    del out
    logits, grads = _head_result(evaluator, task, feats, labels, {}, dtype=torch.float64)
    return {"batch": batch, "labels": labels, "feats": feats, "logits": logits, "grads": grads}


def _numerics(evaluator, encoders, task, options, reference):
    """The variant's relative errors against ``reference`` (see the module docstring)."""
    key, prepare, *_ = _new_head(evaluator, task)
    out = _encode(evaluator, encoders, options, reference["batch"])
    feats = prepare(feature(out, key))
    del out
    head_logits, head_grads = _head_result(evaluator, task, reference["feats"], reference["labels"], options)
    end_to_end_logits, _ = _head_result(evaluator, task, feats, reference["labels"], options)
    return {
        "encoder_err": _relative_error(feats, reference["feats"]),
        "head_logits_err": _relative_error(head_logits, reference["logits"]),
        "head_grad_err": _relative_error(head_grads, reference["grads"]),
        "end_to_end_err": _relative_error(end_to_end_logits, reference["logits"]),
    }


def _head_result(evaluator, task, feats, labels, options, dtype=None):
    """Logits and all parameter gradients (flattened) of one forward and backward of the freshly initialized head;
    with ``dtype``, head, input and labels are cast to it first."""
    _, _, head, _, loss_fn = _new_head(evaluator, task)
    if dtype is not None:
        head, feats, labels = head.to(dtype), feats.to(dtype), labels.to(dtype)
    head.train()
    logits = _head_step(head, loss_fn, feats, labels, options, evaluator.device.type == "cuda")
    grads = torch.cat([p.grad.flatten() for p in head.parameters() if p.grad is not None])
    return logits.detach().double(), grads.double()


def _relative_error(value, reference):
    reference = reference.double()
    return (torch.linalg.vector_norm(value.double() - reference) / torch.linalg.vector_norm(reference)).item()


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
        self.started = None

    def start(self):
        _sync(self.device)
        self.active, self.started = True, time.perf_counter()

    def stop(self):
        if self.active:
            _sync(self.device)
            self.wall = time.perf_counter() - self.started
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


_NUMERICS_COLUMNS = {"enc err": "encoder_err", "head grad err": "head_grad_err", "e2e err": "end_to_end_err"}


def _summary_table(report):
    columns = (
        "wall",
        *PHASES,
        "train pass min",
        "warmup s",
        "enc TFLOP/s",
        "peak alloc GiB",
        "retries",
        *_NUMERICS_COLUMNS,
    )
    rows = []
    for name, run in report["runs"].items():
        if "error" in run:
            rows.append((name, run["error"][:60], *([""] * (len(columns) - 1))))
            continue
        phases = run.get("phases_ms_per_exam") or {}
        numerics = run.get("numerics") or {}
        cells = [
            f"{run['wall_ms_per_exam']:.1f}",
            *(f"{phases[phase]:.1f}" if phase in phases else "-" for phase in PHASES),
            f"{run['train_pass_min']:.1f}",
            _fmt(run.get("warmup_s")),
            _fmt(run.get("encoder_tflops")),
            _fmt(run.get("peak_allocated_gib")),
            str(run.get("alloc_retries", "-")),
            *(f"{numerics[key]:.1e}" if key in numerics else "-" for key in _NUMERICS_COLUMNS.values()),
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
    parser.add_argument("--numerics-exams", type=int, default=2, help="exams of the numerics check (0: skip it)")
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
        numerics_exams=args.numerics_exams,
    )


# Required: with num_workers > 0 the DataLoader may spawn/forkserver workers, which re-import this module.
if __name__ == "__main__":
    main()
