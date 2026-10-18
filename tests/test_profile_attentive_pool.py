from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import DEPTH, FakeLabeledDataset
from kneeno.evaluation import classification
from kneeno.evaluation.dataset import KneeNoEvalDataset

import profile_attentive_pool

MICRO = 2  # head micro-batch (exams)


@pytest.fixture(autouse=True)
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enough train exams for every run: 12 runs x (1 + 2 + 1) micro-batches x 2 exams + 2 numerics exams = 98 of
    # the 160 x 0.8 = 128.
    dataset = KneeNoEvalDataset(
        FakeLabeledDataset(prefix="t", num_exams=160), {"internal": FakeLabeledDataset(num_exams=6, prefix="i")}
    )
    monkeypatch.setattr(classification.KneeNoEvalDataset, "from_config", staticmethod(lambda data_cfg, seed: dataset))


def _config(tmp_path: Path, tiny_curia_dir: Path, num_workers: int, grad_accum_steps: int = 1) -> Path:
    head = {"batch_size": MICRO * grad_accum_steps, "grad_accum_steps": grad_accum_steps, "num_heads": 2}
    config: dict[str, Any] = {
        "model_name": str(tiny_curia_dir),
        "eval": {
            "extraction_batch_size": 1,  # two encoder calls per micro-batch
            "data": {"test_sets": ["internal"], "num_workers": num_workers, "series_depth": DEPTH},
            "attentive_pool": head,
            "attentive_slice_pool": head,
        },
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("task", profile_attentive_pool.TASKS)
def test_every_run_reports_its_phases(tmp_path: Path, tiny_curia_dir: Path, task: str) -> None:
    out_dir = tmp_path / "out"
    report = profile_attentive_pool.profile(
        _config(tmp_path, tiny_curia_dir, num_workers=2), out_dir, task=task, warmup=1, steps=2, profile_steps=1
    )

    assert list(report["runs"]) == ["kneeno", *profile_attentive_pool.VARIANTS]
    assert report["runs"]["kneeno"]["exams"] == 2 * MICRO
    assert report["runs"]["kneeno"]["phases_s"] is None
    for name in profile_attentive_pool.VARIANTS:
        run = report["runs"][name]
        assert run["exams"] == 2 * MICRO
        assert set(run["phases_ms_per_exam"]) == set(profile_attentive_pool.PHASES)
        # The phases happen one after another inside the timed window.
        assert sum(run["phases_s"].values()) <= run["wall_s"]
        assert (out_dir / name / "kernels.txt").read_text()
        assert (out_dir / name / "trace.json").exists()
    assert json.loads((out_dir / "report.json").read_text()) == report
    assert (out_dir / "params.yaml").exists()

    numerics = {name: report["runs"][name]["numerics"] for name in profile_attentive_pool.VARIANTS}
    # On the CPU there is no autocast: the baseline encoder is the float32 reference itself.
    assert numerics["baseline"]["encoder_err"] == 0
    assert numerics["encoder_bf16"]["encoder_err"] > 1e-4
    for name in ("baseline", "pin_memory", "free_prev", "math_sdpa", "exact"):
        # float32 against float64: rounding only
        assert numerics[name]["head_grad_err"] < 1e-5, name
        assert numerics[name]["end_to_end_err"] < 1e-4, name
    assert numerics["pin_memory"] == numerics["free_prev"] == numerics["baseline"]


def test_grad_accumulation_without_profiling(tmp_path: Path, tiny_curia_dir: Path) -> None:
    config = _config(tmp_path, tiny_curia_dir, num_workers=0, grad_accum_steps=2)
    report = profile_attentive_pool.profile(
        config, tmp_path / "out", variants=("baseline",), warmup=0, steps=3, profile_steps=0, numerics_exams=0
    )

    assert report["runs"]["baseline"]["exams"] == 3 * MICRO
    assert "numerics" not in report["runs"]["baseline"]
    assert not (tmp_path / "out" / "baseline").exists()


def test_a_failing_variant_leaves_the_others(
    tmp_path: Path, tiny_curia_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken_options = {"math_sdpa": True}
    monkeypatch.setitem(profile_attentive_pool.VARIANTS, "broken", broken_options)
    real_run_variant = profile_attentive_pool._run_variant

    def run_variant(evaluator: Any, encoders: Any, task: str, indices: Any, options: dict, *rest: Any) -> Any:
        if options is broken_options:
            raise RuntimeError("CUDA out of memory")
        return real_run_variant(evaluator, encoders, task, indices, options, *rest)

    monkeypatch.setattr(profile_attentive_pool, "_run_variant", run_variant)
    config = _config(tmp_path, tiny_curia_dir, num_workers=0)
    report = profile_attentive_pool.profile(
        config, tmp_path / "out", variants=("broken", "baseline"), warmup=0, steps=1, profile_steps=0
    )

    assert report["runs"]["broken"] == {"error": "RuntimeError: CUDA out of memory"}
    assert report["runs"]["baseline"]["exams"] == MICRO


def test_analytic_flops_of_curia_b() -> None:
    from transformers import Dinov2Config

    config = Dinov2Config(hidden_size=768, num_hidden_layers=12, patch_size=16, num_channels=1, use_swiglu_ffn=True)
    head = {"patch_sa_depth": 0, "qa_depth": 1}
    flops = profile_attentive_pool._analytic_flops(config, 512, 4, 24, head)

    # ~213 GFLOP per 512² slice, 96 slices per exam.
    assert flops["encoder"] / 96 == pytest.approx(213e9, rel=0.01)
    assert flops["head_tokens"] == 96 * 1024
    assert flops["head_train"] == 12 * 96 * 1024 * 768**2


def test_refuses_too_few_exams_and_an_existing_out_dir(tmp_path: Path, tiny_curia_dir: Path) -> None:
    config = _config(tmp_path, tiny_curia_dir, num_workers=0)
    with pytest.raises(ValueError, match="train exams"):
        profile_attentive_pool.profile(config, tmp_path / "a", warmup=0, steps=10, profile_steps=0)
    (tmp_path / "b").mkdir()
    with pytest.raises(FileExistsError):
        profile_attentive_pool.profile(config, tmp_path / "b")


def test_shipped_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from kneeno.evaluation import load_eval_config

    monkeypatch.setenv("TMP", "/tmp")
    config = profile_attentive_pool.eval_curia_kneeno.load_config(
        Path(__file__).resolve().parents[1] / "eval" / "configs" / "profile-attentive-pool.yaml"
    )
    eval_cfg = load_eval_config(config["eval"])
    assert eval_cfg["extraction_batch_size"] == 16
    assert eval_cfg["attentive_pool"]["batch_size"] == 16
