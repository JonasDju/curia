from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import DEPTH, fake_eval_dataset
from kneeno.evaluation import ClassificationEvaluator, classification, load_eval_config
from kneeno.evaluation.config import ALL_TASKS, TRAINED_TASKS
from kneeno.evaluation.splits import write_manifest_for

import eval_curia_kneeno


@pytest.fixture(autouse=True)
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The evaluator builds its dataset from eval.data (KneeNoEvalDataset.from_config); build the fakes instead.
    monkeypatch.setattr(
        classification.KneeNoEvalDataset,
        "from_config",
        staticmethod(lambda data_cfg: fake_eval_dataset(tuple(data_cfg["test_sets"]))),
    )


def _out_dir(tmp_path: Path) -> Path:
    return tmp_path / "curia-b" / "eval" / "internal"


@pytest.fixture
def config(tmp_path: Path, tiny_curia_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    attentive = {"max_epochs": 1, "batch_size": 4, "num_heads": 2}
    monkeypatch.setenv("CURIA_TEST_OUT", str(_out_dir(tmp_path)))
    # Every test set builds the same fake train part, so one manifest serves them all.
    manifest = write_manifest_for(fake_eval_dataset(), tmp_path / "split_manifest.json")
    return {
        "model_name": str(tiny_curia_dir),
        "eval": {
            "split": {"manifest": str(manifest), "level": 1.0, "rep": 0},
            "data": {"test_sets": ["internal", "external"], "num_workers": 0, "series_depth": DEPTH},
            "logging": {
                "tensorboard_dir": "${CURIA_TEST_OUT}/tb",
                "per_label_dir": "$CURIA_TEST_OUT/eval_per_label",
            },
            # Every task explicit: a missing key keeps KneeNo's default frequency.
            "freq": {task: 1 for task in ALL_TASKS},
            "knn": {"k": [3]},
            "extraction_batch_size": 4,
            "linear": {"max_epochs": 1, "batch_size": 4},
            "linear_pool": {"max_epochs": 1, "batch_size": 4},
            "attentive_pool": attentive,
            "linear_slice_cls": {"max_epochs": 1, "batch_size": 4},
            "attentive_slice_cls": attentive,
            "attentive_slice_pool": attentive,
        },
    }


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _only(config: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    config["eval"]["freq"] = {task: 1 if task in tasks else None for task in ALL_TASKS}
    return config


def test_evaluate(tmp_path: Path, config: dict[str, Any]) -> None:
    eval_curia_kneeno.evaluate(_write(tmp_path, _only(config, ["knn", "linear"])))
    out_dir = _out_dir(tmp_path)
    metrics = json.loads((out_dir / "results.json").read_text())
    # Every test set's metrics, plus the trained head's best val metric and lr, and k-NN's val metric and temperature.
    assert {key.rsplit("/", 1)[0] for key in metrics} == {
        "internal/knn/k3",
        "external/knn/k3",
        "internal/linear",
        "external/linear",
        "evaluation_val/linear",
        "evaluation_val/knn/k3",
    }
    # params.yaml is the config file with env vars expanded, in the file's key order.
    params = (out_dir / "params.yaml").read_text()
    assert "CURIA_TEST_OUT" not in params
    assert list(yaml.safe_load(params)) == ["model_name", "eval"]
    assert yaml.safe_load(params)["eval"]["logging"]["per_label_dir"] == str(out_dir / "eval_per_label")
    assert any((out_dir / "eval_per_label").iterdir())
    assert any((out_dir / "tb").iterdir())


def test_evaluate__all_tasks(tmp_path: Path, config: dict[str, Any]) -> None:
    # Curia serves every task: it has a cls token, and one per slice.
    eval_curia_kneeno.evaluate(_write(tmp_path, config))
    metrics = json.loads((_out_dir(tmp_path) / "results.json").read_text())
    assert {key.split("/")[1] for key in metrics} == set(ALL_TASKS)
    assert {key.split("/")[0] for key in metrics} == {"internal", "external", "evaluation_val"}
    assert all(math.isfinite(value) for value in metrics.values())


def test_evaluate__missing_freq_is_off(
    tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A task left out of freq keeps KneeNo's default frequency, which is null (off) for every task since KneeNo 77839f0.
    config["eval"]["freq"] = {"linear_slice_cls": 1}
    calls = []
    monkeypatch.setattr(ClassificationEvaluator, "evaluate", lambda self, model, **kwargs: calls.append(kwargs) or {})
    eval_curia_kneeno.evaluate(_write(tmp_path, config))
    assert calls[0]["tasks"] == ["linear_slice_cls"]
    assert "log_every_head_epoch" not in calls[0]


@pytest.mark.parametrize("existing", ["results.json", "params.yaml", "eval_per_label", "tb"])
def test_evaluate__refuses_existing_results(
    tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True)
    if existing.endswith((".json", ".yaml")):
        (out_dir / existing).write_text("{}")
    else:
        (out_dir / existing).mkdir()
    # Refused before the model is loaded.
    monkeypatch.setattr(eval_curia_kneeno.AutoModel, "from_pretrained", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="earlier evaluation"):
        eval_curia_kneeno.evaluate(_write(tmp_path, config))


def test_load_config__unset_env_var(tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURIA_TEST_OUT")
    with pytest.raises(ValueError, match=r"not set: \['\$CURIA_TEST_OUT', '\$\{CURIA_TEST_OUT\}'\]"):
        eval_curia_kneeno.load_config(_write(tmp_path, config))
    assert not _out_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("model_name"),
        lambda c: c.update(checkpoint="x.ckpt"),
        lambda c: c.update(model_name=["raidium/curia"]),
        lambda c: c.update(eval=None),
    ],
    ids=["missing_model_name", "unknown_key", "model_name_not_a_string", "eval_not_a_mapping"],
)
def test_load_config__invalid(tmp_path: Path, config: dict[str, Any], change: Any) -> None:
    change(config)
    with pytest.raises(ValueError, match="Eval config"):
        eval_curia_kneeno.load_config(_write(tmp_path, config))


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda e: e.update(freq={task: None for task in ALL_TASKS}), "No task is enabled"),
        (lambda e: e.update(encoder="online"), "single set of weights"),
        (lambda e: e["logging"].update(per_label_dir=None), "per_label_dir must be set"),
    ],
    ids=["no_task", "online_encoder", "no_per_label_dir"],
)
def test_evaluate__invalid_eval_block(tmp_path: Path, config: dict[str, Any], change: Any, match: str) -> None:
    change(config["eval"])
    with pytest.raises(ValueError, match=match):
        eval_curia_kneeno.evaluate(_write(tmp_path, config))
    assert not (_out_dir(tmp_path) / "params.yaml").exists()


def test_evaluate__split_error_leaves_no_params(tmp_path: Path, config: dict[str, Any]) -> None:
    # The evaluator builds the dataset and the split; a config they reject leaves nothing that blocks the rerun.
    config["eval"]["split"]["rep"] = 99
    with pytest.raises(ValueError, match="no split with level"):
        eval_curia_kneeno.evaluate(_write(tmp_path, config))
    assert not _out_dir(tmp_path).exists()


def test_shipped_config() -> None:
    # The shipped config loads (with $TMP set, as in the job) and enables every task but attentive_pool, which is
    # deliberately off (attentive_slice_pool sees the same tokens and knows their slice).
    path = Path(__file__).resolve().parents[1] / "eval" / "configs" / "eval-MI-curia-b.yaml"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TMP", "/tmp/job")
        mp.setenv("SPLIT_LEVEL", "0.05")  # set by the submit script (default 1.0 / 0)
        mp.setenv("SPLIT_REP", "3")
        config = eval_curia_kneeno.load_config(path)
    assert config["model_name"] == "raidium/curia"
    assert config["eval"]["data"]["data_root"] == {
        "internal": "/tmp/job/kneeno_data/internal",
        "external": "/tmp/job/kneeno_data/external",
    }
    assert config["eval"]["data"]["test_sets"] == ["internal", "external"]
    assert config["eval"]["data"]["series_depth"] > 0  # the slice-wise adapter keeps the depth
    assert config["eval"]["split"] == {
        "manifest": "/hpcwork/p0021834/workspace_roman/jonas/metadata/internal_split_manifest.json",
        "level": "0.05",  # a string until KneeNo's load_eval_config parses it
        "rep": "3",
    }
    for key in ("tensorboard_dir", "per_label_dir"):  # every (level, rep) has its own output folder
        assert "/level-0.05.rep-3/" in config["eval"]["logging"][key]
    eval_cfg = load_eval_config(config["eval"])
    assert (eval_cfg["split"]["level"], eval_cfg["split"]["rep"]) == (0.05, 3)
    assert eval_curia_kneeno.enabled_tasks(eval_cfg["freq"]) == [task for task in ALL_TASKS if task != "attentive_pool"]
    # No early stopping, except for the two heads that re-encode every head epoch (no feature cache).
    assert eval_cfg["training"]["early_stop_patience_global"] == 0
    assert {task: eval_cfg[task]["early_stop_patience"] for task in TRAINED_TASKS} == {
        task: 5 if task in ("attentive_pool", "attentive_slice_pool") else None for task in TRAINED_TASKS
    }


def test_submit_script__split_defaults() -> None:
    # SPLIT_LEVEL / SPLIT_REP select the eval config's split; unset, the whole train split's rep 0.
    script = (Path(__file__).resolve().parents[1] / "eval" / "submit_eval_curia_kneeno.sh").read_text()
    assert 'export SPLIT_LEVEL="${SPLIT_LEVEL:-1.0}"' in script
    assert 'export SPLIT_REP="${SPLIT_REP:-0}"' in script
