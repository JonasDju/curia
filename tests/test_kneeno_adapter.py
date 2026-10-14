from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import CROP_SIZE, DEPTH, EMBED_DIM, NUM_PATCHES, REPO_DIR, SLICE_SHAPE, FakeLabeledDataset, fake_eval_dataset
from kneeno.evaluation.dataset import LabeledExternalKneeMRIDataset, LabeledInternalKneeMRIDataset
from kneeno.evaluation import ClassificationEvaluator
from kneeno.evaluation.adapter import EncoderAdapter
from kneeno.evaluation.config import ALL_TASKS

from kneeno_adapter import CuriaAdapter


@pytest.fixture(scope="module")
def adapter(tiny_curia_dir: Path) -> CuriaAdapter:
    return CuriaAdapter(model_name=str(tiny_curia_dir), embed_dim=EMBED_DIM)


def _volume(depth: int = DEPTH, dtype: torch.dtype = torch.uint8, high: int = 256, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, high, (1, depth, *SLICE_SHAPE), generator=gen).to(dtype)


def _orientations(dataset_cls: type) -> tuple[str, ...]:
    # The datasets' `orientations` property only reads class attributes, so no data is needed.
    return dataset_cls.orientations.fget(dataset_cls)


# -- class attributes --------------------------------------------------------------------------------------


def test_is_encoder_adapter(adapter: CuriaAdapter) -> None:
    assert isinstance(adapter, EncoderAdapter)
    assert adapter.embed_dim == EMBED_DIM


def test_has_cls_and_slice_tokens() -> None:
    # Curia is a 2D DINOv2: one cls token per slice, so "linear" and the slice tasks are available.
    assert CuriaAdapter.has_cls_token
    assert CuriaAdapter.has_slice_tokens


def test_picklable(adapter: CuriaAdapter) -> None:
    # Runs in DataLoader workers, which pickle it under spawn / forkserver.
    restored = pickle.loads(pickle.dumps(adapter))
    volume = _volume()
    torch.testing.assert_close(restored.prepare_input(volume), adapter.prepare_input(volume))


# -- prepare_input ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.int16, torch.float32])
def test_prepare_input__shape_and_dtype(adapter: CuriaAdapter, dtype: torch.dtype) -> None:
    # uint8: internal dataset; uint16 / float32: external NIfTIs, which come back unscaled.
    out = adapter.prepare_input(_volume(dtype=dtype, high=256 if dtype == torch.uint8 else 4000))
    assert out.shape == (1, DEPTH, CROP_SIZE, CROP_SIZE)
    assert out.dtype == torch.float32


def test_prepare_input__single_slice(adapter: CuriaAdapter) -> None:
    assert adapter.prepare_input(_volume(depth=1)).shape == (1, 1, CROP_SIZE, CROP_SIZE)


def test_prepare_input__zscores_the_whole_volume(adapter: CuriaAdapter) -> None:
    # One mean / std for the whole volume, as in Curia's own 3D evaluation, not one per slice: a slice brighter
    # than the others stays brighter.
    volume = _volume()
    volume[0, 2] = volume[0, 2] // 4 + 190
    out = adapter.prepare_input(volume)
    assert out.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert out.std().item() == pytest.approx(1.0, abs=1e-4)
    slice_means = out[0].mean(dim=(1, 2))
    assert slice_means[2] > 1.0
    assert (slice_means[[0, 1, 3, 4]] < 0).all()


def test_prepare_input__is_affine_invariant(adapter: CuriaAdapter) -> None:
    # Why internal (0..255) and external (raw) volumes need no different handling.
    volume = _volume().float()
    torch.testing.assert_close(
        adapter.prepare_input(volume * 7.5 + 300), adapter.prepare_input(volume), atol=1e-4, rtol=1e-4
    )


def test_prepare_input__keeps_slice_order(adapter: CuriaAdapter) -> None:
    # Slice i has constant value i: the output's slice means must still increase.
    volume = torch.arange(DEPTH, dtype=torch.float32)[None, :, None, None].expand(1, DEPTH, *SLICE_SHAPE)
    slice_means = adapter.prepare_input(volume.clone())[0].mean(dim=(1, 2))
    assert (slice_means.diff() > 0).all()


def test_prepare_input__keeps_spatial_layout(adapter: CuriaAdapter) -> None:
    # Bright top rows and a bright left column stay there: (H, W) is neither transposed nor flipped.
    volume = torch.zeros(1, DEPTH, *SLICE_SHAPE)
    volume[:, :, : SLICE_SHAPE[0] // 4, :] = 200
    volume[:, :, :, :2] = 100
    out = adapter.prepare_input(volume)[0]
    quarter = CROP_SIZE // 4
    assert out[:, :quarter].mean() > out[:, -quarter:].mean() + 1
    assert out[:, quarter:, :2].mean() > out[:, quarter:, -2:].mean() + 1


def test_prepare_input__matches_processor_slice_by_slice(adapter: CuriaAdapter) -> None:
    # Every slice is resized on its own; the z-score uses the whole volume's statistics.
    volume = _volume()
    resized = torch.nn.functional.interpolate(
        volume[0, :, None].float(), size=(CROP_SIZE, CROP_SIZE), mode="bicubic", align_corners=False, antialias=True
    )[:, 0]
    expected = (resized - resized.mean()) / resized.std()
    torch.testing.assert_close(adapter.prepare_input(volume)[0], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dataset_cls", [LabeledInternalKneeMRIDataset, LabeledExternalKneeMRIDataset])
def test_prepare_input__accepts_dataset_orientations(adapter: CuriaAdapter, dataset_cls: type) -> None:
    # Both labeled datasets return sagittal LIP, coronal PIL and transversal SPL: in-plane IP / IL / PL, which is
    # what Curia expects.
    orientations = _orientations(dataset_cls)
    assert {code[1:] for code in orientations} == {"IP", "IL", "PL"}
    volume = _volume()
    for orientation in orientations:
        torch.testing.assert_close(adapter.prepare_input(volume, orientation), adapter.prepare_input(volume))


@pytest.mark.parametrize("orientation", ["LPI", "PLI", "RPI", "SAL"])
def test_prepare_input__rejects_other_orientations(adapter: CuriaAdapter, orientation: str) -> None:
    with pytest.raises(ValueError, match="Curia expects slices"):
        adapter.prepare_input(_volume(), orientation)


# -- forward_features ------------------------------------------------------------------------------------


def test_forward_features__shapes(adapter: CuriaAdapter, tiny_model: torch.nn.Module) -> None:
    batch = torch.randn(3, 1, DEPTH, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        out = adapter.forward_features(tiny_model, batch)
    assert set(out) == {"slice_cls", "slice_patches"}
    assert out["slice_cls"].shape == (3, DEPTH, EMBED_DIM)
    assert out["slice_patches"].shape == (3, DEPTH, NUM_PATCHES, EMBED_DIM)


def test_forward_features__matches_per_slice_encoding(adapter: CuriaAdapter, tiny_model: torch.nn.Module) -> None:
    batch = torch.randn(2, 1, DEPTH, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        out = adapter.forward_features(tiny_model, batch)
        for b in range(2):
            for d in range(DEPTH):
                tokens = tiny_model(pixel_values=batch[b : b + 1, :, d]).last_hidden_state[0]
                torch.testing.assert_close(out["slice_cls"][b, d], tokens[0], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(out["slice_patches"][b, d], tokens[1:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("raw", [False, True])
def test_prepare_input__collate__forward_features(adapter: CuriaAdapter, tiny_model: torch.nn.Module, raw: bool) -> None:
    volumes, _, source = FakeLabeledDataset(raw=raw)[0]
    inputs = [adapter.prepare_input(v, o, source) for v, o in zip(volumes, FakeLabeledDataset.orientations)]
    batch = adapter.collate(inputs)
    assert batch.shape == (4, 1, DEPTH, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        out = adapter.forward_features(tiny_model, batch)
    assert out["slice_patches"].shape == (4, DEPTH, NUM_PATCHES, EMBED_DIM)
    assert torch.isfinite(out["slice_patches"]).all()


def test_evaluator__all_tasks_with_workers(adapter: CuriaAdapter, tiny_model: torch.nn.Module) -> None:
    # Every task (the slice tasks included) through KneeNo's evaluator, with real DataLoader workers: the adapter
    # runs in forked workers there, which num_workers=0 would never exercise.
    attentive = {"max_epochs": 1, "batch_size": 4, "num_heads": 2}
    config = {
        "data": {"num_workers": 2, "series_depth": DEPTH},
        "freq": {task: 1 for task in ALL_TASKS},
        "knn": {"k": [3], "batch_size": 4},
        "linear": {"max_epochs": 1, "batch_size": 4},
        "linear_pool": {"max_epochs": 1, "batch_size": 4},
        "attentive_pool": attentive,
        "linear_slice_cls": {"max_epochs": 1, "batch_size": 4},
        "attentive_slice_cls": attentive,
        "attentive_slice_pool": attentive,
    }
    dataset = fake_eval_dataset(("internal", "external"))  # uint8 and raw volumes in one evaluation
    evaluator = ClassificationEvaluator(config=config, adapter=adapter, dataset=dataset)
    try:
        metrics = evaluator.evaluate(tiny_model, tasks=list(ALL_TASKS))
    finally:
        evaluator.cleanup()
    assert {key.split("/")[1] for key in metrics} == set(ALL_TASKS)
    assert all(np.isfinite(value) for value in metrics.values())


# -- the real checkpoint (only if raidium/curia is in the local Hugging Face cache) ---------------------------


def _cached_curia_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download("raidium/curia", local_files_only=True))
    except Exception:
        return None


CURIA_DIR = _cached_curia_dir()
needs_curia = pytest.mark.skipif(CURIA_DIR is None, reason="raidium/curia is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def curia_adapter() -> CuriaAdapter:
    return CuriaAdapter(model_name=str(CURIA_DIR), embed_dim=768)


@needs_curia
def test_curia__prepare_input_matches_curia_trainer(curia_adapter: CuriaAdapter) -> None:
    # Curia's own 3D benchmarks (trainer.py::extract_features) pass float32 numpy volumes in (H, W, D) layout.
    volume = _volume(depth=3)
    expected = curia_adapter.processor([volume[0].permute(1, 2, 0).numpy().astype(np.float32)], return_tensors="pt")
    out = curia_adapter.prepare_input(volume)
    assert out.shape == (1, 3, 512, 512)
    torch.testing.assert_close(out, expected["pixel_values"].squeeze(2), atol=1e-4, rtol=1e-4)


@needs_curia
def test_curia__repo_processor_matches_hub(curia_adapter: CuriaAdapter) -> None:
    # The tiny test checkpoint uses the repo's curia_image_processor.py; pin that it behaves like the Hub's.
    spec = importlib.util.spec_from_file_location("repo_curia_image_processor", REPO_DIR / "curia_image_processor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo_processor = module.CuriaImageProcessor(crop_size=512)
    volume = _volume(depth=3)[0].permute(1, 2, 0)
    torch.testing.assert_close(
        repo_processor(volume, return_tensors="pt")["pixel_values"],
        curia_adapter.processor(volume, return_tensors="pt")["pixel_values"],
    )


@needs_curia
def test_curia__forward_features(curia_adapter: CuriaAdapter) -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(CURIA_DIR).eval()
    assert model.config.hidden_size == curia_adapter.embed_dim
    batch = curia_adapter.prepare_input(_volume(depth=2))[None]  # (1, 1, 2, 512, 512)
    with torch.no_grad():
        out = curia_adapter.forward_features(model, batch)
    assert out["slice_cls"].shape == (1, 2, 768)
    assert out["slice_patches"].shape == (1, 2, (512 // 16) ** 2, 768)
    assert torch.isfinite(out["slice_patches"]).all()
