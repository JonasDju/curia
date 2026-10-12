"""Shared fixtures: an offline tiny Curia "checkpoint" and a fake labeled KneeNo dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from kneeno.evaluation.dataset import LabeledInternalKneeMRIDataset
from transformers import Dinov2Config, Dinov2Model

REPO_DIR = Path(__file__).resolve().parents[1]

CROP_SIZE = 32  # the tiny processor's crop_size (Curia: 512)
PATCH_SIZE = 8
NUM_PATCHES = (CROP_SIZE // PATCH_SIZE) ** 2
EMBED_DIM = 32
DEPTH = 5  # slices per volume = eval.data.series_depth
SLICE_SHAPE = (20, 11)  # (H, W): non-square, to catch an axis-order swap


@pytest.fixture(scope="session")
def tiny_curia_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local stand-in for ``raidium/curia``: a random tiny ``Dinov2Model`` with Curia's architecture choices
    (1 channel, SwiGLU, no registers) and Curia's processor code from the repo root, with a small crop size.

    Loads through the same ``AutoModel`` / ``AutoImageProcessor(trust_remote_code=True)`` calls as the real one,
    without network access.
    """
    path = tmp_path_factory.mktemp("tiny_curia")
    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=EMBED_DIM,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=CROP_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=1,
        use_swiglu_ffn=True,
    )
    Dinov2Model(config).save_pretrained(path)
    shutil.copy(REPO_DIR / "curia_image_processor.py", path / "curia_image_processor.py")
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoImageProcessor": "curia_image_processor.CuriaImageProcessor"},
                "clip_below_air": False,
                "crop_size": CROP_SIZE,
                "do_normalize": True,
                "do_resize": True,
                "eps": 1e-6,
                "image_processor_type": "CuriaImageProcessor",
            }
        )
    )
    return path


@pytest.fixture(scope="session")
def tiny_model(tiny_curia_dir: Path) -> Dinov2Model:
    return Dinov2Model.from_pretrained(tiny_curia_dir).eval()


class FakeLabeledDataset(torch.utils.data.Dataset):
    """Stands in for the labeled KneeNo datasets, which need data on the cluster.

    ``raw=True`` mimics LabeledExternalKneeMRIDataset: uint16 volumes far outside 0..255 instead of the internal
    dataset's uint8 ones, and the source ``"external"``. ``orientations`` are the internal dataset's.
    """

    num_classes = 4
    # One item is an exam: one volume per sequence, in this order.
    sequences = LabeledInternalKneeMRIDataset.SEQUENCES
    orientations = LabeledInternalKneeMRIDataset.orientations.fget(LabeledInternalKneeMRIDataset)

    def __init__(self, raw: bool = False) -> None:
        self.raw = raw

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(index)
        # One depth for all volumes, as with eval.data.series_depth > 0: the adapter keeps the depth.
        volumes = []
        for _ in self.sequences:
            volume = torch.randint(0, 4000 if self.raw else 256, (1, DEPTH, *SLICE_SHAPE), generator=gen)
            volumes.append(torch.from_numpy(volume.numpy().astype(np.uint16 if self.raw else np.uint8)))
        label = (torch.rand(self.num_classes, generator=gen) > 0.5).float()
        return tuple(volumes), label, "external" if self.raw else "internal"
