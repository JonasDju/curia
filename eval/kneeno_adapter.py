"""Bridges the official Curia encoder to ``kneeno.evaluation``'s unified feature format.

``CuriaAdapter`` is the ``kneeno.evaluation.adapter.EncoderAdapter`` implementation handed to
``kneeno.evaluation.ClassificationEvaluator`` by ``eval/eval_curia_kneeno.py``. Curia is a 2D (slice-wise)
DINOv2 ViT, a stock ``transformers`` ``Dinov2Model``, so the adapter treats a ``(1, D, H, W)`` volume as ``D``
independent slices, like lightly-train's 2D ``DINOv2Adapter``, and returns per-slice tokens
(``has_slice_tokens``). KneeNo derives the volume-level views from them and also runs its slice tasks.
"""

import torch
from kneeno.evaluation.adapter import EncoderAdapter
from transformers import AutoImageProcessor

#: The in-plane ``(H, W)`` orientation Curia expects for each view: axial, coronal, sagittal.
CURIA_SLICE_ORIENTATIONS = ("PL", "IL", "IP")


class CuriaAdapter(EncoderAdapter):

    has_cls_token = True
    has_slice_tokens = True

    def __init__(self,
        model_name: str,
        embed_dim: int,
    ) -> None:
        """
        Args:
            model_name:
                Name of the huggingface model (e.g. raidium/curia), or a local directory holding Curia's
                ``preprocessor_config.json`` and ``curia_image_processor.py``.
            embed_dim:
                Feature dimension of the encoder: 768 for Curia-B and 1024 for Curia-L, i.e. the model's
                ``config.hidden_size``.
        """

        self._embed_dim = embed_dim

        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def prepare_input(self, volume, orientation=None, source=None):
        """``(1, D, H, W)`` raw volume (in [0, 255] for internal / unscaled for external)
        -> ``(1, D, crop_size, crop_size)`` float32 with crop_size being the default AutoImageProcessor value (512).

        In contrast to other KneeNo adapters, there is no need to prepare volumes from the internal / external
        dataset in different ways, since Curia's image processor performs z-score normalization and therefore expects
        "just raw values or normalized image[s]" according to the docs. ``source`` (the dataset the volume came
        from) is therefore ignored.

        The processor resizes every slice on its own (bicubic, antialiased), but for a 3D input it z-scores the
        **whole volume** with one mean and std, not each slice (only its 2D path normalizes per image). This is
        exactly what Curia's own 3D benchmarks do (``trainer.py::extract_features`` passes ``(H, W, D)`` volumes).
        The processor's ``do_normalize`` flag is ignored by this code path.

        The images (here: slices) need to be of shape (H, W) and in:
            - PL for axial
            - IL for coronal
            - IP for sagittal
        Read as axcodes (the direction each axis increases toward), as KneeNo's ``orientations`` are. Both
        labeled KneeNo datasets return ``(D, H, W)`` volumes in LIP (sagittal), PIL (coronal) and SPL
        (transversal), whose in-plane codes are exactly these, so there is nothing to do except for preprocessing
        the images via the preprocessor. ``orientation`` is only checked, so that a change on KneeNo's side
        cannot go unnoticed.
        """
        if orientation is not None and orientation[1:] not in CURIA_SLICE_ORIENTATIONS:
            raise ValueError(
                f"Curia expects slices (H, W) in one of {CURIA_SLICE_ORIENTATIONS}, got a volume in "
                f"{orientation!r} (D, H, W order)"
            )

        # The processor is capable of dealing with 3D volumes, but requires shape (H, W, D)
        volume = volume[0].permute(1, 2, 0)  # (H, W, D)
        processed_volume = self.processor(volume, return_tensors="pt")
        pixel_values = processed_volume["pixel_values"]  # (1, D, 1, crop_size, crop_size)

        return pixel_values.squeeze(dim=2)  # (1, D, crop_size, crop_size)

    def forward_features(self, model, batch):
        """``batch``: ``(B, 1, D, crop_size, crop_size)`` -> per-slice tokens
        ``{"slice_cls": (B, D, embed_dim), "slice_patches": (B, D, P, embed_dim)}``.
        """
        b, c, d, h, w = batch.shape

        # (B, 1, D, H, W) -> (B*D, 1, H, W)
        batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Curia is never used with use_n_blocks > 1, so there is no need for the hidden states and its logic
        features = model(pixel_values=batch, output_hidden_states=False).last_hidden_state

        # (B*D, P+1, embed_dim) -> (B, D, P+1, embed_dim), P = patH * patW
        features = features.reshape(b, d, -1, self._embed_dim)

        # CLS tokens and patches
        slice_cls = features[:, :, 0, :]
        slice_patches = features[:, :, 1:, :]

        return {"slice_cls": slice_cls, "slice_patches": slice_patches}
