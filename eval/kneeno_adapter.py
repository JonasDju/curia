import torch
from kneeno.evaluation.adapter import EncoderAdapter
from transformers import AutoImageProcessor


class CuriaAdapter(EncoderAdapter):

    has_cls_token = True
    has_slice_tokens = True

    def __init__(self,
        model_name: str,
        embed_dim: int,
    ) -> None:
        """
        Args:
            embed_dim:
                Feature dimension of the encoder: 768 for Curia-B and 1024 for Curia-L.
            model_name:
                Name of the huggingface model (e.g. raidium/curia)
        """

        self._embed_dim = embed_dim

        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def prepare_input(self, volume, orientation=None):
        """``(1, D, H, W)`` raw volume (in [0, 255] for internal / unscaled for external)
        -> ``(1, D, crop_size, crop_size)`` float32 with crop_size being the default AutoImageProcessor value (512).

        In contrast to other KneeNo adapters, there is no need to prepare volumes from the internal / external
        dataset in different ways, since Curia's image processor performs z-score normalization and therefore expects
        "just raw values or normalized image[s]" according to the docs.

        The images (here: slices) need to be of shape (H, W) and in:
            - PL for axial
            - IL for coronal
            - IP for sagittal
        Since all of these conditions already hold, there is nothing to do except for preprocessing the images via
        the preprocessor.
        """

        # The processor is capable of dealing with 3D volumes, but requires shape (H, W, D)
        volume = volume.permute(0, 2, 3, 1).squeeze() # (H, W, D)
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