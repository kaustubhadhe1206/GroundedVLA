"""SmolVLA with grounding tokens appended right after the state token (Step B2, V3).

Written against lerobot==0.4.4 where the prefix is built by
`VLAFlowMatching.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)` and padded
to `config.prefix_length` at the end. We build the unpadded prefix, append K grounding tokens
with the same attention treatment as the state token (att_mask=1), then pad.

Identity check (B2 "done when"): with `grounding_enabled=False` the grounding tokens get
pad_mask=False, so they are not attended to and do not advance position ids -> V3 == V2
exactly. Note a zero-valued token that is still attended is *not* an identity (it takes
softmax mass and shifts RoPE positions), so test with the flag, not just the zero init.
"""
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, pad_tensor

from groundedvla.models.grounding_adapter import GroundingAdapter

GROUNDING_KEY = "observation.grounding"


class GroundedFlowMatching(VLAFlowMatching):
    def __init__(self, config, num_grounding_tokens: int = 2, **kwargs):
        super().__init__(config, **kwargs)
        hidden = self.vlm_with_expert.config.text_config.hidden_size
        self.grounding_adapter = GroundingAdapter(hidden, num_tokens=num_grounding_tokens)
        self.grounding_enabled = True
        self._grounding: torch.Tensor | None = None  # set by the policy before each call

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        prefix_length, self.prefix_length = self.prefix_length, 0  # defer padding
        try:
            embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        finally:
            self.prefix_length = prefix_length

        if self._grounding is not None:
            g_tok = self.grounding_adapter(self._grounding.to(embs.dtype)).to(embs.dtype)
            bsize, k = g_tok.shape[:2]
            g_pad = torch.full((bsize, k), self.grounding_enabled, dtype=torch.bool, device=embs.device)
            g_att = torch.ones((bsize, k), dtype=torch.bool, device=embs.device)  # same as state token
            embs = torch.cat([embs, g_tok], dim=1)
            pad_masks = torch.cat([pad_masks, g_pad], dim=1)
            att_masks = torch.cat([att_masks, g_att], dim=1)

        if pad_masks.shape[1] < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        return embs, pad_masks, att_masks


class GroundedSmolVLAPolicy(SmolVLAPolicy):
    """Routes `observation.grounding` from the batch into the flow-matching model."""

    def __init__(self, config, num_grounding_tokens: int = 2, **kwargs):
        super().__init__(config, **kwargs)
        # TODO(B2): this builds the VLM twice (parent init + here). Fine for a smoke test;
        # avoid it before training on Colab to save load time and memory.
        self.model = GroundedFlowMatching(config, num_grounding_tokens=num_grounding_tokens,
                                          rtc_processor=self.rtc_processor)

    def _set_grounding(self, batch):
        self.model._grounding = batch.get(GROUNDING_KEY)

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        self._set_grounding(batch)
        return super().forward(batch, noise=noise, time=time, reduction=reduction)

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        self._set_grounding(batch)
        return super()._get_action_chunk(batch, noise, **kwargs)

    def get_optim_params(self):
        # Backbone frozen by config flags; adapter always trainable.
        return [p for p in self.parameters() if p.requires_grad]
