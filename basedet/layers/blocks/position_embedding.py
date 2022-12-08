#!/usr/bin/env python3
import math

import megengine.functional as F
import megengine.module as M


class PositionEmbeddingSine(M.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, eps=1e-6, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = F.cumsum(not_mask.astype(x.dtype), 1)
        x_embed = F.cumsum(not_mask.astype(x.dtype), 2)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = F.arange(self.num_pos_feats, dtype=x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = F.expand_dims(x_embed, -1) / dim_t
        pos_y = F.expand_dims(y_embed, -1) / dim_t
        pos_x = F.flatten(
            F.stack((F.sin(pos_x[:, :, :, 0::2]), F.cos(pos_x[:, :, :, 1::2])), -1), 3
        )
        pos_y = F.flatten(
            F.stack((F.sin(pos_y[:, :, :, 0::2]), F.cos(pos_y[:, :, :, 1::2])), -1), 3
        )
        pos = F.concat((pos_y, pos_x), 3).transpose(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(M.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = M.Embedding(50, num_pos_feats)
        self.col_embed = M.Embedding(50, num_pos_feats)
        self._init_weights()

    def _init_weights(self):
        M.init.uniform_(self.row_embed.weight)
        M.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        B, _, H, W = x.shape
        x_emb = self.col_embed(F.arange(W, dtype="int32"))
        y_emb = self.row_embed(F.arange(H, dtype="int32"))
        pos = F.concat(
            [
                F.repeat(F.expand_dims(x_emb, 0), H, 0),
                F.repeat(F.expand_dims(y_emb, 1), W, 1),
            ],
            axis=-1,
        )
        pos = F.repeat(F.expand_dims(pos.transpose(2, 0, 1), 0), B, 0)
        return pos


def build_pos_embed(pos_embed, N_steps):
    if pos_embed in ("v2", "sine"):
        return PositionEmbeddingSine(N_steps, normalize=True)
    elif pos_embed in ("v3", "learned"):
        return PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {pos_embed}")
