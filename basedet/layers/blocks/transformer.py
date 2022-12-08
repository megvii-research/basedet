#!/usr/bin/env python3
import math

import megengine.functional as F
import megengine.module as M

from basecore.network import get_activation, get_norm

from basedet import layers

__all__ = [
    "Attention",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
]


class Attention(M.Module):

    def __init__(self, dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5
        self.proj_q = M.Linear(dim, dim, bias=bias)
        self.proj_k = M.Linear(dim, dim, bias=bias)
        self.proj_v = M.Linear(dim, dim, bias=bias)
        self.proj = M.Linear(dim, dim, bias=bias)
        self.attn_drop = M.Dropout(dropout) if dropout > 0 else None
        self._init_weights()

    def _init_weights(self):
        # other code init Linear with fan_in = 3 * embed_dim, so we should mul 1 / sqrt(2)
        gain = 1 / math.sqrt(2)
        M.init.xavier_uniform_(self.proj_q.weight, gain)
        M.init.xavier_uniform_(self.proj_k.weight, gain)
        M.init.xavier_uniform_(self.proj_v.weight, gain)
        M.init.xavier_uniform_(self.proj.weight)
        if self.proj_q.bias is not None:
            layers.linear_bias_init(self.proj_q, gain)
            layers.linear_bias_init(self.proj_k, gain)
            layers.linear_bias_init(self.proj_v, gain)
            layers.linear_bias_init(self.proj)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, T, C = query.shape
        _, S, _ = key.shape
        H = self.num_heads
        C //= H

        q = self.proj_q(query).reshape(B, T, H, C).transpose(0, 2, 1, 3)
        k = self.proj_k(key).reshape(B, S, H, C).transpose(0, 2, 1, 3)
        v = self.proj_v(value).reshape(B, S, H, C).transpose(0, 2, 1, 3)

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, T, S)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(B, 1, 1, S)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask |= key_padding_mask

        if attn_mask is not None:
            new_attn_mask = F.zeros(attn_mask.shape, dtype=q.dtype)
            new_attn_mask[attn_mask] = float("-inf")
            attn_mask = new_attn_mask

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, axis=-1)

        if self.attn_drop:
            attn = self.attn_drop(attn)
        output = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, H * C)
        output = self.proj(output)
        return output


class TransformerEncoderLayer(M.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_ffn=2048,
        dropout=0.1,
        act_name="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, dropout=dropout)
        self.linear1 = M.Linear(dim, dim_ffn)
        self.linear2 = M.Linear(dim_ffn, dim)
        self.norm1 = M.LayerNorm(dim)
        self.norm2 = M.LayerNorm(dim)
        if dropout > 0.0:
            self.dropout = M.Dropout(dropout)
        self.act = get_activation(act_name)
        self.normalize_before = normalize_before
        self._init_weights()

    def _init_weights(self):
        layers.linear_init(self.linear1)
        layers.linear_init(self.linear2)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        dropout = getattr(self, "dropout", None)
        if dropout:
            src2 = dropout(src2)
        src = src + src2
        src = self.norm1(src)
        src2 = self.act(self.linear1(src))
        if dropout:
            src2 = dropout(src2)
        src2 = self.linear2(src2)
        if dropout:
            src2 = dropout(src2)
        src = src + src2
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        dropout = getattr(self, "dropout", None)
        if dropout:
            src2 = self.dropout(src2)
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.act(self.linear1(src2))
        if dropout:
            src2 = dropout(src2)
        src2 = self.linear2(src2)
        if dropout:
            src2 = dropout(src2)
        src = src + src2
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        forward_func = self.forward_pre if self.normalize_before else self.forward_post
        return forward_func(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(M.Module):
    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        dim_ffn=2048,
        dropout=0.1,
        act_name="relu",
        normalize_before=False,
        norm_name=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = TransformerEncoderLayer(
                dim, num_heads, dim_ffn, dropout, act_name, normalize_before,
            )
            setattr(self, f"layer{i + 1}", layer)

        if norm_name:
            self.norm = get_norm(norm_name, dim)

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i + 1}")
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        norm = getattr(self, "norm", None)
        if norm:
            output = norm(output)

        return output


class TransformerDecoderLayer(M.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_ffn=2048,
        dropout=0.1,
        act_name="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = Attention(dim, num_heads, dropout=dropout)
        self.cross_attn = Attention(dim, num_heads, dropout=dropout)
        self.linear1 = M.Linear(dim, dim_ffn)
        self.linear2 = M.Linear(dim_ffn, dim)
        self.norm1 = M.LayerNorm(dim)
        self.norm2 = M.LayerNorm(dim)
        self.norm3 = M.LayerNorm(dim)
        if dropout > 0.0:
            self.dropout = M.Dropout(dropout)
        self.act = get_activation(act_name)
        self.normalize_before = normalize_before
        self._init_weights()

    def _init_weights(self):
        layers.linear_init(self.linear1)
        layers.linear_init(self.linear2)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        dropout = getattr(self, "dropout", None)
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt2 = self.act(self.linear1(tgt))
        if dropout:
            tgt2 = dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        dropout = getattr(self, "dropout", None)
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.act(self.linear1(tgt2))
        if dropout:
            tgt2 = dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        if dropout:
            tgt2 = dropout(tgt2)
        tgt = tgt + tgt2
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class TransformerDecoder(M.Module):
    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        dim_ffn=2048,
        dropout=0.1,
        act_name="relu",
        normalize_before=False,
        norm_name=None,
        return_intermediate=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = TransformerDecoderLayer(
                dim,
                num_heads,
                dim_ffn,
                dropout,
                act_name,
                normalize_before,
            )
            setattr(self, f"layer{i + 1}", layer)
        if norm_name:
            self.norm = get_norm(norm_name, dim)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        output = tgt
        intermediate = []
        norm = getattr(self, "norm", None)
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i + 1}")
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                output_media = output
                if norm:
                    output_media = norm(output_media)
                intermediate.append(output_media)
        if norm and not self.return_intermediate:
            output = norm(output)
        if self.return_intermediate:
            return F.stack(intermediate)
        return F.expand_dims(output, 0)


class Transformer(M.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_ffn=2048,
        dropout=0.1,
        act_name="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_encoder_layers,
            dim,
            num_heads,
            dim_ffn,
            dropout,
            act_name,
            normalize_before,
            "LN" if normalize_before else None,
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers,
            dim,
            num_heads,
            dim_ffn,
            dropout,
            act_name,
            normalize_before,
            "LN",
            return_intermediate_dec,
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim > 1:
                if ".proj_q." in name or ".proj_k." in name or ".proj_v." in name:
                    continue
                M.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        B, C, H, W = src.shape
        src = F.flatten(src, 2).transpose(0, 2, 1)
        pos_embed = F.flatten(pos_embed, 2).transpose(0, 2, 1)
        query_embed = F.repeat(F.expand_dims(query_embed, 0), B, axis=0)
        mask = F.flatten(mask, 1)
        tgt = F.zeros_like(query_embed)
        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            pos=pos_embed,
        )
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs, memory.transpose(0, 2, 1).reshape(B, C, H, W)
