from __future__ import annotations

import os
import pickle
import regex as re
import multiprocessing
from typing import IO, Any, BinaryIO, Optional
from collections.abc import Iterable
from collections import defaultdict
from jaxtyping import Float, Int

import numpy.typing as npt
import numpy as np
import torch
from torch import Tensor
import math
import json
import torch.nn as nn
from einops import rearrange, einsum
import einx
import mmap
import heapq
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(in_features, out_features, **factory_kwargs)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight # [*, in_features] @ [in_features, out_features]
        return out

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)
    with torch.no_grad():
        linear.weight.copy_(weights.T)
    return linear(in_features)

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight = torch.empty(num_embedding, embedding_dim, **factory_kwargs)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.weight[token_ids]
        return out

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding(token_ids)

class SiLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = self.adjust_dff(d_model) if d_ff < self.adjust_dff(d_model) else d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)

    def adjust_dff(self, d_model: int) -> int:
        return (int(4 * d_model) + 63) // 64 * 64

    def load_weights(self, w1: torch.Tensor, w2: torch.Tensor):
        w1_adj = adjust_weight(w1, (self.d_ff, self.d_model))
        w2_adj = adjust_weight(w2, (self.d_model, self.d_ff))
        with torch.no_grad():
            self.w1.weight.copy_(w1_adj.T)
            self.w2.weight.copy_(w2_adj.T)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1_x = self.w1(x)
        SiLU_x = run_silu(W1_x)
        out = self.w2(SiLU_x)
        return out


def adjust_weight(weight: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    current_shape = weight.shape
    # Use the same device as weight, but default to CPU if .device is not accessible
    device = weight.device if weight.device.type != 'meta' else torch.device('cpu')
    new_weight = torch.zeros(target_shape, dtype=weight.dtype, device=device)
    min_dim0 = min(current_shape[0], target_shape[0])
    min_dim1 = min(current_shape[1], target_shape[1])
    new_weight[:min_dim0, :min_dim1] = weight[:min_dim0, :min_dim1]
    return new_weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = self.adjust_dff(d_model) if d_ff < self.adjust_dff(d_model) else d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)

    def adjust_dff(self, d_model: int) -> int:
        return (int((8/3) * d_model) + 63) // 64 * 64

    def load_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        w1_adj = adjust_weight(w1, (self.d_ff, self.d_model))
        w2_adj = adjust_weight(w2, (self.d_model, self.d_ff))
        w3_adj = adjust_weight(w3, (self.d_ff, self.d_model))
        with torch.no_grad():
            self.w1.weight.copy_(w1_adj.T)
            self.w2.weight.copy_(w2_adj.T)
            self.w3.weight.copy_(w3_adj.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1_x = self.w1(x)
        W3_x = self.w3(x)
        SiLU_x = run_silu(W1_x)
        out = self.w2((SiLU_x * W3_x))
        return out

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # adjusted_d_ff = (int((8/3) * d_model) + 63) // 64 * 64
    # w1_weight_adj = adjust_weight(w1_weight, (adjusted_d_ff, d_model))
    # w2_weight_adj = adjust_weight(w2_weight, (d_model, adjusted_d_ff))
    # w3_weight_adj = adjust_weight(w3_weight, (adjusted_d_ff, d_model))
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.load_weights(w1_weight, w2_weight, w3_weight)
    # with torch.no_grad():
    #     swiglu.w1.weight.copy_(w1_weight_adj.T)
    #     swiglu.w2.weight.copy_(w2_weight_adj.T)
    #     swiglu.w3.weight.copy_(w3_weight_adj.T)
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    score = torch.einsum("... q d, ... k d -> ... q k", Q, K) / math.sqrt(K.size(-1))
    # score = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
    if mask is not None:
        score = score.masked_fill(mask == 0.0, float('-inf'))
        # score = score.masked_fill(mask == False, float('-inf'))
    score = run_softmax(score, dim=-1)
    att = score @ V
    return att

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)
        self.pos_encode = pos_encode
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take Q, K, V to shape (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        if self.theta is not None:
            Q = self.pos_encode(Q, token_positions)
            K = self.pos_encode(K, token_positions)
        # Construct causal mask
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, device=x.device))
        causal_mask = causal_mask.view(1, 1, sequence_length, sequence_length)

        att = run_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
        # (..., sequence_length, num_heads * d_v).
        att = rearrange(att, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        out = self.o_proj(att)
        return out

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multihead_att = Multihead_self_attention(d_model, num_heads)
    with torch.no_grad():
        multihead_att.q_proj.weight.copy_(q_proj_weight.T)
        multihead_att.k_proj.weight.copy_(k_proj_weight.T)
        multihead_att.v_proj.weight.copy_(v_proj_weight.T)
        multihead_att.o_proj.weight.copy_(o_proj_weight.T)
    return multihead_att(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multihead_att = Multihead_self_attention(d_model, num_heads, RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len), theta)
    with torch.no_grad():
        multihead_att.q_proj.weight.copy_(q_proj_weight.T)
        multihead_att.k_proj.weight.copy_(k_proj_weight.T)
        multihead_att.v_proj.weight.copy_(v_proj_weight.T)
        multihead_att.o_proj.weight.copy_(o_proj_weight.T)
    return multihead_att(in_features, token_positions)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_ik = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k) # [d_k / 2]
        pos = torch.arange(max_seq_len, device=device) # [max_seq_len]
        angles = torch.einsum("i,j->ij", pos, theta_ik) # [max_seq_len, d_k / 2]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2] # [..., d_k / 2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        # out = torch.stack([rotated_x1, rotated_x2], dim=-1).reshape(x.shape) # [..., d_k / 2, 2] -> [..., d_k]
        out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        return out

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        if theta is not None:
            pos_encode = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            self.attn = Multihead_self_attention(d_model=d_model, num_heads=num_heads, pos_encode=pos_encode, theta=theta)
        else:
            self.attn = Multihead_self_attention(d_model=d_model, num_heads=num_heads)
        self.rmsn_1 = RMSNorm(d_model=d_model, eps=1e-5)
        self.rmsn_2 = RMSNorm(d_model=d_model, eps=1e-5)
        # self.pw_ffn = SiLU(d_model=d_model, d_ff=d_ff)
        self.pw_ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(self.rmsn_1(x))
        out1 = x + attn
        out2 = self.pw_ffn(self.rmsn_2(out1))
        out = out1 + out2
        return out

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    block = Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)
    with torch.no_grad():
        block.rmsn_1.weight.copy_(weights['ln1.weight'])
        block.attn.q_proj.weight.copy_(weights['attn.q_proj.weight'].T)
        block.attn.k_proj.weight.copy_(weights['attn.k_proj.weight'].T)
        block.attn.v_proj.weight.copy_(weights['attn.v_proj.weight'].T)
        block.attn.o_proj.weight.copy_(weights['attn.output_proj.weight'].T)
        block.rmsn_2.weight.copy_(weights['ln2.weight'])
        block.pw_ffn.load_weights(weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'])
    return block(in_features)

class Transformer_lm(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float | None = None):
        super().__init__()
        self.context_length = context_length
        self.transformer = nn.ModuleDict(dict(
            token_emb = Embedding(num_embedding=vocab_size, embedding_dim=d_model),
            n_block = nn.ModuleList([Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)]),
            rmsn_l = RMSNorm(d_model=d_model, eps=1e-5)
        ))
        self.linear_emb = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tkemb = self.transformer.token_emb(x)
        for block in self.transformer.n_block:
            tkemb = block(tkemb)
        tkemb = self.transformer.rmsn_l(tkemb)
        out = self.linear_emb(tkemb)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_gen_tokens: int, temperature: float = 1.0, top_p: int | None = None, eos_token_id: int | None = None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_gen_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :]
            temperature_scaled = next_token_logits / temperature
            if top_p:
                sorted_logits, sorted_indices = torch.sort(temperature_scaled, descending=True)
                sorted_probs = run_softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                temperature_scaled = temperature_scaled.masked_fill(mask, float("-inf"))
            probs = run_softmax(temperature_scaled, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        return x[:, original_sequence_length:]

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "model_config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_path, "model.pt")
        state_dict = torch.load(weights_path, weights_only=True)
        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, pretrained_path: str):
        os.makedirs(pretrained_path, exist_ok=True)
        config = {
            "vocab_size": self.transformer["token_emb"].weight.size(0),
            "context_length": self.context_length,
            "num_layers": len(self.transformer["n_block"]),
            "d_model": self.transformer["token_emb"].weight.size(1),
            "num_heads": self.transformer["n_block"][0].num_heads,
            "d_ff": self.transformer["n_block"][0].pw_ffn.d_ff,
            "rope_theta": self.transformer["n_block"][0].theta
        }
        with open(Path(pretrained_path) / "model_config.json", "w") as f:
            json.dump(config, f, indent=4)
        torch.save(self.state_dict(), Path(pretrained_path) / "model.pt")


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    lm = Transformer_lm(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)
    with torch.no_grad():
        lm.transformer.token_emb.weight.copy_(weights['token_embeddings.weight'])
        for i in range(num_layers):
            block = lm.transformer.n_block[i]
            block.rmsn_1.weight.copy_(weights[f'layers.{i}.ln1.weight'])
            block.attn.q_proj.weight.copy_(weights[f'layers.{i}.attn.q_proj.weight'].T)
            block.attn.k_proj.weight.copy_(weights[f'layers.{i}.attn.k_proj.weight'].T)
            block.attn.v_proj.weight.copy_(weights[f'layers.{i}.attn.v_proj.weight'].T)
            block.attn.o_proj.weight.copy_(weights[f'layers.{i}.attn.output_proj.weight'].T)
            block.rmsn_2.weight.copy_(weights[f'layers.{i}.ln2.weight'])
            block.pw_ffn.load_weights(weights[f'layers.{i}.ffn.w1.weight'], weights[f'layers.{i}.ffn.w2.weight'], weights[f'layers.{i}.ffn.w3.weight'])
        lm.transformer.rmsn_l.weight.copy_(weights['ln_final.weight'])
        lm.linear_emb.weight.copy_(weights['lm_head.weight'].T)
    return lm(in_indices)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_x = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / RMS_x * self.weight
        return result.to(in_dtype)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    with torch.no_grad():
        rmsnorm.weight.copy_(weights)
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    st = torch.randint(len(dataset) - context_length, (batch_size,))
    input_seq = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in st])
    target_seq = torch.stack([torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in st])
    if 'cuda' in device:
        input_seq = input_seq.pin_memory().to(device, non_blocking=True)
        target_seq = target_seq.pin_memory().to(device, non_blocking=True)
    else:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
    return input_seq, target_seq


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    dim_max = torch.amax(in_features, dim=dim, keepdim=True)
    dim_exp = torch.exp(in_features - dim_max)
    sum_dim_exp = torch.sum(dim_exp, dim=dim, keepdim=True)
    return dim_exp / sum_dim_exp


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    dim_max = torch.amax(inputs, dim=-1, keepdim=True)
    dim_submax = inputs - dim_max
    dim_logsumexp = dim_submax - torch.log(torch.sum(torch.exp(dim_submax), dim=-1, keepdim=True))
    return torch.mean(torch.gather(input=-dim_logsumexp, dim=-1, index=targets.unsqueeze(-1)))


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = []
    for pt in parameters:
        if pt.grad is not None:
            grads.append(pt.grad)
    grads_l2norm = 0.0
    for gd in grads:
        grads_l2norm += (gd ** 2).sum()
    grads_l2norm = torch.sqrt(grads_l2norm)
    if grads_l2norm >= max_l2_norm:
        ft = max_l2_norm / (grads_l2norm + 1e-6)
        for gd in grads:
            gd *= ft

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, betas: tuple[float, float] = [0.9, 0.999], eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid betas[0]: {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid betas[1]: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lambde = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                m_ = beta1 * m + (1 - beta1) * grad
                v_ = beta2 * v + (1 - beta2) * torch.square(grad)
                alpha_t = alpha * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= alpha_t * (m_ / (torch.sqrt(v_) + eps))
                p.data -= alpha * lambde * p.data
                state["m"] = m_
                state["v"] = v_
                state["t"] = t + 1
        return loss

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    alpha_t = 0.0
    if it < warmup_iters:
        alpha_t = it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        alpha_t = min_learning_rate + 0.5 * (1 + math.cos(((it-warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi)) * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        alpha_t = min_learning_rate
    return alpha_t


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoints = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoints, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoints = torch.load(src)
    model.load_state_dict(checkpoints["model_state"])
    optimizer.load_state_dict(checkpoints["optimizer_state"])
    return checkpoints["iteration"]

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges_rank = {merge: i for i, merge in enumerate(merges)}
        self.special_token_bytes = set()
        if special_tokens:
            self.special_token_bytes = {st.encode('utf-8') for st in special_tokens}
            self.add_special_tokens(special_tokens)
        self.vocab_to_id  = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 加载 vocab.pkl
        with open(vocab_filepath, 'rb') as vf:
            raw_vocab = pickle.load(vf)
        # 转换为 {int: bytes}
        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}
        # 加载 merges.pkl
        with open(merges_filepath, 'rb') as mf:
            raw_merges = pickle.load(mf)
        # 转换为 List[Tuple[bytes, bytes]]
        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encode_list: list[int] = []
        if self.special_token_bytes:
            pattern = '|'.join(re.escape(st.decode('utf-8')) for st in self.special_token_bytes)
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]
        for chunk in chunks:
            if not chunk:
                continue
            bck = chunk.encode('utf-8')
            if bck in self.special_token_bytes:
                encode_list.append(self.vocab_to_id[bck])
            else:
                for tk in gpt2_pre_tokenize(chunk):
                    btk = tk.encode('utf-8')
                    for merge_tk in self.merge_bytes(btk):
                        encode_list.append(self.vocab_to_id[merge_tk])
        return encode_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for it in tqdm(iterable):
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        ret = b"".join(self.vocab[it] for it in ids).decode('utf-8', errors="replace")
        return ret

    def add_special_tokens(self, special_tokens: list[str]):
        exit_vocab = set(self.vocab.values())
        new_special_tokens = [st.encode('utf-8') for st in special_tokens if st.encode('utf-8') not in exit_vocab]
        if new_special_tokens:
            max_key = max(self.vocab.keys())
            self.vocab.update({max_key + i + 1: nst for i, nst in enumerate(new_special_tokens)})

    def merge_bytes(self, token: bytes) -> list[bytes]:
        split = [bytes([tk]) for tk in token]
        while True:
            pair_rank: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            for i in range(len(split) - 1):
                pair_rank[(split[i], split[i + 1])] = self.merges_rank.get((split[i], split[i + 1]), float("inf"))
            if not pair_rank:
                break
            best_pair = min(pair_rank.items(), key=lambda x: (x[1], -int.from_bytes(x[0][0] + x[0][1], "big")))[0]
            if best_pair not in self.merges_rank:
                break
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
        return split

        
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(tok, bytes) for tok in split_special_tokens), (
        "All special tokens must be bytes"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_positions = [
                (mini_chunk.find(tok), tok)
                for tok in split_special_tokens if mini_chunk.find(tok) != -1
            ]

            # Find the special token in the mini chunk
            if found_positions:
                found_at = min(pos for pos, _ in found_positions)
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique
    return sorted(set(chunk_boundaries))

def gpt2_pre_tokenize(text: str) -> list:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return [m.group() for m in re.finditer(PAT, text)]

def tokenize_chunk(chunk_and_special_tokens: tuple[str, list[str]]) -> list[str]:
    chunk, special_tokens = chunk_and_special_tokens
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)

    sub_chunks = [s for s in re.split(f"({special_pat})", chunk) if s]
    all_tokens = []
    for sub_chunk in sub_chunks:
        if sub_chunk in special_tokens:
            all_tokens.append(sub_chunk)
        else:
            all_tokens.extend(gpt2_pre_tokenize(sub_chunk))
    # sub_chunks = [s for s in re.split(special_pat, chunk) if s]
    # # 对每个子块进行预分词
    # all_tokens = []
    # for sub_chunk in sub_chunks:
    #     all_tokens.extend(gpt2_pre_tokenize(sub_chunk))
    return all_tokens

# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """
#     num_processes = 8
#     num_chunks = 4 * num_processes
#     special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}
#     with open(input_path, 'rb') as f:
#         boundaries = find_chunk_boundaries(f, num_chunks, list(special_token_bytes))
#     def count_tokens_in_chunk(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict[bytes, int]:
#         freq: dict[bytes, int] = defaultdict(int)

#         with open(input_path, 'rb') as f:
#             mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#             mmapped.seek(start)
#             chunk = mmapped.read(end - start).decode('utf-8', errors='ignore')
#             tokens = tokenize_chunk((chunk, special_tokens))  # this returns list[str]
#             for tok in tokens:
#                 b = tok.encode('utf-8')
#                 if b in special_token_bytes:
#                     continue
#                 freq[b] += 1
#         return freq
#     results = Parallel(n_jobs=num_processes)(
#         delayed(count_tokens_in_chunk)(start, end, input_path, special_tokens)
#         for start, end in zip(boundaries[:-1], boundaries[1:])
#     )
#         # chunks = []
#         # for start, end in zip(boundaries[:-1], boundaries[1:]):
#         #     f.seek(start)
#         #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         #     chunks.append(chunk)
#         # results = Parallel(n_jobs=num_processes)(delayed(tokenize_chunk)((chunk, special_tokens)) for chunk in chunks)
#         # tokens: list[bytes] = [tok.encode('utf-8') for chunk in results for tok in chunk]
#     vocab: dict[int, bytes] = {}
#     merges: list[tuple[bytes, bytes]] = []
#     token_freqs = defaultdict(int)
#     for local_freq in results:
#         for token, count in local_freq.items():
#             token_freqs[token] += count
#     # for token in tokens:
#     #     if token in special_token_bytes:
#     #         continue
#     #     token_freqs[token] += 1
#     token_id = 0
#     for special_token in special_tokens:
#         vocab[token_id] = special_token.encode('utf-8')
#         token_id += 1
#     for i in range(256):
#         b = bytes([i])
#         if b not in special_token_bytes:
#             vocab[token_id] = b
#             token_id += 1
#     split_freqs: dict[bytes, list[bytes]] = {token: [bytes([t]) for t in token] for token in token_freqs}
#     pair_freqs: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
#     for token, split in split_freqs.items():
#         freq = token_freqs[token]
#         for i in range(len(split) - 1):
#             pair_freqs[(split[i], split[i + 1])] += freq
#     nvocab = len(vocab)
#     num_merges = vocab_size - nvocab
#     for _ in tqdm(range(num_merges)):
#         best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
#         merges.append(best_pair)
#         new_token = best_pair[0] + best_pair[1]
#         vocab[nvocab] = new_token
#         nvocab += 1
#         affected_tokens = defaultdict(int)
#         for token, split in split_freqs.items():
#             if best_pair[0] not in split:
#                 continue
#             i = 0
#             new_split = []
#             changed = False
#             while i < len(split):
#                 if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
#                     new_split.append(new_token)
#                     i += 2
#                     changed = True
#                 else:
#                     new_split.append(split[i])
#                     i += 1
#             if changed:
#                 affected_tokens[token] = new_split
#         for token, old_split in affected_tokens.items():
#             # 移除旧对
#             freq = token_freqs[token]
#             for i in range(len(split_freqs[token]) - 1):
#                 pair = (split_freqs[token][i], split_freqs[token][i + 1])
#                 pair_freqs[pair] -= freq
#                 if pair_freqs[pair] <= 0:
#                     del pair_freqs[pair]

#             # 更新 split
#             split_freqs[token] = old_split

#             # 添加新对
#             for i in range(len(old_split) - 1):
#                 pair = (old_split[i], old_split[i + 1])
#                 pair_freqs[pair] += freq
#     return vocab, merges

def _update_stats(
    word: bytes,
    split: list[bytes],
    word_freq: int,
    pair_freqs: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]],
    is_add: bool
):
    """
    一个辅助函数，用于从频率和索引中添加或删除一个词的所有对。
    is_add=True  -> 添加
    is_add=False -> 移除
    """
    for i in range(len(split) - 1):
        pair = (split[i], split[i + 1])
        delta = word_freq if is_add else -word_freq
        
        pair_freqs[pair] += delta
        if pair_freqs[pair] <= 0:
            # 安全地删除，以防万一
            if pair in pair_freqs:
                del pair_freqs[pair]
        
        # 更新倒排索引，初始时 words_set 等于空集合 set()
        words_set = pair_to_words[pair]
        if is_add:
            # pair_to_words[pair] 的 value 也会变成 word 不再是 set()
            words_set.add(word)
        elif word in words_set:
            words_set.remove(word)
        
        if not words_set:
            if pair in pair_to_words:
                del pair_to_words[pair]

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = 8
    num_chunks = 4 * num_processes
    special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, list(special_token_bytes))
    def count_tokens_in_chunk(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict[bytes, int]:
        freq: dict[bytes, int] = defaultdict(int)

        with open(input_path, 'rb') as f:
            mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mmapped.seek(start)
            chunk = mmapped.read(end - start).decode('utf-8', errors='ignore')
            tokens = tokenize_chunk((chunk, special_tokens))  # this returns list[str]
            for tok in tokens:
                b = tok.encode('utf-8')
                if b in special_token_bytes:
                    continue
                freq[b] += 1
        return freq
    results = Parallel(n_jobs=num_processes)(
        delayed(count_tokens_in_chunk)(start, end, input_path, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )
        # chunks = []
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #     chunks.append(chunk)
        # results = Parallel(n_jobs=num_processes)(delayed(tokenize_chunk)((chunk, special_tokens)) for chunk in chunks)
        # tokens: list[bytes] = [tok.encode('utf-8') for chunk in results for tok in chunk]
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    token_freqs = defaultdict(int)
    for local_freq in results:
        for token, count in local_freq.items():
            token_freqs[token] += count
    # for token in tokens:
    #     if token in special_token_bytes:
    #         continue
    #     token_freqs[token] += 1
    token_id = 0
    for special_token in special_tokens:
        vocab[token_id] = special_token.encode('utf-8')
        token_id += 1
    for i in range(256):
        b = bytes([i])
        if b not in special_token_bytes:
            vocab[token_id] = b
            token_id += 1
    split_freqs: dict[bytes, list[bytes]] = {token: [bytes([b]) for b in token] for token in token_freqs}
    pair_freqs: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

    for word, split in split_freqs.items():
        word_freq = token_freqs[word]
        _update_stats(word, split, word_freq, pair_freqs, pair_to_words, is_add=True)

    nvocab = len(vocab)
    num_merges = vocab_size - nvocab
    
    for i in tqdm(range(num_merges)):
        if not pair_freqs:
            break
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
 
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[nvocab] = new_token
        nvocab += 1
        
        p1, p2 = best_pair
        # 直接从倒排索引中找到所有受影响的词，这是优化的关键
        words_to_update = list(pair_to_words[best_pair])

        for word in words_to_update:
            word_freq = token_freqs[word]
            old_split = split_freqs[word]
            
            # 先完全移除旧对的统计
            _update_stats(word, old_split, word_freq, pair_freqs, pair_to_words, is_add=False)
            
            # 创建新的拆分
            new_split = []
            i = 0
            while i < len(old_split):
                if i < len(old_split) - 1 and old_split[i] == p1 and old_split[i+1] == p2:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1
            
            # 更新词的拆分
            split_freqs[word] = new_split
            # 再完全添加新对的统计
            _update_stats(word, new_split, word_freq, pair_freqs, pair_to_words, is_add=True)
    return vocab, merges