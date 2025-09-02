# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
import math
import time
from typing import List, Optional, Tuple, Union
from flash_attn import flash_attn_func, flash_attn_with_kvcache

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from typing import Callable


import torch._dynamo as dynamo

@dynamo.disable
def flash_attn_with_kvcache_wrapper(query_states, K_Cache, V_Cache, 
                                    key_states, value_states, 
                                    causal=True, cache_seqlens=None):
    # 如果 flash_attn_with_kvcache 是在同一作用域，
    # 可以直接调用它，否则请根据你的导入方式来调用
    return flash_attn_with_kvcache(
        query_states, 
        K_Cache, 
        V_Cache, 
        key_states, 
        value_states, 
        causal=causal, 
        cache_seqlens=cache_seqlens
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    # return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.max_len = 512
        self.prefix_lens = None
        self.softmax_scale = 1 / (self.head_dim ** 0.5)
        self.range_indices = torch.arange(1024)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def set_torch_mask(self, max_len=4096, block_size=4):
        q_idx = torch.arange(max_len).view(-1, 1)
        kv_idx = torch.arange(max_len).view(1, -1)
        self.torch_mask = q_idx // block_size > kv_idx // block_size
        self.torch_mask = self.torch_mask.cuda()
        self.torch_mask[:4, :4] = True

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        exec_type="training",
        tree_mask=None,
    ):
        
        if exec_type in ["prefill", "sa_prefill"]:
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens)
        elif exec_type in ["tree_decoding"]:
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, tree_mask)
        else:
            raise ValueError(f"Unknown inference_type: {exec_type}")
        return y

    def prefill(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        # print("*" * 10)
        # print(q_len, self.max_len)
        # print("*" * 10)

        self.K_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.K_Cache[:, :q_len] = key_states
        self.V_Cache[:, :q_len] = value_states
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        self.range_indices = self.range_indices.to(self.K_Cache.device)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        K_Cache = self.K_Cache
        V_Cache = self.V_Cache
        attn_output = flash_attn_with_kvcache_wrapper(query_states, K_Cache, V_Cache, key_states, 
                                                value_states, causal=True, cache_seqlens=cache_lens.int())

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            tree_mask=None,
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        prefix_o, prefix_lse = flash_attn_with_kvcache(query_states, self.K_Cache, self.V_Cache, cache_seqlens=cache_lens, return_softmax_lse=True)

        # update kv cache
        _, current_kv_len, all_kv_len = tree_mask.size()
        range_indices = cache_lens.unsqueeze(-1) + self.range_indices[all_kv_len - current_kv_len : all_kv_len].unsqueeze(0)
        bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
        # print("=" * 10)
        # print(self.K_Cache.shape)
        # print(bsz_indices)
        # print(range_indices)
        # print(key_states.shape)
        # print(self.K_Cache[bsz_indices, range_indices].shape)
        self.K_Cache[bsz_indices, range_indices] = key_states
        self.V_Cache[bsz_indices, range_indices] = value_states

        all_cache_indices = cache_lens.unsqueeze(-1) + self.range_indices[0 :all_kv_len].unsqueeze(0)
        key_states = self.K_Cache[bsz_indices, all_cache_indices]
        value_states = self.V_Cache[bsz_indices, all_cache_indices]
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute(0, 2, 3, 1)
        value_states = value_states.transpose(1, 2)
        attn_score = torch.matmul(query_states, key_states) * self.softmax_scale
        attn_score = attn_score.masked_fill(tree_mask.unsqueeze(1) == 0, -float('inf'))
        attn_weight = torch.softmax(attn_score.float(), dim=-1).to(query_states.dtype)
        current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
        current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2).to(query_states.dtype) # bsz, seqlen, headnum, 1
        if torch._dynamo.is_compiling():
            prefix_lse = prefix_lse.reshape(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        else:
            prefix_lse = prefix_lse.view(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse.view(bsz, q_len, self.num_heads, -1)).to(query_states.dtype)
        attn_output = prefix_o * weight + current_out * (1 - weight)

        attn_output = attn_output.view(bsz, q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, hidden_states, position_embeddings, cache_lens, exec_type, tree_mask) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            cache_lens,
            exec_type,
            tree_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return hidden_states


class Eagle(nn.Module):

    def __init__(self, config, bias: bool = True):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.stable_kv: List[Tuple[torch.Tensor]] = None
        self.tree_mask: torch.Tensor = None
        self.gradient_checkpointing: bool = False

    def set_max_gen_len(self, max_gen_len):
        for layer in self.layers:
            layer.self_attn.max_len = max_gen_len

    def load_weight(self, path: str):
        print("load eagle model...")
        start = time.perf_counter()
        path = os.path.join(path, "pytorch_model.bin")
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        end = time.perf_counter()
        print("loading ended in {} seconds.".format(end - start))

    def eagle_forward(self, hidden_states, position_embeddings, cache_lens, exec_type, tree_mask=None):
        hidden_state = self.fc(hidden_states)
        for layer in self.layers:
            hidden_state = layer(hidden_state, position_embeddings, cache_lens, exec_type, tree_mask)
        return hidden_state


    @torch.no_grad()
    def repeat_hidden(self, hidden_state: torch.Tensor, repeat_nums: int):
        new_hidden = []
        for id, i in enumerate(repeat_nums):
            new_hidden.append(hidden_state[:, id : id + 1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)

    def sample(self, logits, logits_processor, k=1):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (
                torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device),
                cumulative_sum[:, :-1],
            ),
            dim=-1,
        )

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs, probabilities

    def greedy(self, logits, k=1):
        top = torch.topk(logits, k, dim=-1)
        topk_index, topk_prob = top.indices, top.values
        return topk_index, topk_prob, None

    @torch.no_grad()
    def topk_genrate(
        self, hidden_states, input_ids, head, logits_processor: Callable = None, top_k: int = 10
    ):
        self.tree_mask = None  # reset tree mask
        
        hidden_states = hidden_states[None]
        input_ids = input_ids[None, 1:]
        ss_token = []
        ss_prob = []
        ss_op = []
        
        out_hidden, past_key_values = self(
            hidden_states,
            input_ids=input_ids,
            past_key_values=self.stable_kv,
            use_cache=True,
        )
        self.stable_kv = past_key_values
        position_len = self.stable_kv[0][0].shape[-2]
        last_hidden = out_hidden[:, -1]
        last_headout = head(last_hidden)
        for i in range(len(self.tree_buffer["tree_indices"])):
            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(
                    last_headout, logits_processor, k=top_k
                )
            else:
                topk_index, topk_prob, op = self.greedy(last_headout, k=top_k)

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            topk_index = topk_index.view(-1)
            select_index = topk_index[self.tree_buffer["tree_indices"][i]]
            input_ids = select_index[None, :]
            if i == 0:
                hidden_states = out_hidden[:, -1:]
            else:
                hidden_states = out_hidden
            hidden_states = self.repeat_hidden(
                hidden_states, self.tree_buffer["repeat_nums"][i]
            )
            self.tree_mask = self.tree_buffer["attn_mask"][i]
            position_ids = position_len + self.tree_buffer["position_ids"][i]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            position_len += 1
            last_headout = head(out_hidden[0])
        if logits_processor is not None:
            topk_index, topk_prob, op = self.sample(
                last_headout, logits_processor, k=top_k
            )
        else:
            topk_index, topk_prob, op = self.greedy(last_headout, k=top_k)
        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)
        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op)