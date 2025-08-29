from typing import List, Optional, Tuple, Union

import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from fairscale.nn.model_parallel import initialize as mpu
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, StaticCache
from qwen import Qwen2ForCausalLM, Qwen2RotaryEmbedding, Qwen2RMSNorm, Qwen2MLP, apply_rotary_pos_emb
from dataclasses import dataclass
from transformers.utils import (
    logging,
)
import random
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from functools import lru_cache, partial
from transformers.utils import ModelOutput
import math
from torch import nn
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os
from types import SimpleNamespace
import json

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_single_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    llm_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

class GlideAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

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
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.max_len = 512
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.softmax_scale = 1 / (self.head_dim ** 0.5)
        self.range_indices = torch.arange(1024)
        
        # self.flex_attn = torch.compile(flex_attention)
        self.set_torch_mask()

    def set_flex_attn(self, seqlen):
        def block_mask(b, h, q_idx, kv_idx):
            q_block = q_idx // 4
            kv_block = kv_idx // 4
            return q_block > kv_block
        mask = create_block_mask(block_mask, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen,  _compile=True)
        return mask
    
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
        flex_attn=None,
        exec_type="training",
        k_cache=None,
        v_cache=None,
        llm_kv_len=None,
        tree_mask=None,
    ):
        
        if exec_type in ["prefill", "sa_prefill"]:
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "sa_decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens)
        elif exec_type in ["sa_tree_decoding"]:
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
        attn_output = flash_attn_with_kvcache(query_states, K_Cache, V_Cache, key_states, 
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

class Qwen2GlideDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.self_attn = GlideAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._init_weights
        self.config = config
        # self.apply(self._init_weights)
    

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        exec_type=None,
        tree_mask=None,
    ):

        residual = hidden_states
        # print(hidden_states.shape, self.input_layernorm.weight.shape)
        # torch.cuda.synchronize()
        # record_time = time.time()
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            exec_type="sa_" + exec_type,
            tree_mask=tree_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # torch.cuda.synchronize()
        # print("ffn time", time.time() - record_time)
        # record_time = time.time()
        return hidden_states

class QwenEagle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.layers = nn.ModuleList([Qwen2GlideDecoderLayer(config, layer_idx=0)]) 
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        print("this eagle code is only used for inference.")

    def set_max_gen_len(self, max_gen_len):
        for layer in self.layers:
            layer.self_attn.max_len = max_gen_len

    def load_weight(self, path: str):
        
        print("load eagle model...")
        start = time.perf_counter()
        path = os.path.join(path, "pytorch_model.bin")
        state_dict = torch.load(path)
        print(state_dict.keys())
        missing_keys, unexpected_keys = self.load_state_dict(state_dict)
        print(missing_keys)
        print(unexpected_keys)
        end = time.perf_counter()
        print("loading ended in {} seconds.".format(end - start))

    def eagle_forward(self, hidden_states, position_embeddings, cache_lens, exec_type, tree_mask=None):
        hidden_state = self.fc(hidden_states)
        for layer in self.layers:
            hidden_state = layer(hidden_state, position_embeddings, cache_lens, exec_type, tree_mask)
        return hidden_state

