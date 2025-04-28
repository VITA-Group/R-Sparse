import os
import sys
import pdb
import tqdm
import math
import copy
import types
import torch
import numpy as np
from typing import Optional, Tuple

from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache


__all__ = ["LlamaForCausalLM_R_Sparse", "R_Sparse_Linear"]


class R_Sparse_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(R_Sparse_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.prefill_ratio = 0.5
        self.flag_getting_threshold = False
        self.target_sparsity = 0

        self.rank = None
        self.threshold = None
        self.sparse_ratio = None

    def _getting_threshold(self, input):
        nelements = input.numel()

        if self.sparse_ratio == 1:
            threshold = torch.topk(input.abs().view(-1), int(nelements * self.target_sparsity), largest=False)[0][-1]
            estimated_sparsity = (input.abs() < threshold).float().mean().item()
        else:
            scale_input = input * self.scale.unsqueeze(0).unsqueeze(0)
            threshold = torch.topk(scale_input.abs().view(-1), int(nelements * self.target_sparsity), largest=False)[0][-1]
            estimated_sparsity = (scale_input.abs() < threshold).float().mean().item()

        output = F.linear(input, self.weight, self.bias)
        self.threshold = threshold.item()
        print('Setting threshold: ', self.threshold, 'Estimated sparsity: ', estimated_sparsity, 'Target sparsity: ', self.target_sparsity)
        self.flag_getting_threshold = False
        return output

    def _setting_mode(self):
        if self.channels is not None and self.rank is not None:
            self.mode = 'r_sparse'
        elif self.channels is not None:
            self.mode = 'sparse'
        elif self.rank is not None:
            self.mode = 'low_rank'
        else:
            self.mode = 'dense'

    def _load_low_rank_module(self, path):
        if os.path.exists(path):
            usv = torch.load(path, map_location='cpu')
            u = usv[0][:, :self.rank]
            s = usv[1][:self.rank]
            v = usv[2][:, :self.rank]
            scale = usv[3]
            self.register_buffer('U', u.to(self.weight.dtype).to(self.weight.device))
            self.register_buffer('S', s.to(self.weight.dtype).to(self.weight.device))
            self.register_buffer('V', v.to(self.weight.dtype).to(self.weight.device))
            self.register_buffer('scale', scale.to(self.weight.dtype).to(self.weight.device))

        else:
            print('Can not find SVD decomposition file')

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.flag_getting_threshold:
            output = self._getting_threshold(input)
            return output

        num_tokens = input.size(1)
        if self.prefill_ratio == 1:
            if num_tokens > 1: # prefilling stage
                output = F.linear(input, self.weight, self.bias)
            else:
                if self.mode == 'dense':
                    output = F.linear(input, self.weight, self.bias)
                elif self.mode == 'sparse': # sparse forward
                    s_mask = input.abs().gt(self.threshold).to(input.dtype)
                    output = F.linear(input * s_mask, self.weight, self.bias)
                elif self.mode == 'low_rank': # low rank forward
                    output = input @ (self.V[:, :self.rank] @ torch.diag(self.S[:self.rank])) 
                    output = low_rank_output @ self.U[:, :self.rank].T
                elif self.mode == 'r_sparse': # R-Sparse forward
                    scale_input = input * self.scale.unsqueeze(0).unsqueeze(0)
                    s_mask = scale_input.abs().gt(self.threshold).to(input.dtype)

                    sparse_input = input * s_mask
                    sparse_output = F.linear(sparse_input, self.weight, self.bias)

                    low_rank_input = input * (1 - s_mask)
                    low_rank_output = low_rank_input @ (self.V[:, :self.rank] @ torch.diag(self.S[:self.rank])) 
                    low_rank_output = low_rank_output @ self.U[:, :self.rank].T

                    output = sparse_output + low_rank_output
                else:
                    raise NotImplementedError

        else:
            assert num_tokens > 1

            decoding_tokens = int(num_tokens * (1 - self.prefill_ratio))
            input_prefill = input[:, :decoding_tokens, :]
            input_decoding = input[:, decoding_tokens:, :]

            output_prefill = F.linear(input_prefill, self.weight, self.bias)

            if self.mode == 'dense':
                output_decoding = F.linear(input_decoding, self.weight, self.bias)
                output = torch.cat([output_prefill, output_decoding], dim=1)
            elif self.mode == 'sparse': # sparse forward
                s_mask = input_decoding.abs().gt(self.threshold).to(input.dtype)
                sparse_input = input_decoding * s_mask
                sparse_output = F.linear(sparse_input, self.weight, self.bias)
                output = torch.cat([output_prefill, sparse_output], dim=1)
            elif self.mode == 'low_rank': # low rank forward
                low_rank_output = input_decoding @ (self.V[:, :self.rank] @ torch.diag(self.S[:self.rank])) 
                low_rank_output = low_rank_output @ self.U[:, :self.rank].T
                output = torch.cat([output_prefill, low_rank_output], dim=1)
            elif self.mode == 'r_sparse': # R-Sparse forward
                scale_input = input_decoding * self.scale.unsqueeze(0).unsqueeze(0)
                s_mask = scale_input.abs().gt(self.threshold).to(input.dtype)

                sparse_input = input_decoding * s_mask
                sparse_output = F.linear(sparse_input, self.weight, self.bias)

                low_rank_input = input_decoding * (1 - s_mask)
                low_rank_output = low_rank_input @ (self.V[:, :self.rank] @ torch.diag(self.S[:self.rank])) 
                low_rank_output = low_rank_output @ self.U[:, :self.rank].T

                output = torch.cat([output_prefill, sparse_output + low_rank_output], dim=1)
            else:
                raise NotImplementedError

        return output




class LlamaForCausalLM_R_Sparse(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            original_linear_layer = self.model.layers[layer_idx].mlp.gate_proj
            gate_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            gate_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.gate_proj = gate_proj

            original_linear_layer = self.model.layers[layer_idx].mlp.up_proj
            up_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            up_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.up_proj = up_proj

            original_linear_layer = self.model.layers[layer_idx].mlp.down_proj
            down_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            down_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.down_proj = down_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.q_proj
            q_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            q_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.q_proj = q_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.k_proj
            k_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            k_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.k_proj = k_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.v_proj
            v_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            v_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.v_proj = v_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.o_proj
            o_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, bias=False)
            o_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.o_proj = o_proj

    def _load_low_rank_module(self, config):
        num_layers = len(self.model.layers)
        for layer_idx in tqdm.tqdm(range(num_layers)):
            self.model.layers[layer_idx].mlp.gate_proj._load_low_rank_module(config.gate_svd_path[layer_idx])
            self.model.layers[layer_idx].mlp.up_proj._load_low_rank_module(config.up_svd_path[layer_idx])
            self.model.layers[layer_idx].mlp.down_proj._load_low_rank_module(config.down_svd_path[layer_idx])
            self.model.layers[layer_idx].self_attn.q_proj._load_low_rank_module(config.q_svd_path[layer_idx])
            self.model.layers[layer_idx].self_attn.k_proj._load_low_rank_module(config.k_svd_path[layer_idx])
            self.model.layers[layer_idx].self_attn.v_proj._load_low_rank_module(config.v_svd_path[layer_idx])
            self.model.layers[layer_idx].self_attn.o_proj._load_low_rank_module(config.o_svd_path[layer_idx])

































class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True



        self.prefill_ratio = 0.5

        self.flag_getting_threshold = True
        self.target_sparsity = 0.5

        self.rank = 0
        self.threshold = None
        self.sparse_ratio = 1
        self.mode = 'sparse'



        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            assert False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # if self.flag_getting_threshold:

        #     self._getting_threshold(input)

        #     self.flag_getting_threshold = False
        # else:
        #     # sparsity

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value





















