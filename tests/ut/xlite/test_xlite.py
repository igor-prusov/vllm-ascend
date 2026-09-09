#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""Unit tests for the xlite adapter weight/state helpers and hybrid config building."""

import types

import pytest
import torch

try:
    from vllm_ascend.xlite.xlite import (
        AscendMetadata,
        XliteWrapper,
        _flatten_expert_weights,
        _gemma_norm_to_xlite,
        _pack_mha_qkv_with_gate,
        _resolve_full_attention_interval,
        _split_linear_ba,
        _split_linear_qkvz,
    )
except ImportError:  # pragma: no cover - depends on optional xlite/vllm install
    pytest.skip("xlite/vllm not available", allow_module_level=True)


def test_gemma_norm_to_xlite():
    weight = torch.randn(8, dtype=torch.float16)
    result = _gemma_norm_to_xlite(weight)
    expected = (weight.float() + 1.0).to(weight.dtype).contiguous()
    torch.testing.assert_close(result, expected)
    assert result.dtype == weight.dtype
    assert result.is_contiguous()


def test_flatten_expert_weights():
    packed = torch.randn(3, 4, 5)
    flat_2d = torch.randn(6, 7)
    flat = _flatten_expert_weights([packed, flat_2d])
    assert len(flat) == 4
    assert [tuple(t.shape) for t in flat] == [(4, 5), (4, 5), (4, 5), (6, 7)]
    for t in flat:
        assert t.is_contiguous()
    torch.testing.assert_close(flat[0], packed[0])
    torch.testing.assert_close(flat[3], flat_2d)


def test_pack_mha_qkv_with_gate():
    num_heads, head_dim, num_kv_heads = 2, 3, 1
    d = 5
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    q_gate = torch.randn(num_heads, 2, head_dim, d)
    q_w = q_gate[:, 0].reshape(q_size, d)
    g_w = q_gate[:, 1].reshape(q_size, d)
    k_w = torch.randn(kv_size, d)
    v_w = torch.randn(kv_size, d)
    fused = torch.cat([q_gate.reshape(q_size * 2, d), k_w, v_w], dim=0)

    packed, packed_bias = _pack_mha_qkv_with_gate(fused, head_dim, num_heads, num_kv_heads)
    assert packed_bias is None
    torch.testing.assert_close(packed, torch.cat([q_w, k_w, v_w, g_w], dim=0))


def test_pack_mha_qkv_with_gate_bias():
    num_heads, head_dim, num_kv_heads = 2, 3, 1
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    q_gate_b = torch.randn(num_heads, 2, head_dim)
    k_b = torch.randn(kv_size)
    v_b = torch.randn(kv_size)
    bias = torch.cat([q_gate_b.reshape(-1), k_b, v_b])
    qkv_weight = torch.randn(q_size * 2 + kv_size * 2, 5)

    _, packed_bias = _pack_mha_qkv_with_gate(qkv_weight, head_dim, num_heads, num_kv_heads, bias)
    expected = torch.cat([q_gate_b[:, 0].reshape(-1), k_b, v_b, q_gate_b[:, 1].reshape(-1)])
    torch.testing.assert_close(packed_bias, expected)


def test_pack_mha_qkv_with_gate_invalid_rows():
    with pytest.raises(ValueError, match="Unexpected fused QKV rows"):
        _pack_mha_qkv_with_gate(torch.randn(7, 4), 2, 2, 1)


def test_split_linear_qkvz():
    key_dim, value_dim, d = 4, 5, 6
    qkv = torch.randn(key_dim * 2 + value_dim, d)
    z = torch.randn(value_dim, d)
    weight = torch.cat([qkv, z], dim=0)
    qkv_out, z_out = _split_linear_qkvz(weight, key_dim, value_dim)
    torch.testing.assert_close(qkv_out, qkv)
    torch.testing.assert_close(z_out, z)
    assert qkv_out.is_contiguous() and z_out.is_contiguous()


def test_split_linear_qkvz_invalid_rows():
    with pytest.raises(ValueError, match="Unexpected in_proj_qkvz rows"):
        _split_linear_qkvz(torch.randn(9, 3), 4, 5)


def test_split_linear_ba():
    weight = torch.randn(10, 4)
    b_w, a_w = _split_linear_ba(weight)
    torch.testing.assert_close(b_w, weight[:5])
    torch.testing.assert_close(a_w, weight[5:])
    assert b_w.is_contiguous() and a_w.is_contiguous()


def test_split_linear_ba_invalid_rows():
    with pytest.raises(ValueError, match="Unexpected in_proj_ba rows"):
        _split_linear_ba(torch.randn(7, 4))


def test_resolve_full_attention_interval_from_attribute():
    cfg = types.SimpleNamespace(full_attention_interval=2)
    assert _resolve_full_attention_interval(cfg) == 2


def test_resolve_full_attention_interval_from_layer_types():
    cfg = types.SimpleNamespace(
        full_attention_interval=None, layer_types=["linear_attention", "linear_attention", "full_attention"]
    )
    assert _resolve_full_attention_interval(cfg) == 3


def test_resolve_full_attention_interval_default():
    cfg = types.SimpleNamespace()
    assert _resolve_full_attention_interval(cfg) == 4


def test_pick_attn_metadata_prefers_largest_block_tables():
    m_short = AscendMetadata(block_tables=torch.zeros(2, 3))
    m_long = AscendMetadata(block_tables=torch.zeros(2, 7))
    picked = XliteWrapper._pick_attn_metadata({"a": m_short, "b": m_long}, AscendMetadata)
    assert picked is m_long


def test_pick_attn_metadata_single_candidate():
    m = AscendMetadata(block_tables=torch.zeros(2, 3))
    assert XliteWrapper._pick_attn_metadata({"a": m}, AscendMetadata) is m


def test_pick_attn_metadata_none_on_type_mismatch():
    assert XliteWrapper._pick_attn_metadata({"a": object()}, AscendMetadata) is None
    assert XliteWrapper._pick_attn_metadata(object(), AscendMetadata) is None
