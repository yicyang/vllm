# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlyDSL non-spec GDN decode must match the packed Triton decode for the
inputs _forward_core_decode_non_spec passes: strided q/k/v views of the conv
output, strided a/b, and CUDA-graph padding rows with state slot 0."""

import inspect

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER FlyDSL decode is ROCm-only", allow_module_level=True)

lak = pytest.importorskip("aiter.ops.flydsl.linear_attention_kernels")
if "min_valid_index" not in inspect.signature(lak.flydsl_gdr_decode).parameters:
    pytest.skip(
        "AITER lacks flydsl_gdr_decode(min_valid_index)", allow_module_level=True
    )

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402


@pytest.mark.parametrize("num_heads", [(16, 32), (16, 48)])
@pytest.mark.parametrize("batch,num_real", [(1, 1), (4, 3), (8, 8), (16, 9)])
def test_flydsl_decode_matches_packed(num_heads, batch, num_real):
    torch.set_default_device("cuda")
    set_random_seed(0)
    hk, hv = num_heads
    d = 128
    qkv = torch.randn(batch, 2 * hk * d + hv * d).bfloat16()
    ab = torch.randn(batch, 2 * hv).bfloat16()
    b, a = ab[:, :hv], ab[:, hv:]
    A_log = torch.randn(hv).bfloat16()
    dt_bias = torch.randn(hv).bfloat16()
    state = torch.randn(batch + 8, hv, d, d) * 0.1
    idx = (torch.randperm(batch + 7)[:batch] + 1).to(torch.int32)
    idx[num_real:] = 0  # CUDA-graph padding rows use the reserved slot 0

    ref_state = state.clone()
    ref_out = torch.empty(batch, 1, hv, d).bfloat16()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=d**-0.5,
        initial_state=ref_state,
        out=ref_out,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    out = torch.empty(batch, 1, hv, d).bfloat16()
    lak.flydsl_gdr_decode(
        qkv[:, : hk * d].view(batch, 1, hk, d),
        qkv[:, hk * d : 2 * hk * d].view(batch, 1, hk, d),
        qkv[:, 2 * hk * d :].view(batch, 1, hv, d),
        a.view(batch, 1, hv),
        b.view(batch, 1, hv),
        dt_bias,
        A_log,
        idx,
        state,
        out,
        use_qk_l2norm=True,
        need_shuffle_state=False,
        min_valid_index=1,
    )
    torch.testing.assert_close(
        out[:num_real].float(), ref_out[:num_real].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(state, ref_state, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not hasattr(lak, "flydsl_gdr_mtp"), reason="AITER lacks flydsl_gdr_mtp"
)
@pytest.mark.parametrize("num_heads", [(16, 32), (16, 48)])
@pytest.mark.parametrize("batch,num_real", [(1, 1), (4, 3), (8, 8), (16, 13)])
def test_flydsl_mtp_matches_fused_sigmoid_gating(num_heads, batch, num_real):
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.set_default_device("cuda")
    set_random_seed(0)
    hk, hv = num_heads
    d, t = 128, 4
    qkv = torch.randn(batch * t, 2 * hk * d + hv * d).bfloat16()
    ab = torch.randn(batch * t, 2 * hv).bfloat16()
    b, a = ab[:, :hv], ab[:, hv:]
    A_log = torch.randn(hv).bfloat16()
    dt_bias = torch.randn(hv).bfloat16()
    state = torch.randn(batch * t + 8, hv, d, d) * 0.1
    idx = (torch.randperm(batch * t + 7)[: batch * t] + 1).to(torch.int32)
    idx = idx.view(batch, t).contiguous()
    nacc = torch.randint(1, t + 1, (batch,), dtype=torch.int32)
    idx[num_real:] = 0  # CUDA-graph padding rows: null block, num_accepted 1
    nacc[num_real:] = 1
    cu = torch.arange(0, batch * t + 1, t, dtype=torch.int32)
    q = qkv[:, : hk * d]
    k = qkv[:, hk * d : 2 * hk * d]
    v = qkv[:, 2 * hk * d :]

    ref_state = state.clone()
    ref_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.contiguous().view(1, batch * t, hk, d),
        k=k.contiguous().view(1, batch * t, hk, d),
        v=v.contiguous().view(1, batch * t, hv, d),
        initial_state=ref_state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        use_qk_l2norm_in_kernel=True,
    )
    out = torch.empty(batch, t, hv, d).bfloat16()
    lak.flydsl_gdr_mtp(
        q.view(batch, t, hk, d),
        k.view(batch, t, hk, d),
        v.view(batch, t, hv, d),
        a.view(batch, t, hv),
        b.view(batch, t, hv),
        dt_bias,
        A_log,
        idx,
        nacc,
        state,
        out,
        use_qk_l2norm=True,
        min_valid_index=1,
    )
    real = num_real * t
    torch.testing.assert_close(
        out.view(batch * t, hv, d)[:real].float(),
        ref_out.view(batch * t, hv, d)[:real].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(state, ref_state, atol=1e-3, rtol=1e-3)
