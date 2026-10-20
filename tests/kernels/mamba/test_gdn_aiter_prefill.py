# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER chunk_gated_delta_rule_opt_vk must match FLA for the inputs that
ChunkGatedDeltaRule.forward_aiter passes (gdn_prefill_backend="aiter").
seq_lens_cpu needs an AITER build with host-side chunk metadata."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER GDN prefill is ROCm-only", allow_module_level=True)

aiter_gdr = pytest.importorskip("aiter.ops.triton.gated_delta_net.gated_delta_rule")

from vllm.third_party.flash_linear_attention.ops.chunk import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402


@pytest.mark.parametrize("seq_lens", [[300], [4096], [1000, 77, 2050, 64]])
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(16, 32), (16, 48)])
@pytest.mark.parametrize("use_chunk_flydsl", [False, True])
@pytest.mark.parametrize("pass_seq_lens_cpu", [False, True])
def test_aiter_opt_vk_matches_fla(
    seq_lens, num_k_heads, num_v_heads, use_chunk_flydsl, pass_seq_lens_cpu
):
    torch.set_default_device("cuda")
    set_random_seed(0)
    K = V = 128
    L = sum(seq_lens)
    q = F.normalize(torch.randn(1, L, num_k_heads, K), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, L, num_k_heads, K), dim=-1).bfloat16()
    v = torch.randn(1, L, num_v_heads, V).bfloat16()
    g = -F.softplus(torch.randn(1, L, num_v_heads)) * 0.5
    beta = torch.randn(1, L, num_v_heads).sigmoid()
    h0 = torch.randn(len(seq_lens), num_v_heads, V, K) * 0.1
    # vLLM passes int32 prefill_query_start_loc.
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    cu = cu.to(torch.int32)

    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
    )
    o_ref, h_ref = chunk_gated_delta_rule(initial_state=h0.clone(), **common)
    extra = {"seq_lens_cpu": seq_lens} if pass_seq_lens_cpu else {}
    o, h = aiter_gdr.chunk_gated_delta_rule_opt_vk(
        initial_state=h0.clone(),
        use_chunk_flydsl=use_chunk_flydsl,
        state_dtype=h0.dtype,
        **extra,
        **common,
    )
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(h.float(), h_ref.float(), atol=1e-2, rtol=1e-2)
