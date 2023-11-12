# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Callable, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .moa_layer import get_fused_cumsum_sub_one
from .top2gate import entropy, one_hot

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2


def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moa_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    moa_eval_capacity_length=None,
    use_tutel=False,
    prefix_tokens=None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()

    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moa_eval_capacity_token_fraction > 0.0 and eval_mode:
        if moa_eval_capacity_length is None:
            capacity = math.ceil(moa_eval_capacity_token_fraction * num_tokens)
        else:
            capacity = math.ceil(
                moa_eval_capacity_token_fraction * moa_eval_capacity_length
            )
    else:
        # capacity = capacity_factor * S/E
        capacity = int(capacity_factor * math.ceil(num_tokens / num_experts))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = (
        100
        * torch.histc(
            (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = (
        torch.sort(expert1_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    gates1_s = (gates * mask1).sum(dim=1)

    # Compute locations in capacity buffer
    locations1 = get_fused_cumsum_sub_one(use_tutel)(mask1)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    if use_tutel:
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return (
            l_aux,
            metadata,
            capacity,
            num_experts,
            [
                indices1_s,
            ],
            [
                locations1_s,
            ],
            [
                gates1_s,
            ],
        )

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    dispatch_mask = combine1_sec.bool()
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata


from fairseq.modules.linear import Linear


class MOATop1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moa_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        use_tutel=False,
        init_model_on_gpu=False,
        method="token",
        num_langs=0,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()
        self.wg = Linear(
            model_dim, num_experts, bias=False, init_model_on_gpu=init_model_on_gpu
        )
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moa_eval_capacity_token_fraction = moa_eval_capacity_token_fraction
        self.use_tutel = use_tutel
        self.method=method
        self.num_langs=num_langs

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        moa_eval_capacity_length: Optional[int] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
        lang_features: Optional[torch.Tensor] = None,
        input_shape: Optional[List] = None,
        lang_ids: Optional[List] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        if self.method == "lang":
            reshape_input_size = input.shape
            del input
            ## lang_features: [bz, dim]
            ## input_shape: [bz, seq_len, dim]
            lang_features = lang_features.unsqueeze(1).expand(input_shape)
            lang_features = lang_features.reshape(-1, input_shape[-1])
            input = torch.zeros(
                reshape_input_size,
                dtype=lang_features.dtype,
                layout=lang_features.layout,
                device=lang_features.device,
            )
            input[:lang_features.shape[0], :] = lang_features

        if self.method == "lang-wise":
            logits = one_hot(lang_ids.unsqueeze(-1), self.num_langs)
            logits = torch.where(logits==0, -torch.inf, logits)
            logits = logits.unsqueeze(1).expand(input_shape[0], input_shape[1], logits.shape[-1])
            logits = logits.reshape(-1, logits.shape[-1])
            new_logits = torch.zeros(
                input.shape[0], logits.shape[-1],
                dtype=input.dtype,
                layout=logits.layout,
                device=logits.device,
            )
            new_logits[:logits.shape[0], :] = logits
            logits = new_logits
        else:
            logits = self.wg(input)

        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            capacity_factor=self.capacity_factor,
            eval_mode=not self.training,
            moa_eval_capacity_token_fraction=self.moa_eval_capacity_token_fraction,
            moa_eval_capacity_length=moa_eval_capacity_length,
            use_tutel=self.use_tutel,
            prefix_tokens=prefix_tokens,
        )
