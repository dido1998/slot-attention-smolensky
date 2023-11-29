import math
from typing import Any, Dict, Optional

import numpy
import torch
from sklearn import cluster
from torch import nn

class SlotAttentionSmolensky(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        num_fillers: int,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        learnable_roles: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.num_fillers = num_fillers

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_fillers = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp
        self.learnable_roles = learnable_roles

        # This is just a dummy implementation of how learnable roles could be implemented based on my understanding
        # (which could be wrong). I am just adding it here so that you can point out any errors in my implementation
        # or understanding.

        if learnable_roles:
            self.num_roles = num_roles
            self.role_dim = num_fillers

            self.role_embedding = nn.Parameter(torch.randn(1, num_fillers, num_fillers) * num_fillers**-0.5)


    def step(self, fillers, k, v, masks=None):
        bs, n_fillers, _ = fillers.shape
        fillers_prev = fillers

        fillers = self.norm_fillers(fillers)
        q = self.to_q(fillers).view(bs, n_fillers, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_fillers, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_fillers, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        fillers = self.gru(updates.reshape(-1, self.kvq_dim), fillers_prev.reshape(-1, self.dim))

        fillers = fillers.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            fillers = self.ff_mlp(fillers)

        return fillers, attn_before_reweighting.mean(dim=2)

    def iterate(self, fillers, k, v, masks=None):
        for _ in range(self.iters):
            fillers, attn = self.step(fillers, k, v, masks)
        return fillers, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        fillers = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)


        fillers, attn = self.iterate(fillers, k, v, masks)

        if self.learnable_roles:
            # expanding the learnable roles along the batch dimension
            roles = self.role_embedding.repeat(b, 1, 1)
            
            # each role should be a distribution over features
            roles = torch.softmax(roles, dim=-1)

            # get final slots by taking outer product between roles and fillers
            slots = torch.einsum("bij,bjd->bid", roles, fillers)
        else:
            # roles will just be an identity matrix of size [B, NUM_FILLERS, NUM_FILLERS]
            roles = torch.eye(self.num_fillers).to(inputs.device).unsqueeze(0).repeat(b, 1, 1)

            slots = torch.einsum("bij,bjd->bid", roles, fillers)

        return slots, attn

sa = SlotAttentionSmolensky(5, 128, 128, learnable_roles = False).cuda() # 5 fillers, 128 filler_dim and feature_dim
initial_fillers = torch.randn(32, 5, 128).cuda()
encoder_features = torch.randn(32, 1024, 128).cuda()

slots, attn = sa(encoder_features, initial_fillers)
print(slots.shape)

sa = SlotAttentionSmolensky(5, 128, 128, learnable_roles = True, num_roles = 10).cuda() # 5 fillers, 10 roles, learnable roles, 128 filler_dim and feature_dim
initial_fillers = torch.randn(32, 5, 128).cuda()
encoder_features = torch.randn(32, 1024, 128).cuda()

slots, attn = sa(encoder_features, initial_fillers)
print(slots.shape)
