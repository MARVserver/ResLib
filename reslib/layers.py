import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_orthogonal_matrix

class ResMoELoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, reservoir_size: int, num_experts: int, top_k: int = 0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.reservoir_size = reservoir_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Shared Reservoir (A matrix) - Frozen
        # Shape: (reservoir_size, in_features)
        A_matrix = get_orthogonal_matrix(reservoir_size, self.in_features)
        self.register_buffer("A", A_matrix)

        # Readout Experts (B matrices) - Trainable
        # Shape: (num_experts, out_features, reservoir_size)
        self.B = nn.Parameter(torch.zeros(num_experts, self.out_features, reservoir_size))

        # Dynamic Router - Trainable
        self.router = nn.Linear(self.in_features, num_experts, bias=False)

        # Initialize B with small values or zeros.
        # Standard LoRA initializes B with zeros to ensure the adapter starts as an identity mapping.
        nn.init.zeros_(self.B)
        # Initialize router weights
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, in_features) or (batch_size, in_features)

        # 1. Base layer computation
        base_output = self.base_layer(x)

        # 2. Project to Reservoir space
        # res_hidden: (..., reservoir_size)
        res_hidden = F.linear(x, self.A)

        # 3. Dynamic Routing
        # router_logits: (..., num_experts)
        router_logits = self.router(x)

        if self.top_k > 0 and self.top_k < self.num_experts:
            routing_weights = F.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

            # Mask out non-top-k experts
            mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_indices, 1.0)
            routing_weights = routing_weights * mask
            # Re-normalize
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            routing_weights = F.softmax(router_logits, dim=-1)

        # 4. Expert computation and blending
        # res_hidden: (batch, seq, reservoir_size) -> (batch, seq, 1, reservoir_size)
        # self.B: (num_experts, out_features, reservoir_size)

        # We compute B_i @ res_hidden for each expert
        # Efficient way:
        # res_hidden is (..., R)
        # B is (N, D_out, R)
        # We want sum_i routing_weights_i * (res_hidden @ B_i.T)

        # res_hidden @ B_i.T -> (..., D_out)
        # We can use einsum or matmul

        # res_hidden: (..., R)
        # B: (N, D_out, R)
        # routing_weights: (..., N)

        # Let's use einsum for clarity and potentially efficiency
        # b: batch/seq dimensions, r: reservoir_size, n: num_experts, d: out_features
        # x_res: (...r), weights: (...n), B: (ndr)
        # result: (...d)

        # delta = torch.einsum("...r,...n,ndr->...d", res_hidden, routing_weights, self.B)

        # Since einsum can be tricky with variadic dimensions, let's be more explicit if needed.
        # But "...r,...n,ndr->...d" should work for any leading dimensions.

        delta = torch.einsum("...r,...n,ndr->...d", res_hidden, routing_weights, self.B)

        return base_output + delta
