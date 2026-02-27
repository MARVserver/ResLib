import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_orthogonal_matrix

try:
    import reslib_cpp
except ImportError:
    reslib_cpp = None


class ResMoELoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, reservoir_size: int, num_experts: int, top_k: int = 0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.reservoir_size = reservoir_size
        self.num_experts = num_experts
        self.top_k = top_k

        # --- dtype/device を base_layer に合わせる ---
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Shared Reservoir (A matrix) - Frozen
        A_matrix = get_orthogonal_matrix(reservoir_size, self.in_features).to(device=device, dtype=dtype)
        self.register_buffer("A", A_matrix)

        # Readout Experts (B matrices) - Trainable
        self.B = nn.Parameter(
            torch.zeros(num_experts, self.out_features, reservoir_size, device=device, dtype=dtype)
        )

        # Dynamic Router - Trainable
        self.router = nn.Linear(self.in_features, num_experts, bias=False, device=device, dtype=dtype)

        nn.init.zeros_(self.B)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        # Optimize: Only cast if necessary
        A = self.A
        if A.dtype != x.dtype:
            A = A.to(dtype=x.dtype)

        B = self.B
        if B.dtype != x.dtype:
            B = B.to(dtype=x.dtype)

        router_weight = self.router.weight
        if router_weight.dtype != x.dtype:
            router_weight = router_weight.to(dtype=x.dtype)

        if reslib_cpp is not None:
            delta = reslib_cpp.forward(x, A, B, router_weight, self.top_k)
        else:
            res_hidden = F.linear(x, A) # (..., reservoir_size)
            router_logits = F.linear(x, router_weight) # (..., num_experts)

            if 0 < self.top_k < self.num_experts:
                routing_weights = F.softmax(router_logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

                # Re-normalize top-k weights
                top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

                # Optimized selective expert computation to avoid huge memory allocation
                batch_dims = res_hidden.shape[:-1]
                res_hidden_flat = res_hidden.reshape(-1, self.reservoir_size)
                top_k_weights_flat = top_k_weights.reshape(-1, self.top_k)
                top_k_indices_flat = top_k_indices.reshape(-1, self.top_k)

                delta_flat = torch.zeros(res_hidden_flat.size(0), self.out_features, device=x.device, dtype=x.dtype)

                # Iterate over top-k to keep memory usage low
                for k in range(self.top_k):
                    w = top_k_weights_flat[:, k:k+1] # (TT, 1)
                    idx = top_k_indices_flat[:, k] # (TT)

                    unique_experts = idx.unique()
                    for expert_idx in unique_experts:
                        mask = (idx == expert_idx)
                        if not mask.any(): continue

                        expert_out = F.linear(res_hidden_flat[mask], B[expert_idx])
                        delta_flat[mask] += w[mask] * expert_out

                delta = delta_flat.reshape(*batch_dims, self.out_features)
            else:
                routing_weights = F.softmax(router_logits, dim=-1)
                # Optimized MoE blending using matmuls with proper broadcasting
                orig_shape = res_hidden.shape
                res_hidden_flat = res_hidden.reshape(-1, 1, 1, self.reservoir_size)
                # B.transpose(1, 2) is (N, R, D)
                # (Tokens, 1, 1, R) @ (1, N, R, D) -> (Tokens, N, 1, D)
                experts_out = torch.matmul(res_hidden_flat, B.transpose(1, 2).unsqueeze(0))

                # routing_weights: (..., N) -> (Tokens, 1, N)
                routing_weights_flat = routing_weights.reshape(-1, 1, self.num_experts)
                # (Tokens, 1, N) @ (Tokens, N, D) -> (Tokens, 1, D)
                delta_flat = torch.matmul(routing_weights_flat, experts_out.squeeze(-2)).squeeze(1)
                delta = delta_flat.reshape(*orig_shape[:-1], self.out_features)

        return base_output + delta
