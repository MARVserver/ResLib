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

        # --- dtype を x に強制統一 ---
        A = self.A.to(dtype=x.dtype)
        B = self.B.to(dtype=x.dtype)
        router_weight = self.router.weight.to(dtype=x.dtype)

        if reslib_cpp is not None:
            delta = reslib_cpp.forward(x, A, B, router_weight, self.top_k)
        else:
            res_hidden = F.linear(x, A)
            router_logits = F.linear(x, router_weight)

            if 0 < self.top_k < self.num_experts:
                routing_weights = F.softmax(router_logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
                mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_indices, 1.0)
                routing_weights = routing_weights * mask
                routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                routing_weights = F.softmax(router_logits, dim=-1)

            delta = torch.einsum("...r,...n,ndr->...d", res_hidden, routing_weights, B)

        return base_output + delta
