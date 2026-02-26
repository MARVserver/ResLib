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

        # Shared Reservoir (A matrix) - Frozen
        A_matrix = get_orthogonal_matrix(reservoir_size, self.in_features)
        self.register_buffer("A", A_matrix)

        # Readout Experts (B matrices) - Trainable
        self.B = nn.Parameter(torch.zeros(num_experts, self.out_features, reservoir_size))

        # Dynamic Router - Trainable
        self.router = nn.Linear(self.in_features, num_experts, bias=False)

        nn.init.zeros_(self.B)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer computation
        base_output = self.base_layer(x)

        # Try to use C++ extension for performance
        if reslib_cpp is not None:
            delta = reslib_cpp.forward(x, self.A, self.B, self.router.weight, self.top_k)
        else:
            # Fallback to Python/Einsum
            res_hidden = F.linear(x, self.A)
            router_logits = self.router(x)

            if self.top_k > 0 and self.top_k < self.num_experts:
                routing_weights = F.softmax(router_logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
                mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_indices, 1.0)
                routing_weights = routing_weights * mask
                routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                routing_weights = F.softmax(router_logits, dim=-1)

            delta = torch.einsum("...r,...n,ndr->...d", res_hidden, routing_weights, self.B)

        return base_output + delta
