import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_orthogonal_matrix

try:
    import reslib_cpp
except ImportError:
    reslib_cpp = None


class ResMoELoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        reservoir_size: int,
        num_experts: int,
        top_k: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        activation: str = "identity"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.reservoir_size = reservoir_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / reservoir_size

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.activation_name = activation.lower()
        if self.activation_name == "tanh":
            self.activation = torch.tanh
        elif self.activation_name == "relu":
            self.activation = torch.relu
        else:
            self.activation = nn.Identity()

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

        # C++ extension now supports activation but not dropout
        # We use C++ if dropout is Identity (eval mode or p=0)
        use_cpp = (reslib_cpp is not None and
                   isinstance(self.lora_dropout, nn.Identity))

        if use_cpp:
            if hasattr(reslib_cpp, "forward_v2"):
                delta = reslib_cpp.forward_v2(x, A, B, router_weight, self.top_k, self.activation_name)
            else:
                # Fallback for old extension if activation is identity
                if self.activation_name == "identity":
                    delta = reslib_cpp.forward(x, A, B, router_weight, self.top_k)
                else:
                    # Fallback to Python if activation is used but forward_v2 is missing
                    return self._forward_python(x, base_output)

            delta = delta * self.scaling
        else:
            return self._forward_python(x, base_output)

        return base_output + delta

    def _forward_python(self, x, base_output):
        A = self.A.to(dtype=x.dtype)
        B = self.B.to(dtype=x.dtype)
        router_weight = self.router.weight.to(dtype=x.dtype)

        res_hidden = F.linear(x, A) # (..., reservoir_size)
        res_hidden = self.activation(res_hidden)
        res_hidden = self.lora_dropout(res_hidden)

        router_logits = F.linear(x, router_weight) # (..., num_experts)

        if 0 < self.top_k < self.num_experts:
            routing_weights = F.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

            # Re-normalize top-k weights
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

            batch_dims = res_hidden.shape[:-1]
            res_hidden_flat = res_hidden.reshape(-1, self.reservoir_size)
            top_k_weights_flat = top_k_weights.reshape(-1, self.top_k)
            top_k_indices_flat = top_k_indices.reshape(-1, self.top_k)

            delta_flat = torch.zeros(res_hidden_flat.size(0), self.out_features, device=x.device, dtype=x.dtype)

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
            orig_shape = res_hidden.shape
            res_hidden_flat = res_hidden.reshape(-1, 1, 1, self.reservoir_size)
            experts_out = torch.matmul(res_hidden_flat, B.transpose(1, 2).unsqueeze(0))

            routing_weights_flat = routing_weights.reshape(-1, 1, self.num_experts)
            delta_flat = torch.matmul(routing_weights_flat, experts_out.squeeze(-2)).squeeze(1)
            delta = delta_flat.reshape(*orig_shape[:-1], self.out_features)

        delta = delta * self.scaling
        return base_output + delta
