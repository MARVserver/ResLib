#include <torch/extension.h>
#include <vector>

torch::Tensor res_moelora_forward(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor router_weight,
    int64_t top_k) {

  // 1. Project to Reservoir space
  // x: (..., in_features)
  // A: (reservoir_size, in_features)
  auto res_hidden = torch::linear(x, A); // (..., reservoir_size)

  // 2. Dynamic Routing
  // router_weight: (num_experts, in_features)
  auto router_logits = torch::linear(x, router_weight); // (..., num_experts)

  torch::Tensor routing_weights;
  int64_t num_experts = B.size(0);

  if (top_k > 0 && top_k < num_experts) {
    auto softmax_logits = torch::softmax(router_logits, -1);
    auto topk_out = torch::topk(softmax_logits, top_k, -1);
    auto top_k_weights = std::get<0>(topk_out);
    auto top_k_indices = std::get<1>(topk_out);

    routing_weights = torch::zeros_like(softmax_logits);
    routing_weights.scatter_(-1, top_k_indices, 1.0);
    routing_weights = softmax_logits * routing_weights;
    routing_weights = routing_weights / (routing_weights.sum(-1, true) + 1e-6);
  } else {
    routing_weights = torch::softmax(router_logits, -1);
  }

  // 3. Expert computation and blending
  // res_hidden: (..., R)
  // routing_weights: (..., N)
  // B: (N, D, R)
  // Output: (..., D)

  // torch::einsum in LibTorch
  return torch::einsum("...r,...n,ndr->...d", {res_hidden, routing_weights, B});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &res_moelora_forward, "ResMoELoRA forward pass (C++)");
}
