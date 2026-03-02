#include <torch/extension.h>
#include <vector>
#include <string>

torch::Tensor res_moelora_forward(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor router_weight,
    int64_t top_k) {

  // 1. Project to Reservoir space
  auto res_hidden = torch::linear(x, A);

  // 2. Dynamic Routing
  auto router_logits = torch::linear(x, router_weight);

  int64_t num_experts = B.size(0);
  int64_t out_features = B.size(1);
  int64_t reservoir_size = B.size(2);

  torch::Tensor routing_weights;
  if (top_k > 0 && top_k < num_experts) {
    auto softmax_logits = torch::softmax(router_logits, -1);
    auto topk_out = torch::topk(softmax_logits, top_k, -1);
    auto top_k_weights = std::get<0>(topk_out);
    auto top_k_indices = std::get<1>(topk_out);

    // Re-normalize top-k weights
    top_k_weights = top_k_weights / (top_k_weights.sum(-1, true) + 1e-6);

    routing_weights = torch::zeros_like(softmax_logits);
    routing_weights.scatter_(-1, top_k_indices, top_k_weights);
  } else {
    routing_weights = torch::softmax(router_logits, -1);
  }

  // 3. Optimized MoE blending
  auto res_hidden_shape = res_hidden.sizes().vec();
  int64_t total_tokens = 1;
  for (size_t i = 0; i < res_hidden_shape.size() - 1; ++i) {
      total_tokens *= res_hidden_shape[i];
  }

  auto res_hidden_flat = res_hidden.reshape({total_tokens, 1, 1, reservoir_size});
  auto B_t = B.transpose(1, 2).unsqueeze(0); // (1, N, R, D)

  // (Tokens, 1, 1, R) @ (1, N, R, D) -> (Tokens, N, 1, D)
  auto experts_out = torch::matmul(res_hidden_flat, B_t).squeeze(-2); // (Tokens, N, D)

  auto routing_weights_flat = routing_weights.reshape({total_tokens, 1, num_experts});
  // (Tokens, 1, N) @ (Tokens, N, D) -> (Tokens, 1, D)
  auto delta_flat = torch::matmul(routing_weights_flat, experts_out).squeeze(1);

  auto final_shape = res_hidden_shape;
  final_shape.back() = out_features;
  return delta_flat.reshape(final_shape);
}

// Extended forward with activation
torch::Tensor res_moelora_forward_v2(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor router_weight,
    int64_t top_k,
    std::string activation) {

  // 1. Project to Reservoir space
  auto res_hidden = torch::linear(x, A);

  // Apply Activation
  if (activation == "tanh") {
      res_hidden = torch::tanh(res_hidden);
  } else if (activation == "relu") {
      res_hidden = torch::relu(res_hidden);
  }

  // 2. Dynamic Routing
  auto router_logits = torch::linear(x, router_weight);

  int64_t num_experts = B.size(0);
  int64_t out_features = B.size(1);
  int64_t reservoir_size = B.size(2);

  torch::Tensor routing_weights;
  if (top_k > 0 && top_k < num_experts) {
    auto softmax_logits = torch::softmax(router_logits, -1);
    auto topk_out = torch::topk(softmax_logits, top_k, -1);
    auto top_k_weights = std::get<0>(topk_out);
    auto top_k_indices = std::get<1>(topk_out);

    top_k_weights = top_k_weights / (top_k_weights.sum(-1, true) + 1e-6);

    routing_weights = torch::zeros_like(softmax_logits);
    routing_weights.scatter_(-1, top_k_indices, top_k_weights);
  } else {
    routing_weights = torch::softmax(router_logits, -1);
  }

  // 3. Optimized MoE blending
  auto res_hidden_shape = res_hidden.sizes().vec();
  int64_t total_tokens = 1;
  for (size_t i = 0; i < res_hidden_shape.size() - 1; ++i) {
      total_tokens *= res_hidden_shape[i];
  }

  auto res_hidden_flat = res_hidden.reshape({total_tokens, 1, 1, reservoir_size});
  auto B_t = B.transpose(1, 2).unsqueeze(0);

  auto experts_out = torch::matmul(res_hidden_flat, B_t).squeeze(-2);

  auto routing_weights_flat = routing_weights.reshape({total_tokens, 1, num_experts});
  auto delta_flat = torch::matmul(routing_weights_flat, experts_out).squeeze(1);

  auto final_shape = res_hidden_shape;
  final_shape.back() = out_features;
  return delta_flat.reshape(final_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &res_moelora_forward, "ResMoELoRA forward pass (C++)");
  m.def("forward_v2", &res_moelora_forward_v2, "ResMoELoRA forward pass v2 with activation (C++)");
}
