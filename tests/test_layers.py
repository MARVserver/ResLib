import torch
import torch.nn as nn
from reslib.layers import ResMoELoRALinear

def test_layer_initialization():
    in_features = 64
    out_features = 32
    reservoir_size = 128
    num_experts = 4

    base_layer = nn.Linear(in_features, out_features)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts)

    assert res_layer.in_features == in_features
    assert res_layer.out_features == out_features
    assert res_layer.reservoir_size == reservoir_size
    assert res_layer.num_experts == num_experts
    assert res_layer.A.shape == (reservoir_size, in_features)
    assert res_layer.B.shape == (num_experts, out_features, reservoir_size)
    assert torch.all(res_layer.B == 0)

def test_layer_forward():
    in_features = 64
    out_features = 32
    reservoir_size = 128
    num_experts = 4
    batch_size = 4

    base_layer = nn.Linear(in_features, out_features)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts)

    x = torch.randn(batch_size, in_features)
    output = res_layer(x)

    assert output.shape == (batch_size, out_features)
    # Since B is initialized to 0, output should be identical to base_layer
    assert torch.allclose(output, base_layer(x), atol=1e-6)

def test_layer_top_k():
    in_features = 64
    out_features = 32
    reservoir_size = 128
    num_experts = 4
    batch_size = 4
    top_k = 2

    base_layer = nn.Linear(in_features, out_features)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts, top_k=top_k)

    # Set B to non-zero
    res_layer.B.data.fill_(0.1)

    x = torch.randn(batch_size, in_features)
    output = res_layer(x)
    assert output.shape == (batch_size, out_features)

def test_gradients():
    in_features = 64
    out_features = 32
    reservoir_size = 128
    num_experts = 4

    base_layer = nn.Linear(in_features, out_features)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts)

    # Set B to non-zero to ensure gradients flow
    res_layer.B.data.normal_()

    x = torch.randn(2, in_features)
    output = res_layer(x)
    loss = output.sum()
    loss.backward()

    assert res_layer.B.grad is not None
    assert res_layer.router.weight.grad is not None
