import torch
import torch.nn as nn
from reslib.layers import ResMoELoRALinear

def test_scaling():
    in_features = 16
    out_features = 16
    reservoir_size = 32
    num_experts = 1
    lora_alpha = 64 # scaling = 64 / 32 = 2.0

    base_layer = nn.Linear(in_features, out_features)
    base_layer.weight.data.zero_()
    base_layer.bias.data.zero_()

    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts, lora_alpha=lora_alpha, lora_dropout=0.0)
    res_layer.B.data.fill_(1.0)
    res_layer.router.weight.data.fill_(0.0) # all experts equal weight

    x = torch.ones(1, in_features)
    # Project to reservoir: x @ A.T. A is orthogonal.
    # We can't easily predict A, but we can compare with and without scaling.

    output = res_layer(x)

    res_layer_no_scale = ResMoELoRALinear(base_layer, reservoir_size, num_experts, lora_alpha=reservoir_size, lora_dropout=0.0)
    res_layer_no_scale.B.data.copy_(res_layer.B.data)
    res_layer_no_scale.A.copy_(res_layer.A)
    res_layer_no_scale.router.weight.data.copy_(res_layer.router.weight.data)

    output_no_scale = res_layer_no_scale(x)

    assert torch.allclose(output, output_no_scale * 2.0, atol=1e-5)

def test_activation():
    in_features = 16
    out_features = 16
    reservoir_size = 32
    num_experts = 1

    base_layer = nn.Linear(in_features, out_features)
    base_layer.weight.data.zero_()
    base_layer.bias.data.zero_()

    x = torch.randn(1, in_features)

    # Tanh activation
    res_layer_tanh = ResMoELoRALinear(base_layer, reservoir_size, num_experts, activation="tanh", lora_dropout=0.0)
    res_layer_tanh.B.data.fill_(1.0)

    output_tanh = res_layer_tanh(x)

    # Manual check
    res_hidden = torch.tanh(torch.nn.functional.linear(x, res_layer_tanh.A))
    expected = torch.nn.functional.linear(res_hidden, res_layer_tanh.B[0]) * res_layer_tanh.scaling

    assert torch.allclose(output_tanh, expected, atol=1e-5)

def test_dropout():
    in_features = 16
    out_features = 16
    reservoir_size = 128
    num_experts = 1

    base_layer = nn.Linear(in_features, out_features)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts, lora_dropout=0.5)
    res_layer.B.data.fill_(1.0)

    x = torch.ones(1, in_features)

    res_layer.train()
    output_train = res_layer(x)

    res_layer.eval()
    output_eval = res_layer(x)

    # In training, some elements should be zeroed (scaled by 1/(1-p))
    # In eval, none should be zeroed.
    assert not torch.allclose(output_train, output_eval)

if __name__ == "__main__":
    test_scaling()
    test_activation()
    test_dropout()
    print("All new feature tests passed!")
