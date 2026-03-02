import torch
import torch.nn as nn
from reslib.layers import ResMoELoRALinear
import reslib_cpp

def test_cpp_forward_v2():
    print("Testing C++ forward_v2...")
    in_features = 16
    out_features = 16
    reservoir_size = 32
    num_experts = 2
    top_k = 1

    base_layer = nn.Linear(in_features, out_features)
    base_layer.weight.data.zero_()
    base_layer.bias.data.zero_()

    x = torch.randn(2, in_features)

    # Test Identity (v2)
    res_layer = ResMoELoRALinear(base_layer, reservoir_size, num_experts, top_k=top_k, lora_dropout=0.0, activation="identity")
    res_layer.eval() # Ensure dropout is Identity

    output_python = res_layer._forward_python(x, base_layer(x))
    output_cpp = res_layer(x)

    assert torch.allclose(output_python, output_cpp, atol=1e-5), "C++ Identity output mismatch"
    print("C++ Identity (v2) passed")

    # Test Tanh (v2)
    res_layer_tanh = ResMoELoRALinear(base_layer, reservoir_size, num_experts, top_k=top_k, lora_dropout=0.0, activation="tanh")
    res_layer_tanh.eval()

    output_python_tanh = res_layer_tanh._forward_python(x, base_layer(x))
    output_cpp_tanh = res_layer_tanh(x)

    assert torch.allclose(output_python_tanh, output_cpp_tanh, atol=1e-5), "C++ Tanh output mismatch"
    print("C++ Tanh (v2) passed")

    # Test ReLU (v2)
    res_layer_relu = ResMoELoRALinear(base_layer, reservoir_size, num_experts, top_k=top_k, lora_dropout=0.0, activation="relu")
    res_layer_relu.eval()

    output_python_relu = res_layer_relu._forward_python(x, base_layer(x))
    output_cpp_relu = res_layer_relu(x)

    assert torch.allclose(output_python_relu, output_cpp_relu, atol=1e-5), "C++ ReLU output mismatch"
    print("C++ ReLU (v2) passed")

if __name__ == "__main__":
    if reslib_cpp is not None:
        test_cpp_forward_v2()
        print("All C++ tests passed!")
    else:
        print("C++ extension not found, skipping tests.")
