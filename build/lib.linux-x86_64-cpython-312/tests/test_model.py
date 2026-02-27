import torch
import torch.nn as nn
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora, save_res_adapter, load_res_adapter
import os

def test_model_injection():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 5)
    )
    config = ResMoELoRAConfig(reservoir_size=16, num_experts=2, target_modules=["0"])

    inject_res_moelora(model, config)

    from reslib.layers import ResMoELoRALinear
    assert isinstance(model[0], ResMoELoRALinear)
    assert isinstance(model[1], nn.Linear)

    # Check freezing
    # Base model parameters should be frozen
    assert not model[1].weight.requires_grad
    assert not model[1].bias.requires_grad
    assert not any(p.requires_grad for n, p in model[0].base_layer.named_parameters())

    # Res-MoELoRA parameters should be trainable
    assert model[0].B.requires_grad
    assert model[0].router.weight.requires_grad

def test_save_load_adapter():
    model = nn.Sequential(nn.Linear(10, 10))
    config = ResMoELoRAConfig(reservoir_size=16, num_experts=2)
    inject_res_moelora(model, config)

    # Randomize B
    model[0].B.data.normal_()
    original_B = model[0].B.clone()

    save_res_adapter(model, "test_save.res")
    assert os.path.exists("test_save.res")

    # New model
    model2 = nn.Sequential(nn.Linear(10, 10))
    inject_res_moelora(model2, config)
    # Initially B is 0
    assert torch.all(model2[0].B == 0)

    load_res_adapter(model2, "test_save.res")
    assert torch.allclose(model2[0].B, original_B)

    if os.path.exists("test_save.res"):
        os.remove("test_save.res")
