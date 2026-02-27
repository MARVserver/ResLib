import torch
import torch.nn as nn
import os
import json
from .layers import ResMoELoRALinear

def save_as_peft(model: nn.Module, path: str, expert_idx: int = 0):
    """
    Saves a specific expert's weights in a format compatible with HF PEFT (LoRA).
    This allows loading the Res-MoELoRA expert as a standard LoRA adapter.
    """
    os.makedirs(path, exist_ok=True)

    peft_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, ResMoELoRALinear):
            # PEFT LoRA expects 'lora_A' and 'lora_B'
            # In Res-MoELoRA: A is (R, In), B is (N, Out, R)
            # PEFT LoRA A is (R, In), B is (Out, R)
            peft_state_dict[f"base_model.model.{name}.lora_A.weight"] = module.A.data
            peft_state_dict[f"base_model.model.{name}.lora_B.weight"] = module.B.data[expert_idx]

    torch.save(peft_state_dict, os.path.join(path, "adapter_model.bin"))

    # Minimal adapter_config.json
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": model.config.reservoir_size if hasattr(model, "config") else 128,
        "target_modules": [], # User should fill or we can infer
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "base_model_name_or_path": ""
    }
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Expert {expert_idx} saved as PEFT-compatible adapter in {path}")
