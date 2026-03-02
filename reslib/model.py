import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
import os
from .config import ResMoELoRAConfig
from .layers import ResMoELoRALinear

def inject_res_moelora(model: nn.Module, config: ResMoELoRAConfig):
    """
    Recursively replaces target linear layers with ResMoELoRALinear.

    Args:
        model (nn.Module): The model to modify.
        config (ResMoELoRAConfig): Configuration for Res-MoELoRA.

    Returns:
        nn.Module: The modified model.
    """
    target_modules = config.target_modules
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # Freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False

    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # If target_modules is None, replace all Linear layers
            if not target_modules or any(target in name for target in target_modules):
                modules_to_replace.append(name)

    for name in modules_to_replace:
        # Navigate to the parent module
        path = name.split(".")
        parent = model
        for part in path[:-1]:
            parent = getattr(parent, part)

        leaf_name = path[-1]
        old_layer = getattr(parent, leaf_name)

        # Create the Res-MoELoRA layer
        new_layer = ResMoELoRALinear(
            old_layer,
            config.reservoir_size,
            config.num_experts,
            config.top_k,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            activation=config.activation
        )

        # Replace the layer
        setattr(parent, leaf_name, new_layer)

    return model

def save_res_adapter(model: nn.Module, path: str):
    """
    Saves the ResMoELoRA adapter weights (B matrices and router weights) to a .res file.

    Args:
        model (nn.Module): The model containing ResMoELoRA layers.
        path (str): The file path to save to (should end in .res).
    """
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, ResMoELoRALinear):
            state_dict[f"{name}.B"] = module.B.data
            state_dict[f"{name}.router.weight"] = module.router.weight.data
            state_dict[f"{name}.A"] = module.A.data

    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state_dict, path)

def load_res_adapter(model: nn.Module, path: str):
    """
    Loads ResMoELoRA adapter weights from a .res file.

    Args:
        model (nn.Module): The model to load weights into.
        path (str): The path to the .res file.

    Returns:
        nn.Module: The model with loaded weights.
    """
    state_dict = torch.load(path, weights_only=True)
    model_dict = model.state_dict()

    # Only load keys that exist in the model
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    if len(filtered_state_dict) < len(state_dict):
        missing = set(state_dict.keys()) - set(filtered_state_dict.keys())
        print(f"Warning: Some weights in the adapter file were not found in the model: {missing}")

    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)

    return model
