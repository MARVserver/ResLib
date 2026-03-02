from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class ResMoELoRAConfig:
    """
    Configuration for Res-MoELoRA.

    Args:
        reservoir_size (int): Dimension of the shared reservoir (A matrix).
        num_experts (int): Number of readout experts (B matrices).
        target_modules (Union[List[str], str], optional): List of module names to apply Res-MoELoRA to.
        top_k (int): Number of experts to select for each input (default is 0, which means all experts are used).
        lora_alpha (int): The alpha parameter for LoRA scaling.
        lora_dropout (float): The dropout probability for LoRA layers.
        activation (str): The activation function to use after the reservoir (e.g., "identity", "tanh", "relu").
    """
    reservoir_size: int = 256
    num_experts: int = 4
    target_modules: Optional[Union[List[str], str]] = None
    top_k: int = 0  # 0 means use all experts (weighted sum)
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    activation: str = "identity" # Default is identity as in standard LoRA
