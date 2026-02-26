# ResLib

ResLib is a high-performance library for ultra-fast adaptation of LLMs using **Res-MoELoRA**.

## Features
- **C++ Backend**: Optimized forward pass with LibTorch.
- **Shared Reservoir**: Frozen orthogonal A matrix reduces trainable parameters.
- **Mixture of Experts**: Multiple trainable B matrices for domain specialization.
- **Dynamic Routing**: Context-aware expert selection (with Top-k support).
- **PEFT Compatibility**: Export experts as standard LoRA adapters.

## Installation
```bash
pip install torch transformers
pip install -e . --no-build-isolation
```

## Quick Start

### 1. Inject Res-MoELoRA into a Model
```python
from transformers import AutoModelForCausalLM
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = ResMoELoRAConfig(reservoir_size=128, num_experts=4)
model = inject_res_moelora(model, config)
```

### 2. Save and Load Adapters
```python
from reslib.model import save_res_adapter, load_res_adapter

# Save ResLib adapter (.res)
save_res_adapter(model, "my_adapter.res")

# Load ResLib adapter
load_res_adapter(model, "my_adapter.res")
```

### 3. Export to PEFT (Hugging Face)
```python
from reslib.peft_compat import save_as_peft

# Save Expert 0 as a standard LoRA adapter
save_as_peft(model, "peft_lora_adapter", expert_idx=0)
```

## Technical Advantages
- **Convergence**: Faster than standard LoRA due to frozen Reservoir.
- **Memory**: Lower VRAM usage during training.
- **Switching**: Millisecond-level adapter swapping for multi-tenant apps.

## Advanced Training Examples
We provide examples for common fine-tuning tasks:
- **SFT (Supervised Fine-Tuning)**: See [examples/sft_training.py](examples/sft_training.py)
- **GRPO (Group Relative Policy Optimization)**: See [examples/grpo_training.py](examples/grpo_training.py)


