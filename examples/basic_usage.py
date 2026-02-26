import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora, save_res_adapter

def main():
    # Using OPT-125m as it uses standard nn.Linear layers
    model_name = "facebook/opt-125m"
    print(f"Loading {model_name}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Falling back to a mock model for demonstration.")
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Linear(768, 768)
        )
        model.config = type('obj', (object,), {'hidden_size': 768})()
        tokenizer = None

    # Configure Res-MoELoRA
    # We target the attention projection layers
    config = ResMoELoRAConfig(
        reservoir_size=128,
        num_experts=4,
        target_modules=["q_proj", "v_proj"] if "opt" in model_name else None
    )

    print("Injecting Res-MoELoRA...")
    model = inject_res_moelora(model, config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage: {100 * trainable_params / total_params:.4f}%")

    # Prepare input
    if tokenizer:
        text = "ResLib is a high-performance library designed for ultra-fast adaptation"
        inputs = tokenizer(text, return_tensors="pt")

        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"Output logits shape: {outputs.logits.shape}")
    else:
        x = torch.randn(1, 10, 768)
        output = model(x)
        print(f"Output shape: {output.shape}")

    # Save adapter
    save_res_adapter(model, "opt_reslib_adapter.res")
    print("Adapter saved to opt_reslib_adapter.res")

if __name__ == "__main__":
    main()
