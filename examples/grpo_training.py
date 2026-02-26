import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora

def dummy_reward_func(prompts, completions, **kwargs):
    """
    A simple reward function: gives higher reward for longer completions.
    """
    rewards = [float(len(c)) / 100.0 for c in completions]
    return rewards

def main():
    model_id = "facebook/opt-125m"

    # 1. Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Inject Res-MoELoRA
    config = ResMoELoRAConfig(
        reservoir_size=128,
        num_experts=4,
        target_modules=["q_proj", "v_proj"]
    )
    model = inject_res_moelora(model, config)

    # 3. Load Dataset (Prompts only for GRPO)
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:100]")

    # GRPO expects a 'prompt' column
    dataset = dataset.map(lambda x: {"prompt": x["prompt"]})

    # 4. Define GRPO Configuration
    training_args = GRPOConfig(
        output_dir="./res_grpo_output",
        per_device_train_batch_size=2,
        num_generations=4, # Number of completions per prompt
        max_prompt_length=128,
        max_completion_length=128,
        learning_rate=1e-5,
        logging_steps=1,
        num_train_epochs=1,
        report_to="none"
    )

    # 5. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=dummy_reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    # 6. Train
    print("Starting GRPO training...")
    trainer.train()

    # 7. Save Adapter
    from reslib import save_res_adapter
    save_res_adapter(model, "grpo_res_adapter.res")
    print("Adapter saved to grpo_res_adapter.res")

if __name__ == "__main__":
    main()
