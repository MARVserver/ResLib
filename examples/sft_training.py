import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora

def main():
    model_id = "facebook/opt-125m"
    dataset_name = "timdettmers/openassistant-guanaco"

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

    # 3. Load Dataset
    dataset = load_dataset(dataset_name, split="train[:1000]") # Small subset for demo

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./res_sft_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=100,
        fp16=True,
        report_to="none"
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Train
    print("Starting SFT training...")
    trainer.train()

    # 7. Save Adapter
    from reslib import save_res_adapter
    save_res_adapter(model, "sft_res_adapter.res")
    print("Adapter saved to sft_res_adapter.res")

if __name__ == "__main__":
    main()
