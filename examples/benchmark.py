import torch
import time
import os
import gc
import argparse
import resource
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from reslib.config import ResMoELoRAConfig
from reslib.model import inject_res_moelora

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # Get RSS memory in MB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def benchmark(model, input_ids, num_steps=20, label="Model"):
    print(f"Benchmarking {label}...")

    # Warmup
    for _ in range(3):
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()

    for i in range(num_steps):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()

    end_time = time.time()

    avg_time = (end_time - start_time) / num_steps
    peak_mem = get_memory_usage()

    throughput = input_ids.shape[0] * input_ids.shape[1] / avg_time

    return {
        "Label": label,
        "Avg Time (s)": f"{avg_time:.4f}",
        "Throughput (tokens/s)": f"{throughput:.2f}",
        "Peak Memory (MB)": f"{peak_mem:.2f}"
    }

def print_table(results):
    if not results:
        return
    keys = results[0].keys()
    header = " | ".join(keys)
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        print(" | ".join(str(res[k]) for k in keys))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len)).to(device)

    results = []

    # 1. Base Model
    print("Loading Base Model...")
    model_base = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    results.append(benchmark(model_base, input_ids, args.steps, "Base Model"))
    del model_base
    gc.collect()

    # 2. LoRA
    print("Loading LoRA...")
    model_lora = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model_lora = get_peft_model(model_lora, lora_config)
    results.append(benchmark(model_lora, input_ids, args.steps, "LoRA (r=16)"))
    del model_lora
    gc.collect()

    # 3. ResLib
    print("Loading ResLib...")
    model_res = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    res_config = ResMoELoRAConfig(
        reservoir_size=128,
        num_experts=4,
        target_modules=["q_proj", "v_proj"]
    )
    model_res = inject_res_moelora(model_res, res_config)
    results.append(benchmark(model_res, input_ids, args.steps, "ResLib (res=128, exp=4)"))
    del model_res
    gc.collect()

    print_table(results)

if __name__ == "__main__":
    main()
