from ..verify import reward
import argparse
import gc
import os
import random
import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from .NudgeRLTrainer import NudgeRLTrainer
import wandb



reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



def has_saved_tokenizer(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "tokenizer_config.json"))


def main():
    parser = argparse.ArgumentParser(description="Build training dataset from hints")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--eps_high", required=False, type=float, default=0.2, help="Epsilon high for NudgeRL")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    model_name = cfg["model_name"]
    max_seq_length = cfg["max_prompt_length"] + cfg["max_completion_length"]
    lora_rank = cfg.get("lora_rank", 16)
    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    nudgerl_cfg = cfg["nudgerl"]
    eps_high = args.eps_high if args.eps_high is not None else nudgerl_cfg["epsilon_high"]
    max_steps = cfg.get("max_steps", 300)
    save_steps = cfg.get("save_steps", 50)
    set_seed(seed)
    print(f"Training with seed {seed}")

    model, tokenizer = load_model_and_tokenizer(model_name, max_seq_length)

    model_id = model_name.split("/")[-1]
    output_root = cfg.get("output_root", "outputs/models")
    baseline_save_dir = os.path.join(output_root, model_id, "baseline")
    algorithm_name = f"NudgeRL_{nudgerl_cfg['num_hints']}x{nudgerl_cfg['rollouts_per_hint']}_eps{eps_high*100:.0f}"
    algorithm_name += "_random" if nudgerl_cfg["sampler_type"] == "random" else ""
    model_save_dir = os.path.join(output_root, model_id, algorithm_name)
    os.makedirs(baseline_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    lora_config = LoraConfig(
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    if use_gradient_checkpointing:
        if hasattr(model, "base_model"):
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    print(f"Gradient checkpointing: {'enabled' if use_gradient_checkpointing else 'disabled'}")
    model.print_trainable_parameters()
    print("Starting NudgeRL training...")
    dataset = load_dataset(
        "json",
        data_files="data/dapo17k_contexts_5_samples_500.jsonl",
        split="train",
    )
    dataset = dataset.map(
        lambda x: {
            "problem": x["prompt"],
            "answer": x["solution"],
        }
    )
    dataset = dataset.select(range(250))
    print(f"Loaded train dataset with {len(dataset)} samples")
    wandb_api_key = (
        cfg.get("wandb_api_key")
        or cfg.get("WANDB_API_KEY")
        or os.environ.get("WANDB_API_KEY")
    )
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "Hint_RL_PoC")
    os.environ["WANDB_NAME"] = algorithm_name
    wandb.login()
    use_vllm = cfg["use_vllm"]
    vllm_mode = cfg["vllm_mode"]
    generation_kwargs = {}
    if use_vllm:
        generation_kwargs.update(
            {
                "stop": [solution_end, tokenizer.eos_token],
                "include_stop_str_in_output": True,
            }
        )

    training_args = GRPOConfig(
        seed=cfg["random_state"],
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=cfg["temperature"],
        min_p=cfg["min_p"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        generation_kwargs=generation_kwargs,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        loss_type="grpo",
        beta=cfg["beta"],
        epsilon_high=eps_high,
        optim=cfg["optimizer"],
        logging_steps=cfg["logging_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=nudgerl_cfg["num_hints"]*nudgerl_cfg["rollouts_per_hint"],
        generation_batch_size=cfg["generation_batch_size"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        max_steps=max_steps,
        save_steps=save_steps,
        remove_unused_columns=False,
        report_to=["wandb"],
        output_dir="ckpts",

    )
    trainer = NudgeRLTrainer(
        model=model,
        processing_class=tokenizer,
        reward_fn=reward,
        reward_funcs=reward,
        train_dataset=dataset,
        args=training_args,
        system_prompt=system_prompt,
        adv_eps=nudgerl_cfg["adv_eps"],
        adv_lbd=nudgerl_cfg["adv_lbd"],
        rollout_per_hint=nudgerl_cfg["rollouts_per_hint"],
        p_dropout=nudgerl_cfg["p_dropout"],
        num_hint=nudgerl_cfg["num_hints"],
        sampler_type=nudgerl_cfg["sampler_type"],
        distill_coeff=nudgerl_cfg["distill_coeff"],

    )
    trainer.train()
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)


if __name__ == "__main__":
    main()
