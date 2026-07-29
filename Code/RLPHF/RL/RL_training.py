
import os


import argparse
import inspect
import json
import math
from pathlib import Path

import pandas as pd
import torch
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from dynamic_data_selection import (
    DEMOGRAPHIC_COLUMNS,
    DynamicSelectionCallback,
    DynamicSubsetDataset,
    SelectionAwareGRPOTrainer,
    SubgroupAwareDynamicSelector,
    add_selection_metadata,
    compute_capacity_capped_group_quotas,
    compute_dataset_china_difference_lower_bounds,
    compute_dataset_group_proportions,
    compute_group_quotas,
    load_aligned_china_benchmark_proportions,
)
from reward_function import (
    answer_reward_function,
    llm_reward_function as llm_reward_fn_template,
    load_reward_model_if_exists,
    rule_reward_function,
)


BASE_MODEL = "Path/to/SFT/Model"
DATA_ROOT  = "Code/RLPHF/RL/dataset"
REWARD_CKPT_DIR = "./Model/Reward-Model"
DEFAULT_OUTPUT_DIR          = "./Model/RLPHF-Model"

TRAIN_FILE = DATA_ROOT / "RL_traindataset_que+pro_v1.csv"
EVAL_FILE = DATA_ROOT / "RL_testdataset_que+pro_v1.csv"
COMPARISON_DATA_FILE = Path(__file__).resolve().parent / "comparison_data.csv"

NUM_GENERATIONS = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
REWARD_WEIGHTS = (0.25, 0.25, 0.50)  # process, rule, answer


def process_device_map():
    """Keep each torchrun worker on its own GPU; preserve auto placement otherwise."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return "auto"
    return {"": int(local_rank)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-interval", type=int, default=25)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Number of candidates per round. Default is 2*K; -1 uses all 7,220 rows.",
    )
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--lambda-conflict", type=float, default=1.0)
    parser.add_argument(
        "--china-correction-alpha",
        type=float,
        default=0.2,
        help=(
            "Weight for the mapped China-minus-data difference in the "
            "demographic lower-bound formula."
        ),
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one optimizer step with one small dynamic-selection round.",
    )
    parser.add_argument(
        "--validate-selection-only",
        action="store_true",
        help="Validate metadata, values, and quotas on CPU without loading either model.",
    )
    return parser.parse_args()


def validate_trl_interface() -> None:
    """Fail early if a future TRL upgrade removes the two small hooks we use."""
    required_methods = ("_calculate_rewards", "_generate_and_score_completions")
    for method_name in required_methods:
        if not hasattr(SelectionAwareGRPOTrainer, method_name):
            raise RuntimeError(f"Installed TRL lacks required method {method_name}.")
    parameters = inspect.signature(GRPOTrainer._calculate_rewards).parameters
    expected = {
        "self",
        "inputs",
        "prompts",
        "completions",
        "completion_ids_list",
    }
    if set(parameters) != expected:
        raise RuntimeError(
            "TRL _calculate_rewards signature changed; "
            f"installed TRL={trl.__version__}, parameters={list(parameters)}"
        )


def init_tokenizer_and_policy():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map=process_device_map(),
        output_hidden_states=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    return tokenizer, base_model, peft_config


class FirstStepGradientGuard(TrainerCallback):
    """Fail before the first optimizer update if the policy has no real gradient."""

    def __init__(self):
        self.checked = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.checked:
            return control
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError(
                "Policy has no trainable parameters before the first optimizer step."
            )
        for name, parameter in trainable:
            if parameter.grad is None:
                continue
            grad_norm = float(parameter.grad.detach().float().norm())
            if math.isfinite(grad_norm) and grad_norm > 0.0:
                print(
                    "[INFO] first-step gradient guard passed: "
                    f"parameter={name}, grad_norm={grad_norm:.8g}"
                )
                self.checked = True
                return control
        raise RuntimeError(
            "All trainable policy gradients are missing or zero before the first "
            "optimizer step."
        )


def _to_hf_dataset(dataframe: pd.DataFrame) -> Dataset:
    columns = [
        "prompt",
        "human_output",
        "human_num",
        "question_id",
        "sample_id",
        *DEMOGRAPHIC_COLUMNS,
    ]
    renamed = dataframe[columns].rename(
        columns={
            "human_output": "chosen",
            "human_num": "real_num",
            "question_id": "question_index",
        }
    )
    return Dataset.from_pandas(renamed, preserve_index=False)


def load_datasets_and_metadata():
    train_df = add_selection_metadata(pd.read_csv(TRAIN_FILE))
    eval_df = add_selection_metadata(pd.read_csv(EVAL_FILE))
    comparison_data = pd.read_csv(COMPARISON_DATA_FILE)

    if len(train_df) != 7220:
        raise ValueError(f"Expected 7,220 train rows, found {len(train_df)}.")
    if train_df["prompt"].nunique() == len(train_df):
        raise AssertionError(
            "Expected duplicated prompt text with distinct row-level supervision."
        )

    full_train_dataset = _to_hf_dataset(train_df)
    eval_dataset = _to_hf_dataset(eval_df)
    metadata = train_df[
        ["sample_id", *DEMOGRAPHIC_COLUMNS, "question_id"]
    ].rename(columns={"question_id": "question_index"})
    calibrated_targets = compute_dataset_china_difference_lower_bounds(
        train_df,
        comparison_data,
        alpha=ARGS.china_correction_alpha,
    )

    print(
        f"训练样本数: {len(full_train_dataset)}, "
        f"不同 prompt 文本: {train_df['prompt'].nunique()}"
    )
    print(f"验证样本数: {len(eval_dataset)}")
    return (
        full_train_dataset,
        eval_dataset,
        metadata,
        calibrated_targets,
        comparison_data,
    )


def write_demographic_calibration_report(
    output_dir: Path,
    metadata: pd.DataFrame,
    comparison_data: pd.DataFrame,
    calibrated_targets,
    selection_size: int,
) -> None:
    """Persist exact data/China/target distributions and per-round quotas."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    dataset_proportions = compute_dataset_group_proportions(metadata)
    china_proportions = load_aligned_china_benchmark_proportions(
        comparison_data,
        dataset_proportions,
    )
    desired_quotas = compute_group_quotas(selection_size, calibrated_targets)
    applied_quotas = compute_capacity_capped_group_quotas(
        selection_size,
        calibrated_targets,
        metadata,
    )
    payload = {
        "formula": (
            "lower_bound_mass = (1-alpha) * p_data "
            "+ alpha * (p_china - p_data)"
        ),
        "alpha": ARGS.china_correction_alpha,
        "training_data": str(TRAIN_FILE),
        "china_benchmark_data": str(COMPARISON_DATA_FILE),
        "selection_size": selection_size,
        "dataset_proportions": dataset_proportions,
        "china_proportions_aligned": china_proportions,
        "calibrated_lower_bound_mass": calibrated_targets,
        "desired_quotas_before_capacity_cap": desired_quotas,
        "applied_quotas": applied_quotas,
        "quota_capacity_adjustments": {
            feature: {
                group: {
                    "desired": desired_quotas[feature][group],
                    "applied": applied_quotas[feature][group],
                }
                for group in desired_quotas[feature]
                if desired_quotas[feature][group] != applied_quotas[feature][group]
            }
            for feature in desired_quotas
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "demographic_calibration.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"人口校准报告已保存: {report_path}")


def get_stage1_reward_funcs(reward_model, reward_tokenizer):
    def process_reward(prompts, completions, **kwargs):
        return llm_reward_fn_template(
            prompts,
            completions,
            tokenizer=reward_tokenizer,
            model=reward_model,
            normalize=True,
        )

    def answer_reward(prompts, completions, real_num, question_index, **kwargs):
        return answer_reward_function(
            prompts,
            completions,
            truths=real_num,
            question_index=question_index,
        )

    return [process_reward, rule_reward_function, answer_reward]


def build_config(
    output_dir: Path,
    full_train_size: int,
    smoke_test: bool,
) -> tuple[GRPOConfig, int, int]:
    interval = 1 if smoke_test else ARGS.selection_interval
    config = GRPOConfig(
        output_dir=str(output_dir),
        remove_unused_columns=False,
        max_prompt_length=512,
        num_generations=NUM_GENERATIONS,
        max_completion_length=512,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        # Candidate scoring has no backward pass. This equals the normal
        # per-process generation batch and avoids scoring one prompt at a time.
        per_device_eval_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        temperature=0.8,
        learning_rate=1e-5,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards=False,
        num_iterations=1,
        mask_truncated_completions=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=-1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=interval,
        eval_strategy="no",
        fp16=False,
        # GRPOConfig otherwise infers bf16=True from fp16=False. Keep that
        # baseline behavior for GPU runs, but permit CPU-only selector checks.
        bf16=False if ARGS.validate_selection_only else None,
        report_to="wandb",
        reward_weights=list(REWARD_WEIGHTS),
        seed=ARGS.seed,
        data_seed=ARGS.seed,
        dataloader_num_workers=0,
    )

    unique_prompts_per_update = config.generation_batch_size // NUM_GENERATIONS
    selection_size = interval * unique_prompts_per_update
    if selection_size > full_train_size:
        raise ValueError(
            f"K={selection_size} exceeds training size={full_train_size}."
        )
    baseline_max_steps = math.ceil(
        full_train_size / unique_prompts_per_update
    )
    config.max_steps = 1 if smoke_test else baseline_max_steps
    config.num_train_epochs = math.ceil(config.max_steps / interval)

    print(
        "动态选择批次计算: "
        f"world_size={config.world_size}, "
        f"generation_batch_size={config.generation_batch_size}, "
        f"每 update 不同样本数={unique_prompts_per_update}, "
        f"T={interval}, K={selection_size}, max_steps={config.max_steps}"
    )
    return config, interval, selection_size


def validate_selection_only(
    metadata: pd.DataFrame,
    calibrated_lower_bounds,
    selection_size: int,
) -> None:
    """Exercise the real selector on deterministic synthetic rollout rewards."""
    from dynamic_data_selection import (
        build_candidate_pool,
        compute_capacity_capped_group_quotas,
        compute_prompt_values,
        subgroup_aware_greedy_select,
        unmet_quotas,
    )

    quotas = compute_capacity_capped_group_quotas(
        selection_size,
        calibrated_lower_bounds,
        metadata,
    )
    candidate_size = min(len(metadata), 2 * selection_size)
    candidates = build_candidate_pool(
        metadata=metadata,
        candidate_size=candidate_size,
        round_id=0,
        previous_selected=[],
        last_scored_round={},
        quotas=quotas,
        seed=ARGS.seed,
    )
    rng = torch.Generator().manual_seed(ARGS.seed)
    records = []
    for sample_id in candidates:
        reward_tensor = torch.rand(
            (NUM_GENERATIONS, 3),
            generator=rng,
        )
        for reward_row in reward_tensor.tolist():
            records.append(
                {
                    "sample_id": sample_id,
                    "process_reward": reward_row[0],
                    "rule_reward": reward_row[1],
                    "answer_reward": reward_row[2],
                }
            )
    scores = compute_prompt_values(
        pd.DataFrame(records),
        num_generations=NUM_GENERATIONS,
        lambda_conflict=ARGS.lambda_conflict,
        reward_weights=REWARD_WEIGHTS,
    )
    selected = subgroup_aware_greedy_select(
        scores=scores,
        metadata=metadata,
        quotas=quotas,
        selection_size=selection_size,
        gamma=ARGS.gamma,
    )
    deficits = unmet_quotas(selected, metadata, quotas)
    if deficits:
        raise AssertionError(f"CPU selector validation failed: {deficits}")
    candidate_mean = float(scores["final_value"].mean())
    selected_mean = float(
        scores.loc[scores.sample_id.isin(selected), "final_value"].mean()
    )
    if selected_mean <= candidate_mean:
        raise AssertionError(
            "Selected samples should have higher mean value than the candidate pool "
            f"in this deterministic validation: {selected_mean} <= {candidate_mean}"
        )
    print(
        "CPU selection validation passed: "
        f"candidates={len(candidates)}, selected={len(selected)}, "
        f"candidate_mean_U={candidate_mean:.6f}, "
        f"selected_mean_U={selected_mean:.6f}"
    )


def main() -> None:
    validate_trl_interface()
    (
        full_train_dataset,
        eval_dataset,
        metadata,
        calibrated_targets,
        comparison_data,
    ) = (
        load_datasets_and_metadata()
    )

    output_dir = ARGS.output_dir
    if ARGS.smoke_test:
        output_dir = Path(f"{output_dir}_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ARGS.validate_selection_only and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Fix the NVIDIA driver/library mismatch before "
            "running model loading or GRPO training."
        )

    config, interval, selection_size = build_config(
        output_dir=output_dir,
        full_train_size=len(full_train_dataset),
        smoke_test=ARGS.smoke_test,
    )
    write_demographic_calibration_report(
        output_dir=output_dir,
        metadata=metadata,
        comparison_data=comparison_data,
        calibrated_targets=calibrated_targets,
        selection_size=selection_size,
    )

    if ARGS.validate_selection_only:
        validate_selection_only(metadata, calibrated_targets, selection_size)
        return

    tokenizer, base_policy, peft_config = init_tokenizer_and_policy()
    reward_tokenizer, reward_model, loaded_path = load_reward_model_if_exists(
        str(REWARD_CKPT_DIR),
        whether_8bit=False,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if reward_model is None:
        raise RuntimeError(
            "Dynamic process-choice selection requires a valid reasoning reward model."
        )
    reward_model.eval()
    print(
        f"[INFO] reasoning reward model: {loaded_path}, "
        f"device={next(reward_model.parameters()).device}"
    )

    if ARGS.candidate_pool_size is None:
        candidate_size = min(
            len(full_train_dataset),
            int(math.ceil(ARGS.candidate_multiplier * selection_size)),
        )
    else:
        candidate_size = ARGS.candidate_pool_size
    if candidate_size != -1 and candidate_size < selection_size:
        raise ValueError(
            f"candidate_pool_size={candidate_size} must be >= K={selection_size}."
        )

    selector = SubgroupAwareDynamicSelector(
        metadata=metadata,
        group_targets=calibrated_targets,
        selection_size=selection_size,
        candidate_size=candidate_size,
        output_dir=output_dir,
        num_generations=NUM_GENERATIONS,
        lambda_conflict=ARGS.lambda_conflict,
        gamma=ARGS.gamma,
        seed=ARGS.seed,
        resume_state=ARGS.resume_from_checkpoint is not None,
    )
    dynamic_train_dataset = DynamicSubsetDataset(
        full_train_dataset,
        active_ids=selector.initial_active_ids(),
    )
    reward_funcs = get_stage1_reward_funcs(reward_model, reward_tokenizer)
    trainer = SelectionAwareGRPOTrainer(
        model=base_policy,
        args=config,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        train_dataset=dynamic_train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    if not any(parameter.requires_grad for parameter in trainer.model.parameters()):
        raise RuntimeError("TRL preparation left the policy with no trainable LoRA.")
    trainer.add_callback(
        DynamicSelectionCallback(
            trainer=trainer,
            selector=selector,
            dynamic_dataset=dynamic_train_dataset,
            full_dataset=full_train_dataset,
            selection_interval=interval,
        )
    )
    trainer.add_callback(FirstStepGradientGuard())

    print(
        "开始动态数据选择 GRPO: "
        f"candidate_size={candidate_size}, K={selection_size}, interval={interval}"
    )
    trainer.train(
        resume_from_checkpoint=(
            str(ARGS.resume_from_checkpoint)
            if ARGS.resume_from_checkpoint is not None
            else None
        )
    )

    model_dir = output_dir / "final_grpo_model"
    tokenizer_dir = output_dir / "final_grpo_model_tokenizer"
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"训练结束，adapter 已保存到 {model_dir}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()
