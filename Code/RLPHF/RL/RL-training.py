
import os

import torch, gc, random, json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from reward_function import (
    load_reward_model_if_exists,
    llm_reward_function as llm_reward_fn_template,
    rule_reward_function,
    answer_reward_function
)


BASE_MODEL = "Path/to/SFT/Model"
DATA_ROOT  = "Code/RLPHF/RL/dataset"
REWARD_CKPT_DIR = "./Model/Reward-Model"
FINAL_SAVE          = "./Model/RLPHF-Model"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                       
    llm_int8_threshold=6.0,                  
    llm_int8_skip_modules=["lm_head"],       
)
# quantization_config = None


PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)


def init_tokenizer_and_policy():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        output_hidden_states=True,
    )
    policy = get_peft_model(base_model, PEFT_CONFIG)
    policy.print_trainable_parameters()
    return tokenizer, policy




def load_stage1_datasets():
    train_df = pd.read_csv(DATA_ROOT + "RL_traindataset_que+pro_v1.csv")
    eval_df  = pd.read_csv(DATA_ROOT + "RL_testdataset_que+pro_v1.csv")
    train_ds = Dataset.from_dict({
        "prompt": train_df["prompt"].tolist(),
        "chosen": train_df["human_output"].tolist(),
        "real_num": train_df["human_num"].tolist(),
        "question_index": train_df["question_id"].tolist(),
    })
    eval_ds = Dataset.from_dict({
        "prompt": eval_df["prompt"].tolist(),
        "chosen": eval_df["human_output"].tolist(),
        "real_num": eval_df["human_num"].tolist(),
        "question_index": eval_df["question_id"].tolist(),
    })
    print(f"Training Sample: {len(train_ds)}")
    print(f"Testing Sample: {len(eval_ds)}")
    return train_ds, eval_ds

def get_llm_reward_func(reward_model, reward_tokenizer):
    """包装 LLM reward"""
    def llm_reward(prompts, completions, **kwargs):
        if reward_model is None:
            return [0.0] * len(completions)
        return llm_reward_fn_template(prompts, completions,
                                      tokenizer=reward_tokenizer,
                                      model=reward_model,
                                      normalize=True)
    return llm_reward

def get_stage1_reward_funcs(reward_model, reward_tokenizer):
    llm_reward = get_llm_reward_func(reward_model, reward_tokenizer)
    def answer_reward_batch(prompts, completions, real_num, question_index, **kwargs):
        return answer_reward_function(prompts, completions,
                                      truths=real_num,
                                      question_index=question_index)
    def diversity_reward_batch(prompts, completions, real_num, question_index, **kwargs):
        return diversity_reward_function(prompts, completions,
                                         truths=real_num,
                                         question_index=question_index)
    return [llm_reward, rule_reward_function,
            answer_reward_batch]#, diversity_reward_batch]

def get_stage2_reward_funcs(reward_model, reward_tokenizer):
    llm_reward = get_llm_reward_func(reward_model, reward_tokenizer)
    return [llm_reward, rule_reward_function]



def build_trainer(stage, model, tokenizer, reward_funcs, train_ds, eval_ds):
    
    num_gen, grad_acc, save_steps, eval_steps = 4, 16, 25, 15
    reward_weights = [0.25, 0.25, 0.5 ]#0.25]
    eval_strategy = "no"
    remove_unused = False
    out_dir = FINAL_SAVE
    os.makedirs(FINAL_SAVE, exist_ok=True)
    max_completion = 512


    config = GRPOConfig(
        output_dir=out_dir,
        remove_unused_columns=remove_unused,
        max_prompt_length=512,
        num_generations=num_gen,
        max_completion_length=max_completion,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_acc,
        temperature=0.8,
        learning_rate=1e-5,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards=False,
        num_iterations=1,
        mask_truncated_completions=True,
        max_grad_norm=1.0,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        fp16=False,
        report_to="wandb",
        reward_weights=reward_weights,
        )
    return GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )


def main():
    
    tokenizer, policy = init_tokenizer_and_policy()

    reward_tokenizer, reward_model, loaded_path = load_reward_model_if_exists(
        REWARD_CKPT_DIR,
        base_model_name=BASE_MODEL,
        whether_8bit=False,
    )
    if reward_model is None:
        print("[WARN] no reward model checkpoint。")
    else:
        print(f"[INFO] loading reward model：{loaded_path}, device={next(reward_model.parameters()).device}")

    print("\n>>>>>>>>>> Taining <<<<<<<<<<")
    train_ds1, eval_ds1 = load_stage1_datasets()
    reward_funcs1 = get_stage1_reward_funcs(reward_model, reward_tokenizer)
    trainer1 = build_trainer("stage1", policy, tokenizer,
                             reward_funcs1, train_ds1, eval_ds1)
                             
    trainer1.train()
    os.makedirs(FINAL_SAVE, exist_ok=True)
    policy.save_pretrained(FINAL_SAVE+"final_grpo_model")
    tokenizer.save_pretrained(FINAL_SAVE+"final_grpo_model_tokenizer")
    print(f"Stage1 结束，adapter 已保存到 {FINAL_SAVE}+final_grpo_model")


if __name__ == "__main__":
    main()



