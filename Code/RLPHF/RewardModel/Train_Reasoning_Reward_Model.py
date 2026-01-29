import os
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7" 
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from trl import RewardTrainer, RewardConfig


data_folder = './Code/RLPHF/RewardModel'
train_data = pd.read_csv(data_folder+"trainquestion_dataset_0915.csv")  
eval_data = pd.read_csv(data_folder+"testquestion_dataset_0915.csv")    

train_dataset = Dataset.from_dict({
    "prompt": train_data["prompt"].tolist(),
    "chosen": train_data["正样本数据"].tolist(),
    "rejected": train_data["负样本数据"].tolist()
})

# 创建验证数据集
eval_dataset = Dataset.from_dict({
    "prompt": eval_data["prompt"].tolist(),
    "chosen": eval_data["正样本数据"].tolist(),
    "rejected": eval_data["负样本数据"].tolist()
})

print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(eval_dataset)}")



model_name = "/Path/to/Qwen3-8B"


rm_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1, 
   
    device_map="auto"
) 
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj"]
)

rm_model = get_peft_model(rm_model, peft_config)
rm_model.print_trainable_parameters()

rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_model.config.pad_token_id = rm_tokenizer.pad_token_id


def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    

    chosen_texts = [p + c for p, c in zip(examples["prompt"], examples["chosen"])]
    tokenized_chosen = rm_tokenizer(
        chosen_texts, 
        padding="max_length",  
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    new_examples["input_ids_chosen"] = tokenized_chosen["input_ids"]
    new_examples["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    
    rejected_texts = [p + r for p, r in zip(examples["prompt"], examples["rejected"])]
    tokenized_rejected = rm_tokenizer(
        rejected_texts, 
        padding="max_length",  
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    new_examples["input_ids_rejected"] = tokenized_rejected["input_ids"]
    new_examples["attention_mask_rejected"] = tokenized_rejected["attention_mask"]
    
    return new_examples

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,  
    batch_size=16  
) 
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,  
    batch_size=16 
)


training_args = RewardConfig(
    output_dir="./Save/Path/reward_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    gradient_accumulation_steps=8,
    eval_strategy="steps",  
    eval_steps=100, 
    save_strategy="steps",
    save_steps=100,
    report_to="wandb",
    fp16=True,
    logging_steps=10,
    remove_unused_columns=False,  
    disable_dropout=False  
)


trainer = RewardTrainer(
    model=rm_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=rm_tokenizer,
)



trainer.train()
trainer.save_model("./Save/Path/trained_reward_model")
