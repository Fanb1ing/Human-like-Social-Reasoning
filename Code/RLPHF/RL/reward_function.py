# reward_functions.py
import os
import torch
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# --------- 辅助：自动选择 reward model 路径 ----------
def find_latest_checkpoint(base_dir):
    # 1. 优先 final dir
    final_dir = os.path.join(base_dir, "trained_reward_model")
    if os.path.isdir(final_dir):
        return final_dir

    # 2. 扫描子目录寻找 checkpoint
    cand = []
    base_dir = os.path.join(base_dir,"reward_model")
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and ("checkpoint" in name or name.startswith("checkpoint-")):
            cand.append((os.path.getmtime(p), p))
    if not cand:
        return None
    latest = sorted(cand, key=lambda x: x[0], reverse=True)[0][1]
    return latest

from peft import PeftModel
from transformers import BitsAndBytesConfig

BASE_MODEL_NAME="/data5/fanbingbing/Behave-Benchmark-RL/Data/Model/Qwen3-8B"
def load_reward_model_if_exists(base_dir, base_model_name=BASE_MODEL_NAME,whether_8bit = False,device=None):

    path = find_latest_checkpoint(base_dir)
    if path is None:
        return None, None, None


    if whether_8bit: 
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    adapter_config_path = os.path.join(path, "adapter_config.json")
    model_bin_path = os.path.join(path, "pytorch_model.bin")
    safetensors_path = os.path.join(path, "model.safetensors")

    tokenizer = AutoTokenizer.from_pretrained(path if os.path.exists(model_bin_path) or os.path.exists(safetensors_path) else base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(adapter_config_path):
        if base_model_name is None:
            raise ValueError(" LoRA adapter，but no base_model_name！")

        print(f"[INFO] Loading LoRA adapter from {path} with base {base_model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            device_map="auto",
            
        quantization_config = quantization_config

        )
        model = PeftModel.from_pretrained(base_model, path)
        model = model.merge_and_unload()
    else:
        print(f"[INFO] Loading merged model directly from {path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=1,
            device_map="auto",#{"": device.index if device.type == "cuda" else "cpu"}  
            
        quantization_config = quantization_config
        )
    
    return tokenizer, model, path

# --------- Reward 函数 1：基于 LLM 的 reward model ----------
def llm_reward_function(prompts, completions, tokenizer, model, normalize=True):
    
    device = next(model.parameters()).device #?
    texts = [f"{p} {c}" for p, c in zip(prompts, completions)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.squeeze(-1).cpu().tolist()  # list of floats

    if normalize:
    
        import math
        normalized = [1/(1+math.exp(-x)) for x in logits]
        return normalized
    else:
        return logits


# --------- Reward：rule_reward_function----------
def thinking_answer_split(text):
    '''
    提取...</think> + json "选择..." 之间的内容
    '''
    if text is None:
        return "", ""
    text = str(text)
    
    think_match = re.search(r'(.*?)</think>', text, re.DOTALL)
    thinking_answer = think_match.group(1) if think_match else ""
    
    text_without_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    filt_str = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9=()（）]', '', text_without_think)
    
    match = re.search(r'选择(.*?)$', filt_str)
    if match:
        choice = match.group(1)
    else:
        choice = ''
    
    return thinking_answer, choice

def ngram_repeat_penalty_cn(text, n=4):
    chars = re.sub(r'\s+', '', text)
    if len(chars) < n:
        return 0.0
    grams = [chars[i:i+n] for i in range(len(chars)-n+1)]
    total = len(grams)
    if total == 0:
        return 0.0
    freq = Counter(grams)
    max_repeat = max(freq.values())
    repeat_ratio = (max_repeat - 1) / total
    return min(repeat_ratio, 1.0)

def rule_reward_function(completions, **kwargs):
    rewards = []
    for completion in completions:
        thinking_text, answer_text = thinking_answer_split(completion)
        if thinking_text==  '':
            # 如果提取thinking失败就是-2分
            rewards.append(0)
            continue
        text_length = len(thinking_text)
        if text_length < 30 or text_length > 200:
            rewards.append(0)
            continue
        reward = 1.0 - ngram_repeat_penalty_cn(thinking_text, n=4)
        rewards.append(float(reward))
    return rewards

def build_risk_question(sys_prompt: str, text: dict) -> str:

    def num_to_text(pa):
        if pa > 0:
            return f"收益{pa}元"
        elif pa == 0:
            return "既不收益也不损失"
        else:
            return f"损失{-pa}元"
        
    pa = text["A"][0][1]
    pHa = text["A"][0][0]
    la = text["A"][1][1]
    pLa = text["A"][1][0]
    
    pb = text["B"][0][1]
    pHb = text["B"][0][0]
    lb = text["B"][1][1]
    pLb = text["B"][1][0]
    
    text_a_h, text_a_l = num_to_text(pa), num_to_text(la)
    text_b_h, text_b_l = num_to_text(pb), num_to_text(lb)
    
    if pHa != 1 and pHa != 0:
        text_a = f"以{int(pHa*100)}%的概率{text_a_h}，以{100-int(pHa*100)}%的概率{text_a_l}；"
    elif pHa == 1:
        text_a = f"以100%的概率{text_a_h}；"
    else:
        text_a = f"以100%的概率{text_a_l}；"
    
    if pHb != 1 and pHb != 0:
        text_b = f"以{int(pHb*100)}%的概率{text_b_h}，以{100-int(pHb*100)}%的概率{text_b_l}；"
    elif pHb == 1:
        text_b = f"以100%的概率{text_b_h}；"
    else:
        text_b = f"以100%的概率{text_b_l}；"
    
    return sys_prompt.format(text_a=text_a, text_b=text_b)


import json
# 读入json文件,并准备好题目demo_questions
question_path =  "/data5/fanbingbing/Human-like-Social-Reasoning/Code/Data/Human-Data-Question-list.json"    
with open(question_path, 'r', encoding='utf-8') as f:
    questions_dict = json.load(f)
demo_questions = []
question_index = 0

for key,value in questions_dict.items():
    for question_id,question_value in value.items():
        question = question_value['question']['situation']
        options = question_value['question']['option']
        options = [option for option in options if option != 'nan']
        options_str  =  "、".join(options)
        question = question + f"\n可以选的选项有：{options_str}\n"
        demo_questions.append({
        "type": key,
        "question_id":question_id,
        "question_index": question_index,
        "question": question,
        "option": options
        })
        question_index += 1


question_json_path = 'Code/Data/Augment_Question.json'
with open(question_json_path, 'r', encoding='utf-8') as f:
        questions_dict = json.load(f)

for key, value in questions_dict.items():
    # 跳过不需要的问题类型
    if key in ['心智理论', '弱智吧', '逻辑推理']:
        continue
    for question_id, question_value in value['question'].items():
        question = question_value['situation']
        # 处理风险决策问题的特殊格式
        if key == '风险决策':
            question = build_risk_question(value['system_prompt'], question)
        # 构建选项字符串
        options = question_value['option']
        options = [option for option in options if option != 'nan']
        options_str = "、".join(options)
        question = question + f"\n可以选的选项有：{options_str}\n"
        demo_questions.append({
            "type": key,
            "question_id": question_id,
            "question_index": question_index,
            "question": question,
            "option": options
        })
        question_index += 1



def choicemap(answer_text,question_id):
    question = demo_questions[question_id]
    option_len = len(question['option'])
    option_to_num = {option.replace('，', ''): idx for idx, option in enumerate(question['option'])}
    if question_id in [6,7,8]:

        option_to_num['应该']=0
        option_to_num['不应该']=1
    if question_id == 9:
        new_dict = option_to_num.copy()
        for key, value in option_to_num.items():
            number = key.split('=')[1]  
            new_dict[number] = value
        option_to_num = new_dict.copy()
    

    def custom_map(value):
        if not isinstance(value, str): 
            return value 
        for key, num in option_to_num.items():
            if key in value:
                return num
        return np.nan  
    answer_num = custom_map(answer_text)
    answer_num = answer_num/(option_len - 1)
    return answer_num

def choicemap_index(answer_text,question_id):
    question = demo_questions[question_id]
    option_len = len(question['option'])
    option_to_num = {option.replace('，', ''): idx for idx, option in enumerate(question['option'])}
    if question_id in [6,7,8]:

        option_to_num['应该']=0
        option_to_num['不应该']=1
    if question_id == 9:
        new_dict = option_to_num.copy()
        for key, value in option_to_num.items():
            number = key.split('=')[1]  
            new_dict[number] = value
        option_to_num = new_dict.copy()
    
    
    def custom_map(value):
        if not isinstance(value, str):  
            return value  
        for key, num in option_to_num.items():
            if key in value:
                return num
        return np.nan  
    answer_num = custom_map(answer_text)
    return answer_num

def answer_reward_function(prompts, completions, truths=None, question_index=None, **kwargs):
  
    rate_list = []
    # 每一个都这样处理
    for i,completion in enumerate(completions):
        thinking_text, answer_text = thinking_answer_split(completion)
        if answer_text=='':
            # 如果提取失败就是-2分
            rate_list.append(0)
            continue
        qid = question_index[i] #if question_index is not None else i
        answer_num = choicemap(answer_text, qid)
        if np.isnan(answer_num): 
            # 如果choice不符合规定就是-1分
            rate_list.append(0)
            continue
        rate_list.append(1-abs(answer_num-truths[i]))

    return rate_list


