# main.py
# from Code.llm_api_client import config,utils
from config import IDEN,MODELS
from utils import *
import random
import json
import os
import time
import torch
from unsloth import FastLanguageModel

# 加载模型和分词器
model_name = "/Path/to/Centaur-8B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=32768,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

prompt_centaur_ch = """
我将给你一个人的基本信息，请你从他/她的视角出发回答以下题目。\n
基本信息为：{identity}\n
所给的问题是：{question}\n
你选择：<<
""" # 直接输出结果
prompt_centaur_en = """
I will give you a person's basic information. Please answer the following question from his/her perspective.\n
The basic information is: {identity}\n
The question is: {question}\n
You choose:<<
"""


def main():
    # 初始化
    iden_df = InitDetailIden()
    llm_model = model  # 使用本地模型

    # 读入json文件,并准备好题目demo_questions
    question_path = QUESTIONPATH
    with open(question_path, 'r', encoding='utf-8') as f:
        questions_dict = json.load(f)
    demo_questions = []
    for key,value in questions_dict.items():
        for question_id,question_value in value.items():
            question = question_value['question']['situation']
            options = question_value['question']['option']
            options = [option for option in options if option != 'nan']
            options_str  =  "、".join(options)
            # question = question + f"\n可以选的选项有：{options_str}\n"
            question = question +f"\n Choice list: {options_str}\n" #change this language align with prompt
            demo_questions.append({
            "type": key,
            "question_id":question_id,
            "question": question
            })
            

    save_time = time.time()
    save_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(save_time))
    os.makedirs(f"{SAVEPATH}/questions_{save_time}_{MODEL_SELECT}_{args.prompt_used}", exist_ok=True)

    question_save_path = os.path.join(f"{SAVEPATH}/questions_{save_time}_{MODEL_SELECT}_{args.prompt_used}", "model+questions.json")
    with open(question_save_path, "w", encoding="utf-8") as file:
        json.dump({'model_select':MODEL_SELECT},file,ensure_ascii=False,indent=4)
        json.dump(demo_questions, file, ensure_ascii=False,indent=4)

    prompt_save_path = os.path.join(f"{SAVEPATH}/questions_{save_time}_{MODEL_SELECT}_{args.prompt_used}", "prompt.json")
    with open(prompt_save_path, "w", encoding="utf-8") as file:
        json.dump({"prompt_ch":PROMPT_USED}, file, ensure_ascii=False,indent=4)
        
    result_df = []
    for index, row in iden_df.iterrows():
        print(index)
        identity = IdenText(row)
        answer_list = []
        for question_index,question_value in enumerate(demo_questions):
            question_str = question_value["question"]
            prompt = PROMPT_USED.format(identity=identity,question=question_str)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_length=512, temperature=0.8)
            response_llm = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_list.append(response_llm)
            answer_list.append(None)
            
        combined_list = row.tolist() + answer_list
        result_df.append(combined_list)

        if len(result_df) >=CHUNK_SIZE:
            result_df = pd.DataFrame(result_df)
            cur_time = time.time()
            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
            csv_save_path = os.path.join(f"{SAVEPATH}/questions_{save_time}_{MODEL_SELECT}_{args.prompt_used}", f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
            result_df.to_csv(csv_save_path,index=False)
            result_df = []
        
    if len(result_df)>0:
        result_df = pd.DataFrame(result_df)
        cur_time = time.time()
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
        csv_save_path = os.path.join(f"{SAVEPATH}/questions_{save_time}_{MODEL_SELECT}_{args.prompt_used}", f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
        result_df.to_csv(csv_save_path,index=False)
        result_df = []


import argparse
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp5")
    choices_list = list(MODELS.keys())
    parser.add_argument("--model_select", type=str, default="DeepSeek-R1-trained", choices=choices_list, help="Model to use")
    parser.add_argument("--prompt_used", type=str, default="prompt_ch_direct", choices=["prompt_ch_direct_centaur"], help="Prompt to use")
    parser.add_argument("--chunk_size", type=int, default=10, help="Chunk size for saving results")

    args = parser.parse_args()

    QUESTIONPATH = "./Code/Data/SBR-Question-list.json"    
    SAVEPATH=f"/Path/to/save/"
    MODEL_SELECT = args.model_select
    PROMPT_USED = globals()[args.prompt_used]  
    CHUNK_SIZE = args.chunk_size
    main()
