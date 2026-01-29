import os
import json
import time
import tqdm
import pandas as pd
import argparse
from config import IDEN,MODELS
from utils import *

prompt_direct_en = """
I will give you a person's basic information. Please answer the following question from his/her perspective.\n
The basic information is: {identity}\n
The question is: {question}\n
Please provide your choice and output it in JSON format:\n
1. "Choice": Your selected option\n
Please note, the output JSON format must be correct, and the field name must be consistent with the one above. Do not provide any other content.\n
"""

prompt_direct = """
我将给你一个人的基本信息，请你从他/她的视角出发回答以下题目。\n
基本信息为：{identity}\n
所给的问题是：{question}\n
请你给出选择，并按照json格式输出：\n
1. "选择"：你选择的选项\n
请注意，输出的json格式必须正确，且字段名称必须与上述一致，除此之外不要回答其他内容。\n
""" 

def main():
   
    model_dict = MODELS[MODEL_SELECT]
    llm_model = MODEL_SELECT

    iden_df = pd.read_csv(IDEN_PATH)

    question_path = QUESTIONPATH
    print(f"Loading questions from {question_path}")
    with open(question_path, 'r', encoding='utf-8') as f:
        questions_dict = json.load(f)
    demo_questions = []
    for key,value in questions_dict.items():
        demo_questions.append({
            "index": key,
            "question": value['text']
            })
            
    save_time = time.time()
    save_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(save_time))
    save_folder = f"{SAVEPATH}/{MODEL_SELECT}_{save_time}"
    os.makedirs(save_folder, exist_ok=True)

    question_save_path = os.path.join(save_folder, "model+prompt+question_file.json")
    with open(question_save_path, "w", encoding="utf-8") as file:
        json.dump({'model_select':MODEL_SELECT,'prompt':PROMPT_USED,'question_file':QUESTIONPATH},file,ensure_ascii=False,indent=4)

    result_df = []
    print(f"Starting inference with model: {MODEL_SELECT}")
    for index,row in tqdm.tqdm(iden_df.iterrows(), total=len(iden_df)):
        identity = row['profile_str']
        answer_list = []
        for question_value in demo_questions:
            question_str = question_value["question"]
            prompt = PROMPT_USED.format(identity=identity,question=question_str)
        
            response_llm,reasoning_llm  = call_llm(llm_model, prompt,temp = 0.8)
            answer_list.append(response_llm)
            answer_list.append(reasoning_llm)
           
            
        combined_list = row.tolist() + answer_list
        result_df.append(combined_list) 

        if len(result_df) >=CHUNK_SIZE:
            result_df_pd = pd.DataFrame(result_df)
            cur_time = time.time()
            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
            csv_save_path = os.path.join(save_folder, f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
            result_df_pd.to_csv(csv_save_path,index=False)
            result_df = []
        
    if len(result_df)>0:
        result_df_pd = pd.DataFrame(result_df)
        cur_time = time.time()
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
        csv_save_path = os.path.join(save_folder, f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
        result_df_pd.to_csv(csv_save_path,index=False)
        result_df = []
    
    print(f"Inference complete. Results saved to {save_folder}")

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WVS Benchmark Inference")
    choices_list = list(MODELS.keys())
    parser.add_argument("--model_select", type=str, default="Model", choices=choices_list, help="Model to use")
    parser.add_argument("--prompt_used", type=str, default="prompt_direct", choices=["prompt_direct"], help="Prompt to use")
    parser.add_argument("--chunk_size", type=int, default=5, help="Chunk size for saving results")
    args = parser.parse_args()

    # Paths - Adapted for WVS Benchmark
    QUESTIONPATH = "./Code/Data/wvs_questions.json" # e.g., "/path/to/wvs_questions.json"
    SAVEPATH = "/path/to/Results" # e.g., "/path/to/Results"
    IDEN_PATH = "./Code/Data/wvs_china_200_benchmark.csv" # e.g., "/path/to/wvs_china_200_benchmark.csv"
    
    MODEL_SELECT = args.model_select
    PROMPT_USED = globals()[args.prompt_used]  # 获取对应的prompt变量
    CHUNK_SIZE = args.chunk_size
    
    main()
