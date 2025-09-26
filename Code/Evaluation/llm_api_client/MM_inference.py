from config import IDEN,MODELS
from utils import *
import random
import json
import os
import time
import tqdm
import pandas as pd

prompt_direct_en = """
I will give you a person's basic information. Please answer the following question from his/her perspective.\n
The basic information is: {identity}\n
The question is: {question}\n
Please provide your choice and output it in JSON format:\n
1. "Choice": Your selected option\n
Please note, the output JSON format must be correct, and the field name must be consistent with the one above. Do not provide any other content.\n
"""

prompt_direct_ch = """
我将给你一个人的基本信息，请你从他/她的视角出发回答以下题目。\n
基本信息为：{identity}\n
所给的问题是：{question}\n
请你给出选择，并按照json格式输出：\n
1. "选择"：你选择的选项\n
请注意，输出的json格式必须正确，且字段名称必须与上述一致，除此之外不要回答其他内容。\n
""" 

def main():
    llm_model = MODEL_SELECT#MODELS[MODEL_SELECT] 
    question_path = QUESTIONPATH
    demo_questions = pd.read_csv(question_path)
    save_time = time.time()
    save_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(save_time))
    save_folder = f"{SAVEPATH}/{MODEL_SELECT}_{save_time}"
    os.makedirs(save_folder, exist_ok=True)

    question_save_path = os.path.join(save_folder, "model+prompt+question_file.json")
    with open(question_save_path, "w", encoding="utf-8") as file:
        json.dump({'model_select':MODEL_SELECT,'prompt':PROMPT_USED,'question_file':QUESTIONPATH},file,ensure_ascii=False,indent=4)

    result_df = []
    for index,question_value in tqdm.tqdm(demo_questions.iterrows()): 
        question_str = question_value["prompt_str"]
        # option_str = "可以选的选项有:应该、不应该"
        option_str = "Choice list: yes, no" #change this language align with prompt
        profile_str = question_value['profile_text_ch']+option_str
        prompt = PROMPT_USED.format(identity=profile_str,question=question_str)
    
        response_llm,reasoning_llm  = call_llm(llm_model, prompt,temp = 0.8)
        question_value['llm_answer'] = response_llm
        question_value['llm_reason'] = reasoning_llm
        
        result_df.append(question_value) 

        if len(result_df) >=CHUNK_SIZE:
            result_df = pd.DataFrame(result_df)
            cur_time = time.time()
            cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
            csv_save_path = os.path.join(save_folder, f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
            result_df.to_csv(csv_save_path,index=False)
            result_df = []
        
    if len(result_df)>0:
        result_df = pd.DataFrame(result_df)
        cur_time = time.time()
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(cur_time))
        csv_save_path = os.path.join(save_folder, f"result_index_{int(index/CHUNK_SIZE)}_{cur_time}.csv")
        result_df.to_csv(csv_save_path,index=False)
        result_df = []


import argparse
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MM")
    choices_list = list(MODELS.keys())
    parser.add_argument("--model_select", type=str, default="DeepSeek-R1-8B-SFT", choices=choices_list, help="Model to use")
    parser.add_argument("--prompt_used", type=str, default="prompt_direct", choices=["prompt_direct"], help="Prompt to use")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size for saving results")
    args = parser.parse_args()

    
    QUESTIONPATH = "/Code/Data/Chinese_profiletext_prompttext_nosignal_3scenario.csv"    
    SAVEPATH=f"/Path/to/save/"
    MODEL_SELECT = args.model_select
    PROMPT_USED = globals()[args.prompt_used] 
    CHUNK_SIZE = args.chunk_size
    main()
