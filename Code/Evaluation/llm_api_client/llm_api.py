# llm_api.py
import requests
import httpx
import logging
import time
from config import MODELS, MAX_RETRIES
import os  
from openai import OpenAI

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "llm_api.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class LLMClient:
    def __init__(self, model_name):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = MODELS[model_name]["model"]
        self.api_key = MODELS[model_name]["api_key"]
        self.api_url = MODELS[model_name]["api_url"]
        self.timeout = MODELS[model_name]["timeout"]
        self.max_retries = MAX_RETRIES
        self.client = OpenAI(
                api_key = self.api_key,
                base_url = self.api_url,
            )
        self.stream = MODELS[model_name].get("stream", False)
        self.enable_think = MODELS[model_name].get("enable_thinking", None)

    def chat_once(self,msgs,model_name,temp = 0.3):
        try:
            if self.enable_think is not None:
                response = self.client.chat.completions.create(
                    model = model_name,
                    messages = msgs,
                    temperature = temp,
                    timeout=45,
                    extra_body={"enable_thinking": self.enable_think},
                    stream = self.stream
                )
            else:
                response = self.client.chat.completions.create(
                    model = model_name,
                    messages = msgs,
                    temperature = temp,
                    timeout=45,
                    stream = self.stream
                )
            
            if self.stream:
                full_content = ""
                full_reason = ""
                for chunk in response:
                    if chunk.choices:
                        if chunk.choices[0].delta.content is not None:
                            full_content += chunk.choices[0].delta.content
                        if  chunk.choices[0].delta.model_extra['reasoning_content'] is not None:
                            full_reason += chunk.choices[0].delta.model_extra['reasoning_content']
                return full_content,full_reason

            answer = response.choices[0].message.content
            if 'reasoning_content' in response.choices[0].message.model_extra:
                reasoning = response.choices[0].message.model_extra['reasoning_content']
            else:
                reasoning = None
            return answer, reasoning

        except httpx.TimeoutException:
            print("No Response")
            return None, None
        except httpx.HTTPError as e:
            print(f"Error：{e}")
            return None, None
    
    def call_llm(self, prompt, temperature=0.3):
        
        messages = [{"role": "system", "content": "You are an assistant capable of portraying characters with specific demographic attributes and backgrounds."}]
        messages = [{"role": "system", "content": "你是一个能够扮演具有特定人口属性背景的角色的助手。"}]
        messages.append({
            "role": "user",
            "content": prompt,	
        })
        st_time = time.time()  
        for i in range(self.max_retries):
            # print(f"Attempts: {i+1}/{self.max_retries}")
            try:
                response,reasoning = self.chat_once(messages,self.model_name,temp = temperature)
                ed_time = time.time()
                # print("Query Succuess!")
                print(f"Query Time: {ed_time-st_time}")
                return response, reasoning
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
        print("Query Failed.")
        return "Fail", "Fail"