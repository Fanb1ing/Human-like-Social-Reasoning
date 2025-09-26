# config.py
MODELS = {
"DeepSeek-R1": {
        "api_key": "your-api-key",
        "model":"deepseek-r1",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "timeout": 5,  
        "reasoning": True  
    }, #take DeepSeek (aliyun server as a sample)

"Local-Model": {
        "api_key": "your-api-key",
        "model":"your-model-local-server-name",
        "api_url": "your-model-local-server-url",
        "timeout": 5,   
    } #or you can run a local llm server
}


MAX_RETRIES = 3 
LOG_FILE = "logs/llm_api.log"  