
from config import *
from llm_api import LLMClient
import pandas as pd


def InitDetailIden():
    file_path = '/Path/to/dataset.xlsx'  
    df = pd.read_excel(file_path, engine='openpyxl')
    return df.iloc[:,:5]

def IdenText(row):
    iden_str =''
    for column_name, column_value in row.items():
        iden_str += f"{column_name}{column_value}\n"
    return iden_str

def call_llm(model_name, prompt,temp):
    try:
        client = LLMClient(model_name=model_name)
        answ_get = client.call_llm(prompt,temperature=temp)
        if answ_get:
            return answ_get
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error in {model_name} : {e}")
