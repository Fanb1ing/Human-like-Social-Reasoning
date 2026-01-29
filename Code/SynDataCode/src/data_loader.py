import pandas as pd
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, human_data_path: str, question_json_path: str, profile_columns: List[str]):
        self.human_data_path = human_data_path
        self.question_json_path = question_json_path
        self.profile_columns = profile_columns
        self.human_data = None
        self.questions = None
        
    def load_human_data(self) -> pd.DataFrame:
        """加载原始人类数据"""
        try:
            human_df = pd.read_csv(self.human_data_path)
            
            delete_columns = [str(i)+'_choice_num' for i in range(15)] + [str(i) for i in range(5)]+ [str(i)+'_reason' for i in range(15)]
            human_df.drop(columns=delete_columns, errors='ignore', inplace=True)
            self.human_data = human_df
            logger.info(f"成功加载人类数据，共 {len(self.human_data)} 行")
            
            # 添加source列标记为真人数据
            self.human_data['source'] = 'real'
            self.human_data['round'] = 0
            
            return self.human_data
        except Exception as e:
            logger.error(f"加载人类数据失败: {e}")
            raise
    
    def load_questions(self) -> List[Dict]:
        """加载问题数据"""
        try:
            with open(self.question_json_path, 'r', encoding='utf-8') as f:
                questions_dict = json.load(f)
            
            demo_questions = []
            index = 0
            for key, value in questions_dict.items():
                for question_id, question_value in value.items():
                    question = question_value['question']['situation']
                    options = question_value['question']['option']
                    options = [option for option in options if option != 'nan']
                    options_str = "、".join(options)
                    question = question + f"\n可以选的选项有：{options_str}\n"
                    demo_questions.append({
                        "type": key,
                        "question_index": index,
                        "question": question
                    })
                    index+=1
            
            self.questions = demo_questions
            logger.info(f"成功加载 {len(self.questions)} 个问题")
            return self.questions
        except Exception as e:
            logger.error(f"加载问题数据失败: {e}")
            raise
    
    def get_response_columns(self) -> Tuple[List[str], List[str]]:
        """获取所有回答相关的列名"""
        choice_columns = [f"{i}_choice" for i in range(15)]
        reason_columns = [f"{i}_reason_clean" for i in range(15)]
        return choice_columns, reason_columns
    
    def load_all(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """加载所有数据"""
        self.load_human_data()
        self.load_questions()
        logger.info("所有数据加载完成")
        return self.human_data, self.questions