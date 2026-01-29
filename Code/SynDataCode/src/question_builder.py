import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class QuestionBuilder:
    """从RLstage2_dataset.ipynb中复用的问题构建逻辑"""
    
    def __init__(self, question_json_path: str):
        self.question_json_path = question_json_path
        self.questions = None
    
    def load_questions(self) -> List[Dict]:
        """加载并构建问题列表"""
        try:
            with open(self.question_json_path, 'r', encoding='utf-8') as f:
                questions_dict = json.load(f)
            
            demo_questions = []
            question_index = 0
            
            for key, value in questions_dict.items():
                # 跳过不需要的问题类型
                if key in ['心智理论', '弱智吧', '逻辑推理']:
                    continue
                
                for question_id, question_value in value['question'].items():
                    question = question_value['situation']
                    
                    # 处理风险决策问题的特殊格式
                    if key == '风险决策':
                        question = self._build_risk_question(value['system_prompt'], question)
                    
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
                        "options": options
                    })
                    question_index += 1
            
            self.questions = demo_questions
            logger.info(f"成功加载 {len(self.questions)} 个问题")
            return self.questions
            
        except Exception as e:
            logger.error(f"加载问题数据失败: {e}")
            raise
    
    def _build_risk_question(self, sys_prompt: str, text: dict) -> str:
        """构建风险决策问题的文本"""
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