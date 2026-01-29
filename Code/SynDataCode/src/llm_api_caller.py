import logging
import time
from typing import Tuple, Optional, List, Dict
from openai import OpenAI, RateLimitError, APIError
from utils import _thinking_answer_split

logger = logging.getLogger(__name__)

class LLMApiCaller:
    def __init__(self, api_key: str,api_url:str, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 8296, temperature: float = 0.7,
                 timeout: int = 30, max_retries: int = 3, retry_delay: int = 2):
        self.client = OpenAI(api_key=api_key,base_url = api_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate_thinking_and_choice(self, question: str, profile: Dict, 
                                     neighbor_data: List[Dict], choice_columns: List[str],
                                     question_index: int) -> Tuple[str, str]:
        """
        调用LLM生成思考过程和选择
        question_index: 当前问题的索引（0-14）
        """
        # 构建prompt
        prompt = self._build_prompt(question, profile, neighbor_data, choice_columns, question_index)
        
        # 调用API（带重试）
        response = self._call_api_with_retry(prompt)
        
        if response is None:
            logger.error("LLM API调用失败，返回空结果")
            return '', ''
        
        # 提取思考过程和选择
        thinking, choice = _thinking_answer_split(response)
        
        return thinking, choice
    
    def _build_prompt(self, question: str, profile: Dict, neighbor_data: List[Dict], 
                      choice_columns: List[str], question_index: int) -> str:
        """构建prompt"""
        profile_str = self._format_profile(profile)
        neighbors_str = self._format_neighbors(neighbor_data, question_index, profile)
        
        prompt = f"""我将给你一个人的基本信息，请你从他/她的视角出发回答以下题目。

【基本信息】
{profile_str}

【问题】
{question}

请你站在这个人的角度进行思考，然后做出选择，并按照以下格式输出：

思考过程：[你从这个人的角度思考，以第一人称给出做出这个选择的原因]
选择：[你选择的选项]

【参考邻居数据】
为了帮助你更好的理解这个人的基本信息，以下提供了与该人仅有一个信息属性不同的邻居对该问题的回答示例：
{neighbors_str}

请注意：
1. "思考过程"中不要出现自述我的身份等字样（比如"作为一名18-27岁的女性"），而是直接将自己带入这个角色，给出你选择该选项的原因。
2. 输出格式必须包含"思考过程："和"选择："两个字段。
"""

        return prompt
    
    def _format_profile(self, profile: Dict) -> str:
        """格式化profile"""
        profile_str = ""
        for key, value in profile.items():
            profile_str += f"{key}{value}\n"
        return profile_str.strip()
    
    def _format_neighbors(self, neighbor_data: List[Dict], question_index: int, target_profile: Dict) -> str:
        """格式化邻居数据，只包含当前问题的回答"""
        neighbors_str = ""
        choice_col = f"{question_index}_chocice"
        reason_col = f"{question_index}_reason_clean"
        
        for i, neighbor in enumerate(neighbor_data, 1):
            # 获取邻居的profile信息
            neighbor_profile_str = ""
            for key in target_profile.keys():
                if key in neighbor:
                    neighbor_profile_str += f"{key}{neighbor[key]}\n"
            
            # 获取邻居对当前问题的回答
            if choice_col in neighbor and reason_col in neighbor:
                choice = neighbor[choice_col]
                thinking = neighbor[reason_col]
                
                neighbors_str += f"\n邻居 {i}:\n"
                neighbors_str += f"背景信息：\n{neighbor_profile_str}"
                neighbors_str += f"思考过程：{thinking}\n"
                neighbors_str += f"选择：{choice}\n"
        
        return neighbors_str.strip() if neighbors_str else "暂无邻居数据"
    
    def _call_api_with_retry(self, prompt: str) -> Optional[str]:
        """带重试的API调用"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用LLM API (尝试 {attempt + 1}/{self.max_retries})")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个能够扮演具有特定人口属性背景的角色的助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"LLM API调用成功")
                return result
                
            except RateLimitError:
                logger.warning(f"触发速率限制，等待 {self.retry_delay} 秒后重试...")
                time.sleep(self.retry_delay)
            except APIError as e:
                logger.warning(f"API错误: {e}，等待 {self.retry_delay} 秒后重试...")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"调用LLM API失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"LLM API调用失败，已重试 {self.max_retries} 次")
        return None
    
class LLMApiCallerQuestion:
    """用于新问题回答生成的LLM调用器"""
    
    def __init__(self, api_key: str, api_url: str, model: str = "gpt-3.5-turbo",
                 max_tokens: int = 8296, temperature: float = 0.7,
                 timeout: int = 30, max_retries: int = 3, retry_delay: int = 2):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate_thinking_and_choice(self, question: str, profile: Dict,
                                     existing_answers: Dict, old_questions: List[Dict] = None) -> Tuple[str, str]:
        """
        基于profile和已有问题的回答，生成新问题的思考过程和选择
        existing_answers: 包含已有问题的回答信息 {question_index: {'choice': ..., 'thinking': ...}}
        old_questions: 原始问题列表，用于提供完整的题目上下文
        """
        prompt = self._build_prompt(question, profile, existing_answers, old_questions)
        response = self._call_api_with_retry(prompt)
        
        if response is None:
            logger.error("LLM API调用失败，返回空结果")
            return '', ''
        
        thinking, choice = _thinking_answer_split(response)
        return thinking, choice
    
    def _build_prompt(self, question: str, profile: Dict, existing_answers: Dict, 
                      old_questions: List[Dict] = None) -> str:
        """构建prompt"""
        profile_str = self._format_profile(profile)
        existing_answers_str = self._format_existing_answers(existing_answers, old_questions)
        
        prompt = f"""我将给你一个人的基本信息和他/她对之前问题的回答，请你从他/她的视角出发回答以下新题目。

【基本信息】
{profile_str}

【该人对之前问题的回答示例】
{existing_answers_str}

【新问题】
{question}

请你站在这个人的角度进行思考，然后做出选择，并按照以下格式输出：

思考过程：[你从这个人的角度思考，以第一人称给出做出这个选择的原因，结合该人的基本信息和之前的回答风格]
选择：[你选择的选项]

请注意：
1. "思考过程"中不要出现自述我的身份等字样（比如"作为一名18-27岁的女性"），而是直接将自己带入这个角色。
2. 输出格式必须包含"思考过程："和"选择："两个字段。
3. 参考该人之前的回答风格和逻辑，保持一致性。
"""
        return prompt
    
    def _format_profile(self, profile: Dict) -> str:
        """格式化profile"""
        profile_str = ""
        for key, value in profile.items():
            profile_str += f"{key}{value}\n"
        return profile_str.strip()
    
    def _format_existing_answers(self, existing_answers: Dict, old_questions: List[Dict] = None) -> str:
        """格式化已有的回答，包含原始题目"""
        if not existing_answers:
            return "暂无之前的回答数据"
        
        # 构建问题索引映射
        question_map = {}
        if old_questions:
            for q in old_questions:
                question_map[q['question_index']] = q['question']
        
        answers_str = ""
        for idx, (q_idx, answer_info) in enumerate(existing_answers.items(), 1):
            answers_str += f"\n示例 {idx}:\n"
            
            # 添加原始题目
            if q_idx in question_map:
                answers_str += f"题目：{question_map[q_idx]}\n"
            
            answers_str += f"思考过程：{answer_info.get('thinking', '')}\n"
            answers_str += f"选择：{answer_info.get('choice', '')}\n"
        
        return answers_str.strip()
    
    def _call_api_with_retry(self, prompt: str) -> Optional[str]:
        """带重试的API调用"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用LLM API (尝试 {attempt + 1}/{self.max_retries})")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个能够扮演具有特定人口属性背景的角色的助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"LLM API调用成功")
                return result
                
            except RateLimitError:
                logger.warning(f"触发速率限制，等待 {self.retry_delay} 秒后重试...")
                time.sleep(self.retry_delay)
            except APIError as e:
                logger.warning(f"API错误: {e}，等待 {self.retry_delay} 秒后重试...")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"调用LLM API失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"LLM API调用失败，已重试 {self.max_retries} 次")
        return None