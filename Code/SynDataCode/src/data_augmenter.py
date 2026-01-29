import pandas as pd
import logging
import queue
import threading
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_api_caller import LLMApiCaller
from neighbor_retriever import NeighborRetriever
from profile_generator import ProfileGenerator

logger = logging.getLogger(__name__)

class DataAugmenter:
    def __init__(self, human_data: pd.DataFrame, questions: List[Dict], 
                 profile_columns: List[str], llm_caller: LLMApiCaller,
                 max_neighbors: int, neighbor_threshold: int, real_ratio_threshold: float,
                 num_workers: int = 5):
        self.human_data = human_data
        self.questions = questions
        self.profile_columns = profile_columns
        self.llm_caller = llm_caller
        self.max_neighbors = max_neighbors
        self.neighbor_threshold = neighbor_threshold
        self.real_ratio_threshold = real_ratio_threshold
        self.num_workers = num_workers
        self.choice_columns, self.reason_columns = self._get_response_columns()
    
    def _get_response_columns(self) -> Tuple[List[str], List[str]]:
        """获取回答相关的列名"""
        choice_columns = [f"{i}_choice" for i in range(15)]
        reason_columns = [f"{i}_reason_clean" for i in range(15)]
        return choice_columns, reason_columns
    
    def augment_data(self, round_num: int = 1, use_synthetic: bool = False) -> pd.DataFrame:
        """
        增强数据
        """
        logger.info(f"开始第 {round_num} 轮数据增强")
        
        # 生成新profiles
        profile_generator = ProfileGenerator(self.human_data, self.profile_columns)
        new_profiles = profile_generator.generate_new_profiles()
        
        logger.info(f"准备为 {len(new_profiles)} 个新profile生成数据")
        
        # 创建邻居检索器
        neighbor_retriever = NeighborRetriever(
            self.human_data, self.profile_columns,
            self.max_neighbors, self.neighbor_threshold, self.real_ratio_threshold
        )
        
        # 多线程处理
        augmented_records = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for profile in new_profiles:
                future = executor.submit(
                    self._process_single_profile,
                    profile, neighbor_retriever, round_num
                )
                futures[future] = profile
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"已处理 {completed}/{len(new_profiles)} 个profile")
                
                try:
                    result = future.result()
                    if result is not None:
                        augmented_records.append(result)
                except Exception as e:
                    logger.error(f"处理profile失败: {e}")
        
        logger.info(f"成功生成 {len(augmented_records)} 条增强数据")
        
        # 转换为DataFrame
        augmented_df = pd.DataFrame(augmented_records)
        return augmented_df
    
    def _process_single_profile(self, profile: Dict, neighbor_retriever: NeighborRetriever, 
                                round_num: int) -> Optional[Dict]:
        """处理单个profile"""
        try:
            # 检索邻居
            neighbor_indices, total_neighbors, provided_neighbors, real_ratio, provided_real_ratio = \
                neighbor_retriever.retrieve_neighbors(profile)
            
            if len(neighbor_indices) == 0:
                # logger.warning(f"无法为profile {profile}检索到足够的邻居")
                return None
            
            # 获取邻居数据
            neighbor_data = neighbor_retriever.get_neighbor_data(neighbor_indices)
            
            # 为每个问题生成回答
            record = {col: profile.get(col) for col in self.profile_columns}
            record['round'] = round_num
            record['neighbor_list'] = str(neighbor_indices)  # 转换为字符串保存
            record['total_neighbors'] = total_neighbors
            record['provided_neighbors'] = provided_neighbors
            record['real_ratio'] = real_ratio
            record['provided_real_ratio'] = provided_real_ratio
            record['source'] = 'synthetic'
            
            for i, question_info in enumerate(self.questions):
                question = question_info['question']
                
                # logger.info(f"为profile生成第 {i} 个问题的回答")
                
                # 调用LLM生成回答，传入question_index
                thinking, choice = self.llm_caller.generate_thinking_and_choice(
                    question, profile, neighbor_data, self.choice_columns, question_index=i
                )
                
                record[f'{i}_choice'] = choice
                record[f'{i}_reason_clean'] = thinking
            
            logger.info(f"成功为profile生成所有问题的回答")
            return record
            
        except Exception as e:
            logger.error(f"生成新profile回答问题时出错: {e}", exc_info=True)
            return None
    
    def save_augmented_data(self, augmented_df: pd.DataFrame, output_path: str):
        """保存增强数据"""
        try:
            augmented_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"增强数据已保存到 {output_path}")
        except Exception as e:
            logger.error(f"保存增强数据失败: {e}")
            raise