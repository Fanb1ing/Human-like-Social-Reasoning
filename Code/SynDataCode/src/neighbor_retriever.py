import pandas as pd
import logging
import random
from typing import List, Dict, Tuple
from utils import calculate_profile_diff

logger = logging.getLogger(__name__)

class NeighborRetriever:
    def __init__(self, data: pd.DataFrame, profile_columns: List[str], 
                 max_neighbors: int, neighbor_threshold: int, real_ratio_threshold: float):
        self.data = data
        self.profile_columns = profile_columns
        self.max_neighbors = max_neighbors
        self.neighbor_threshold = neighbor_threshold
        self.real_ratio_threshold = real_ratio_threshold
    
    def retrieve_neighbors(self, target_profile: Dict) -> Tuple[List[int], int, int, float, float]:
        """
        检索邻居，采用随机机制：
        1. 先挑选所有真人邻居
        2. 如果真人邻居超过max_neighbors，随机挑选max_neighbors个真人邻居
        3. 如果真人邻居不足max_neighbors，随机挑选合成邻居补充
        
        返回: (邻居索引列表, 邻居总数, 提供邻居数, 真人占比, 提供真人占比)
        """
        real_neighbors = []
        synthetic_neighbors = []
        
        # 找到所有仅有一个profile不同的邻居，分别存储真人和合成数据
        for idx, row in self.data.iterrows():
            diff_count = calculate_profile_diff(
                {col: target_profile.get(col) for col in self.profile_columns},
                {col: row[col] for col in self.profile_columns},
                self.profile_columns
            )
            
            if diff_count == 1:
                neighbor_info = {
                    'index': idx,
                    'row': row,
                    'is_real': row['source'] == 'real'
                }
                
                if neighbor_info['is_real']:
                    real_neighbors.append(neighbor_info)
                else:
                    synthetic_neighbors.append(neighbor_info)
        
        # 计算总邻居数和真人比例
        total_neighbors = len(real_neighbors) + len(synthetic_neighbors)
        real_ratio = len(real_neighbors) / total_neighbors if total_neighbors > 0 else 0.0
        
        # 检查邻居数量是否满足门限
        if total_neighbors < self.neighbor_threshold:
            # logger.warning(f"邻居总数 {total_neighbors} 小于门限 {self.neighbor_threshold}，放弃生成")
            return [], 0, 0, 0.0, 0.0
        
        # 检查真人比例
        if real_ratio < self.real_ratio_threshold:
            # logger.warning(f"真人比例 {real_ratio:.2f} 小于门限 {self.real_ratio_threshold}，放弃生成")
            return [], total_neighbors, 0, real_ratio, 0.0
        
        # 随机选择邻居
        selected_neighbors = self._select_neighbors_randomly(
            real_neighbors, synthetic_neighbors, self.max_neighbors
        )
        
        selected_real_count = sum(1 for n in selected_neighbors if n['is_real'])
        selected_real_ratio = selected_real_count / len(selected_neighbors) if len(selected_neighbors) > 0 else 0.0
        
        neighbor_indices = [n['index'] for n in selected_neighbors]
        
        logger.info(f"检索到 {len(selected_neighbors)} 个邻居（真人: {selected_real_count}, 合成: {len(selected_neighbors) - selected_real_count}），真人占比 {selected_real_ratio:.2f}")
        
        return neighbor_indices, total_neighbors, len(selected_neighbors), real_ratio, selected_real_ratio
    
    def _select_neighbors_randomly(self, real_neighbors: List[Dict], 
                                   synthetic_neighbors: List[Dict], 
                                   max_neighbors: int) -> List[Dict]:
        """
        随机选择邻居的逻辑：
        1. 如果真人邻居 >= max_neighbors，随机选择max_neighbors个真人邻居
        2. 如果真人邻居 < max_neighbors，选择所有真人邻居，然后随机补充合成邻居
        """
        selected_neighbors = []
        
        if len(real_neighbors) >= max_neighbors:
            # 情况1：真人邻居足够，随机选择max_neighbors个真人邻居
            selected_neighbors = random.sample(real_neighbors, max_neighbors)
            logger.info(f"真人邻居充足，随机选择 {max_neighbors} 个真人邻居")
        else:
            # 情况2：真人邻居不足，选择所有真人邻居，然后补充合成邻居
            selected_neighbors = real_neighbors.copy()
            remaining_slots = max_neighbors - len(real_neighbors)
            
            if len(synthetic_neighbors) > 0:
                # 随机选择合成邻居补充
                num_synthetic_to_select = min(remaining_slots, len(synthetic_neighbors))
                selected_synthetic = random.sample(synthetic_neighbors, num_synthetic_to_select)
                selected_neighbors.extend(selected_synthetic)
                logger.info(f"真人邻居不足，选择所有 {len(real_neighbors)} 个真人邻居，随机补充 {num_synthetic_to_select} 个合成邻居")
            else:
                logger.warning(f"没有合成邻居可用，仅使用 {len(real_neighbors)} 个真人邻居")
        
        return selected_neighbors
    
    def get_neighbor_data(self, neighbor_indices: List[int]) -> List[Dict]:
        """获取邻居的完整数据"""
        neighbor_data = []
        for idx in neighbor_indices:
            neighbor_data.append(self.data.iloc[idx].to_dict())
        return neighbor_data