import pandas as pd
import logging
import itertools
from typing import List, Dict
from utils import get_all_profile_values

logger = logging.getLogger(__name__)

class ProfileGenerator:
    def __init__(self, data: pd.DataFrame, profile_columns: List[str]):
        self.data = data
        self.profile_columns = profile_columns
        self.profile_values = get_all_profile_values(data, profile_columns)
    
    def generate_new_profiles(self) -> List[Dict]:
        """
        生成所有可能的新profile组合
        """
        new_profiles = []
        
        # 获取所有可能的profile组合，笛卡尔积
        all_combinations = itertools.product(*[self.profile_values[col] for col in self.profile_columns])
        
        existing_profiles = set()
        for _, row in self.data.iterrows():
            profile_tuple = tuple(row[col] for col in self.profile_columns)
            existing_profiles.add(profile_tuple)
        
        for combination in all_combinations:
            if combination not in existing_profiles:
                profile = {}
                for col, value in zip(self.profile_columns, combination):
                    profile[col] = value
                new_profiles.append(profile)
        
        logger.info(f"生成了 {len(new_profiles)} 个新的profile组合")
        return new_profiles
    
    def get_existing_profiles(self) -> List[Dict]:
        """获取所有现有的profile"""
        existing_profiles = []
        for _, row in self.data.iterrows():
            profile = {}
            for col in self.profile_columns:
                profile[col] = row[col]
            existing_profiles.append(profile)
        return existing_profiles