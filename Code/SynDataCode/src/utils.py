import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def _thinking_answer_split(text: str) -> Tuple[str, str]:
    """
    提取 "思考过程" 和 "选择" 之间的内容
    处理特殊情况：如果最后一个"选择"前是"经常影响"，则使用倒数第二个"选择"
    """
    if not isinstance(text, str):
        logger.warning("Input is not a string, returning empty results.")
        return '', ''
    
    # 1. 创建过滤版本用于定位关键词
    filt_str = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    
    # 2. 查找关键词位置
    thinking_idx = filt_str.find('思考过程')
    
    if thinking_idx == -1:
        logger.warning(f"未能从文本中提取内容")
        return '', ''
    
    # 3. 找到所有"选择"的位置
    choice_indices = []
    start = 0
    while True:
        idx = filt_str.find('选择', start)
        if idx == -1:
            break
        choice_indices.append(idx)
        start = idx + 1
    
    if not choice_indices:
        logger.warning(f"未能从文本中提取内容")
        return '', ''
    
    # 4. 判断最后一个"选择"前是否是"经常影响"
    choice_idx = choice_indices[-1]
    
    # 检查最后一个"选择"前面是否有"经常影响"
    if len(choice_indices) > 1:
        # 获取最后一个"选择"前的内容
        last_choice_before = filt_str[max(0, choice_idx - 10):choice_idx]
        if '经常影响' in last_choice_before:
            # 使用倒数第二个"选择"
            choice_idx = choice_indices[-2]
    
    # 5. 在原文本中定位这些关键词的位置
    thinking_pos = 0
    choice_pos = 0
    filt_count = 0
    
    for i, char in enumerate(text):
        if re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]', char):
            if filt_count == thinking_idx:
                thinking_pos = i
            if filt_count == choice_idx:
                choice_pos = i
            filt_count += 1
    
    # 6. 在原文本中找到"思考过程"和"选择"的实际位置
    thinking_keyword_pos = text.find('思考过程', thinking_pos)
    choice_keyword_pos = text.find('选择', choice_pos)
    
    if thinking_keyword_pos == -1 or choice_keyword_pos == -1:
        logger.warning(f"未能在原文本中定位关键词")
        return '', ''
    
    # 7. 提取内容：从"思考过程"之后到"选择"之前
    thinking_start = thinking_keyword_pos + len('思考过程')
    thinking_from_json = text[thinking_start:choice_keyword_pos].strip()
    
    # 从"选择"之后到末尾
    choice_start = choice_keyword_pos + len('选择')
    choice = text[choice_start:].strip()
    
    # 8. 清理引号、冒号
    thinking_from_json = thinking_from_json.translate(str.maketrans('', '', '"\'：:'))
    choice = choice.translate(str.maketrans('', '', '"\'：:'))
    
    logger.info(f"成功提取思考过程和选择")
    return thinking_from_json, choice


def calculate_profile_diff(profile1: dict, profile2: dict, profile_columns: list) -> int:
    """计算两个profile之间的差异数"""
    diff_count = 0
    for col in profile_columns:
        if profile1.get(col) != profile2.get(col):
            diff_count += 1
    return diff_count


def get_all_profile_values(data, profile_columns: list) -> dict:
    """获取所有profile列的所有可能取值"""
    profile_values = {}
    for col in profile_columns:
        profile_values[col] = data[col].unique().tolist()
    return profile_values