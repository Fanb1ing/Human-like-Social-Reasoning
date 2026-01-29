import os
import yaml
import logging
import sys
from pathlib import Path
from data_loader import DataLoader
from llm_api_caller import LLMApiCaller
from data_augmenter import DataAugmenter
from logger_config import setup_logger

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 设置日志
    setup_logger()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("开始数据增强流程")
    logger.info("=" * 50)
    
    try:
        # 加载配置
        config_path = "./SynData/config/hyperparameters.yamlhyperparameters.yaml"
        config = load_config(config_path)
        logger.info(f"配置已加载")
        
        # 创建输出目录
        output_dir = config['DATA']['OUTPUT_DIR']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        logger.info("加载原始数据...")
        data_loader = DataLoader(
            config['DATA']['HUMAN_DATA_PATH'],
            config['DATA']['QUESTION_JSON_PATH'],
            config['PROFILE']['COLUMNS']
        )
        human_data, questions = data_loader.load_all()
        
        # 初始化LLM调用器
        logger.info("初始化LLM API调用器...")
        api_key = config['LLM']['API_KEY']
        if not api_key:
            logger.error("未配置LLM API密钥，退出程序")
            sys.exit(1)
        
        llm_caller = LLMApiCaller(
            api_key=api_key,
            api_url=config['LLM']['API_URL'],
            model=config['LLM']['MODEL'],
            max_tokens=config['LLM']['MAX_TOKENS'],
            temperature=config['LLM']['TEMPERATURE'],
            timeout=config['LLM']['TIMEOUT'],
            max_retries=config['LLM']['MAX_RETRIES'],
            retry_delay=config['LLM']['RETRY_DELAY']
        )
        
        # 逐轮增强数据
        current_data = human_data.copy()
        
        for round_config in config['ROUNDS']:
            round_num = round_config['round']
            logger.info(f"\n{'=' * 50}")
            logger.info(f"第 {round_num} 轮数据增强")
            logger.info(f"{'=' * 50}")
            logger.info(f"配置: max_neighbors={round_config['max_neighbors']}, "
                       f"neighbor_threshold={round_config['neighbor_threshold']}, "
                       f"real_ratio_threshold={round_config['real_ratio_threshold']}")
            
            # 创建数据增强器
            augmenter = DataAugmenter(
                current_data, questions,
                config['PROFILE']['COLUMNS'],
                llm_caller,
                max_neighbors=round_config['max_neighbors'],
                neighbor_threshold=round_config['neighbor_threshold'],
                real_ratio_threshold=round_config['real_ratio_threshold'],
                num_workers=config['THREADING']['NUM_WORKERS']
            )
            
            # 增强数据
            augmented_df = augmenter.augment_data(
                round_num=round_num,
                use_synthetic=round_config['use_synthetic']
            )
            
            # 保存增强数据
            output_path = os.path.join(output_dir, f"round_{round_num}_augmented.csv")
            augmenter.save_augmented_data(augmented_df, output_path)
            
            # 合并数据用于下一轮
            current_data = pd.concat([current_data, augmented_df], ignore_index=True)
            logger.info(f"第 {round_num} 轮完成，当前数据总数: {len(current_data)}，新生成数据数: {len(augmented_df)}")
        
        logger.info(f"\n{'=' * 50}")
        logger.info("数据增强流程完成")
        logger.info(f"{'=' * 50}")
        
    except Exception as e:
        logger.error(f"数据增强流程失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import pandas as pd
    main()