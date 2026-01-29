import os
import yaml
import logging
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import DataLoader
from llm_api_caller import LLMApiCallerQuestion
from question_builder import QuestionBuilder
from logger_config import setup_logger

logger = logging.getLogger(__name__)


class QuestionAnswerGenerator:
    """为现有人口属性生成新问题的回答"""
    
    def __init__(self, human_data: pd.DataFrame, questions: list,
                 profile_columns: list, llm_caller: LLMApiCallerQuestion,
                 old_questions: list = None, num_workers: int = 5):
        self.human_data = human_data
        self.questions = questions
        self.profile_columns = profile_columns
        self.llm_caller = llm_caller
        self.old_questions = old_questions or []
        self.num_workers = num_workers
    
    def generate_answers(self) -> pd.DataFrame:
        """为所有人生成新问题的回答"""
        logger.info(f"开始为 {len(self.human_data)} 个人生成新问题的回答")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for idx, (_, row) in enumerate(self.human_data.iterrows()):
                future = executor.submit(self._process_single_person, row, idx)
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"已处理 {completed}/{len(self.human_data)} 个人")
                
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理人员失败: {e}")
        
        logger.info(f"成功生成 {len(results)} 个人的新问题回答")
        return pd.DataFrame(results)
    
    def _process_single_person(self, row: pd.Series, person_idx: int) -> dict:
        """处理单个人的新问题回答生成"""
        try:
            # 提取profile信息
            profile = {col: row[col] for col in self.profile_columns if col in row.index}
            
            # 提取已有的回答信息（用于参考）
            existing_answers = self._extract_existing_answers(row)
            
            # 从原始行数据中复制所有信息（profile + 原始回答）
            record = row.to_dict()
            
            # 为每个新问题生成回答
            for question_info in self.questions:
                q_idx = question_info['question_index'] + 15
                question = question_info['question']
                
                # 调用LLM生成回答，传入原始问题列表
                thinking, choice = self.llm_caller.generate_thinking_and_choice(
                    question, profile, existing_answers, self.old_questions
                )
                
                record[f'{q_idx}_chocice'] = choice
                record[f'{q_idx}_reason_clean'] = thinking
            
            logger.info(f"成功为第 {person_idx} 个人生成所有新问题的回答")
            return record
            
        except Exception as e:
            logger.error(f"生成第 {person_idx} 个人的回答时出错: {e}", exc_info=True)
            return None
    
    def _extract_existing_answers(self, row: pd.Series) -> dict:
        """从行数据中提取已有的回答信息"""
        existing_answers = {}
        
        # 提取所有问题的回答作为参考
        for i in range(15):
            choice_col = f'{i}_chocice'
            reason_col = f'{i}_reason_clean'
            
            if choice_col in row.index and reason_col in row.index:
                choice = row[choice_col]
                thinking = row[reason_col]
                
                # 检查是否为有效的回答
                if pd.notna(choice) and pd.notna(thinking):
                    existing_answers[i] = {
                        'choice': str(choice),
                        'thinking': str(thinking)
                    }
        
        return existing_answers
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """保存生成的结果"""
        try:
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"结果已保存到 {output_path}")
            logger.info(f"保存数据大小: {results_df.shape}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise



def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 设置日志
    setup_logger()
    logger.info("=" * 50)
    logger.info("开始新问题回答生成流程")
    logger.info("=" * 50)
    
    try:
        # 加载配置
        config_path = "./SynData/config/hyperparameters.yaml"
        config = load_config(config_path)
        logger.info("配置已加载")
        
        # 创建输出目录
        output_dir = config['DATA']['OUTPUT_DIR']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载原始人类数据和原始问题（使用DataLoader）
        logger.info("加载原始人类数据和原始问题...")
        data_loader = DataLoader(
            config['DATA']['HUMAN_DATA_PATH'],
            config['DATA']['QUESTION_JSON_PATH'],
            config['PROFILE']['COLUMNS']
        )
        human_data, old_questions = data_loader.load_all()
        
        # 构建新问题（使用QuestionBuilder）
        logger.info("构建新问题...")
        question_builder = QuestionBuilder(config['DATA']['QUESTION_NEW_JSON_PATH'])
        questions = question_builder.load_questions()
        
        # 初始化LLM调用器
        logger.info("初始化LLM API调用器...")
        api_key = config['LLM']['API_KEY']
        if not api_key:
            logger.error("未配置LLM API密钥，退出程序")
            sys.exit(1)
        
        llm_caller = LLMApiCallerQuestion(
            api_key=api_key,
            api_url=config['LLM']['API_URL'],
            model=config['LLM']['MODEL'],
            max_tokens=config['LLM']['MAX_TOKENS'],
            temperature=config['LLM']['TEMPERATURE'],
            timeout=config['LLM']['TIMEOUT'],
            max_retries=config['LLM']['MAX_RETRIES'],
            retry_delay=config['LLM']['RETRY_DELAY']
        )
        
        # 生成新问题的回答
        logger.info("生成新问题的回答...")
        generator = QuestionAnswerGenerator(
            human_data, questions,
            config['PROFILE']['COLUMNS'],
            llm_caller,
            old_questions=old_questions,
            num_workers=config['THREADING']['NUM_WORKERS']
        )
        
        results_df = generator.generate_answers()
        
        # 保存结果
        output_path = os.path.join(output_dir, "104profile_30question_augmented_v1.csv")
        generator.save_results(results_df, output_path)
        
        logger.info("=" * 50)
        logger.info("新问题回答生成流程完成")
        logger.info(f"生成数据大小: {results_df.shape}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"新问题回答生成流程失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()