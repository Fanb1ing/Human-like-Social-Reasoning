# Human-like-Social-Reasoning

This repository provides the code and datasets used in our paper "Eliciting Human-like Social Reasoning in Large Language Models".

### Social-Behavior-Reasoning Dataset
- Dataset
Bilingual (Chinese & English) datasets for social-behavior reasoning.
- Survey
The corresponding bilingual questionnaires used to collect the datasets above.

### LHBRM
Contains the fine-tuned LHBRM checkpoints.
Base model: DeepSeek-R1-Distill-Llama-8B.

### Code

- User Study: User Study Questionnaire and corresponding 35 response
- Evaluation:  
1.  `llm_api_client/`: Scripts to query models via URL-based APIs.
Run: `python SBR_inference.py(or MM_inference.py) --model_select YOUR_MODEL_NAME --chunk_size YOUR_CHUNK_SIZE`
2.  `evaluation_inference_result/`: Scripts that compute final metrics from the intermediate outputs generated above.

- Reasoning-Enhanced-SFT: Training configuration files for Reasoning-Enhanced Supervised Fine-Tuning.
