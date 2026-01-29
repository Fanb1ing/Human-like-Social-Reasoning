# Human-like-Social-Reasoning

This repository provides the code and datasets used in our paper "Why, Not Just What: Improving Large Language Models' Diversity in Social Simulation via Process-level Human Alignment".

### PSRP Dataset
- Human Data
Bilingual (Chinese & English) collected human datasets.
- Survey
The corresponding bilingual questionnaires used to collect the datasets above.
- Syn Data
Data generated from data augmentation.

### Model
- Reward-Model
LoRA Adapter parameter of Reasoning Reward Model
Base model：Qwen3-8B
- RLPHF-Model
LoRA Adapter parameter of RLPHF Model
Base model: DeepSeek-R1-Distill-Llama-8B.

### Code
- Data   
Quesiton list of PSRP and WorldValuesBench; The groundtruth value of WorldValuesBench
- Evaluation  
1.  `llm_api_client/`: Scripts to query models via URL-based APIs.
Run: `python SBR_inference.py(or MM_inference.py) --model_select YOUR_MODEL_NAME --chunk_size YOUR_CHUNK_SIZE`
2.  `evaluation_inference_result/`: Scripts that compute final metrics from the intermediate outputs generated above.

- RLPHF  
Training configuration files and code files of RLPHF method

- SynDataCode   
Code of data augmentation.

### User Study
User Study Questionnaire and corresponding 42 response