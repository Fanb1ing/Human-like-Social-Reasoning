# Human-like Social Reasoning

Code, data, and model adapters for our paper:
**"Align the Reasoning, Not Just the Answer: Process-Level Human Preference Alignment for Persona-Conditioned Social Simulation"**.

> Important: some experiments query third-party LLM APIs. Exact bitwise reproducibility is not guaranteed due to model/version changes and non-determinism of hosted services. We provide: prompts, inference scripts, and evaluation notebooks to reproduce the reported metrics as closely as possible.

---

## 1. Repository structure

- `Code/`
  - `Data/`
    - `Human-Data-Question-list.json`: PSRP questionnaire items.
    - `Augment_Question.json`: augmentation question pool.
    - `wvs_questions.json`, `wvs_china_200_benchmark.csv`: WVS benchmark questions and China-200 subset.
  - `SynDataCode/`: synthetic data generation / augmentation pipeline.
  - `Evaluation/`
    - `llm_api_client/`: OpenAI-compatible API client + inference scripts.
    - `evaluation_inference_result/`: analysis notebooks to compute final metrics.
  - `RLPHF/`: training code (SFT + Reward Model + RL).
- `PSRP Dataset/`
  - `Survey/`: bilingual questionnaires (PDF).
  - `Human Data/`: bilingual human response spreadsheets.
  - `Syn Data/`: released synthetic data (CSV).
- `User Study/`
  - `Questionnaire/`: user study questionnaire PDFs.
  - `Questionnaire-Response/`: anonymized responses CSV.
- `Model/`
  - `Reward-Model/`: LoRA adapter for the reasoning reward model.
  - `RLPHF-Model/`: LoRA adapter for the final aligned model.

---

## 2. Environment setup

### 2.1 Python / dependencies

Recommended: Python >= 3.9.

Install dependencies from `requirements.txt`:
- `pip install -r requirements.txt`


### 2.2 GPU / training
Training scripts under `Code/RLPHF/` are designed for GPU machines.

---

## 3. Reproducing the experiments (recommended order)

This repository contains multiple components. The recommended reproduction order is:

1) **Synthetic data augmentation** (optional, if you want to regenerate syn data)
2) **Train the reasoning reward model**
3) **Train SFT (via LLaMA-Factory) + RL (RLPHF)**
4) **Inference**
5) **Evaluation (metrics)**

### 3.1 Step 1: Synthetic data generation (SynDataCode)

Configuration:
- `Code/SynDataCode/config/hyperparameters.yaml`

Entry scripts:
- Profile augmentation: `Code/SynDataCode/src/main_profile.py`
- New-question answering: `Code/SynDataCode/src/main_question.py`

Before running:
1) Set `DATA.HUMAN_DATA_PATH` and `DATA.OUTPUT_DIR` in `hyperparameters.yaml`.
2) Set `LLM.API_KEY`, `LLM.API_URL`, and `LLM.MODEL`.

Outputs:
- Generated CSV files will be saved under `DATA.OUTPUT_DIR`.

### 3.2 Step 2: Train the reasoning reward model

Entry script:
- `Code/RLPHF/RewardModel/Train_Reasoning_Reward_Model.py`

Inputs:
- Training pairs are stored in:
  - `Code/RLPHF/RewardModel/trainquestion_dataset_0915.csv`
  

Output:
- The script saves checkpoints under the configured `output_dir`.

### 3.3 Step 3: Train SFT (LLaMA-Factory) + RL (RLPHF)

#### SFT (LLaMA-Factory)
Configuration file:
- `Code/RLPHF/SFT/LlamaFactory_SFT.yaml`

Please edit:
- `model_name_or_path`
- `dataset`
- `output_dir`

#### RL (RLPHF)
Entry script:
- `Code/RLPHF/RL/RL-training.py`

Please edit paths in the script before running:
- `BASE_MODEL` (SFT model path)
- `REWARD_CKPT_DIR` (reward model adapter/ckpt)
- `DATA_ROOT` (`Code/RLPHF/RL/dataset/`)
- `FINAL_SAVE`

Pretrained adapters (if you prefer not to retrain):
- `Model/Reward-Model/`
- `Model/RLPHF-Model/`

**Compute resources (reference)**

- As a reference, our RLPHF training typically requires **2× NVIDIA A100-SXM4-80GB** and takes **~40 hours** end-to-end.
- Example machine specification used in our runs:
  - CPU: **AMD EPYC 7742 64-Core Processor** (2 sockets, 256 logical CPUs)
  - GPU: **8× NVIDIA A100-SXM4-80GB** (we used 2 GPUs for RLPHF training)
  - Memory: **~1.0 TiB RAM**
  - Storage (visible mounts): `/` ~876G; `/data5` ~28T (others may vary by cluster)

### 3.4 Step 4: Inference (query models)

#### PSRP inference
Entry script: `Code/Evaluation/llm_api_client/PSRP_inference.py`

1) Configure endpoints in `Code/Evaluation/llm_api_client/config.py` (API key, base URL, model name).
2) Run:
- `python PSRP_inference.py --model_select <MODEL_KEY_IN_CONFIG> --chunk_size 5`

Outputs are saved as CSV shards under a timestamped folder (see `SAVEPATH` inside the script).

#### WVS inference
Entry script: `Code/Evaluation/llm_api_client/WVS_inference.py`

- `python WVS_inference.py --model_select <MODEL_KEY_IN_CONFIG> --chunk_size 5`


### 3.5 Step 5: Evaluation (compute metrics)

Notebooks:
- `Code/Evaluation/evaluation_inference_result/PSRP-inference-analyse.ipynb`
- `Code/Evaluation/evaluation_inference_result/WVS-inference-analyse.ipynb`

These notebooks load the inference CSV outputs and compute the final metrics/figures.

---

## 4. New assets: dataset & user study

### 4.1 PSRP Dataset
Location: `PSRP Dataset/`

Contents:
- Bilingual questionnaires (`Survey/`).
- Human responses (`Human Data/`).
- Synthetic responses (`Syn Data/`).

### 4.2 User Study
Location: `User Study/`

- Questionnaire PDFs and anonymized responses.

### 4.3 Privacy note:
- Please ensure you comply with local ethics/IRB requirements if you redistribute or extend the human-subject components.
- This dataset contains de-identified and aggregated human questionnaire responses. Before use, ensure you have read and comply with your institution’s and local laws and regulations regarding ethics, privacy, and human-subjects research.
- We encourage responsible use for academic research, education, or transparent and compliant engineering purposes only. Do not attempt re-identification, reverse-engineering of anonymized data, or linking this dataset with other sources to identify individuals.
- If you plan to publicly release derived data or analysis results, confirm whether additional ethics approvals or data-use agreements are required, and clearly describe the data processing and de-identification measures in any release.
- For commercial use, use in restricted environments, or sharing with third parties, contact the original data providers or project leads and obtain written permission before proceeding.
- Please acknowledge and cite this dataset and the project in related publications, reports, or software that use the data.

### 4.4 Third-party assets & compliance

This repository may include or reference third-party assets. We use existing assets in accordance with their respective licenses and usage terms.

- **WVS data**: we comply with the WVS data **Terms of Use** when using the related question sets / benchmarks.
- **BeFM model**: we obtained authorization to use the released BeFM model parameters for our experiments.


## 5. License

- **Code**: MIT License.
- **Data**: CC BY-NC 4.0
  - `PSRP Dataset/LICENSE`
  - `User Study/LICENSE`

---
