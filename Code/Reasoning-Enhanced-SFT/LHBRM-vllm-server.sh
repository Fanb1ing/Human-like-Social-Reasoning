#!/usr/bin/env bash
# run_api_vllm.sh
export CUDA_VISIBLE_DEVICES=3          
export API_PORT=8000

llamafactory-cli api \
  --model_name_or_path  /Path/To/UntrianedModel \
  --adapter_name_or_path /Path/To/SFTAdapter \
  --template Corresponding-Template \
  --infer_backend vllm \
  --vllm_maxlen 1024 \
  --vllm_gpu_util 0.95