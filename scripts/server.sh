#!/usr/bin/bash

#SBATCH -J vllm-server
#SBATCH -t 1-00:00:00
#SBATCH -o log/vllm-server-%j.log
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:nvidia_rtx_a6000:4

. .venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model /home/share/models/Meta-Llama-3.1-70B-Instruct \
    --served-model-name llama3.1-70b \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port 58111
# python -m vllm.entrypoints.openai.api_server \
#     --model /home/share/models/Meta-Llama-3.1-8B-Instruct \
#     --served-model-name llama3.1-8b \
#     --trust-remote-code \
#     --tensor-parallel-size 1 \
#     --max-model-len 32768 \
#     --port 58111
# python -m vllm.entrypoints.openai.api_server \
#     --model /home/share/models/Qwen2-72B-Instruct \
#     --served-model-name qwen2-72b \
#     --trust-remote-code \
#     --tensor-parallel-size 4 \
#     --max-model-len 32768 \
#     --port 58000
# python -m vllm.entrypoints.openai.api_server \
#     --model /home/share/models/Qwen2-7B-Instruct \
#     --served-model-name qwen2-7b \
#     --trust-remote-code \
#     --tensor-parallel-size 1 \
#     --max-model-len 32768 \
#     --port 58000
