#!/usr/bin/bash

#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G

. .venv/bin/activate

if [ $model = "gpt-4o-mini" ] || [ $model = "gpt-4o" ]; then
    python inference.py --method $method --model $model
elif [ $model = "qwen2-72b" ] || [ $model = "qwen2-7b" ]; then
    python inference.py --method $method --model $model --host gpu07 --port 58000
elif [ $model = "llama3.1-70b" ] || [ $model = "llama3.1-8b" ]; then
    python inference.py --method $method --model $model --host gpu07 --port 58111
else
    echo "Unknown model: $model"
fi
