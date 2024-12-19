#!/usr/bin/bash

#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G

. .venv/bin/activate

export PYTHONPATH=$(pwd)

python downstream/inference-mbti.py \
    --round $round \
    --model $model \
    --method $method \
    --type $type
