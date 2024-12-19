#!/usr/bin/bash

#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16G

. .venv/bin/activate

export PYTHONPATH=$(pwd)

if [ $type = "none" ]; then
    python downstream/inference-task.py --round $round --model $model                          --method $method              --host gpu07 --port 58111
elif [ $type = "soft" ] || [ $type = "hard" ]; then
    python downstream/inference-task.py --round $round --model $model --mbti_model $mbti_model --method $method --type $type --host gpu07 --port 58111
else
    echo "Invalid type: $type"
fi
