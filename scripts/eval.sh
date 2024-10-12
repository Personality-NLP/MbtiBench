methods=(
    zero-shot
    step-by-step
    few-shot
    psycot
)

models=(
    llama3.1-8b
    qwen2-72b
    llama3.1-70b
)
    # gpt-4o-mini
    # gpt-4o
    # qwen2-7b

for m in ${methods[@]}; do
    for b in ${models[@]}; do
        python evaluate.py --model $b --method $m
    done
done
