methods=(
    zero-shot
    step-by-step
    few-shot
    psycot
)

models=(
    gpt-4o-mini
    gpt-4o
    qwen2-7b
    qwen2-72b
    llama3.1-8b
    llama3.1-70b
)

types=(
    soft
)
    # hard

for t in ${types[@]}; do
    for b in ${models[@]}; do
        for m in ${methods[@]}; do
            python evaluate.py --model $b --method $m --type $t
        done
    done
done
