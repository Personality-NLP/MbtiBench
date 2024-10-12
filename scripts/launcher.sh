methods=(
    zero-shot
    step-by-step
)
    # few-shot
    # psycot

models=(
    llama3.1-70b
)
    # qwen2-72b
    # llama3.1-8b
    # gpt-4o-mini
    # gpt-4o
    # qwen2-7b

for m in ${methods[@]}; do
    for b in ${models[@]}; do
        sbatch -J mbtibench_${b}_${m} \
            -o log/mbtibench_${b}_${m}_%j.log \
            --export=method=$m,model=$b \
            scripts/run.sh
    done
done
