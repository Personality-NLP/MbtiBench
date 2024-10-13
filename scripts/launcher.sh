methods=(
    step-by-step
)
    # zero-shot
    # few-shot
    # psycot

models=(
    llama3.1-70b
    qwen2-72b
    gpt-4o-mini
)
    # qwen2-7b
    # llama3.1-8b
    # gpt-4o

types=(
    hard
    soft
)

for t in ${types[@]}; do
    for m in ${methods[@]}; do
        for b in ${models[@]}; do
            sbatch -J mbtibench_${b}_${m}_${t} \
                -o log/mbtibench_${b}_${m}_${t}_%j.log \
                --export=method=$m,model=$b,type=$t \
                scripts/run.sh
        done
    done
done
