methods=(
    zero-shot
    step-by-step
    few-shot
    psycot
)

models=(
    gpt-4o-mini
    gpt-4o
    llama3.1-8b
    llama3.1-70b
    qwen2-7b
    qwen2-72b
)

types=(
    soft
)
    # hard

rounds=(
    1
    2
    3
    4
    5
)

for r in ${rounds[@]}; do
    for t in ${types[@]}; do
        for m in ${methods[@]}; do
            for b in ${models[@]}; do
                sbatch -J mbtibench_${b}_${m}_${t} \
                    -o log/mbtibench_${b}_${m}_${t}_%j.log \
                    --export=method=$m,model=$b,type=$t,round=$r \
                    scripts/run.sh
            done
        done
    done
done
