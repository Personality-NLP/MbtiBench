methods=(
    zero-shot
)

models=(
    gpt-4o
)

types=(
    soft
    hard
)

rounds=(
    0
)

for r in ${rounds[@]}; do
    for t in ${types[@]}; do
        for m in ${methods[@]}; do
            for b in ${models[@]}; do
                sbatch -J downstream_${b}_${m}_${t} \
                    -o log/downstream_mbti_${b}_${m}_${t}_%j.log \
                    --export=method=$m,model=$b,type=$t,round=$r \
                    scripts/downstream-mbti-run.sh
            done
        done
    done
done
