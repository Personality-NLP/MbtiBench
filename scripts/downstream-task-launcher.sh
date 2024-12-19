methods=(
    zero-shot
)

models=(
    llama3.1-70b
)

mbti_models=(
    gpt-4o
)

types=(
    none
    soft
    hard
)

rounds=(
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
)

for r in ${rounds[@]}; do
    for t in ${types[@]}; do
        for m in ${methods[@]}; do
            for b in ${models[@]}; do
                for bb in ${mbti_models[@]}; do
                    sbatch -J downstream_${b}_${m}_${t} \
                        -o log/downstream_${tt}_${b}_${m}_${t}_%j.log \
                        --export=method=$m,model=$b,mbti_model=$bb,type=$t,round=$r \
                        scripts/downstream-task-run.sh
                done
            done
        done
    done
done
