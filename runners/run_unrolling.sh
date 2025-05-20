LLM_MODEL=$1

CATEGORIES=("put", "puttwo")

for CATEGORY_NAME in "${CATEGORIES[@]}"
do 
    echo "Running Unrolling Expt"

    python runners/runner_unrolling.py --llm_model=$LLM_MODEL --pillar=1 --category=$CATEGORY_NAME --variation=2

done