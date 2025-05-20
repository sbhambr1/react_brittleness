# Runs a given model for all pillars, category, variations 


# export PYTHONPATH=/workspaces/react_brittleness
# export ANTHROPIC_API_KEY=mykey
# export OPENAI_API_KEY=mykey

LLM_MODEL=$1

CATEGORIES=("put" "clean" "heat" "cool" "examine" "puttwo")

for CATEGORY_NAME in "${CATEGORIES[@]}"
do 
    echo "Running Pillar 1"

    python runners/runner.py --llm_model=$LLM_MODEL --pillar=1 --category=$CATEGORY_NAME --variation=1
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=1 --category=$CATEGORY_NAME --variation=2
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=1 --category=$CATEGORY_NAME --variation=3

    echo "Running Pillar 2"

    python runners/runner.py --llm_model=$LLM_MODEL --pillar=2 --category=$CATEGORY_NAME --variation=1
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=2 --category=$CATEGORY_NAME --variation=2

    echo "Running Pillar 3"

    python runners/runner.py --llm_model=$LLM_MODEL --pillar=3 --category=$CATEGORY_NAME --variation=1
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=3 --category=$CATEGORY_NAME --variation=2
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=3 --category=$CATEGORY_NAME --variation=3
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=3 --category=$CATEGORY_NAME --variation=4

    echo "Running Pillar 4"

    python runners/runner.py --llm_model=$LLM_MODEL --pillar=4 --category=$CATEGORY_NAME --variation=1
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=4 --category=$CATEGORY_NAME --variation=2
    python runners/runner.py --llm_model=$LLM_MODEL --pillar=4 --category=$CATEGORY_NAME --variation=3

done