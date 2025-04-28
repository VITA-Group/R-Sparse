# Environment Setup
conda env create -f env.yml
# task in winogrande piqa sciq openbookqa boolq arc_easy arc_challenge hellaswag


# Generate Low-Rank Weights (Offline)
python -u utils/prepare_low_rank_weight.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --output_dir ../low_rank_models/llama-3-8b

# Baseline: Full evaluation
GPU=0
model=meta-llama/Meta-Llama-3-8B
for task in piqa; do
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation.py \
    --tasks $task \
    --num_fewshot 0 \
    --model_name ${model} \
    --method full
done



# Baseline: Relufiction
GPU=0
model=meta-llama/Meta-Llama-3-8B
for task in piqa; do
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation.py \
    --tasks $task \
    --num_fewshot 0 \
    --model_name ${model} \
    --method relufiction
done


# R-Sparse
GPU=0
model=meta-llama/Meta-Llama-3-8B
for task in piqa; do
CUDA_VISIBLE_DEVICES=${GPU} python -u evaluation.py \
    --tasks $task \
    --num_fewshot 0 \
    --model_name ${model} \
    --method r_sparse \
    --sparse_config_file config/llama3_sparsity_50_evolutionary_search.npy \
    --config_file config/llama-3-8b_default.json
done









