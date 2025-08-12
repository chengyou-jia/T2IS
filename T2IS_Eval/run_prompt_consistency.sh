# Set basic parameters
IMAGE_DIR="/home/chengyou/AutoT2IS/output_images/layout_deepseek-reasoner"  # Target directory
IMAGE_PATTERN=".png"  # Image format
OUTPUT_JSON="/home/chengyou/Qwen/results/layout_deepseek-reasoner.json"  # Output JSON file path

# Process specified directory
echo "Evaluating images in ${IMAGE_DIR}..."
CUDA_VISIBLE_DEVICES=0 python test_prompt_consistency.py \
    --dataset_path "./ChengyouJia/T2IS-Bench/T2IS-Bench.json" \ 
    --criteria_path "./ChengyouJia/T2IS-Bench/prompt_consistency.json" \
    --image_base_path "${IMAGE_DIR}" \
    --image_pattern "${IMAGE_PATTERN}" \
    --output_path "${OUTPUT_JSON}"
echo "Evaluation completed for ${IMAGE_DIR}!" 