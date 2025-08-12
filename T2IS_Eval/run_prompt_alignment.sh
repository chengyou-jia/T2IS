IMAGE_DIR="/home/chengyou/AutoT2IS/output_images/layout_deepseek-reasoner" # Target directory
OUTPUT_JSON="/home/chengyou/results/layout_deepseek-reasoner.json"  # Output JSON file path

cd /home/chengyou/clipscore/t2v_metrics
# Process specified directory
# see https://github.com/linzhiqiu/t2v_metrics for t2v_metrics

echo "Evaluating images in ${IMAGE_DIR}..."
CUDA_VISIBLE_DEVICES=0 python vqa_alignment.py \
    --image_dir "${IMAGE_DIR}" \
    --image_format png \
    --prompt_json "./ChengyouJia/T2IS-Bench/prompt_alignment.json" \
    --output_json "${OUTPUT_JSON}" \
    --start_idx 0

echo "Evaluation completed for ${IMAGE_DIR}!" 