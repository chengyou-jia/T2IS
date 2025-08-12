import t2v_metrics
import json
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
dimensions = ['Entity', 'Attribute', 'Relation']
# Initialize the scoring model
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')



# Function to get image path from task_id and sub_id
def get_image_path(task_id, sub_id, image_dir, image_format):
    return os.path.join(image_dir, f"{task_id}_{sub_id}.{image_format}")

# Function to evaluate a single image against its criteria
def evaluate_image(image_path, criteria):
    if not os.path.exists(image_path):
        return None
    
    # Load the image
    image = Image.open(image_path)
    
    # Prepare texts for each dimension
    texts = []
    for dimension in dimensions:
        texts.extend(criteria[dimension])
    
    # Calculate scores
    scores = clip_flant5_score(images=[image_path], texts=texts)
    
    # Calculate average score for each dimension
    dimension_scores = {}
    start_idx = 0
    for dimension in dimensions:
        num_criteria = len(criteria[dimension])
        dimension_scores[dimension] = sum(scores[0][start_idx:start_idx + num_criteria]) / num_criteria
        start_idx += num_criteria
    
    return dimension_scores

# Main evaluation loop
def main_evaluation(image_dir, image_format, prompt_json, output_json, start_idx=1):

    # Load the prompt alignment JSON file 
    with open(prompt_json, 'r') as f:
        prompt_data = json.load(f)
    
    overall_dimension_scores = {dimension: 0 for dimension in dimensions}
    total_images = 0
    results = {}

    # Load existing results from the JSON file if it exists
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for task_id, task_data in prompt_data.items():
        print(f"\nEvaluating task: {task_id}")
        
        for idx, (sub_id, criteria) in enumerate(task_data.items(), start=start_idx):
            formatted_sub_id = f"{idx:04d}"
            result_key = f"{task_id}_{formatted_sub_id}"
            
            # Skip processing if the result already exists
            if result_key in results:
                print(f"Skipping already processed image: {result_key}")
                continue
            
            image_path = get_image_path(task_id, formatted_sub_id, image_dir, image_format)
            print(f"\nEvaluating image: {image_path}")
            
            scores = evaluate_image(image_path, criteria)
            if scores:
                print("Dimension scores:")
                for dimension, score in scores.items():
                    print(f"{dimension}: {score:.4f}")
                    overall_dimension_scores[dimension] += score
                total_images += 1
                results[result_key] = scores
                
                # Save the updated results to the JSON file after each image
                with open(output_json, 'w') as f:
                    json.dump(results, f, indent=4, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)
            else:
                print("Image not found")

    # Calculate overall average scores for each dimension
    if total_images > 0:
        for dimension in overall_dimension_scores:
            overall_dimension_scores[dimension] /= total_images

        print("\nOverall average dimension scores:")
        for dimension, score in overall_dimension_scores.items():
            print(f"{dimension}: {score:.4f}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate image dimensions.")
    parser.add_argument("--image_dir", type=str, default="/home/chengyou/GroupGen/baseline/results/flux/", help="Directory containing the images.")
    parser.add_argument("--image_format", type=str, default="png", choices=["png", "jpg"], help="Format of the images.")
    parser.add_argument("--output_json", type=str, default="dimension_scores.json", help="Path to the output JSON file.")
    parser.add_argument("--start_idx", type=int, default=1, help="Starting index for image enumeration.")
    parser.add_argument("--prompt_json", type=str, default="/home/chengyou/GroupGen/evaluation/prompt_alignment.json", help="Path to the prompt alignment JSON file.")
    args = parser.parse_args()

    main_evaluation(args.image_dir, args.image_format, args.prompt_json, args.output_json, args.start_idx)