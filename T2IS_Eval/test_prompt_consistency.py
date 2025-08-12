import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig

def load_model(model_path):
    # Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor

def evaluate_prompt_consistency(dataset_path, criteria_path, image_base_path, image_pattern, output_path=None):
    # Load the model
    model, processor = load_model("./Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load the filtered dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"Total number of cases in dataset: {len(dataset)}")

    # Load the prompt consistency criteria
    with open(criteria_path, 'r') as f:
        criteria_data = json.load(f)
    print(f"Loaded prompt consistency criteria")

    # Get folder name for output file
    folder_name = os.path.basename(image_base_path)
    print(f"\n{'='*50}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*50}\n")

    # Set output path if not provided
    if output_path is None:
        output_path = f"evaluation_scores_{folder_name}.json"
    
    # Initialize results
    results = {}
    
    # Check if results file already exists and load it
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results from {output_path}")
    else:
        results = {
            "case_results": {}
        }

    # Process each case in the dataset
    for case_id, case_data in tqdm(dataset.items(), desc=f"Processing cases"):
        # Skip if this case has already been processed
        if case_id in results["case_results"]:
            print(f"\nSkipping already processed case: {case_id}")
            continue
    
        print(f"\nProcessing case: {case_id}")

        # Get criteria for this case
        case_criteria = criteria_data.get(case_id, {})
        if not case_criteria:
            print(f"Warning: No criteria found for case {case_id}. Skipping.")
            continue

        # Get all image paths for this case based on the provided pattern
        all_files = os.listdir(image_base_path)
        image_paths = []
        
        # Parse the image pattern to extract case_id and file extension
        ext = image_pattern
        
        for f in all_files:
            if f.startswith(case_id) and f.endswith(ext):
                try:
                    # Extract the last number from filename (e.g., 0001 from 0001_0001_0001.png)
                    num = int(f.split('_')[-1].replace(ext, ''))
                    image_paths.append((num, os.path.join(image_base_path, f)))
                except ValueError:
                    continue

        # Sort by the extracted number and get only the paths
        image_paths = [path for _, path in sorted(image_paths)]

        if len(image_paths) == 0:
            print(f"Warning: Expected {case_data['output_image_count']} images, but found {len(image_paths)}. Skipping this case.")
            continue

        # Initialize dimension scores
        dimension_scores = {
            "Style": {"scores": [], "criteria": []},
            "Identity": {"scores": [], "criteria": []},
            "Logic": {"scores": [], "criteria": []}
        }

        # Process each dimension
        for dimension in ["Style", "Identity", "Logic"]:
            if dimension not in case_criteria:
                continue

            # Get criteria for this dimension
            dimension_criteria = case_criteria[dimension][0]  # Get the first (and only) dictionary in the list
            dimension_scores[dimension]["criteria"] = list(dimension_criteria.values())

            # Process each criterion in this dimension
            for criterion_text in dimension_criteria.values():
                softmax_values = []
                
                # Compare each pair of images
                for i in range(len(image_paths)):
                    for j in range(i + 1, len(image_paths)):
                        messages = []
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_paths[i], "resized_height": 512, "resized_width": 512},
                                    {"type": "image", "image": image_paths[j], "resized_height": 512, "resized_width": 512},
                                    {"type": "text", "text": f"Do images meet the following criteria? {criterion_text} Please answer Yes or No."},
                                ],
                            }
                        )

                        # Prepare for inference
                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")

                        generated_ids = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, output_logits=True)
                        sequences = generated_ids.sequences
                        scores = generated_ids.scores
                        logits = generated_ids.logits

                        no_logits = logits[0][0][2753]
                        yes_logits = logits[0][0][9454]

                        # Calculate softmax
                        logits = torch.tensor([no_logits, yes_logits])
                        softmax = torch.nn.functional.softmax(logits, dim=0)
                        yes_softmax = softmax[1].item()

                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sequences)
                        ]

                        output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )

                        # Append the current yes_softmax value to the list
                        softmax_values.append(yes_softmax)

                # Calculate average score for this criterion
                if softmax_values:
                    average_softmax = sum(softmax_values) / len(softmax_values)
                    dimension_scores[dimension]["scores"].append(average_softmax)

        # Calculate overall scores for each dimension
        dimension_averages = {}
        for dimension, data in dimension_scores.items():
            if data["scores"]:
                dimension_averages[dimension] = sum(data["scores"]) / len(data["scores"])
            else:
                dimension_averages[dimension] = 0.0

        # Calculate overall average across all dimensions
        overall_average = sum(dimension_averages.values()) / len(dimension_averages) if dimension_averages else 0.0

        # Store the results for this case
        results["case_results"][case_id] = {
            "task_name_case_id": case_data["task_name_case_id"],
            "num_images": case_data["output_image_count"],
            "dimension_scores": dimension_scores,
            "dimension_averages": dimension_averages,
            "overall_average": overall_average
        }

        print(f"\nScores for case {case_id}:")
        for dimension, avg in dimension_averages.items():
            print(f"{dimension} average: {avg:.4f}")
        print(f"Overall average: {overall_average:.4f}")
        
        # Save results after each case is processed
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Updated results saved to: {output_path}")

    # Final save of results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nFinal results saved to: {output_path}")
    
    # Calculate overall dimension scores across all cases
    print(f"\nCalculating overall dimension scores...")
    
    # Initialize dimension totals
    dimension_totals = {}
    dimension_counts = {}
    
    # Aggregate scores across all cases
    for case_id, case_result in results["case_results"].items():
        for dimension, avg in case_result["dimension_averages"].items():
            if dimension not in dimension_totals:
                dimension_totals[dimension] = 0.0
                dimension_counts[dimension] = 0
            if avg > 0:  # Only add non-zero scores
                dimension_totals[dimension] += avg
                dimension_counts[dimension] += 1
    
    # Calculate final averages for each dimension
    final_dimension_averages = {}
    for dimension, total in dimension_totals.items():
        if dimension_counts[dimension] > 0:
            final_dimension_averages[dimension] = total / dimension_counts[dimension]
        else:
            final_dimension_averages[dimension] = 0.0
    
    # Calculate overall average across all dimensions
    final_overall_average = sum(final_dimension_averages.values()) / len(final_dimension_averages) if final_dimension_averages else 0.0
    
    # Add final scores to results
    results["final_dimension_averages"] = final_dimension_averages
    results["final_overall_average"] = final_overall_average
    
    # Save updated results with final scores
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final scores
    print(f"\nFinal dimension scores:")
    for dimension, avg in final_dimension_averages.items():
        print(f"{dimension} final average: {avg:.4f}")
    print(f"Final overall average: {final_overall_average:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt consistency for generated images")
    parser.add_argument("--dataset_path", type=str, default="/home/chengyou/GroupGen/baseline/filtered_responses_sorted.json", help="Path to the dataset JSON file")
    parser.add_argument("--criteria_path", type=str, default="/home/chengyou/Qwen/prompt_consistency.json", help="Path to the prompt consistency criteria JSON file")
    parser.add_argument("--image_base_path", type=str, default="/home/chengyou/sx/Gemini/gemini", help="Path to the directory containing generated images")
    parser.add_argument("--image_pattern", type=str, default="jpg", help="Pattern for image filenames, use {} as placeholder for case_id")
    parser.add_argument("--output_path", type=str, help="Path to save the evaluation results")
    
    args = parser.parse_args()
    
    evaluate_prompt_consistency(
        args.dataset_path,
        args.criteria_path,
        args.image_base_path,
        args.image_pattern,
        args.output_path
    )

if __name__ == "__main__":
    main()
