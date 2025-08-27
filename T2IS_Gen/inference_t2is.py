import torch
import argparse
import json
import os
from t2is_pipeline_flux import T2IS_FluxPipeline
from PIL import Image
from utils import calculate_layout_dimensions, calculate_cutting_layout

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    parser.add_argument('--output_dir', type=str, default="./output_images/", help="Output directory")
    parser.add_argument('--json_file', type=str, default="./data/gallery.json", help="Input JSON file")
    return parser.parse_args()

pipe = T2IS_FluxPipeline.from_pretrained("/home/chengyou/hugging/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

args = parse_arguments()

if args.idx is not None:
    # Load parameters from JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    item = data[args.idx]
    
    idx = item["idx"]
    task_name_case_id = item["task_name_case_id"] 
    prompt = item["prompt"]
    Divide_prompt_list = item["Divide_prompt_list"]
    
    # Calculate dimensions
    num_prompts = len(Divide_prompt_list)
    sub_height = 512
    sub_width = 512
    height, width = calculate_layout_dimensions(num_prompts, sub_height, sub_width)
    
    Divide_replace = 2
    seed = 1234

else:
    # Default parameters
    idx = "0001_0007"
    prompt = "THREE-PANEL Images with a 1x3 grid layout a teenage boy with short spiky black hair, a slight build, and dark brown eyes in hyper-realistic style.All images maintain hyper-realistic digital painting style with consistent character design, emphasizing the boy's distinct features and naturalistic lighting across varied environments. [LEFT]:The boy stands at a science fair, surrounded by project displays and glowing holographic models. He holds a blueprint, his expression bright with curiosity. The background features blurred crowds and colorful experiment stations under bright indoor lighting. [MIDDLE]:The boy crouches in a sunlit garden, digging soil with a trowel. Dirt stains his hands and casual clothes, with scattered gardening tools nearby. His focused gaze and slightly parted lips suggest discovery, sunlight casting sharp shadows on the earthy textures. [RIGHT]:The boy wears a green knitted hat in a snowy urban park, breath visible in cold air. Frosted trees frame the scene as he clutches a steaming drink. The hat's yarn details contrast with his spiky hair, while distant ice-skating figures blur into the winter haze."
    Divide_prompt_list = [
        "The boy stands at a science fair, surrounded by project displays and glowing holographic models. He holds a blueprint, his expression bright with curiosity. The background features blurred crowds and colorful experiment stations under bright indoor lighting.",
        "The boy crouches in a sunlit garden, digging soil with a trowel. Dirt stains his hands and casual clothes, with scattered gardening tools nearby. His focused gaze and slightly parted lips suggest discovery, sunlight casting sharp shadows on the earthy textures.",
        "The boy wears a green knitted hat in a snowy urban park, breath visible in cold air. Frosted trees frame the scene as he clutches a steaming drink. The hat's yarn details contrast with his spiky hair, while distant ice-skating figures blur into the winter haze."
    ]
    
    sub_height = 512
    sub_width = 512
    height, width = calculate_layout_dimensions(3, sub_height, sub_width)
    
    Divide_replace = 2
    seed = 1234

# Create output directory
base_output_path = args.output_dir
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

seed_output_path = os.path.join(base_output_path, f"seed_{seed}")
if not os.path.exists(seed_output_path):
    os.makedirs(seed_output_path)

print(f"Generating with seed {seed}:")
try:
    image = pipe(
        Divide_prompt_list=Divide_prompt_list,
        Divide_replace=Divide_replace,
        seed=seed,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=20,
        guidance_scale=3.5,
    ).images[0]
    
    # Save image
    output_filename = os.path.join(seed_output_path, f"{idx}_merge_seed{seed}.png")
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
except Exception as e:
    print(f"Error processing {idx} with seed {seed}: {str(e)}")
