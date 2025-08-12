# T2IS Evaluation Scripts

This directory contains evaluation scripts for Text-to-ImageSet (T2IS) models, including prompt alignment and visual consistency evaluation.

## Prerequisites

Before running the evaluation scripts, you need to download the T2IS-Bench dataset from Hugging Face:

1. Visit [ChengyouJia/T2IS-Bench](https://huggingface.co/datasets/ChengyouJia/T2IS-Bench) on Hugging Face
2. Download the following files:
   - `T2IS-Bench.json`: Main dataset file containing prompts and metadata
   - `prompt_alignment.json`: Alignment evaluation criteria
   - `prompt_consistency.json`: Consistency evaluation criteria
3. Place the downloaded files in a directory named `ChengyouJia/T2IS-Bench/` relative to the evaluation scripts


## Scripts Overview

### 1. Prompt Alignment Evaluation (`run_prompt_alignment.sh`)

**Purpose**: Evaluates the alignment between generated images and their corresponding prompts using VQAScore metrics.

**Features**:
- Uses VQAScore to measure semantic similarity between images and text prompts
- Processes PNG images from a specified directory
- Outputs evaluation results in JSON format
- Configurable start index for batch processing

**Usage**:
```bash
# Make the script executable
chmod +x run_prompt_alignment.sh

# Run the evaluation
./run_prompt_alignment.sh
```

**Configuration**:
- **Image Directory**: Set `IMAGE_DIR` to the path containing your generated images
- **Output Path**: Set `OUTPUT_JSON` to where you want to save the results
- **Image Format**: Currently set to PNG format
- **Start Index**: Set `start_idx` for batch processing (default: 0)

**Requirements**:
- See details in https://github.com/linzhiqiu/t2v_metrics.

### 2. Prompt Consistency Evaluation (`run_prompt_consistency.sh`)

**Purpose**: Evaluates the consistency of generated images with their prompts using Qwen model analysis.

**Features**:
- Uses Qwen2.5VL model to analyze image-prompt consistency
- Processes images with configurable pattern matching
- Outputs detailed consistency analysis in JSON format
- Supports various image formats through pattern matching

**Usage**:
```bash
# Make the script executable
chmod +x run_prompt_consistency.sh

# Run the evaluation
./run_prompt_consistency.sh
```

**Configuration**:
- **Image Directory**: Set `IMAGE_DIR` to the path containing your generated images
- **Image Pattern**: Set `IMAGE_PATTERN` to match your image files (e.g., ".png", ".jpg")
- **Output Path**: Set `OUTPUT_JSON` to where you want to save the results

**Requirements**:
- See details in https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.
