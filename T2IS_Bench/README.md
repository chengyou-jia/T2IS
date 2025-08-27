See https://huggingface.co/datasets/ChengyouJia/T2IS-Bench for more details.


### Dataset Overview

**T2IS-Bench** is a comprehensive benchmark designed to evaluate generative models' performance in text-to-image set generation tasks. It includes **596 carefully constructed tasks** across **five major categories** （26 sub-categories）, each targeting different aspects of set-level consistency such as identity preservation, style uniformity, and logical coherence. These tasks span a wide range of real-world applications, including character creation, visual storytelling, product mockups, procedural illustrations, and instructional content.

T2IS-Bench provides a scalable evaluation framework that assesses image sets across **three critical consistency dimensions**: identity, style, and logic. Each of the **596 tasks** is paired with structured natural language instructions and evaluated using **LLM-driven criteria generation**, enabling automatic, interpretable, and fine-grained assessment. This design supports benchmarking generative models' ability to produce coherent visual outputs beyond prompt-level alignment, and reflects real-world requirements for controllability and consistency in multi-image generation.



### Supported Tasks

The dataset comprises five main categories, each with a set of associated tasks and unique task IDs as listed below:

#### **Character Generation**

- `0001` – Multi-Scenario
- `0002` – Multi-Expression
- `0003` – Portrait Design
- `0004` – Multi-view
- `0005` – Multi-pose

#### **Design Style Generation**

- `0006` – Creative Style
- `0007` – Poster Design
- `0008` – Font Design
- `0009` – IP Product
- `0010` – Home Decoration

#### **Story Generation**

- `0011` – Movie Shot
- `0012` – Comic Story
- `0013` – Children Book
- `0014` – News Illustration
- `0015` – Hist. Narrative

#### **Process Generation**

- `0016` – Growth Process
- `0017` – Draw Process
- `0018` – Cooking Process
- `0019` – Physical Law
- `0020` – Arch. Building
- `0021` – Evolution Illustration

#### **Instruction Generation**

- `0022` – Education Illustration
- `0023` – Historical Panel
- `0024` – Product Instruction
- `0025` – Travel Guide
- `0026` – Activity Arrange



### Use Cases

**T2IS-Bench** is designed for evaluating generative models on multi-image consistency tasks, testing capabilities such as aesthetics, prompt alignment (including entity, attribute, and relation understanding), and visual consistency (covering identity, style, and logic) across image sets. It is suitable for benchmarking text-to-image models, diffusion transformers, and multimodal generation systems in real-world applications like product design, storytelling, and instructional visualization.

## Dataset Format and Structure

### Data Organization

1. **`T2IS-Bench.json`**  
   A JSON file providing all of the cases. The structure of `T2IS-Bench.json` is as follows:

```json
{    
    ......
    "0018_0001": {
        "task_name": "Cooking Process",
        "num_of_cases": 27,
        "uid": "0018",
        "output_image_count": 4,
        "case_id": "0001",
        "task_name_case_id": "cooking_process_0001",
        "category": "Process Generation",
        "instruction": "Please provide a detailed guide on melting chocolate, including 4 steps. For each step, generate an image.",
        "sub_caption": [
            "A glass bowl filled with chopped dark chocolate pieces sits on top of a pot of simmering water. Steam rises gently around the bowl, and a thermometer is visible in the chocolate. The kitchen counter shows other baking ingredients in the background.",
            "Hands holding a silicone spatula are gently stirring melting chocolate in a glass bowl. The chocolate is partially melted, with some pieces still visible. The bowl is positioned over a steaming pot on a stovetop.",
            "A close-up view of a digital thermometer inserted into fully melted, glossy chocolate. The thermometer display shows a temperature of 88°F (31°C). The melted chocolate has a rich, dark color and smooth texture.",
            "A hand is seen removing the bowl of melted chocolate from the double boiler setup. The chocolate appears smooth and shiny. Next to the stove, various dessert items like strawberries, cookies, and a cake are ready for dipping or coating."
        ]
    }
	...... 
}
```

- task_name: Name of the task.
- num_of_cases: The number of individual cases in the task.
- uid: Unique identifier for the task.
- output_image_count: Number of images expected as output.
- case_id: Identifier for this case. 
- task_name_case_id: Unique identifier for each specific case within a task, combining the task name and case ID.
- category: The classification of the task.
- instruction: The task's description, specifying what needs to be generated.
- sub_caption: Descriptions for each image in the task by feeding instruction into LLM.



2. **`prompt_alignment_criterion.json`**

   This file contains evaluation criteria for assessing prompt alignment in image generation tasks. Each entry corresponds to a specific task and is organized by steps, with each step evaluated based on three key aspects: **Entity**, **Attribute**, and **Relation**.

   - **Entity** defines the key objects or characters required in the scene.
   - **Attribute** describes the properties or conditions that these entities must possess.
   - **Relation** outlines how the entities interact or are positioned within the scene.

   This structured format helps evaluate the accuracy of the generated images in response to specific prompts.

3. **`prompt_consistency_criterion.json`**

   This file defines evaluation criteria for assessing *intra-sequence consistency* in image generation tasks. Each entry corresponds to a specific task and outlines standards across three core aspects: **Style**, **Identity**, and **Logic**.

   - **Style** evaluates the visual coherence across all generated images, including consistency in rendering style, color palette, lighting conditions, and background detail. It ensures that all images share a unified artistic and atmospheric aesthetic.

   - **Identity** focuses on maintaining character integrity across scenes. This includes preserving key facial features, body proportions, attire, and expressions so that the same individual or entity is clearly represented throughout the sequence.

   - **Logic** ensures semantic and physical plausibility across images. This includes spatial layout consistency, realistic actions, appropriate interactions with the environment, and coherent scene transitions.

     This structured format enables a systematic evaluation of how well generated images maintain consistency within a task.
