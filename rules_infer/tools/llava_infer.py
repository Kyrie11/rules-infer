import os
import json
import base64
import requests
from tqdm import tqdm
from pathlib import Path

EVENTS_BASE_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
# VLM Model ID
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

PROMPT_TEMPLATE = """
You are an expert autonomous driving safety analyst and social behavior researcher. Your task is to analyze a sequence of traffic scene images where an AI's trajectory prediction model has failed, and deduce the underlying social reasons for the failure.

I will provide you with a "Case File" in JSON format that describes the event, and a series of corresponding images.
- The Case File lists the images available for each frame of the event.
- The key agent whose trajectory was mispredicted is highlighted in a RED box.
- Any other relevant interacting agents are in BLUE boxes.

**Your Goal:**
Follow a strict Chain of Thought to explain the failure. You MUST structure your analysis in three distinct steps as detailed below.

**Chain of Thought Instructions:**
    
    **Step 1: Direct Observation (Phenomenon)**
    - **Task:** Based on the sequence of provided images (ordered by frame number), objectively describe what is happening. Focus on the actions of the key agent (RED box) and its interactions with other agents (BLUE boxes) or the environment (e.g., traffic lights, pedestrians, road layout).
    
    **Step 2: Causal Inference (Physical Reason)**
    - **Task:** Based on your observations, infer the immediate, direct, physical-world cause for the key agent's behavior. For example: "The key agent braked suddenly because a pedestrian stepped onto the crosswalk."
    
    **Step 3: Social/Behavioral Inference (Implicit Rule)**
    - **Task:** Go one level deeper. Analyze the social context. What underlying social norm, local driving culture, or unwritten rule explains why the causal event happened? For example: "The pedestrian crossed against their red light, but the driver chose to yield anyway. This suggests a local driving culture of 'defensive driving' or 'prioritizing pedestrian safety over right-of-way'."
    
    ---
    
    **Output Format:**
    You MUST provide your response in a single, clean, and parsable JSON object. Do not include any text outside of this JSON block. Use the following structure:
        
    {{
      "analysis": {{
        "direct_observation": "...",
        "causal_inference": "...",
        "social_inference": "..."
      }},
      "implicit_rule_summary": "A concise, one-sentence summary of the discovered implicit social rule or behavior."
    }}

"""

def analyze_event(event_dir):
    manifest_path = event_dir / 'manifest.json'
    if not manifest_path.exists():
        tqdm.write(f" [Warning] Manifest not found in {event_dir}. Skipping.")
    return
    print("correct")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    images_base64 = []
    # 按帧号排序，以保证时序
    print("sort frames")
    sorted_frames = sorted(manifest['frames'].keys())
    for frame_key in sorted_frames:
        for image_filename in manifest['frames'][frame_key]:
            image_path = event_dir / image_filename
            if image_path.exists():
                images_base64.append(encode_image_to_base64(image_path))
            else:
                tqdm.write(f"  [Warning] Image {image_filename} not found in {event_dir}. Skipping image.")

    if not images_base64:
        tqdm.write(f"  [Warning] No valid images found for event {event_dir.name}. Skipping event.")
        return

    final_prompt = PROMPT_TEMPLATE.format(case_file_json=json.dumps(prompt_manifest, indent=2))

    payload = {
        "model": MODEL_NAME,
        "prompt": final_prompt,
        "images": images_base64,  # 一次性提交所有图片的Base64编码
        "stream": False,
        "format": "json"  # 请求Ollama直接返回JSON格式
    }

    try:
        # 使用较长的超时时间，因为处理多张图片可能很慢
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()

        response_data = response.json()

        # 'response'字段包含了模型生成的JSON字符串，需要再次解析
        analysis_content_str = response_data.get('response')
        if not analysis_content_str:
            raise ValueError("VLM returned an empty response.")

        analysis_content = json.loads(analysis_content_str)

        # 5. 保存分析结果
        output_path = event_dir / 'vlm_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis_content, f, indent=4)
            tqdm.write("return corrected result")
        return True, output_path

    except requests.exceptions.RequestException as e:
        tqdm.write(f"  [Error] API request failed for {event_dir.name}: {e}")
    except json.JSONDecodeError:
        tqdm.write(f"  [Error] Failed to parse VLM response for {event_dir.name}. Raw response: {analysis_content_str}")
    except Exception as e:
        tqdm.write(f"  [Error] An unexpected error occurred for {event_dir.name}: {e}\n{traceback.format_exc()}")

    return False, None

if __name__=="__main__":
    event_dirs = [d for d in Path(EVENTS_BASE_DIR).iterdir() if d.is_dir()]
    for event_dir in tqdm(event_dirs, desc="Analyzing Events with VLM"):
        if (event_dir / 'vlm_analysis.json').exists():
            tqdm.write(f"Event {event_dir.name} already analyzed. Skipping.")
            continue

        tqdm.write(f"\nProcessing event: {event_dir.name}")
        success, result_path = analyze_event(event_dir)
        if success:
            tqdm.write(f"  -> Analysis successful. Result saved to {result_path}")

    print("\nVLM analysis complete.")
