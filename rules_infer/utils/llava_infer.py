import os
import json
import base64
import requests
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml
import traceback
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.helper import PredictHelper

from eventbank import run_qc_and_resolve

nusc = NuScenes(version='v1.0-trainval', dataroot='/data0/senzeyu2/dataset/nuscenes', verbose=False)
helper = PredictHelper(nusc)
maps_root = "/data0/senzeyu2/dataset/nuscenes"

EVENTS_JSON_PATH = 'social_events_2.json'
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'

EVENTS_BASE_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
# VLM Model ID
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# ==== 地图缓存 (性能关键) ====
# 避免在循环中重复加载几百兆的地图文件
map_cache = {}
def get_map_cached(location):
    if location not in map_cache:
        # tqdm.write(f"Loading map for {location}...")
        map_cache[location] = NuScenesMap(dataroot=maps_root, map_name=location)
    return map_cache[location]

PROMPT_TEMPLATE = r"""
You are an expert autonomous driving safety analyst and social behavior researcher.
Your task is to analyze a sequence of traffic scene images where an AI's trajectory prediction model has failed,
and infer the underlying *social interaction pattern* of the scene.

I will provide you with:
- A **Case File** in YAML format describing:
  - key agent and interacting agents' kinematics.
  - **short_id**: A short identifier (e.g., "1a2b") for each agent.
  - **environment**: Whether a traffic light structure is present at the location.
- A **sequence of images** (multiple frames × multiple camera views).
  - The **key agent** is in a **RED** box (marked "KEY" or with its ID).
  - **Interacting agents** are in **BLUE** boxes. 
  - **IMPORTANT**: The text ID on the BLUE box (e.g., "1a2b") CORRESPONDS EXACTLY to the `short_id` in the Case File. Use this to link the visual agent to its data.

You must use BOTH the structured context AND the visual evidence.

---
## Social Event Taxonomy (Closed Set)
You MUST treat the following list as the **closed-set label space** for `social_event`:

1. **Right-of-Way Exchange**
   - `intersection_yield`, `crosswalk_yield`, `roundabout_merge_yield`, `bus_stop_merge_yield`, `U_turn_yield`
2. **Competition / Merge**
   - `merge_compete`, `zipper_merge`
3. **Relative Trajectory Relations**
   - `cut_in`, `cut_out`, `follow_gap_opening`
4. **Obstacle & Bypassing**
   - `double_parking_avoidance`, `blocked_intersection_clear`, `dooring_avoidance`
5. **Pedestrian / Micromobility**
   - `jaywalking_response`, `bike_lane_merge_yield`
6. **Emergency & Courtesy**
   - `emergency_vehicle_yield`, `courtesy_stop`
7. **Congestion State**
   - `congestion_stop`, `queue_discharge`
8. **Compliance / Violation**
   - `red_light_stop`, `stop_sign_yield`, `priority_violation`
9. **Open Class**
   - `novel_event` (only if none of the above fit)

---

**Case File & Scene Context:**
Below is the **structured context** for this event, including agent dynamics and map/traffic-light information:

```yaml
[[SCENE_CONTEXT_YAML]]

Visual Evidence:
Images are ordered chronologically.
[[VISUAL_EVIDENCE_SECTION]]

**Reasoning Instructions (Chain of Thought):**

    **1: Describe Direct Observation (Phenomenon)**: Match the BLUE boxes/RED box in images to the short_id/'key' label in the YAML to understand their speed and acceleration,
    then try to describe exactly what happened， eg. The red vehicle accelerated, changed lanes, and overtook the car in front..
    
    **2: Identify the social event**: Based on the actual phenomena observed, identify the social event caused by/happened to the key agent 
    (1)First, evaluate ALL closed-set labels and compute a confidence score (0.0–1.0) for each relevant candidate.
    (2)If at least one closed-set label matches reasonably, choose the best one as social_event_primary.
    (3)If NO closed-set label fits well (all confidences are low), then:
        ·Set social_event_primary = "novel_event",
        ·Fill the novel_event object with:
            ·is_novel = true
            ·proposed_label = short free-text name (e.g. "parking_lot_reverse_negotiation")
            ·nearest_parent = one of the 8 parent groups listed above (e.g. "Right-of-Way Exchange", "Congestion State", etc.)
            ·rationale = short explanation of why this proposed label and parent were chosen.
         
            
    **3: Social/Behavioral Inference**: Infer the underlying social norm / implicit rule / driving culture: e.g. defensive driving, aggressive merge, courtesy stop, jaywalking tolerance, etc.
    
    ---
    
    **Output Format:**
    You MUST output one single JSON object. No markdown, no text outside JSON.
           
    {
    "direct_observation": "Natural language description of what happens in the scene.",
    "causal_inference": "Natural language explanation of the physical cause of the behavior / failure.",
    "social_event_primary": "ONE label from the closed set above, OR "novel_event"",
    "closed_set_candidates": [
    { "label": "intersection_yield", "confidence": 0.78 },
    { "label": "congestion_stop", "confidence": 0.21 }
    ],
    "novel_event": {
    "is_novel": false/true,
    "proposed_label": null,
    "nearest_parent": null,
    "rationale": null
    },
    "event_span": {"start_frame": 12,"end_frame": 24},
    "actors": [
    {
    "track_id": "key_agent_token_or_short_id",
    "type": "vehicle / pedestrian / cyclist / bus / emergency_vehicle / other",
    "role": "yielding / cutting_in / crossing / merging / blocking / leading / following / stopped / other"
    }
    ],
    "context": {
    "junction_type": "none / unsignalized / signalized / roundabout / merge / unknown",
    "tl_state": "red / yellow / green / no_light / unknown",
    "crosswalk_present": true,
    "traffic_density": "free_flow / moderate / congested / stop_and_go",
    "notes": "..."
    },
    "evidence": [
    "Short bullet-like phrases that justify your chosen social_event_primary.",
    "Refer to both frames and agent behaviors when possible."
    ],
    "confidence_overall": 0.83,
    }
"""


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_agent_dynamics(kinematics_list):
    dynamics = []

    for frame_data in kinematics_list:

        # 只处理包含有效运动学数据的帧
        if not frame_data:
            continue

        formatted_frame = {"frame": frame_data['frame']}

        # 速度 (speed)
        if 'speed' in frame_data:
            formatted_frame['velocity'] = round(frame_data['speed'], 2)  # m/s

        # 加速度 (acceleration magnitude)
        if 'acceleration' in frame_data and frame_data['acceleration'] is not None:
            accel_vec = np.array(frame_data['acceleration'])
            lon_acc = accel_vec[0]
            lat_acc = accel_vec[1]
            formatted_frame['lon_acc'] = round(float(lon_acc), 2)
            formatted_frame['lat_acc'] = round(float(lat_acc), 2)

        if frame_data.get('tl_present'):
            formatted_frame['tl_present'] = True

        # 偏航角速度 (yaw rate)
        if 'angular_velocity_yaw' in frame_data and frame_data['angular_velocity_yaw'] is not None:
            formatted_frame['yaw_rate'] = round(frame_data['angular_velocity_yaw'], 3)  # rad/s
        # 只有在提取到至少一项运动数据时才添加
        if len(formatted_frame) > 1:
            dynamics.append(formatted_frame)

    return dynamics


def get_traffic_light_status_for_scene(nusc, sample_token):
    """获取一个sample中所有交通灯的状态"""
    sample = nusc.get('sample', sample_token)
    light_statuses = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if 'traffic_light' in ann['category_name']:
            # 属性token指向一个attribute记录，该记录的name字段是 'vehicle_state.green' 等
            if ann['attribute_tokens']:
                attr_token = ann['attribute_tokens'][0]
                attr = nusc.get('attribute', attr_token)
                light_statuses.append(attr['name'].split('.')[-1])  # 提取 'green', 'red', 'yellow'

    # 实际应用中需要更复杂的逻辑来判断哪个灯对key-agent有效
    # 这里我们简化为报告场景中存在的所有灯的状态
    if not light_statuses:
        return "No traffic lights detected"
    return ", ".join(set(light_statuses))  # e.g., "green, red"


def get_sample_token_by_frame_index(nusc: NuScenes, scene: dict, frame_index: int) -> str:
    """
    Given a scene and a frame index, return the corresponding sample_token.

    :param nusc: NuScenes object.
    :param scene: A scene record from nusc.scene.
    :param frame_index: The zero-based index of the sample in the scene.
    :return: The sample_token.
    """
    if not (0 <= frame_index < scene['nbr_samples']):
        raise ValueError(f"Frame index {frame_index} is out of bounds for scene with {scene['nbr_samples']} samples.")

    sample_token = scene['first_sample_token']
    # Iterate through the linked list of samples
    for _ in range(frame_index):
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
        if not sample_token:  # Should not happen if frame_index is in bounds
            raise IndexError("Reached end of scene unexpectedly.")

    return sample_token

def build_scene_index(nusc: NuScenes, scene: dict) -> list[str]:
    """Builds a list of sample_tokens for a scene, indexed by frame number."""
    scene_index = []
    current_token = scene['first_sample_token']
    while current_token:
        scene_index.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
    return scene_index

def analyze_event(event, event_dir, tau=0.6):
    manifest_path = os.path.join(event_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        tqdm.write(f" [Warning] Manifest not found in {event_dir}. Skipping.")
        return
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    images_base64 = []
    visual_evidence_lines = []
    # 按帧号排序，以保证时序
    sorted_frames = sorted(manifest['frames'].keys())
    prompt_manifest = {"event_id": manifest["event_id"], "frames": {}}

    for frame_key in sorted_frames:
        frame_idx = int(frame_key.split("_")[1])
        visual_evidence_lines.append(f"**Frame {frame_idx:03d}:**")
        line_parts = []
        for image_filename in manifest['frames'][frame_key]:
            image_path = Path(event_dir) / image_filename

            if image_path.exists():
                images_base64.append(encode_image_to_base64(image_path))
                line_parts.append(f"{image_filename}:<image>")
            else:
                tqdm.write(f"  [Warning] Image {image_filename} not found in {event_dir}. Skipping image.")
        visual_evidence_lines.append("|".join(line_parts))
        visual_evidence_lines.append("")


    if not images_base64:
        tqdm.write(f"  [Warning] No valid images found for event {Path(event_dir).name}. Skipping event.")
        return

    peak_frame = event['peak_error_frame']
    # 查找 Key Agent 在 Peak Frame 的状态以确定环境
    key_agent_kinematics = event['kinematics']['key_agent']
    peak_state = next((f for f in key_agent_kinematics if f['frame'] == peak_frame), None)
    tl_at_peak = False
    if peak_state and peak_state.get('tl_present'):
        tl_at_peak = True

    event_kinematics = event['kinematics']

    context = {}
    key_agent_token = event['key_agent_token']
    context = {
        "event_id": manifest['event_id'],
        "meta": {
            "peak_frame": peak_frame,
            "traffic_light_structure_present": tl_at_peak,
            "instruction": "Verify TL color visually if structure_present is true."
        },
        "key_agent": {
            "short_id": event.get('key_agent_short_id', "KEY"),
            "token": event['key_agent_token'],
            "dynamics_snippet": get_agent_dynamics(key_agent_kinematics)
        },
        "interacting_agents": []
    }

    # 填充交互 Agents
    inter_short_ids = event.get('interacting_agents_short_ids', {})
    for agent_token, k_list in event['kinematics']['interacting_agents'].items():
        s_id = inter_short_ids.get(agent_token, agent_token[:4])
        context['interacting_agents'].append({
            "short_id": s_id,
            "dynamics_snippet": get_agent_dynamics(k_list)
        })


    scene_context_yaml = yaml.dump(context, indent=2, sort_keys=False)
    visual_evidence_str = "\n".join(visual_evidence_lines)

    final_prompt = PROMPT_TEMPLATE.replace("[[SCENE_CONTEXT_YAML]]",scene_context_yaml)
    final_prompt = final_prompt.replace("[[VISUAL_EVIDENCE_SECTION]]", visual_evidence_str)

    # tqdm.write(f"the final prompt is {final_prompt}")

    if final_prompt.count('<image>') != len(images_base64):
        tqdm.write(f"[ERROR] Mismatch in event {event_dir}: "
                   f"{final_prompt.count('<image>')} placeholders vs "
                   f"{len(images_base64)} images.")
        return False, None

    payload = {
        "model": MODEL_NAME,
        "prompt": final_prompt,
        "images": images_base64,  # 一次性提交所有图片的Base64编码
        "stream": False,
        "format": "json",  # 请求Ollama直接返回JSON格式

    }

    try:
        # 使用较长的超时时间，因为处理多张图片可能很慢
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()

        # 解析响应
        resp_json = response.json()
        content = resp_json.get('response', resp_json)
        if isinstance(content, str):
            content = json.loads(content)
        # 5. QC & Post-processing
        final_result = run_qc_and_resolve(content, tau=tau)

        # Save
        output_path = Path(event_dir) / 'vlm_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(final_result, f, indent=4)

        return True, output_path

    # except requests.exceptions.RequestException as e:
    #     tqdm.write(f"  [Error] API request failed for {event_dir.name}: {e}")
    # except json.JSONDecodeError:
    #     tqdm.write(f"  [Error] Failed to parse VLM response for {event_dir}. Raw response: {final_result}")
    except Exception as e:
        tqdm.write(f"  [Error] An unexpected error occurred for {event_dir}: {e}\n{traceback.format_exc()}")

    return False, None

if __name__=="__main__":
    event_dirs = [d for d in Path(EVENTS_BASE_DIR).iterdir() if d.is_dir()]

    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events = json.load(f)

    for event in tqdm(all_events):
        scene_token = event['scene_token']
        key_agent_token = event['key_agent_token']
        start_frame, end_frame = event['event_start_frame'], event['event_end_frame']
        event_id = f"{scene_token}_{key_agent_token}_{start_frame}-{end_frame}"
        event_output_dir = os.path.join(OUTPUT_DIR, event_id)
        if not os.path.exists(event_output_dir):
            tqdm.write(f" not found event {event_id}. Skipping.")
            continue

        if os.path.exists(os.path.join(event_output_dir, 'vlm_analysis.json')):
            tqdm.write(f"Event {event_id} already analyzed. Skipping.")
            continue

        result = analyze_event(event, event_output_dir)
        if result != None:
            success, result_path = result
            if success:
                tqdm.write(f"  -> Analysis successful. Result saved to {result_path}")
        else:
            tqdm.write("not success")


    print("\nVLM analysis complete.")
