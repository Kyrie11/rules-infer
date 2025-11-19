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

EVENTS_JSON_PATH = 'social_events.json'
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'

EVENTS_BASE_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
# VLM Model ID
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

PROMPT_TEMPLATE = r"""
You are an expert autonomous driving safety analyst and social behavior researcher.
Your task is to analyze a sequence of traffic scene images where an AI's trajectory prediction model has failed,
and infer the underlying *social interaction pattern* of the scene.

I will provide you with:
- A **Case File** in YAML/JSON-like format describing:
  - key agent and interacting agents' kinematics over time,
  - basic map/traffic-light context at the peak error frame.
- A **sequence of images** (multiple frames × multiple camera views).
  - The **key agent** whose trajectory was mispredicted is highlighted in a **RED** box.
  - Any **interacting agents** are highlighted in **BLUE** boxes. Every BLUE box in the images has a small text ID like "1a2b". This ID exactly matches the short_id field in the Case File for each interacting agent.

You must use BOTH the structured context AND the visual evidence.

---
## Social Event Taxonomy (Closed Set)
You MUST treat the following list as the **closed-set label space** for `social_event`:

1. **Right-of-Way Exchange**
   - `intersection_yield`
   - `crosswalk_yield`
   - `roundabout_merge_yield`
   - `bus_stop_merge_yield`
   - `U_turn_yield`

2. **Competition / Merge**
   - `merge_compete`
   - `zipper_merge`

3. **Relative Trajectory Relations**
   - `cut_in`
   - `cut_out`
   - `follow_gap_opening`

4. **Obstacle & Bypassing**
   - `double_parking_avoidance`
   - `blocked_intersection_clear`
   - `dooring_avoidance`

5. **Pedestrian / Micromobility**
   - `jaywalking_response`
   - `bike_lane_merge_yield`

6. **Emergency & Courtesy**
   - `emergency_vehicle_yield`
   - `courtesy_stop`

7. **Congestion State**
   - `congestion_stop`
   - `queue_discharge`

8. **Compliance / Violation**
   - `red_light_stop`
   - `stop_sign_yield`
   - `priority_violation`

9. **Open Class**
   - `novel_event` (used only when the above labels do not fit; see instructions below)

You MUST NOT invent new closed-set labels outside the list above.
If no label in this closed set fits well, you will use the `novel_event` mechanism.
---

**Case File & Scene Context:**
Below is the **structured context** for this event, including agent dynamics and map/traffic-light information:

```yaml
{scene_context_yaml}


Below is the visual evidence for the event.
Images are ordered chronologically by frame index. Each token like image_XXX:<image> corresponds to one image that you can see:

{visual_evidence_section}


**Your Goal:**
You must internally go through the following reasoning steps, but do NOT include these steps explicitly in the final JSON (only include the final summarized fields):
**Chain of Thought Instructions:**
    
    **Step 1: Direct Observation (Phenomenon)**
    - **Task:** Describe what is happening in the sequence:
        (1)motion of the key agent (RED box),
        (2)interactions with BLUE agents,
        (3)road geometry, crosswalks, intersections, lane structure,
        (4)any traffic lights, signs, or blockages that matter.
    
    **Step 2: Causal Inference (Physical Reason)**
    - **Task:** Infer the immediate physical-world cause of the key agent's behavior and/or the model failure. e.g. "The key agent brakes because another car cuts in front."
    
    **Step 3: Social/Behavioral Inference (Implicit Rule)**
    - **Task:** Infer the underlying social norm / implicit rule / driving culture: e.g. defensive driving, aggressive merge, courtesy stop, jaywalking tolerance, etc.
    
    **Step 3: Social Event Classification (Closed Set + Open World)**
    - **Task:** Using the taxonomy above:
    (1)First, evaluate ALL closed-set labels and compute a confidence score (0.0–1.0) for each relevant candidate.
    (2)If at least one closed-set label matches reasonably, choose the best one as social_event_primary.
    (3)If NO closed-set label fits well (all confidences are low), then:
        ·Set social_event_primary = "novel_event",
        ·Fill the novel_event object with:
            ·is_novel = true
            ·proposed_label = short free-text name (e.g. "parking_lot_reverse_negotiation")
            ·nearest_parent = one of the 8 parent groups listed above (e.g. "Right-of-Way Exchange", "Congestion State", etc.)
            ·rationale = short explanation of why this proposed label and parent were chosen.
    ---
    
    **Output Format:**
    You MUST output one single JSON object and NOTHING else (no markdown, no commentary).
    
    Use exactly this schema (keys MUST exist; you may set null or empty lists if unknown):        
    {
    "direct_observation": "Natural language description of what happens in the scene.",
    "causal_inference": "Natural language explanation of the physical cause of the behavior / failure.",
    "social_event_primary": "ONE label from the closed set above, OR "novel_event"",
    "closed_set_candidates": [
    { "label": "intersection_yield", "confidence": 0.78 },
    { "label": "congestion_stop", "confidence": 0.21 }
    ],
    "novel_event": {
    "is_novel": false,
    "proposed_label": null,
    "nearest_parent": null,
    "rationale": null
    },
    "event_span": {
    "start_frame": 12,
    "end_frame": 24
    },
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
    "notes": "any relevant high-level scene context"
    },
    "evidence": [
    "Short bullet-like phrases that justify your chosen social_event_primary.",
    "Refer to both frames and agent behaviors when possible."
    ],
    "map_refs": {
    "lane_ids": ["optional_lane_id_1", "optional_lane_id_2"],
    "crosswalk_ids": ["optional_crosswalk_id"],
    "other_map_features": []
    }, 
    
    【=="confidence_overall": 0.83,
    "alternatives": [
    "zipper_merge",
    "queue_discharge"
    ]
    }
    
    Important:
        (1)All strings must be double-quoted.
        (2)Booleans must be true or false.
        (3)Do NOT include comments or trailing commas.
        (4)Do NOT wrap the JSON in markdown.
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

        # 偏航角速度 (yaw rate)
        if 'angular_velocity_yaw' in frame_data and frame_data['angular_velocity_yaw'] is not None:
            formatted_frame['yaw_rate'] = round(frame_data['angular_velocity_yaw'], 3)  # rad/s
        # 只有在提取到至少一项运动数据时才添加
        if len(formatted_frame) > 1:
            dynamics.append(formatted_frame)

    return dynamics


def get_map_context_for_agent(nusc, nusc_map, helper, agent_token, scene_token, frame):
    """获取agent在某一帧的地图上下文"""
    try:
        scene = nusc.get('scene', scene_token)
        sample_token = helper.get_sample_token_for_scene(scene_token, frame)
        ann = helper.get_sample_annotation(agent_token, sample_token)

        pose = ann['translation'][:2]  # x, y

        # 检查是否在可行驶区域/路口
        is_on_drivable = nusc_map.is_on_layer(pose[0], pose[1], 'drivable_area')
        is_at_intersection = nusc_map.is_on_layer(pose[0], pose[1], 'intersection')

        # 获取最近的车道信息
        lane_record = nusc_map.get_closest_lane(pose[0], pose[1], radius=2.0)
        lane_info = nusc.get('lane', lane_record) if lane_record else {}

        return {
            "is_on_drivable_surface": is_on_drivable,
            "is_at_intersection": is_at_intersection,
            "lane_connectivity": lane_info.get('turn_direction', 'Unknown'),  # e.g., 'straight', 'left', 'right'
        }
    except Exception:
        return {}  # 如果找不到标注或出错，返回空字典


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

    visual_evidence_section = "\n".join(visual_evidence_lines)

    if not images_base64:
        tqdm.write(f"  [Warning] No valid images found for event {Path(event_dir).name}. Skipping event.")
        return

    scene_token = event['scene_token']
    scene = nusc.get('scene', scene_token)
    log = nusc.get('log', scene['log_token'])
    nusc_map = NuScenesMap(dataroot=maps_root, map_name=log['location'])

    start_frame = event['event_start_frame']
    end_frame = event['event_end_frame']
    peak_frame = event['peak_error_frame']

    event_kinematics = event['kinematics']

    context = {}
    key_agent_token = event['key_agent_token']
    context['key_agent'] = {
        'token': key_agent_token,
        'dynamics': get_agent_dynamics(event_kinematics['key_agent']),
        'map_context_at_peak': get_map_context_for_agent(nusc, nusc_map, helper, key_agent_token, scene_token,
                                                         peak_frame)
    }

    context['interacting_agents'] = []
    for agent_token, kinematics_list in event_kinematics['interacting_agents'].items():
        context['interacting_agents'].append({
            'token': agent_token,
            'dynamics': get_agent_dynamics(kinematics_list)
        })

    try:
        peak_sample_token = get_sample_token_by_frame_index(nusc, scene, peak_frame)
        context['environment_at_peak'] = {
            'traffic_lights': get_traffic_light_status_for_scene(nusc, peak_sample_token)
        }
    except (ValueError, IndexError) as e:
        tqdm.write(f"[Warning] Could not get peak_sample_token for event {Path(event_dir).name}: {e}")
        context['environment_at_peak'] = {
            'traffic_lights': 'Error retrieving status'
        }

    scene_context_yaml = yaml.dump(context, indent=2, sort_keys=False)

    final_prompt = PROMPT_TEMPLATE.format(scene_context_yaml=scene_context_yaml,
    visual_evidence_section=visual_evidence_section)

    tqdm.write(f"the final prompt is {final_prompt}")

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
        "format": "json"  # 请求Ollama直接返回JSON格式
    }

    try:
        # 使用较长的超时时间，因为处理多张图片可能很慢
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()

        response_data = response.json()

        # 'response'字段包含了模型生成的JSON字符串，需要再次解析
        analysis_content_str = response_data.get('response', response_data)
        if not analysis_content_str:
            raise ValueError("VLM returned an empty response.")
        if isinstance(analysis_content_str, str):
            analysis_content_str = json.loads(analysis_content_str)

        # ---- EventBank QC & Resolve ----
        qc_pack = run_qc_and_resolve(analysis_content_str, tau=tau)

        # 5. 保存分析结果
        output_path = Path(event_dir) / 'vlm_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(qc_pack, f, indent=4)
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


    print("\nVLM analysis complete.")
