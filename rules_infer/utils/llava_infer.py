import os
import json
import base64
import requests
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.helper import PredictHelper

nusc = NuScenes(version='v1.0-trainval', dataroot='/data0/senzeyu2/dataset/nuscenes', verbose=False)
helper = PredictHelper(nusc)
maps_root = "/data0/senzeyu2/dataset/nuscenes"

EVENTS_JSON_PATH = 'social_events.json'
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events_ins'  # 新的输出目录

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
- The appendix also includes information on the key agent's motion in each frame.  

**Case File & Scene Context:**
Here is the structured context of the scene, including agent dynamics and environmental data. Use this information as the factual basis for your analysis.

```yaml
{scene_context_yaml}


Visual Evidence (Chronological Frames):
The following section presents a frame-by-frame visual breakdown of the event. Each image token corresponds to a cropped image from a specific camera view at a specific moment in time.

{visual_evidence_section}


**Your Goal:**
Based on ALL the information provided (both the YAML context and the visual evidence), follow a strict Chain of Thought to explain the failure.

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
      { "direct_observation": "...", "causal_inference": "...", "social_event": "intersection_yield", 
      "event_span": {"start":"t-2.0s","end":"t+3.0s"}, "actors": [{"track_id":123, "type":"vehicle", "role":"yielding"}, 
      {"track_id":45, "type":"pedestrian", "role":"crossing"}], 
      "context": {"junction":"signalized", "tl_state":"red", "crosswalk":true}, 
      "evidence": ["pedestrian enters crosswalk", "ego decelerates from 8m/s to 0"], 
      "confidence": 0.82, "alternatives": ["congestion_stop"], "map_refs": {"lane_ids":[...], "crosswalk_id":...} }
    }}

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
            # tqdm.write(f"accel_vec:{accel_vec}")
            # tqdm.write(f"lon_acc:{lon_acc}")
            # tqdm.write(f"lat_acc:{lat_acc}")
            # formatted_frame['acceleration'] = round(np.linalg.norm(accel_vec), 2)  # m/s^2
            formatted_frame['lon_acc'] = round(float(lon_acc), 2)
            formatted_frame['lat_acc'] = round(float(lat_acc), 2)
            # tqdm.write(f"formated_frame:{formatted_frame}")

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

def analyze_event(event, event_dir):
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
        analysis_content_str = response_data.get('response')
        if not analysis_content_str:
            raise ValueError("VLM returned an empty response.")

        analysis_content = json.loads(analysis_content_str)

        # 5. 保存分析结果
        output_path = Path(event_dir) / 'vlm_analysis.json'
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

        if os.path.exists(os.path.join(event_output_dir, 'vlm.json')):
            tqdm.write(f"Event {event_id} already analyzed. Skipping.")
            continue

        result = analyze_event(event, event_output_dir)
        if result != None:
            success, result_path = result
            if success:
                tqdm.write(f"  -> Analysis successful. Result saved to {result_path}")


    print("\nVLM analysis complete.")
