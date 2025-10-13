import json
import os
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion

# --- Llava 调用相关的库 ---
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ------------------- CONFIGURATION -------------------
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes/'  # <-- 修改这里
NUSCENES_VERSION = 'v1.0-trainval'
EVENTS_FILE = 'critical_events_with_interactions.json'  # <-- 输入文件
OUTPUT_DIR = 'vlm_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VLM Model ID
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# 可视化颜色 (BGR format for OpenCV)
COLOR_KEY_AGENT = (0, 0, 255)  # Red
COLOR_INTERACTING = (255, 0, 0)  # Blue
COLOR_GT_TRAJ = (0, 255, 0)  # Green
COLOR_PRED_TRAJ = (0, 165, 255)  # Orange


# ------------------- HELPER FUNCTIONS -------------------

def get_full_trajectory_data(nusc, scene_token, instance_token):
    """从NuScenes数据中提取一个agent在整个场景中的完整轨迹和时间戳"""
    scene_rec = nusc.get('scene', scene_token)
    first_sample_token = scene_rec['first_sample_token']

    instance_rec = nusc.get('instance', instance_token)
    current_ann_token = instance_rec['first_annotation_token']

    coords = []
    timestamps = []

    while current_ann_token:
        ann_rec = nusc.get('sample_annotation', current_ann_token)
        sample_rec = nusc.get('sample', ann_rec['sample_token'])

        coords.append(ann_rec['translation'])
        timestamps.append(sample_rec['timestamp'])

        current_ann_token = ann_rec['next']
        if not current_ann_token:
            break

    return np.array(coords), np.array(timestamps)


def select_best_camera_view(nusc, sample_token, agent_coords_3d):
    """
    基于agent在3D空间中的位置，选择最佳的摄像头视角
    agent_coords_3d: list of 3D coordinates [[x,y,z], ...]
    """
    best_cam = None
    max_score = -1

    sample_rec = nusc.get('sample', sample_token)

    for cam_token in sample_rec['data']:
        if 'CAM' not in nusc.get('sample_data', cam_token)['channel']:
            continue

        sd_rec = nusc.get('sample_data', cam_token)
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

        cam_points = []
        visible_agents = 0
        total_area = 0

        for agent_coord in agent_coords_3d:
            # 创建一个虚拟的box用于投影
            box = Box(agent_coord, [1.8, 4.5, 1.5], Quaternion(axis=[0, 0, 1], angle=0))

            if not box_in_image(box, np.array(cs_rec['camera_intrinsic']), (sd_rec['width'], sd_rec['height'])):
                continue

            # 转换到相机坐标系
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # 投影到2D图像
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, np.array(cs_rec['camera_intrinsic']), normalize=True)[:2, :]

            x_min, y_min = corners_2d.min(axis=1)
            x_max, y_max = corners_2d.max(axis=1)

            # 检查是否在图像边界内
            if x_max < 0 or x_min > sd_rec['width'] or y_max < 0 or y_min > sd_rec['height']:
                continue

            visible_agents += 1
            total_area += (x_max - x_min) * (y_max - y_min)

        if visible_agents == 0:
            continue

        # 分数 = 可见agent数 * 平均面积 (简单启发式)
        score = visible_agents * (total_area / len(agent_coords_3d))
        if score > max_score:
            max_score = score
            best_cam = cam_token

    return best_cam if best_cam else sample_rec['data']['CAM_FRONT']


def draw_on_image(nusc, sample_token, cam_token, event_data, full_trajectories):
    """在选定的图像上绘制所有标注"""
    sd_rec = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    image_path = os.path.join(nusc.dataroot, sd_rec['filename'])
    image = cv2.imread(image_path)

    # 获取当前帧的绝对索引
    sample_rec = nusc.get('sample', sample_token)
    scene_rec = nusc.get('scene', sample_rec['scene_token'])
    scene_name = scene_rec['name']

    # 找到当前帧在场景中的索引
    current_sample_token = scene_rec['first_sample_token']
    frame_idx = 0
    while current_sample_token != sample_token:
        current_sample_token = nusc.get('sample', current_sample_token)['next']
        frame_idx += 1

    key_agent_token = event_data['instance_token']
    interacting_tokens = event_data.get('interactions', {}).get(str(frame_idx), [])

    # 绘制锚框
    for agent_token in [key_agent_token] + interacting_tokens:
        color = COLOR_KEY_AGENT if agent_token == key_agent_token else COLOR_INTERACTING
        try:
            ann_token = nusc.get_sample_annotation_token(sample_token, agent_token)
            ann_rec = nusc.get('sample_annotation', ann_token)
            box = Box(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']))

            # 投影到图像
            box.render_cv2(image, view=np.array(cs_rec['camera_intrinsic']), normalize=True,
                           cs_record=cs_rec, sd_record=sd_rec, color=color, thickness=3)
        except KeyError:
            # Agent可能在这一帧不存在
            continue

    # 绘制轨迹 (只在 t0 帧绘制)
    if frame_idx == event_data['peak_fde_frame_in_traj']:
        # 真实轨迹
        gt_traj_world = full_trajectories[key_agent_token][frame_idx: frame_idx + 12]  # 未来12帧
        if gt_traj_world.shape[0] > 1:
            points = np.vstack((gt_traj_world[:, 0], gt_traj_world[:, 1], np.zeros(gt_traj_world.shape[0])))
            points_cam = nusc.map_pointcloud_to_image(points, cam_token)
            points_2d = points_cam[:2, :].astype(np.int32)
            cv2.polylines(image, [points_2d.T], isClosed=False, color=COLOR_GT_TRAJ, thickness=3)

        # 预测轨迹
        pred_traj_world = np.array(event_data['predicted_trajectory'])
        if pred_traj_world.shape[0] > 1:
            points = np.vstack((pred_traj_world[:, 0], pred_traj_world[:, 1], np.zeros(pred_traj_world.shape[0])))
            points_cam = nusc.map_pointcloud_to_image(points, cam_token)
            points_2d = points_cam[:2, :].astype(np.int32)
            cv2.polylines(image, [points_2d.T], isClosed=False, color=COLOR_PRED_TRAJ, thickness=3,
                          lineType=cv2.LINE_AA)

    # 添加时间戳
    relative_time = (frame_idx - event_data['peak_fde_frame_in_traj']) * 0.5  # NuScenes is 2Hz
    timestamp_text = f"t0" if relative_time == 0 else f"t{relative_time:+.1f}s"
    cv2.putText(image, timestamp_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def build_vlm_prompt(case_data, event):
    """构建VLM的文本prompt"""

    # 从NuScenes获取元数据
    scene_rec = nusc.get('scene', case_data['scene_token'])
    log_rec = nusc.get('log', scene_rec['log_token'])

    prompt = f"""You are an expert autonomous driving safety analyst and social behavior researcher. Your task is to analyze a sequence of traffic scene images where an AI's trajectory prediction model has failed, and deduce the underlying social reasons for the failure.

    I will provide you with a "Case File" in JSON format and a series of corresponding images. The key agent whose trajectory was mispredicted is highlighted in a RED box. Any other relevant interacting agents are in BLUE boxes. On the 't0' frame, the model's incorrect prediction is shown as an ORANGE line, and the agent's actual path is a GREEN line.
    
    **Your Goal:**
    Follow a strict Chain of Thought to explain the failure. You MUST structure your analysis in three distinct steps as detailed below.
    
    ---
    
    **Chain of Thought Instructions:**
    
    **Step 1: Direct Observation (Phenomenon)**
    - **Task:** Objectively describe what is happening in the image sequence. Focus on the actions of the key agent (RED box) and its interactions with other agents or the environment.
    
    **Step 2: Causal Inference (Physical Reason)**
    - **Task:** Based on your observations, infer the immediate, direct cause for the key agent's behavior. This is the "physical world" reason.
    
    **Step 3: Social/Behavioral Inference (Implicit Rule)**
    - **Task:** Go one level deeper. Analyze the social context. What underlying social norm, local driving culture, or unwritten rule explains *why* the causal event happened?
    
    ---
    
    **Output Format:**
    You MUST provide your response in a single, clean, and parsable JSON object. Do not include any text outside of this JSON block. Use the following structure:
    ```json
    {{
      "analysis": {{
        "direct_observation": "...",
        "causal_inference": "...",
        "social_inference": "..."
      }},
      "implicit_rule_summary": "A concise, one-sentence summary of the discovered implicit social rule or behavior."
    }}"""


if __name__=="__main__":
    # 加载Llava模型
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    # 2. 加载事件文件
    with open(EVENTS_FILE, 'r') as f:
        critical_events = json.load(f)

    # 3. 遍历和处理每个事件
    for scene_token, events in critical_events.items():
        # 为了演示，我们只处理每个场景的第一个事件
        event = events[0]

        case_id = f"{scene_token}_{event['instance_token']}"
        print(f"\n--- Processing case: {case_id} ---")

        # 预加载场景中所有agent的完整轨迹，以提高效率
        scene_rec = nusc.get('scene', scene_token)
        all_instance_tokens = [nusc.get('sample_annotation', ann_token)['instance_token'] for ann_token in
                               scene_rec['anns']]
        full_trajectories = {
            token: get_full_trajectory_data(nusc, scene_token, token)[0]
            for token in set(all_instance_tokens)
        }

        # 4. 关键时刻采样
        t0_frame = event['peak_fde_frame_in_traj']
        sampled_frames = {
            t0_frame - 4,  # t-2s
            t0_frame,  # t0
            t0_frame + 4  # t+2s
        }

        # 获取场景的sample tokens列表
        sample_tokens = nusc.get_sample_tokens_in_scene(scene_token)

        images_for_vlm = []
        for frame_idx in sorted(list(sampled_frames)):
            if not (event['start_frame'] <= frame_idx < event['end_frame']):
                continue

            sample_token = sample_tokens[frame_idx]

            # 5. 最佳视角选择
            key_agent_coord = full_trajectories[event['instance_token']][frame_idx]
            interacting_coords = [full_trajectories[t][frame_idx] for t in
                                  event.get('interactions', {}).get(str(frame_idx), [])]
            all_relevant_coords = [key_agent_coord] + interacting_coords

            from nuscenes.utils.data_classes import Box  # 需要导入Box

            best_cam_token = select_best_camera_view(nusc, sample_token, all_relevant_coords)

            # 6. 绘制图像
            annotated_image = draw_on_image(nusc, sample_token, best_cam_token, event, full_trajectories)
            images_for_vlm.append(annotated_image)

            # 保存用于调试
            annotated_image.save(os.path.join(OUTPUT_DIR, f"{case_id}_frame_{frame_idx}.png"))

        if not images_for_vlm:
            print(f"Skipping case {case_id} due to no valid images.")
            continue

        # 7. 构建Prompt并调用VLM
        prompt = build_vlm_prompt({"scene_token": scene_token}, event)
        inputs = processor(text=prompt, images=images_for_vlm, return_tensors="pt").to("cuda", torch.float16)

        print("Generating VLM analysis...")
        output = model.generate(**inputs, max_new_tokens=512)
        response_text = processor.decode(output[0], skip_special_tokens=True)

        # 提取JSON部分
        try:
            json_response = response_text[response_text.find('{'):response_text.rfind('}') + 1]
            analysis_data = json.loads(json_response)

            # 保存结果
            output_filepath = os.path.join(OUTPUT_DIR, f"{case_id}_analysis.json")
            with open(output_filepath, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            print(f"Analysis saved to {output_filepath}")

        except Exception as e:
            print(f"Failed to parse VLM response for {case_id}: {e}")
            # 保存原始文本
            output_filepath = os.path.join(OUTPUT_DIR, f"{case_id}_analysis_raw.txt")
            with open(output_filepath, 'w') as f:
                f.write(response_text)
