import json
import os
import numpy as np
import cv2
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.utils.data_classes import Box  # 需要导入Box
from pyquaternion import Quaternion
import textwrap
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
    基于agent在3D空间中的位置，选择最佳的摄像头视角。
    这个版本会优先选择能清晰展示关键agent, 并同时包含尽可能多交互agent的视角。

    Args:
        nusc (NuScenes): NuScenes API instance.
        sample_token (str): The token of the sample to render.
        agent_coords_3d (list): A list of 3D coordinates [[x,y,z], ...].
                                **CRITICAL ASSUMPTION**: The first element agent_coords_3d[0]
                                is the key agent, and the rest are interacting agents.
    Returns:
        str: The token of the best camera sample_data.
    """
    if not agent_coords_3d:
        # 如果没有提供任何agent坐标，直接返回前置摄像头作为安全默认值
        sample_rec = nusc.get('sample', sample_token)
        return sample_rec['data']['CAM_FRONT']

    best_cam_token = None
    max_score = -1.0

    sample_rec = nusc.get('sample', sample_token)

    # 遍历该sample中的所有摄像头数据
    for cam_channel, cam_token in sample_rec['data'].items():
        if not cam_channel.startswith('CAM'):
            continue

        sd_rec = nusc.get('sample_data', cam_token)
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_rec['camera_intrinsic'])
        imsize = (sd_rec['width'], sd_rec['height'])

        # --- 初始化当前视角的评分变量 ---
        key_agent_visible = False
        key_agent_area = 0.0
        visible_interacting_agents_count = 0
        total_interacting_area = 0.0

        # 遍历所有需要关注的agent
        for i, agent_coord in enumerate(agent_coords_3d):
            # 创建一个标准的虚拟box用于投影 (尺寸可以根据agent类型调整，但这里用统一尺寸)
            box = Box(agent_coord, [1.8, 4.5, 1.5], Quaternion(axis=[0, 0, 1], angle=0))

            # 快速检查box是否可能在图像内，以节省计算
            if not box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                continue

            # --- 精确计算2D包围盒 ---
            # 转换到相机坐标系
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # 投影到2D图像
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]

            # 计算2D包围盒并裁剪到图像边界内
            x_min, y_min = np.maximum(0, corners_2d.min(axis=1))
            x_max, y_max = np.minimum(imsize, corners_2d.max(axis=1))

            # 如果裁剪后box的宽或高为0，则跳过
            if x_max <= x_min or y_max <= y_min:
                continue

            area = (x_max - x_min) * (y_max - y_min)

            # 根据是关键agent还是交互agent，记录信息
            if i == 0:  # 关键agent (基于约定)
                key_agent_visible = True
                key_agent_area = area
            else:  # 交互agent
                visible_interacting_agents_count += 1
                total_interacting_area += area

        # --- 核心评分逻辑 ---
        # 规则1: 关键agent必须可见，否则该视角无效
        if not key_agent_visible:
            score = 0.0
        else:
            # 权重可以根据研究需求进行微调
            # W_KEY_AGENT_AREA: 关键agent的面积权重 (最重要)
            # W_INTERACTING_COUNT: 每个可见的交互agent带来的固定奖励 (鼓励包含更多agent)
            # W_INTERACTING_AREA: 交互agent的面积权重 (次要，锦上添花)
            W_KEY_AGENT_AREA = 10.0
            W_INTERACTING_COUNT = 5000.0
            W_INTERACTING_AREA = 0.5

            # 计算总分
            score = (W_KEY_AGENT_AREA * key_agent_area +
                     W_INTERACTING_COUNT * visible_interacting_agents_count +
                     W_INTERACTING_AREA * total_interacting_area)

        # 更新最佳视角
        if score > max_score:
            max_score = score
            best_cam_token = cam_token

    # 如果因为某种原因没有找到任何合适的视角（例如所有agent都在车后），则提供一个默认的前置视角
    return best_cam_token if best_cam_token else sample_rec['data']['CAM_FRONT']


def draw_on_image(nusc, sample_token, cam_token, event_data, full_trajectories, frame_idx):
    """在选定的图像上绘制所有标注"""
    sd_rec = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    image_path = os.path.join(nusc.dataroot, sd_rec['filename'])
    image = cv2.imread(image_path)

    # 获取当前帧的绝对索引
    # sample_rec = nusc.get('sample', sample_token)
    # scene_rec = nusc.get('scene', sample_rec['scene_token'])
    scene_name = scene_rec['name']

    # 找到当前帧在场景中的索引
    # current_sample_token = scene_rec['first_sample_token']
    # frame_idx = 0
    # while current_sample_token != sample_token:
    #     current_sample_token = nusc.get('sample', current_sample_token)['next']
    #     frame_idx += 1

    key_agent_token = event_data['instance_token']
    interacting_tokens = event_data.get('interactions', {}).get(str(frame_idx), [])

    # 绘制锚框
    for agent_token in [key_agent_token] + interacting_tokens:
        color = COLOR_KEY_AGENT if agent_token == key_agent_token else COLOR_INTERACTING
        ann_rec = get_annotation_for_instance(nusc, sample_token, agent_token)
        if ann_rec:  # 确保找到了annotation
            box = Box(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']))
            # ... 后续的渲染代码
        else:
            # Agent可能在这一帧不存在
            print(f"Warning: Instance {agent_token} not found in sample {sample_token}.")
            continue

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
        gt_traj_world = full_trajectories[key_agent_token][frame_idx: frame_idx + 12]
        if gt_traj_world.shape[0] > 1:
            # gt_traj_world 本身就是 (N, 3) 的，直接转置
            points = gt_traj_world.T  # Shape (3, N)
            points_cam = nusc.map_pointcloud_to_image(points, cam_token)
            points_2d = points_cam[:2, :].astype(np.int32)
            cv2.polylines(image, [points_2d.T], isClosed=False, color=COLOR_GT_TRAJ, thickness=3)

        # 预测轨迹
        pred_traj_world = np.array(event_data['predicted_trajectory'])  # 假设这个是 (N, 2)
        if pred_traj_world.shape[0] > 1:
            # 对于预测轨迹，如果只有(x, y)，一个合理的做法是使用当前帧agent的高度作为Z坐标
            # 或者从真实轨迹的第一个点获取Z坐标
            z_coord = full_trajectories[key_agent_token][frame_idx][2]
            z_coords = np.full(pred_traj_world.shape[0], z_coord)
            points = np.vstack((pred_traj_world[:, 0], pred_traj_world[:, 1], z_coords))
            points_cam = nusc.map_pointcloud_to_image(points, cam_token)
            points_2d = points_cam[:2, :].astype(np.int32)
            cv2.polylines(image, [points_2d.T], isClosed=False, color=COLOR_PRED_TRAJ, thickness=3,
                          lineType=cv2.LINE_AA)

    # 添加时间戳
    relative_time = (frame_idx - event_data['peak_fde_frame_in_traj']) * 0.5  # NuScenes is 2Hz
    timestamp_text = f"t0" if relative_time == 0 else f"t{relative_time:+.1f}s"
    cv2.putText(image, timestamp_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_annotation_for_instance(nusc, sample_token, instance_token):
    """在一个sample中查找特定instance的annotation。"""
    sample_rec = nusc.get('sample', sample_token)
    for ann_token in sample_rec['anns']:
        ann_rec = nusc.get('sample_annotation', ann_token)
        if ann_rec['instance_token'] == instance_token:
            return ann_rec
    return None # 如果该instance在这一帧不可见，则返回None

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


def visualize_vlm_result(annotated_images: list, vlm_analysis_json: dict, case_id: str,
                    output_dir: str, timestamps: list, visible: bool = False
):
    """
        将VLM的输入图像和输出分析文本合成为一张图片，用于可视化和调试。

        Args:
            annotated_images (list): 提供给VLM的PIL Image对象列表。
            vlm_analysis_json (dict): 从VLM响应中解析出的JSON对象。
            case_id (str): 案例的唯一标识符，用于命名输出文件。
            output_dir (str): 保存可视化结果的目录。
            timestamps (list): 与annotated_images对应的每个图像的时间戳字符串列表。
            visible (bool): 是否在桌面上显示图像。默认为False。
        """
    if not annotated_images or not vlm_analysis_json:
        print(f"Warning: [Visualizer] Missing images or analysis for case {case_id}. Skipping visualization.")
        return

        # --- 1. 定义布局和样式常量 ---
    PADDING = 20
    HEADER_HEIGHT = 60
    TEXT_BG_COLOR = (40, 40, 40)  # 深灰色背景
    TEXT_COLOR = (255, 255, 255)  # 白色文字
    HEADER_COLOR = (0, 0, 0)  # 黑色标题
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TEXT = 0.7
    FONT_SCALE_HEADER = 1.0
    LINE_THICKNESS = 1
    LINE_SPACING = 30

    # --- 2. 准备图像部分 ---
    # 将PIL图像转换为OpenCV格式 (BGR)
    cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in annotated_images]

    # 假设所有图像尺寸相同
    img_h, img_w, _ = cv_images[0].shape
    num_images = len(cv_images)

    # 水平拼接图像
    stitched_images = np.concatenate(cv_images, axis=1)

    # --- 3. 准备文本部分 ---
    # 格式化VLM分析文本
    analysis_text = vlm_analysis_json.get("analysis", {})
    obs = analysis_text.get("direct_observation", "N/A")
    causal = analysis_text.get("causal_inference", "N/A")
    social = analysis_text.get("social_inference", "N/A")
    summary = vlm_analysis_json.get("implicit_rule_summary", "N/A")

    full_text = (
        f"--- Direct Observation ---\n{obs}\n\n"
        f"--- Causal Inference ---\n{causal}\n\n"
        f"--- Social Inference ---\n{social}\n\n"
        f"--- Implicit Rule Summary ---\n{summary}"
    )

    # 自动换行以适应图像总宽度
    wrapper = textwrap.TextWrapper(width=int(stitched_images.shape[1] / (FONT_SCALE_TEXT * 15)))  # 启发式宽度
    wrapped_lines = []
    for line in full_text.split('\n'):
        wrapped_lines.extend(wrapper.wrap(line) if line.strip() != "" else [""])

    # --- 4. 创建最终的画布 ---
    text_area_height = len(wrapped_lines) * LINE_SPACING + PADDING * 2
    canvas_h = img_h + HEADER_HEIGHT + text_area_height
    canvas_w = stitched_images.shape[1]

    # 创建白色画布
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # --- 5. 组合所有元素到画布上 ---
    # 粘贴拼接好的图像
    canvas[HEADER_HEIGHT: HEADER_HEIGHT + img_h, :] = stitched_images

    # 在每张子图下方添加时间戳
    for i, timestamp in enumerate(timestamps):
        text_size, _ = cv2.getTextSize(timestamp, FONT, 0.8, 2)
        text_x = i * img_w + (img_w - text_size[0]) // 2
        text_y = HEADER_HEIGHT + img_h - PADDING
        cv2.putText(canvas, timestamp, (text_x, text_y), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)  # 带黑色描边
        cv2.putText(canvas, timestamp, (text_x, text_y), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 绘制文本区域背景
    text_bg_y_start = HEADER_HEIGHT + img_h
    cv2.rectangle(canvas, (0, text_bg_y_start), (canvas_w, canvas_h), TEXT_BG_COLOR, -1)

    # 绘制标题
    cv2.putText(canvas, f"VLM Analysis: {case_id}", (PADDING, HEADER_HEIGHT - 20), FONT, FONT_SCALE_HEADER,
                HEADER_COLOR, 2)

    # 逐行写入VLM分析文本
    y_text = text_bg_y_start + PADDING + 20
    for line in wrapped_lines:
        cv2.putText(canvas, line, (PADDING, y_text), FONT, FONT_SCALE_TEXT, TEXT_COLOR, LINE_THICKNESS, cv2.LINE_AA)
        y_text += LINE_SPACING

    # --- 6. 保存并显示 ---
    output_path = os.path.join(output_dir, f"{case_id}_visualization.png")
    cv2.imwrite(output_path, canvas)
    print(f"Visualization saved to {output_path}")

    if visible:
        # 调整显示尺寸以适应屏幕
        display_h, display_w = 1000, 1800
        h, w, _ = canvas.shape
        scale = min(display_h / h, display_w / w)
        if scale < 1.0:
            resized_canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        else:
            resized_canvas = canvas

        cv2.imshow(f'VLM Analysis - {case_id}', resized_canvas)
        print("Press any key in the display window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=="__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

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
        sampled_frame_indices = sorted(list(sampled_frames))
        timestamps_for_vis = []
        for frame_idx in sampled_frame_indices:
            if not (event['start_frame'] <= frame_idx < event['end_frame']):
                continue
            relative_time = (frame_idx - event['peak_fde_frame_in_traj']) * 0.5
            timestamps_for_vis.append(f"t{relative_time:+.1f}s" if relative_time != 0 else "t0")
            sample_token = sample_tokens[frame_idx]

            # 5. 最佳视角选择
            key_agent_coord = full_trajectories[event['instance_token']][frame_idx]
            interacting_coords = [full_trajectories[t][frame_idx] for t in
                                  event.get('interactions', {}).get(str(frame_idx), [])]
            all_relevant_coords = [key_agent_coord] + interacting_coords



            best_cam_token = select_best_camera_view(nusc, sample_token, all_relevant_coords)

            # 6. 绘制图像
            annotated_image = draw_on_image(nusc, sample_token, best_cam_token, event, full_trajectories, frame_idx)
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
            visualize_vlm_result(
                annotated_images=images_for_vlm,
                vlm_analysis_json=analysis_data,
                case_id=case_id,
                output_dir=OUTPUT_DIR,
                timestamps=timestamps_for_vis,
                visible=False  # 在服务器上运行时设为 False
            )

        except Exception as e:
            print(f"Failed to parse VLM response for {case_id}: {e}")
            # 保存原始文本
            output_filepath = os.path.join(OUTPUT_DIR, f"{case_id}_analysis_raw.txt")
            with open(output_filepath, 'w') as f:
                f.write(response_text)
