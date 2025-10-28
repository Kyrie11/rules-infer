import torch
from torch.utils.data import DataLoader, random_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
# 确保 nuscenes-devkit 和其他库已安装
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from collections import defaultdict
import numpy as np
from rules_infer.dataset.nuscenes import NuScenesTrajectoryDataset
from rules_infer.tools.motion_lstm import Encoder, Decoder, Seq2Seq

CONFIG = {
    # --- 数据和模型路径 ---
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! 修改这里 !!!
    'version': 'v1.0-trainval',  # 建议先用 'v1.0-mini' 测试，然后换成 'v1.0-trainval'
    'model_path': 'nuscenes-lstm-model.pt',  # 你保存的模型权重文件
    'output_dir': 'eval_results',  # 保存可视化结果的文件夹

    # --- 模型和数据参数 (必须与训练时一致) ---
    'history_len': 8,
    'future_len': 12,
    'input_dim': 4,  # (x, y, is_near_tl, dist_to_tl) - 如果训练时没用地图，这里是 2
    'hidden_dim': 64,
    'output_dim': 2,
    'n_layers': 2,

    # --- 评估参数 ---
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # --- 地图参数 (如果训练时使用了) ---
    'traffic_light_distance_threshold': 30.0,
    # 定义FDE误差超过多少米被认为是“关键事件”
    'critical_event_threshold_fde': 2.0,  # 比如最终点误差超过2米

    # 定义FDE误差从一帧到下一帧的增量超过多少被认为是“关键事件”
    'critical_event_threshold_spike': 1.5,  # 例如，FDE在0.2秒内（相邻帧）增加了1.5米以上

    # 在识别出的事件窗口前后额外扩展多少帧作为上下文
    'critical_event_context_frames': 10,  # 往前和往后各扩展 10 帧 (即 5s)

    # --- 交互分析参数 ---
    'interaction_proximity_threshold': 25.0,  # 交互距离阈值（米）
    'interaction_heading_threshold_deg': 90.0, # 前方视角阈值（度）

    # 保存最终索引文件的路径
    'critical_event_index_file': 'critical_events.json'
}


def get_agent_history_and_future(instance_token, start_sample, nusc, hist_len, future_len):
    history_xy, future_xy = np.zeros((hist_len, 2)), np.zeros((future_len, 2))
    try:
        current_ann = next(ann for ann in nusc.get('sample', start_sample)['anns'] if
                           nusc.get('sample_annotation', ann)['instance_token'] == instance_token)
    except StopIteration:
        return None, None  # Instance not in this sample

    # Get history
    current_ann_record = nusc.get('sample_annotation', current_ann)
    for i in range(hist_len - 1, -1, -1):
        history_xy[i] = current_ann_record['translation'][:2]
        if not current_ann_record['prev']: return None, None
        current_ann_record = nusc.get('sample_annotation', current_ann_record['prev'])

    # Reset to start and get future
    current_ann_record = nusc.get('sample_annotation', current_ann)
    for i in range(future_len):
        if not current_ann_record['next']: return None, None
        current_ann_record = nusc.get('sample_annotation', current_ann_record['next'])
        future_xy[i] = current_ann_record['translation'][:2]
    return history_xy, future_xy


def transform_to_agent_centric(points, anchor_xy, anchor_yaw):
    rotation_matrix = np.array([[np.cos(anchor_yaw), -np.sin(anchor_yaw)], [np.sin(anchor_yaw), np.cos(anchor_yaw)]])
    return (points - anchor_xy) @ rotation_matrix.T


def transform_to_global(points, anchor_xy, anchor_yaw):
    rotation_matrix = np.array([[np.cos(anchor_yaw), -np.sin(anchor_yaw)], [np.sin(anchor_yaw), np.cos(anchor_yaw)]])
    return points @ rotation_matrix + anchor_xy


def get_heading_from_history(history_xy):
    if np.all(history_xy[-1] == history_xy[-2]): return 0.0
    return np.arctan2(history_xy[-1, 1] - history_xy[-2, 1], history_xy[-1, 0] - history_xy[-2, 0])


class InteractionDetector:
    def __init__(self, model, device, nusc, config):
        self.model = model
        self.device = device
        self.nusc = nusc
        self.config = config
        self.model.eval()

    def detect_events_in_scene(self, scene_token):
        scene = self.nusc.get('scene', scene_token)
        samples = []
        current_token = scene['first_sample_token']
        while current_token:
            samples.append(current_token)
            current_token = self.nusc.get('sample', current_token)['next']

        agent_event_tracker = {}
        detected_events = []

        for frame_idx, sample_token in enumerate(samples):
            sample = self.nusc.get('sample', sample_token)

            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']
                if 'vehicle' not in ann['category_name']: continue

                history, future_gt = get_agent_history_and_future(
                    instance_token, sample_token, self.nusc,
                    self.config['hist_len'], self.config['future_len']
                )
                if history is None or future_gt is None: continue

                anchor_xy, anchor_yaw = history[-1], get_heading_from_history(history)
                history_centric = transform_to_agent_centric(history, anchor_xy, anchor_yaw)
                history_tensor = torch.FloatTensor(history_centric).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    future_pred_centric = self.model(history_tensor, self.config['future_len']).squeeze(0).cpu().numpy()

                future_pred_global = transform_to_global(future_pred_centric, anchor_xy, anchor_yaw)
                fde = np.linalg.norm(future_pred_global[-1] - future_gt[-1])

                if instance_token not in agent_event_tracker:
                    agent_event_tracker[instance_token] = {'is_event': False, 'start_frame_idx': -1, 'peak_error': 0,
                                                           'peak_frame_idx': -1}

                tracker = agent_event_tracker[instance_token]

                if fde > self.config['error_threshold_m']:
                    if not tracker['is_event']:
                        tracker.update({'is_event': True, 'start_frame_idx': frame_idx, 'peak_error': fde,
                                        'peak_frame_idx': frame_idx})
                    elif fde > tracker['peak_error']:
                        tracker.update({'peak_error': fde, 'peak_frame_idx': frame_idx})
                else:
                    if tracker['is_event']:
                        end_idx = frame_idx - 1
                        if end_idx - tracker['start_frame_idx'] + 1 >= self.config['min_event_frames']:
                            peak_sample_token = samples[tracker['peak_frame_idx']]
                            interacting_agents = self._find_interacting_agents(instance_token, peak_sample_token)
                            detected_events.append({
                                'key_agent_token': instance_token,
                                'interacting_agent_tokens': interacting_agents,
                                'start_frame_idx': tracker['start_frame_idx'],
                                'end_frame_idx': end_idx,
                                'peak_error_fde': tracker['peak_error'],
                                'peak_frame_idx': tracker['peak_frame_idx'],
                                'peak_sample_token': peak_sample_token,  # <<<--- 添加此项用于可视化
                                'scene_token': scene_token
                            })
                        tracker['is_event'] = False
        return detected_events

    def _find_interacting_agents(self, key_agent_token, peak_sample_token):
        interacting_tokens = []
        sample = self.nusc.get('sample', peak_sample_token)
        key_agent_ann = next(ann for ann in sample['anns'] if
                             self.nusc.get('sample_annotation', ann)['instance_token'] == key_agent_token)
        key_agent_pos = np.array(self.nusc.get('sample_annotation', key_agent_ann)['translation'][:2])

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == key_agent_token: continue
            distance = np.linalg.norm(key_agent_pos - np.array(ann['translation'][:2]))
            if distance < self.config['proximity_radius_m']:
                interacting_tokens.append(ann['instance_token'])
        return interacting_tokens


def visualize_and_save_event(event, nusc, output_path):
    """为单个事件在误差峰值帧生成并保存可视化图像"""
    peak_sample_token = event['peak_sample_token']
    sample_record = nusc.get('sample', peak_sample_token)

    # 准备要绘制的box和颜色
    boxes_to_draw = []
    colors_to_draw = []

    for ann_token in sample_record['anns']:
        ann_record = nusc.get('sample_annotation', ann_token)
        instance_token = ann_record['instance_token']

        color = None
        if instance_token == event['key_agent_token']:
            color = (255, 0, 0)  # 红色 for key agent
        elif instance_token in event['interacting_agent_tokens']:
            color = (0, 0, 255)  # 蓝色 for interacting agents

        if color:
            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))
            boxes_to_draw.append(box)
            colors_to_draw.append(color)

    # 定义相机顺序和拼接图像
    CAMERAS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    # 渲染每个相机视图
    images = []
    for cam in CAMERAS:
        cam_token = sample_record['data'][cam]
        # 使用 extra_boxes 参数来绘制我们自定义的框
        # with_anns=False 确保不绘制默认的 nuScenes 标注
        img_path, _, _ = nusc.get_sample_data(cam_token, box_vis_level=0)
        img = Image.open(img_path)
        nusc.render_boxes_on_image(img, boxes_to_draw, colors=colors_to_draw)
        images.append(img)

    # 拼接图像
    width, height = images[0].size
    stitched_image = Image.new('RGB', (width * 3, height * 2))

    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        stitched_image.paste(img, (col * width, row * height))

    # 保存图像
    stitched_image.save(output_path)
    # print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    # --- 配置参数 ---
    DATAROOT = '/data0/senzeyu2/dataset/nuscenes'  # <<<--- 修改为你的nuScenes路径
    VERSION = 'v1.0-trainval'  # 或 'v1.0-trainval'
    MODEL_PATH = 'seq2seq_model.pth'  # <<<--- 假设你已经训练好并保存了模型
    OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/critical'  # <<<--- 输出目录

    config = {
        'hist_len': 8,
        'future_len': 12,
        'error_threshold_m': 3.0,
        'proximity_radius_m': 20.0,
        'min_event_frames': 5
    }

    # --- 初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    # --- 加载模型 ---
    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder, device).to(device)

    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Using randomly initialized model.")

    # --- 创建输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 创建检测器并处理所有场景 ---
    detector = InteractionDetector(model, device, nusc, config)

    total_events_found = 0
    for scene_idx, scene in enumerate(tqdm(nusc.scene, desc="Processing Scenes")):
        scene_token = scene['token']
        scene_name = scene['name']

        events = detector.detect_events_in_scene(scene_token)

        if not events:
            continue

        total_events_found += len(events)

        # 为当前场景创建子目录
        scene_output_dir = os.path.join(OUTPUT_DIR, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        for event_idx, event in enumerate(events):
            # 构造文件名
            output_filename = f"event_{event_idx + 1}_peak_frame_{event['peak_frame_idx']}.png"
            output_path = os.path.join(scene_output_dir, output_filename)

            # 生成并保存可视化
            visualize_and_save_event(event, nusc, output_path)

            # 打印事件信息 (可选)
            # print(f"\n--- Detected Event in Scene {scene_name} ---")
            # print(f"  Key Agent: {event['key_agent_token']}")
            # print(f"  Interacting Agents: {event['interacting_agent_tokens']}")
            # print(f"  Saved to: {output_path}")

    print(f"\nProcessing complete. Found a total of {total_events_found} potential interaction events.")
    print(f"Visualizations saved in '{OUTPUT_DIR}' directory.")

