import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes' # nuScenes数据集的根目录
NUSCENES_VERSION = 'v1.0-trainval'            # 使用的数据集版本 ('v1.0-mini' 或 'v1.0-trainval')
EVENTS_JSON_PATH = 'result.json' # 你生成的事件JSON文件
OUTPUT_DIR = '/data0/senzeyu2/dataset/nuscenes/events'         # 保存可视化结果的文件夹

PRIMARY_AGENT_COLOR = (1, 0, 0)
INTERACTING_AGENT_COLOR = (0, 0, 1)

def find_closest_sample(nusc, scene_token, target_timestamp):
    # ... (代码不变)
    scene = nusc.get('scene', scene_token)
    current_sample_token = scene['first_sample_token']
    min_time_diff = float('inf')
    closest_sample_token = current_sample_token
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        time_diff = abs(sample['timestamp'] / 1e6 - target_timestamp)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_sample_token = current_sample_token
        current_sample_token = sample['next']
        if time_diff > min_time_diff + 0.1:
            break
    return nusc.get('sample', closest_sample_token)

def get_annotation_for_instance(nusc, sample, instance_token):
    # ... (代码不变)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None

def visualize_event(nusc, event_data, output_dir):
    # ... (代码不变)
    event_id = event_data['event_id']
    scene_token = event_id.split('_')[0]
    event_timestamp = event_data['timestamp_start']
    primary_agent_instance = event_data['primary_agent']['agent_id']
    interacting_agents_instances = [
        agent['agent_id'] for agent in event_data.get('candidate_interacting_agents', [])[:2]
    ]
    sample = find_closest_sample(nusc, scene_token, event_timestamp)
    primary_ann = get_annotation_for_instance(nusc, sample, primary_agent_instance)
    interacting_anns = [get_annotation_for_instance(nusc, sample, inst) for inst in interacting_agents_instances]
    interacting_anns = [ann for ann in interacting_anns if ann is not None]

    if not primary_ann:
        # print(f"Warning: Could not find primary agent annotation for event {event_id} at timestamp {event_timestamp}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=100)
    axes = axes.ravel()
    cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

    for i, cam_type in enumerate(cam_types):
        ax = axes[i]
        cam_token = sample['data'][cam_type]
        nusc.render_sample_data(cam_token, with_anns=False, ax=ax)
        nusc.render_annotation(primary_ann['token'], ax=ax, box_vis_level=3, color=PRIMARY_AGENT_COLOR, linewidth=3)
        for ann in interacting_anns:
            nusc.render_annotation(ann['token'], ax=ax, box_vis_level=3, color=INTERACTING_AGENT_COLOR, linewidth=2)
        ax.set_title(cam_type.replace('_', ' '))
        ax.set_axis_off()

    fig.suptitle(f'Event: {event_id}\n(Primary: Red, Interacting: Blue)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{event_id}.png")
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == '__main__':
    # ... (main函数部分的代码完全不变)
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("SDK initialized.")
    print(f"Loading events from '{EVENTS_JSON_PATH}'...")
    if not os.path.exists(EVENTS_JSON_PATH):
        print(f"Error: Events file not found at '{EVENTS_JSON_PATH}'")
        exit()
    with open(EVENTS_JSON_PATH, 'r') as f:
        all_events_by_scene = json.load(f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Visualization results will be saved in '{OUTPUT_DIR}/'")
    all_events_flat = []
    for scene_name, events_in_scene in all_events_by_scene.items():
        all_events_flat.extend(events_in_scene)
    print(f"Found a total of {len(all_events_flat)} events to visualize.")
    for event in tqdm(all_events_flat, desc="Visualizing Events"):
        try:
            visualize_event(nusc, event, OUTPUT_DIR)
        except Exception as e:
            # print(f"\nError processing event {event.get('event_id', 'N/A')}: {e}")
            continue
    print("\nVisualization complete!")