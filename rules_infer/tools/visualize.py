import json
import os
import shutil
from typing import List, Tuple

import matplotlib

# ### FIX 2: Force a non-interactive backend for headless servers ###
# This line MUST be before the import of pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
# ### FIX 1: Import BoxVisibility for the corrected function call ###
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CONFIG = {
    # --- Paths ---
    'dataroot': '/data0/senzeyu2/dataset/nuscenes/',  # <--- !!! MUST MATCH your NuScenes path !!!
    'version': 'v1.0-trainval',  # <--- Use 'v1.0-mini' for quick testing
    'json_path': 'critical_events.json',  # Path to the input index file
    'output_dir': '/data0/senzeyu2/dataset/nuscenes/critical_event/',  # Where to save the output images

    # --- Visualization Settings ---
    'camera_channel': 'CAM_FRONT',
    'render_boxes': True,  # Set to True to render 2D bounding boxes on agents
    'colors': {
        'gt_past': '#00B3FF',  # Blue
        'gt_future': '#00FF7F',  # Green
        'prediction': '#FF4D4D',  # Red
        'interaction': '#FFC700'  # Yellow
    }
}


# ==============================================================================
# --- HELPER FUNCTIONS (Unchanged) ---
# ==============================================================================

def get_full_trajectory(nusc: NuScenes, scene_token: str, instance_token: str) -> np.ndarray:
    """
    Extracts the full 2D (x, y) trajectory for a given agent in a given scene.
    This is a simplified version of what your dataset class might do.
    """
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']

    traj = []

    current_sample_token = first_sample_token
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)

        ann_tokens = sample['anns']
        found_ann = None
        for ann_token in ann_tokens:
            ann = nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                found_ann = ann
                break

        if found_ann:
            traj.append(found_ann['translation'][:2])

        current_sample_token = sample['next']
        if not current_sample_token:
            break

    return np.array(traj)


def project_trajectory_to_image(
        traj_global: np.ndarray,
        cam_data: dict,
        nusc: NuScenes
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects a 3D global trajectory onto the 2D image plane of a given camera.
    """
    if traj_global.shape[0] == 0:
        return np.array([]), np.array([])

    traj_3d_global = np.hstack([
        traj_global,
        np.ones((traj_global.shape[0], 1)) * 1.0
    ])

    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor_record = nusc.get('sample_data', cam_data['token'])
    pose_record = nusc.get('ego_pose', sensor_record['ego_pose_token'])

    traj_3d_ego = traj_3d_global - np.array(pose_record['translation'])
    traj_3d_ego = np.dot(Quaternion(pose_record['rotation']).inverse.rotation_matrix, traj_3d_ego.T).T

    traj_3d_cam = traj_3d_ego - np.array(cs_record['translation'])
    traj_3d_cam = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, traj_3d_cam.T).T

    depth = traj_3d_cam[:, 2]
    in_front_mask = depth > 0.1

    if not np.any(in_front_mask):
        return np.array([]), np.array([])

    points_2d_cam = view_points(traj_3d_cam.T, cam_intrinsic, normalize=True)

    points_2d_cam = points_2d_cam[:2, in_front_mask].T
    depth_valid = depth[in_front_mask]

    return points_2d_cam, depth_valid


# ==============================================================================
# --- CORE VISUALIZATION FUNCTION (Corrected) ---
# ==============================================================================

def visualize_critical_event_clip(
        nusc: NuScenes,
        scene_token: str,
        event_data: dict,
        event_idx: int,
        base_output_dir: str
):
    """
    Generates and saves a sequence of images for a single critical event.
    """
    # 1. Parse event data
    key_agent_token = event_data['instance_token']
    start_frame = event_data['start_frame']
    end_frame = event_data['end_frame']
    reason = event_data['reason']
    value = event_data['value']
    pred_traj_global = np.array(event_data['predicted_trajectory'])

    # 2. Create output folder
    folder_name = f"event_{event_idx:02d}_{key_agent_token[:8]}_{reason}"
    event_output_dir = os.path.join(base_output_dir, nusc.get('scene', scene_token)['name'], folder_name)
    os.makedirs(event_output_dir, exist_ok=True)

    # 3. Get all sample tokens for the scene
    scene_record = nusc.get('scene', scene_token)
    sample_tokens = []
    current_token = scene_record['first_sample_token']
    while current_token:
        sample_tokens.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
    end_frame = min(end_frame, len(sample_tokens))

    # 4. Get full ground truth trajectories
    trajectories_gt = {}
    trajectories_gt[key_agent_token] = get_full_trajectory(nusc, scene_token, key_agent_token)

    interacting_agents = set()
    for frame_interactions in event_data.get('interactions', {}).values():
        for token in frame_interactions:
            interacting_agents.add(token)

    for agent_token in interacting_agents:
        if agent_token not in trajectories_gt:
            trajectories_gt[agent_token] = get_full_trajectory(nusc, scene_token, agent_token)

    # 5. Loop through frames and generate images
    for frame_idx in range(start_frame, end_frame):
        sample_token = sample_tokens[frame_idx]
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data'][CONFIG['camera_channel']]
        cam_data = nusc.get('sample_data', cam_token)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # ### FIX 1: Corrected the function call to render the background image ###
        # Replaced `with_category=True` with `box_vis_level=BoxVisibility.ANY`
        # and renamed config option `plot_box` to `render_boxes` for clarity.
        nusc.render_sample_data(
            cam_token,
            with_anns=CONFIG['render_boxes'],
            box_vis_level=BoxVisibility.ANY,  # This renders boxes with their category names
            ax=ax
        )

        # --- Plot Key Agent Trajectories ---
        key_gt_traj = trajectories_gt.get(key_agent_token, np.array([]))
        if key_gt_traj.shape[0] > 0:
            gt_past = key_gt_traj[:frame_idx + 1]
            points_2d, _ = project_trajectory_to_image(gt_past, cam_data, nusc)
            if points_2d.shape[0] > 1:
                ax.plot(points_2d[:, 0], points_2d[:, 1], color=CONFIG['colors']['gt_past'], linewidth=3,
                        label='GT Past')
                ax.scatter(points_2d[-1, 0], points_2d[-1, 1], color=CONFIG['colors']['gt_past'], s=80, zorder=5)

            gt_future = key_gt_traj[frame_idx:]
            points_2d, _ = project_trajectory_to_image(gt_future, cam_data, nusc)
            if points_2d.shape[0] > 1:
                ax.plot(points_2d[:, 0], points_2d[:, 1], color=CONFIG['colors']['gt_future'], linewidth=3,
                        linestyle='--', label='GT Future')

        # --- Plot Predicted Trajectory ---
        points_2d, _ = project_trajectory_to_image(pred_traj_global, cam_data, nusc)
        if points_2d.shape[0] > 1:
            ax.plot(points_2d[:, 0], points_2d[:, 1], color=CONFIG['colors']['prediction'], linewidth=3, linestyle='-.',
                    label='Prediction')

        # --- Plot Interacting Agents' Trajectories ---
        current_interactions = event_data.get('interactions', {}).get(str(frame_idx), [])
        for agent_token in current_interactions:
            inter_gt_traj = trajectories_gt.get(agent_token, np.array([]))
            if inter_gt_traj.shape[0] > 0:
                points_2d, _ = project_trajectory_to_image(inter_gt_traj, cam_data, nusc)
                if points_2d.shape[0] > 1:
                    ax.plot(points_2d[:, 0], points_2d[:, 1], color=CONFIG['colors']['interaction'], linewidth=2,
                            alpha=0.7, label='Interacting Agent' if agent_token == current_interactions[0] else None)

        # --- Add Text Overlay and Final Touches ---
        info_text = (
            f"Scene: {nusc.get('scene', scene_token)['name']}\n"
            f"Frame: {frame_idx} / {len(sample_tokens) - 1}\n"
            f"Reason: {reason.upper()} ({value:.2f})"
        )
        ax.text(10, 50, info_text, fontsize=12, color='white',
                bbox=dict(facecolor='black', alpha=0.5))
        ax.legend(loc='upper right')
        ax.set_xlim(0, cam_data['width'])
        ax.set_ylim(cam_data['height'], 0)
        ax.axis('off')

        # Save the figure to a file
        output_path = os.path.join(event_output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        # Close the figure to free memory and prevent it from being displayed
        plt.close(fig)


# ==============================================================================
# --- MAIN EXECUTION (Unchanged) ---
# ==============================================================================

def main():
    print("--- Starting Critical Event Visualization ---")

    # 1. Load NuScenes dataset
    print(f"Loading NuScenes {CONFIG['version']} from {CONFIG['dataroot']}...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    # 2. Load the critical events JSON file
    if not os.path.exists(CONFIG['json_path']):
        print(f"Error: Critical event file not found at '{CONFIG['json_path']}'")
        return
    with open(CONFIG['json_path'], 'r') as f:
        critical_events = json.load(f)
    print(
        f"Loaded {sum(len(v) for v in critical_events.values())} critical events across {len(critical_events)} scenes.")

    # 3. Setup output directory
    if os.path.exists(CONFIG['output_dir']):
        print(f"Output directory '{CONFIG['output_dir']}' already exists. Removing it.")
        shutil.rmtree(CONFIG['output_dir'])
    os.makedirs(CONFIG['output_dir'])

    # 4. Iterate and visualize
    pbar_scenes = tqdm(critical_events.items(), desc="Processing Scenes")
    for scene_token, events_in_scene in pbar_scenes:
        scene_name = nusc.get('scene', scene_token)['name']
        pbar_scenes.set_description(f"Processing Scene: {scene_name}")

        for event_idx, event_data in enumerate(events_in_scene):
            try:
                visualize_critical_event_clip(
                    nusc=nusc,
                    scene_token=scene_token,
                    event_data=event_data,
                    event_idx=event_idx,
                    base_output_dir=CONFIG['output_dir']
                )
            except Exception as e:
                # This will now catch other potential errors, not just the previous ones
                print(f"\n[ERROR] Failed to process event {event_idx} in scene {scene_name}: {e}")

    print("\n--- Visualization Complete ---")
    print(f"Results saved in: {os.path.abspath(CONFIG['output_dir'])}")


if __name__ == '__main__':
    main()
