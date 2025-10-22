import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

# --- Configuration ---
# Adjust these paths and settings to match your environment
CONFIG = {
    'dataroot': '/data0/senzeyu2/dataset/nuscenes',  # <--- !!! UPDATE THIS PATH !!!
    'version': 'v1.0-trainval',  # Or 'v1.0-mini' if you're testing
    'critical_event_index_file': 'critical_events.json',  # Input JSON file
    'visualization_output_dir': '/data0/senzeyu2/dataset/nuscenes/critical',  # Output folder

    # --- Trajectory Lengths (Must match the data generation script) ---
    'future_len': 12,

    # --- Annotation Colors (OpenCV uses BGR format) ---
    'color_key_agent_box': (0, 0, 255),  # Red
    'color_interaction_agent_box': (255, 0, 0),  # Blue
    'color_gt_trajectory': (0, 0, 255),  # Red
    'color_pred_trajectory': (0, 255, 0),  # Green
    'trajectory_thickness': 2,

    # --- Grid Layout Parameters ---
    'grid_scale': 0.5,  # Scale factor for each camera image
    'grid_layout': [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]
}


def get_full_trajectory(nusc, instance_token):
    """
    Extracts the full 3D world trajectory and corresponding frame indices for an agent.
    """
    traj = []
    frame_indices = []

    instance = nusc.get('instance', instance_token)
    ann_token = instance['first_annotation_token']

    # Iterate through all annotations for this instance
    frame_idx = 0
    while ann_token:
        ann = nusc.get('sample_annotation', ann_token)
        traj.append(ann['translation'])
        frame_indices.append(frame_idx)

        # Move to the next annotation and increment frame count for the scene
        if not ann['next']:
            break
        ann_token = ann['next']

        # To find the correct frame index, we need to walk the scene samples
        sample = nusc.get('sample', ann['sample_token'])
        if sample['next']:
            next_sample = nusc.get('sample', sample['next'])
            # This is a bit of a simplification; a full mapping would be safer
            # but this works for sequential annotations.
            frame_idx += 1

    return np.array(traj), np.array(frame_indices)


def prepare_trajectories_for_event(nusc, event, all_trajs_cache):
    """
    Pre-calculates the GT and Predicted trajectories in aligned 3D world coordinates.
    This is the most critical function for getting the visualization right.
    """
    key_agent_token = event['instance_token']

    # Use cache to avoid repeatedly extracting the same trajectory
    if key_agent_token not in all_trajs_cache:
        all_trajs_cache[key_agent_token] = get_full_trajectory(nusc, key_agent_token)

    full_gt_traj, frame_indices = all_trajs_cache[key_agent_token]

    # This is the frame index *within the agent's own trajectory* where the error was measured
    peak_frame_in_traj = event['peak_fde_frame_in_traj']

    # Find the index in our extracted trajectory array that corresponds to this frame
    try:
        # Find the array index where frame_indices equals peak_frame_in_traj
        anchor_idx = np.where(frame_indices == peak_frame_in_traj - 1)[0][0]
        gt_start_idx = np.where(frame_indices == peak_frame_in_traj)[0][0]
    except IndexError:
        # This can happen if the event is right at the start/end of a scene
        # where the full future trajectory is not available.
        return None, None

    # The prediction was made based on history ending at this point
    anchor_pos_3d = full_gt_traj[anchor_idx]

    # --- Ground Truth Future Trajectory ---
    # Extract the next `future_len` points from the full GT trajectory
    gt_end_idx = min(len(full_gt_traj), gt_start_idx + CONFIG['future_len'])
    gt_future_world_3d = full_gt_traj[gt_start_idx:gt_end_idx]

    # --- Predicted Future Trajectory ---
    pred_future_relative_2d = np.array(event['predicted_trajectory'])

    # The first point of the prediction is its own local origin (0,0) effectively
    # We need to shift the entire predicted path to start at the anchor point.
    pred_start_pos_relative_2d = pred_future_relative_2d[0:1, :]
    anchor_pos_2d = anchor_pos_3d[0:2]

    # Align prediction: shift it so its first point matches the anchor point in the XY plane
    pred_future_aligned_2d = pred_future_relative_2d - pred_start_pos_relative_2d + anchor_pos_2d

    # Add the Z-coordinate from the anchor point to make it 3D
    z_coords = np.full((len(pred_future_aligned_2d), 1), anchor_pos_3d[2])
    pred_future_world_3d = np.hstack([pred_future_aligned_2d, z_coords])

    return gt_future_world_3d, pred_future_world_3d


def project_points(points_3d, cam_intrinsic, cam_to_ego_rot, cam_to_ego_trans,
                   ego_to_world_rot, ego_to_world_trans):
    """Projects 3D points from world coordinates to the 2D image plane."""
    # World to Ego
    points_ego = points_3d - ego_to_world_trans
    points_ego = points_ego @ ego_to_world_rot.T

    # Ego to Camera
    points_cam = points_ego - cam_to_ego_trans
    points_cam = points_cam @ cam_to_ego_rot.T

    # Filter points behind the camera
    depths = points_cam[:, 2]
    points_2d = view_points(points_cam.T, cam_intrinsic, normalize=True)[:2, :]

    return points_2d.T, depths


def main():
    print("Initializing NuScenes...")
    nusc = NuScenes(version=CONFIG['version'], dataroot=CONFIG['dataroot'], verbose=False)

    print(f"Loading critical events from {CONFIG['critical_event_index_file']}...")
    with open(CONFIG['critical_event_index_file'], 'r') as f:
        critical_events = json.load(f)

    os.makedirs(CONFIG['visualization_output_dir'], exist_ok=True)
    total_events = sum(len(events) for events in critical_events.values())
    print(f"Found {len(critical_events)} scenes with a total of {total_events} events. Starting visualization...")

    event_progress = tqdm(total=total_events, desc="Processing Events")

    # --- Pre-calculate canvas dimensions ---
    # Get a sample camera image to determine dimensions
    sample = nusc.sample[0]
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cam_h, cam_w = cam_data['height'], cam_data['width']

    scale = CONFIG['grid_scale']
    scaled_h, scaled_w = int(cam_h * scale), int(cam_w * scale)
    grid_h, grid_w = scaled_h * 2, scaled_w * 3  # 2 rows, 3 columns

    for scene_token, events in critical_events.items():
        scene = nusc.get('scene', scene_token)

        # Create a mapping from frame index to sample_token for the current scene
        frame_idx_to_sample_token = {}
        sample_token = scene['first_sample_token']
        for i in range(scene['nbr_samples']):
            frame_idx_to_sample_token[i] = sample_token
            sample = nusc.get('sample', sample_token)
            if not sample['next']:
                break
            sample_token = sample['next']

        # Cache trajectories within a scene to avoid re-calculating
        all_trajs_cache = {}

        for event_idx, event in enumerate(events):
            event_folder_name = f"scene_{scene['name']}_event_{event_idx}_{event['reason']}"
            event_output_path = os.path.join(CONFIG['visualization_output_dir'], event_folder_name)
            os.makedirs(event_output_path, exist_ok=True)

            # --- 1. Prepare Trajectory Data for the Entire Event ---
            gt_traj_3d, pred_traj_3d = prepare_trajectories_for_event(nusc, event, all_trajs_cache)
            if gt_traj_3d is None or pred_traj_3d is None:
                # print(f"Skipping event {event_idx} in scene {scene['name']} due to missing trajectory data.")
                event_progress.update(1)
                continue

            # --- 2. Render Each Frame in the Event Clip ---
            for frame_idx in range(event['start_frame'], event['end_frame']):
                sample_token = frame_idx_to_sample_token.get(frame_idx)
                if not sample_token:
                    continue

                sample = nusc.get('sample', sample_token)
                key_agent_token = event['instance_token']
                interacting_tokens = event.get('interactions', {}).get(str(frame_idx), [])

                # Create a blank canvas for the 3x2 grid
                grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                # Render each camera view and place it on the grid
                for row_idx, row_cams in enumerate(CONFIG['grid_layout']):
                    for col_idx, cam_name in enumerate(row_cams):
                        cam_token = sample['data'][cam_name]
                        cam_data = nusc.get('sample_data', cam_token)

                        # Load image
                        img = cv2.imread(cam_data['filename'])

                        # --- Draw Bounding Boxes ---
                        # The SDK's get_sample_data function can render boxes directly
                        nusc.render_sample_data(cam_token, with_anns=False, ax=None, out_path=None,
                                                verbose=False)  # This is a bit of a trick

                        # Manually render boxes for coloring
                        for ann_token in sample['anns']:
                            ann = nusc.get('sample_annotation', ann_token)

                            color = None
                            if ann['instance_token'] == key_agent_token:
                                color = CONFIG['color_key_agent_box']
                            elif ann['instance_token'] in interacting_tokens:
                                color = CONFIG['color_interaction_agent_box']

                            if color:
                                box = nusc.get_box(ann_token)
                                box.render_cv2(img, view=np.array(cam_data['calibrated_sensor']['camera_intrinsic']),
                                               normalize=True, colors=(color, color, color))

                        # --- Draw Trajectories ---
                        # Get camera and ego pose information
                        cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
                        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

                        # Pre-calculate transforms
                        cam_to_ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
                        cam_to_ego_trans = np.array(cs_record['translation'])
                        ego_to_world_rot = Quaternion(pose_record['rotation']).rotation_matrix
                        ego_to_world_trans = np.array(pose_record['translation'])

                        # Project and draw GT trajectory
                        if len(gt_traj_3d) > 1:
                            points_2d, depths = project_points(gt_traj_3d, cam_intrinsic, cam_to_ego_rot,
                                                               cam_to_ego_trans, ego_to_world_rot, ego_to_world_trans)
                            valid_points = points_2d[
                                (depths > 1) & (points_2d[:, 0] > 0) & (points_2d[:, 0] < img.shape[1]) & (
                                            points_2d[:, 1] > 0) & (points_2d[:, 1] < img.shape[0])]
                            if len(valid_points) > 1:
                                cv2.polylines(img, [valid_points.astype(np.int32)], isClosed=False,
                                              color=CONFIG['color_gt_trajectory'],
                                              thickness=CONFIG['trajectory_thickness'])

                        # Project and draw Predicted trajectory
                        if len(pred_traj_3d) > 1:
                            points_2d, depths = project_points(pred_traj_3d, cam_intrinsic, cam_to_ego_rot,
                                                               cam_to_ego_trans, ego_to_world_rot, ego_to_world_trans)
                            valid_points = points_2d[
                                (depths > 1) & (points_2d[:, 0] > 0) & (points_2d[:, 0] < img.shape[1]) & (
                                            points_2d[:, 1] > 0) & (points_2d[:, 1] < img.shape[0])]
                            if len(valid_points) > 1:
                                cv2.polylines(img, [valid_points.astype(np.int32)], isClosed=False,
                                              color=CONFIG['color_pred_trajectory'],
                                              thickness=CONFIG['trajectory_thickness'])

                        # --- Place rendered image onto the grid ---
                        resized_img = cv2.resize(img, (scaled_w, scaled_h))
                        cv2.putText(resized_img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        y_offset, x_offset = row_idx * scaled_h, col_idx * scaled_w
                        grid_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_img

                # Save the final grid image for the frame
                output_filename = os.path.join(event_output_path, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(output_filename, grid_image)

            event_progress.update(1)

    event_progress.close()
    print("\nGrid visualization finished!")
    print(f"Results saved to: {CONFIG['visualization_output_dir']}")


if __name__ == '__main__':
    main()
