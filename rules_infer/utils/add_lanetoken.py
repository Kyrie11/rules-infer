import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# ==== 配置路径 ====
NUSCENES_DATAROOT = '/data0/senzeyu2/dataset/nuscenes'
NUSCENES_VERSION = 'v1.0-trainval'
INPUT_JSON_PATH = 'social_events.json'
OUTPUT_JSON_PATH = 'social_events_2.json'

# ==== 缓存地图以避免重复加载 ====
map_cache = {}


def get_map_for_location(location, dataroot):
    """缓存并返回指定位置的 NuScenesMap 对象"""
    if location not in map_cache:
        # print(f"Loading map for location: {location}...")
        map_cache[location] = NuScenesMap(dataroot=dataroot, map_name=location)
    return map_cache[location]


def get_short_id(token):
    """生成前4位字符的ID"""
    return token[:4] if token else "None"


def enrich_frame_data(frame_list, nusc_map):
    """
    遍历运动学帧列表，为每一帧添加 lane_token 和 tl_present
    """
    for frame_data in frame_list:
        # 必须有位置信息才能查地图
        if 'position' not in frame_data:
            continue

        x, y = frame_data['position'][0], frame_data['position'][1]

        # 1. 获取最近的车道 (半径设为3米，避免偏差)
        # get_closest_lane 返回的是 token 字符串或空字符串
        lane_token = nusc_map.get_closest_lane(x, y, radius=3.0)
        frame_data['lane_token'] = lane_token if lane_token else None

        # 2. 获取附近的交通灯 (物理设施)
        # 【修正点】：参数名应为 layer_names
        tl_records = nusc_map.get_records_in_radius(x, y, radius=15.0, layer_names=['traffic_light'])

        # tl_records 是一个字典，key 是图层名，value 是 token 列表
        traffic_lights = tl_records.get('traffic_light', [])
        has_tl = len(traffic_lights) > 0

        frame_data['tl_present'] = has_tl

        # 注意：我们不添加 tl_state，因为 map 里没有颜色信息，留给 VLM 看图


def main():
    print(f"Loading NuScenes metadata from {NUSCENES_DATAROOT}...")
    # 我们需要 nusc 对象来查询 scene 对应的 location (log -> map_name)
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: {INPUT_JSON_PATH} not found.")
        return

    print(f"Reading {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r') as f:
        events = json.load(f)

    print(f"Processing {len(events)} events...")

    # 预先建立 scene_token 到 map_name 的映射，加快速度
    scene_to_location = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        scene_to_location[scene['token']] = log['location']

    for event in tqdm(events):
        scene_token = event['scene_token']
        location = scene_to_location.get(scene_token)

        if not location:
            print(f"Warning: Location not found for scene {scene_token}. Skipping map data.")
            continue

        # 获取对应的地图对象
        nusc_map = get_map_for_location(location, NUSCENES_DATAROOT)

        # ==== 1. 处理 Key Agent ====
        key_token = event['key_agent_token']
        event['key_agent_short_id'] = get_short_id(key_token)

        if 'kinematics' in event and 'key_agent' in event['kinematics']:
            enrich_frame_data(event['kinematics']['key_agent'], nusc_map)

        # ==== 2. 处理 Interacting Agents ====
        # 我们添加一个 lookup 字典，方便 VLM Prompt 构建时直接查找
        event['interacting_agents_short_ids'] = {}

        if 'kinematics' in event and 'interacting_agents' in event['kinematics']:
            # interacting_agents 是一个字典：token -> frame_list
            for agent_token, frames in event['kinematics']['interacting_agents'].items():
                # 添加 short_id 映射
                s_id = get_short_id(agent_token)
                event['interacting_agents_short_ids'][agent_token] = s_id

                # 丰富每一帧的数据
                enrich_frame_data(frames, nusc_map)

    print(f"Saving enriched data to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(events, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
