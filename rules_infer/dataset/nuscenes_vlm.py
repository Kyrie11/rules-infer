import torch
from torch.utils.data import Dataset
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import json

# 1. 定义 VLM 标签到 ID 的映射 (Taxonomy Mapping)
EVENT_TO_ID = {
    "intersection_yield": 0, "crosswalk_yield": 1, "roundabout_merge_yield": 2,"bus_stop_merge_yield":3, "U_turn_yield":4,
    "merge_compete":5, "zipper_merge":6,
    "cut_in":7, "cut_out":8, "follow_gap_opening":9,
    "double_parking_avoidance":10, "blocked_intersection_clear":11, "dooring_avoidance":12,
    "jaywalking_response":13, "bike_lane_merge_yield":14,
    "emergency_vehicle_yield":15, "courtesy_stop":16,
    "congestion_stop":17, "queue_discharge":18,
    "red_light_stop":19, "stop_sign_yield":20, "priority_violation":21,
    # ... 补全你 Taxonomy 中的所有子类
    "novel_event": 99
}
TL_STATE_TO_ID = {"red": 0, "yellow": 1, "green": 2, "unknown": 3, "no_light": 3}
ROLE_TO_ID = {"yielding": 0, "cutting_in": 1, "crossing": 2, "merging": 3, "blocking": 4, "other": 5}


class NeuroSymbolicDataset(Dataset):
    def __init__(self, vlm_json_path, nusc_root, version='v1.0-trainval', split='train'):
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        # 针对每个 location 加载地图 (NuScenes 需按 location 加载)
        self.maps = {
            loc: NuScenesMap(dataroot=nusc_root, map_name=loc)
            for loc in ["singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown", "boston-seaport"]
        }

        # 加载 VLM 标注数据
        with open(vlm_json_path, 'r') as f:
            self.vlm_data = json.load(f)  # list of dicts

        # 过滤 split (假设 vlm_data 里有 split 标记，或者根据 scene_token 过滤)
        # self.vlm_data = [x for x in self.vlm_data if x['split'] == split]

    def __len__(self):
        return len(self.vlm_data)

    def get_agent_history(self, sample_token, instance_token, n_history=20):
        """回溯获取单个 Agent 的历史轨迹"""
        # 实现回溯逻辑，返回 (T, 7) [x, y, vx, vy, ax, ay, heading]
        # 这里需要利用 self.nusc 链表向前遍历 prev token
        # 省略具体链表遍历代码，逻辑同标准 NuScenes 处理
        return np.zeros((n_history, 7))

    def get_local_map(self, nusc_map, ego_pose, radius=50):
        """获取局部车道和 Zone"""
        # 1. 获取范围内的 tokens
        x, y = ego_pose['translation'][:2]
        layers = ['lane', 'road_segment', 'ped_crossing']
        records = nusc_map.get_records_in_radius(x, y, radius, layers)

        # 2. 向量化 Lane (Discretize centerlines)
        lane_polys = []
        for lane_token in records['lane']:
            # 获取 centerline 点集并转换坐标到 ego 系
            # ...
            pass

        # 3. 构建 Zone (Intersection/Crosswalk)
        zone_polys = []
        # ...

        return lane_polys, zone_polys

    def __getitem__(self, idx):
        item = self.vlm_data[idx]

        # === 1. 解析 VLM 元数据 ===
        key_agent_id = item['actors'][0]['track_id']  # 假设第一个是 key agent
        start_frame = item['event_span']['start_frame']
        # 我们通常取 start_frame 作为当前帧 (t=0)，预测未来
        # 需要根据 frame number 找到对应的 sample_token (需预先建立映射或在 json 中存储 sample_token)
        curr_token = item['sample_token_at_start']

        sample = self.nusc.get('sample', curr_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        location = self.nusc.get('log', scene['log_token'])['location']
        nusc_map = self.maps[location]

        # === 2. 确定坐标系 (Key Agent Frame) ===
        # 找到 Key Agent 的标注
        key_ann_token = None
        for ann in sample['anns']:
            ann_rec = self.nusc.get('sample_annotation', ann)
            if ann_rec['instance_token'] == key_agent_id:
                key_ann_token = ann_rec
                break

        if key_ann_token is None:
            # 异常处理：Key agent 在该帧丢失
            return self.__getitem__((idx + 1) % len(self))

        # 构建局部坐标转换矩阵 (Global -> KeyAgent)
        ego_trans = key_ann_token['translation']
        ego_rot = Quaternion(key_ann_token['rotation'])
        # global_to_local_mat = ...

        # === 3. 提取 Agents (Graph Nodes) ===
        # 策略：Key Agent 放第 0 位，Neighbors 放后面
        agents_hist_list = []
        agents_curr_state = []  # 用于 VLM role 匹配

        # 获取 Key Agent 历史
        key_hist = self.get_agent_history(curr_token, key_agent_id)  # [T, 7]
        agents_hist_list.append(key_hist)
        agents_curr_state.append(key_ann_token)  # 存元数据

        # 获取 Neighbors
        for ann in sample['anns']:
            ann_rec = self.nusc.get('sample_annotation', ann)
            if ann_rec['instance_token'] == key_agent_id: continue
            if 'vehicle' not in ann_rec['category_name']: continue  # 或根据需要包含行人

            # 距离过滤
            dist = np.linalg.norm(np.array(ann_rec['translation']) - np.array(ego_trans))
            if dist < 50:
                hist = self.get_agent_history(curr_token, ann_rec['instance_token'])
                agents_hist_list.append(hist)
                agents_curr_state.append(ann_rec)

        # 统一坐标变换 (Batch Transform)
        # agent_seq: [N_agents, T, 7] -> 转换到 KeyAgent 坐标系

        # === 4. 提取地图 (Lane/Zone Nodes) ===
        lane_polys, zone_polys = self.get_local_map(nusc_map, key_ann_token)

        # === 5. 编码 VLM 语义标签 (Semantics) ===
        # Social Event Label
        event_label = item.get('social_event_primary', 'novel_event')
        event_id = EVENT_TO_ID.get(event_label, 99)

        # Context
        tl_state = item['context'].get('tl_state', 'unknown')
        tl_id = TL_STATE_TO_ID.get(tl_state, 3)

        # Role Encoding (对应 agents_curr_state 的顺序)
        roles = []
        vlm_actors = {a['track_id']: a['role'] for a in item['actors']}
        for ag in agents_curr_state:
            tid = ag['instance_token']
            role_str = vlm_actors.get(tid, 'other')
            roles.append(ROLE_TO_ID.get(role_str, 5))
        roles = torch.tensor(roles, dtype=torch.long)

        # GT Future (用于训练)
        # ... 获取 future trajectory

        return {
            "agent_seq": torch.tensor(np.stack(agents_hist_list)).float(),  # [N, T, 7]
            "lane_polys": torch.tensor(np.stack(lane_polys)).float(),  # [Nl, P, 4]
            "zone_polys": torch.tensor(np.stack(zone_polys)).float(),  # [Nz, P, 4]
            "vlm_tags": {
                "event_id": torch.tensor(event_id).long(),
                "tl_id": torch.tensor(tl_id).long(),
                "roles": roles  # [N]
            },
            "gt_trajs": torch.tensor(gt_future).float()
        }