import torch
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint

from mmdet3d.models import build_detector
from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from mmdet3d_plugin.datasets.pipelines import *
from mmdet3d_plugin.models import *

class BEVFormerWrapper:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.cfg = mmcv.Config.fromfile(config_path)
        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None

        self.model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))

        # 加载权重
        load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        self.model.to(device)
        self.model.eval()
        self.device = device

        self.test_pipeline = Compose(self.cfg.data.test.pipeline)

    def run_inference(self, img_paths, lidar2img_matrices, camera_intrinsics):
        """
        输入: 6张图片的路径，以及对应的标定参数
        输出: 解析好的 numpy 格式检测框
        """
        data = dict(
            img_filename = img_paths,
            img_shape=[(900, 1600) for _ in range(6)],  # 假设尺寸
            lidar2img=lidar2img_matrices,
        )

        data = self.test_pipeline(data)
        # 将数据转为 Tensor 并送入 GPU
        # (通常使用 collate_fn，这里简化)
        data['img_metas'] = [data['img_metas'].data]
        data['img'] = [data['img'].data.to(self.device).unsqueeze(0)]

        with torch.no_grad():
            # 这会调用 BEVFormerV2.forward_test -> simple_test
            results = self.model(return_loss=False, rescale=True, **data)

        pts_bbox = results[0]['pts_bbox']
        pred_boxes = pts_bbox['boxes_3d']  # LiDARInstance3DBoxes 对象
        scores = pts_bbox['scores_3d']
        labels = pts_bbox['labels_3d']

        # 4. 过滤低分框 (例如 score > 0.3)
        mask = scores > 0.3
        pred_boxes = pred_boxes[mask]

        tensor_boxes = pred_boxes.tensor.cpu().numpy()  # 转 numpy

        detections_for_tracker = []
        for box in tensor_boxes:
            # box: [x, y, z, dx, dy, dz, rot, vx, vy]
            x, y, z = box[0], box[1], box[2]
            dx, dy, dz = box[3], box[4], box[5]  # l, w, h
            rot = box[6]

            # 封装成我们之前约定的格式 (x, y, l, w, yaw)
            detections_for_tracker.append((x, y, dx, dy, rot))

        return detections_for_tracker


