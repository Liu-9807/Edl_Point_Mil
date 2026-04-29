import os.path as osp
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.fileio import load
from mmengine.utils import scandir

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class PowerGridPointCocoDataset(BaseDetDataset):
    """Power grid dataset with point annotations (geojson/json) and COCO bbox GT.

    This dataset uses point annotations for training instances and can inject
    COCO bbox labels to ``eval_gt_bboxes`` / ``eval_gt_labels`` for evaluation.
    """

    METAINFO = {
        'classes': ('transmission_tower', 'aux_class'),
        'palette': [(220, 20, 60), (0, 180, 255)]
    }

    def __init__(self,
                 point_to_bbox_size: int = 20,
                 geojson_dir: str = 'geojsons',
                 image_root: str = 'images',
                 *args,
                 **kwargs):
        self.point_to_bbox_size = float(point_to_bbox_size)
        self.geojson_dir = geojson_dir
        self.image_root = image_root
        super().__init__(*args, **kwargs)

    @staticmethod
    def _xywh_to_xyxy(box: List[float], width: int, height: int) -> List[float]:
        x, y, w, h = [float(v) for v in box]
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + max(0.0, w))
        y2 = min(float(height), y + max(0.0, h))
        return [x1, y1, x2, y2]

    @staticmethod
    def _norm_rel_image_name(file_name: str, image_root_name: str = 'images') -> str:
        norm_name = file_name.replace('\\', '/')
        marker = f'/{image_root_name}/'
        if marker in norm_name:
            return norm_name.split(marker, 1)[1]
        if norm_name.startswith(f'{image_root_name}/'):
            return norm_name[len(image_root_name) + 1:]
        return norm_name.lstrip('./')

    @staticmethod
    def _extract_pixel_coord(feature: dict) -> Optional[Tuple[float, float]]:
        if not isinstance(feature, dict):
            return None

        geometry = feature.get('geometry', {})
        if geometry.get('type', None) != 'Point':
            return None

        props = feature.get('properties', {})
        pixel_coord = props.get('pixel_coord', None)
        if isinstance(pixel_coord, (list, tuple)) and len(pixel_coord) >= 2:
            try:
                return float(pixel_coord[0]), float(pixel_coord[1])
            except (TypeError, ValueError):
                return None

        return None

    def load_data_list(self) -> List[dict]:
        ann_path = self.ann_file
        if not osp.isabs(ann_path):
            ann_path = osp.join(self.data_root, ann_path)

        coco_data = load(ann_path)
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        categories = coco_data.get('categories', [])

        cat_ids = sorted({int(cat.get('id', 0)) for cat in categories})
        cat2label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        default_label = 0

        anns_by_img_id: Dict[int, List[dict]] = defaultdict(list)
        for ann in annotations:
            image_id = ann.get('image_id', None)
            if image_id is None:
                continue
            anns_by_img_id[int(image_id)].append(ann)

        image_infos: Dict[str, dict] = {}
        basename_to_rel: Dict[str, str] = {}
        for image_info in images:
            image_id = int(image_info.get('id', -1))
            width = int(image_info.get('width', 0))
            height = int(image_info.get('height', 0))
            rel_name = self._norm_rel_image_name(
                str(image_info.get('file_name', '')),
                image_root_name=self.image_root)

            bbox_instances = []
            for ann in anns_by_img_id.get(image_id, []):
                bbox_xyxy = self._xywh_to_xyxy(
                    ann.get('bbox', [0, 0, 0, 0]), width=width, height=height)
                if bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
                    continue

                cat_id = int(ann.get('category_id', 0))
                bbox_instances.append(
                    (bbox_xyxy, cat2label.get(cat_id, default_label)))

            image_infos[rel_name] = {
                'img_id': image_id,
                'width': width,
                'height': height,
                'eval_boxes': bbox_instances,
                'raw_file_name': str(image_info.get('file_name', '')),
            }
            basename_to_rel.setdefault(osp.basename(rel_name), rel_name)

        geo_root = self.geojson_dir
        if not osp.isabs(geo_root):
            geo_root = osp.join(self.data_root, geo_root)

        img_root = self.image_root
        if not osp.isabs(img_root):
            img_root = osp.join(self.data_root, img_root)

        data_list: List[dict] = []
        geo_files = list(scandir(geo_root, suffix=('.json', '.geojson'), recursive=True))
        radius = self.point_to_bbox_size / 2.0

        for geo_rel in sorted(geo_files):
            stem = osp.splitext(geo_rel)[0]
            rel_img_name = f'{stem}.png'

            image_info = image_infos.get(rel_img_name, None)
            if image_info is None:
                image_info = image_infos.get(basename_to_rel.get(osp.basename(rel_img_name), ''), None)
            if image_info is None:
                continue

            width = int(image_info['width'])
            height = int(image_info['height'])

            img_path = osp.join(img_root, rel_img_name)
            if not osp.exists(img_path):
                fallback_rel = basename_to_rel.get(osp.basename(rel_img_name), rel_img_name)
                img_path = osp.join(img_root, fallback_rel)
            if not osp.exists(img_path):
                raw_file_name = image_info.get('raw_file_name', '')
                if osp.isabs(raw_file_name) and osp.exists(raw_file_name):
                    img_path = raw_file_name
            if not osp.exists(img_path):
                continue

            geo_path = osp.join(geo_root, geo_rel)
            with open(geo_path, 'r', encoding='utf-8') as f:
                geo_obj = json.load(f)
            features = geo_obj.get('features', []) if isinstance(geo_obj, dict) else []

            instances = []
            for feat in features:
                pt = self._extract_pixel_coord(feat)
                if pt is None:
                    continue

                x, y = pt
                x = float(np.clip(x, 0.0, max(float(width) - 1.0, 0.0)))
                y = float(np.clip(y, 0.0, max(float(height) - 1.0, 0.0)))

                bbox = [
                    max(0.0, x - radius),
                    max(0.0, y - radius),
                    min(float(width), x + radius),
                    min(float(height), y + radius),
                ]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                instances.append(
                    dict(
                        bbox=bbox,
                        bbox_label=0,
                        point=[x, y],
                        ignore_flag=0))

            eval_gt_bboxes = np.array(
                [bbox for bbox, _ in image_info['eval_boxes']], dtype=np.float32).reshape(-1, 4)
            eval_gt_labels = np.array(
                [label for _, label in image_info['eval_boxes']], dtype=np.int64)

            data_list.append(
                dict(
                    img_path=img_path,
                    img_id=int(image_info['img_id']),
                    height=height,
                    width=width,
                    instances=instances,
                    eval_gt_bboxes=eval_gt_bboxes,
                    eval_gt_labels=eval_gt_labels,
                    eval_gt_source='coco'))

        return data_list
