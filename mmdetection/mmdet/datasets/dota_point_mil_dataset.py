import hashlib
import os.path as osp
import struct
from typing import List, Optional, Tuple

import numpy as np
from mmengine.utils import scandir

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class DotaPointMilDataset(BaseDetDataset):
    """DOTA-v2.0 dataset for PointMIL with points sampled from boxes.

    The dataset reads DOTA ``labelTxt`` annotations, converts oriented boxes to
    horizontal boxes, and attaches one point per object for PointMIL prompts.
    """

    METAINFO = {
        'classes': (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter',
            'container-crane', 'airport', 'helipad'),
        'palette': [
            (165, 42, 42), (0, 192, 0), (196, 196, 196), (190, 153, 153),
            (180, 165, 180), (90, 120, 150), (102, 102, 156),
            (128, 64, 255), (140, 140, 200), (170, 170, 170),
            (250, 170, 160), (96, 96, 96), (230, 150, 140),
            (128, 64, 128), (110, 110, 110), (244, 35, 232),
            (150, 100, 100), (70, 70, 70)
        ]
    }

    def __init__(self,
                 point_seed: int = 42,
                 point_sampling: str = 'random_hbb',
                 ignore_difficult: bool = False,
                 image_suffix: str = '.png',
                 *args,
                 **kwargs) -> None:
        self.point_seed = int(point_seed)
        self.point_sampling = point_sampling
        self.ignore_difficult = bool(ignore_difficult)
        self.image_suffix = image_suffix
        self.cat2label = {
            name: idx
            for idx, name in enumerate(self.METAINFO['classes'])
        }
        super().__init__(*args, **kwargs)

    def _resolve_path(self, path: str) -> str:
        if osp.isabs(path):
            return path
        return osp.join(self.data_root, path)

    def _build_image_index(self, img_root: str) -> dict:
        image_index = {}
        for rel_path in scandir(img_root, suffix=self.image_suffix, recursive=True):
            image_index.setdefault(osp.splitext(osp.basename(rel_path))[0],
                                   osp.join(img_root, rel_path))
        return image_index

    @staticmethod
    def _image_size(img_path: str) -> Tuple[int, int]:
        with open(img_path, 'rb') as f:
            header = f.read(24)
        if header.startswith(b'\x89PNG\r\n\x1a\n') and len(header) >= 24:
            width, height = struct.unpack('>II', header[16:24])
            return int(width), int(height)

        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
            return int(width), int(height)
        except Exception:
            import mmcv
            img = mmcv.imread(img_path, flag='unchanged')
            if img is None:
                raise FileNotFoundError(f'Cannot read image: {img_path}')
            height, width = img.shape[:2]
            return int(width), int(height)

    @staticmethod
    def _obb_to_hbb(coords: List[float], width: int, height: int) -> Optional[List[float]]:
        xs = np.asarray(coords[0::2], dtype=np.float32)
        ys = np.asarray(coords[1::2], dtype=np.float32)
        x1 = float(np.clip(xs.min(), 0.0, float(width)))
        y1 = float(np.clip(ys.min(), 0.0, float(height)))
        x2 = float(np.clip(xs.max(), 0.0, float(width)))
        y2 = float(np.clip(ys.max(), 0.0, float(height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _rng_for_ann(self, label_rel_path: str, ann_idx: int) -> np.random.Generator:
        key = f'{self.point_seed}:{label_rel_path}:{ann_idx}'
        digest = hashlib.md5(key.encode('utf-8')).hexdigest()
        seed = int(digest[:8], 16)
        return np.random.default_rng(seed)

    def _sample_point(self, bbox: List[float], label_rel_path: str,
                      ann_idx: int) -> List[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if self.point_sampling == 'center':
            return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        if self.point_sampling != 'random_hbb':
            raise ValueError(
                f'Unsupported point_sampling={self.point_sampling!r}; '
                "expected 'random_hbb' or 'center'.")

        rng = self._rng_for_ann(label_rel_path, ann_idx)
        x = float(rng.uniform(x1, x2)) if x2 > x1 else x1
        y = float(rng.uniform(y1, y2)) if y2 > y1 else y1
        return [x, y]

    def _parse_label_file(self, label_path: str, label_rel_path: str,
                          width: int, height: int) -> List[dict]:
        instances = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for ann_idx, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue

                try:
                    coords = [float(v) for v in parts[:8]]
                except ValueError:
                    continue

                cls_name = parts[8]
                if cls_name not in self.cat2label:
                    continue
                try:
                    difficulty = int(float(parts[9]))
                except ValueError:
                    difficulty = 0
                if self.ignore_difficult and difficulty > 0:
                    continue

                bbox = self._obb_to_hbb(coords, width=width, height=height)
                if bbox is None:
                    continue

                instances.append(
                    dict(
                        bbox=bbox,
                        bbox_label=self.cat2label[cls_name],
                        point=self._sample_point(bbox, label_rel_path, ann_idx),
                        ignore_flag=0,
                        difficulty=difficulty))
        return instances

    def load_data_list(self) -> List[dict]:
        ann_root = self._resolve_path(self.ann_file)
        img_prefix = self.data_prefix.get('img', '')
        img_root = self._resolve_path(img_prefix)
        image_index = self._build_image_index(img_root)

        data_list = []
        label_files = list(scandir(ann_root, suffix='.txt', recursive=True))
        for img_id, label_rel_path in enumerate(sorted(label_files)):
            stem = osp.splitext(osp.basename(label_rel_path))[0]
            img_path = image_index.get(stem, osp.join(img_root, stem + self.image_suffix))
            if not osp.exists(img_path):
                continue

            width, height = self._image_size(img_path)
            label_path = osp.join(ann_root, label_rel_path)
            instances = self._parse_label_file(
                label_path, label_rel_path, width=width, height=height)

            eval_gt_bboxes = np.array(
                [inst['bbox'] for inst in instances], dtype=np.float32).reshape(-1, 4)
            eval_gt_labels = np.array(
                [inst['bbox_label'] for inst in instances], dtype=np.int64)

            data_list.append(
                dict(
                    img_path=img_path,
                    img_id=img_id,
                    height=height,
                    width=width,
                    instances=instances,
                    eval_gt_bboxes=eval_gt_bboxes,
                    eval_gt_labels=eval_gt_labels,
                    eval_gt_source='dota_hbb'))
        return data_list

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        filter_empty_gt = True
        if self.filter_cfg is not None:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', True)
        if not filter_empty_gt:
            return self.data_list
        return [
            data_info for data_info in self.data_list
            if len(data_info.get('instances', [])) > 0
        ]
