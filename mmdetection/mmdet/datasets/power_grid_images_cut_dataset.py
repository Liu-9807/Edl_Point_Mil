# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class PowerGridImagesCutPointCocoDataset(CocoDataset):
    """COCO bbox annotations with MIL supervision points at bbox centers.

    Each instance gets ``point`` as the horizontal box center so
    ``LoadPointAnnotations`` matches the PointMIL pipeline used for GeoJSON
    point supervision. ``eval_gt_*`` mirrors instance boxes for
    ``PointMilMetric``.
    """

    METAINFO = {
        'classes': ('power', ),
        'palette': [(220, 20, 60)]
    }

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)
        if isinstance(data_info, list):
            for item in data_info:
                self._inject_point_and_eval(item)
        else:
            self._inject_point_and_eval(data_info)
        return data_info

    @staticmethod
    def _inject_point_and_eval(data_info: dict) -> None:
        instances = data_info.get('instances', [])
        eval_bboxes: List[list] = []
        eval_labels: List[int] = []
        for inst in instances:
            bbox = inst['bbox']
            x1, y1, x2, y2 = (float(b) for b in bbox)
            inst['point'] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
            eval_bboxes.append([x1, y1, x2, y2])
            eval_labels.append(int(inst['bbox_label']))
        if len(eval_bboxes) > 0:
            data_info['eval_gt_bboxes'] = np.asarray(
                eval_bboxes, dtype=np.float32).reshape(-1, 4)
            data_info['eval_gt_labels'] = np.asarray(
                eval_labels, dtype=np.int64)
        else:
            data_info['eval_gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            data_info['eval_gt_labels'] = np.zeros((0, ), dtype=np.int64)
        data_info['eval_gt_source'] = 'coco'
