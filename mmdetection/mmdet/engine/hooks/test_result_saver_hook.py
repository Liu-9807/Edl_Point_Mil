import os
import os.path as osp
from typing import List

import numpy as np
import torch
from mmengine.fileio import dump
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class TestResultSaverHook(Hook):
    """
    在 Test 过程中收集预测结果，并在测试结束后将结果以 COCO json 格式写出到指定目录。

    参数:
        out_dir(str|None): 写出目录；如果为 None，则使用 runner.work_dir/<timestamp>/coco_results
        outfile_prefix(str): 写出文件名前缀（不含扩展名）
    """

    def __init__(self, out_dir: str = None, outfile_prefix: str = 'preds') -> None:
        self._user_out_dir = out_dir
        self.out_dir = None
        self.outfile_prefix = outfile_prefix
        self._results: List[dict] = []

    def before_run(self, runner):
        if self._user_out_dir:
            self.out_dir = self._user_out_dir
        else:
            # use timestamped folder to avoid collisions
            self.out_dir = osp.join(runner.work_dir, getattr(runner, 'timestamp', ''), 'coco_results')
        if runner.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)

    def before_test(self, runner):
        # reset results for a fresh test
        self._results = []
        # ensure directory exists (in case runner called separate process)
        if self.out_dir and runner.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _xyxy2xywh(bbox: np.ndarray) -> List[float]:
        _bbox = bbox.tolist()
        return [_bbox[0], _bbox[1], _bbox[2] - _bbox[0], _bbox[3] - _bbox[1]]

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Called per test iteration; collect batch predictions into an internal list.

        outputs: typically a list of DataSample/DetDataSample objects for the batch
        """
        if outputs is None:
            return

        for data_sample in outputs:
            # support both attribute access and dict-like access
            try:
                pred = data_sample.pred_instances
                img_id = getattr(data_sample, 'img_id', None)
                ori_shape = getattr(data_sample, 'ori_shape', None)
            except Exception:
                pred = data_sample['pred_instances']
                img_id = data_sample.get('img_id', None)
                ori_shape = data_sample.get('ori_shape', None)

            result = dict()
            # image id
            result['img_id'] = int(img_id) if img_id is not None else None

            # bboxes / scores / labels
            try:
                bboxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                labels = pred.labels.cpu().numpy()
            except Exception:
                # fallback for dict-like
                bboxes = pred['bboxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()

            result['bboxes'] = bboxes
            result['scores'] = scores
            result['labels'] = labels

            # masks (optional)
            try:
                if hasattr(pred, 'masks') and pred.masks is not None:
                    masks = pred.masks
                    # use encode_mask_results compatible output if necessary later
                    result['masks'] = masks
                elif isinstance(pred, dict) and 'masks' in pred:
                    result['masks'] = pred['masks']
            except Exception:
                pass

            # store original image shape if available
            if ori_shape is not None:
                # ori_shape: (h, w, c) or similar
                try:
                    result['height'] = int(ori_shape[0])
                    result['width'] = int(ori_shape[1])
                except Exception:
                    pass

            self._results.append(result)

    def _flush_results_to_coco(self, runner):
        # Only let rank 0 write files
        if runner.rank != 0:
            return

        dataset = getattr(getattr(runner, 'test_dataloader', None), 'dataset', None)
        if dataset is None:
            runner.logger.warning('TestResultSaverHook: cannot find test dataset, skip writing COCO results')
            return

        # attempt get category mapping from dataset (CocoDataset sets cat_ids)
        cat_ids = getattr(dataset, 'cat_ids', None)
        if cat_ids is None:
            # fallback to identity mapping (assume labels are already COCO ids)
            # create simple mapping 0..N-1
            # try to infer max label
            max_label = 0
            for r in self._results:
                if 'labels' in r and len(r['labels']) > 0:
                    max_label = max(max_label, int(np.max(r['labels'])))
            cat_ids = list(range(max_label + 1))

        bbox_json_results = []
        segm_json_results = []
        for idx, result in enumerate(self._results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']

            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = int(image_id) if image_id is not None else idx
                data['bbox'] = self._xyxy2xywh(np.array(bboxes[i], dtype=float))
                data['score'] = float(scores[i])
                # map label -> category_id via dataset.cat_ids
                try:
                    data['category_id'] = int(cat_ids[int(label)])
                except Exception:
                    data['category_id'] = int(label)
                bbox_json_results.append(data)

            # segm handling if present (assume already RLE/dict format)
            if 'masks' in result and result['masks'] is not None:
                masks = result['masks']
                mask_scores = result.get('mask_scores', scores)
                for i, label in enumerate(labels):
                    data = dict()
                    data['image_id'] = int(image_id) if image_id is not None else idx
                    data['bbox'] = self._xyxy2xywh(np.array(bboxes[i], dtype=float))
                    data['score'] = float(mask_scores[i])
                    try:
                        data['category_id'] = int(cat_ids[int(label)])
                    except Exception:
                        data['category_id'] = int(label)
                    # ensure counts are str if bytes
                    seg = masks[i]
                    if isinstance(seg, dict) and isinstance(seg.get('counts', None), bytes):
                        seg['counts'] = seg['counts'].decode()
                    data['segmentation'] = seg
                    segm_json_results.append(data)

        outfile_prefix = osp.join(self.out_dir, self.outfile_prefix)
        bbox_file = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, bbox_file)

        if len(segm_json_results) > 0:
            segm_file = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, segm_file)

        runner.logger.info(f'TestResultSaverHook: wrote bbox json to {bbox_file}')
        if len(segm_json_results) > 0:
            runner.logger.info(f'TestResultSaverHook: wrote segm json to {segm_file}')

    def after_run(self, runner):
        # flush when the run completes and results exist
        if getattr(runner, 'mode', None) == 'test' or getattr(runner, 'test_dataloader', None) is not None:
            if len(self._results) > 0:
                self._flush_results_to_coco(runner)

    # also provide an explicit after_test hook if runner calls it
    def after_test(self, runner):
        if len(self._results) > 0:
            self._flush_results_to_coco(runner)
