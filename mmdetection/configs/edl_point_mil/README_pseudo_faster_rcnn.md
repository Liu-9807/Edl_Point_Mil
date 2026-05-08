# Pseudo-label Faster R-CNN

This folder contains a minimal config for training Faster R-CNN on offline pseudo labels converted from COCO-format predictions.

## Workflow

1. Run PointMIL / EDL inference and export prediction JSON.
2. Convert the prediction JSON into a COCO training annotation file.
3. Train Faster R-CNN with the converted annotation file.
4. Evaluate on the real validation annotation file with `CocoMetric(metric='bbox')`.

## Conversion example

```bash
python tools/dataset_converters/convert_pseudo_coco_results.py \
  --pred-json work_dirs/coco_results/predictions.bbox.json \
  --source-ann /home/user/Dataset/YouYu-JiangYong/COCO_youyu-jiangyong/annotations/instances_data.json \
  --out-ann /home/user/Dataset/YouYu-JiangYong/COCO_youyu-jiangyong/annotations/pseudo_train.json \
  --score-thr 0.5
```

## Training example

```bash
python tools/train.py \
  configs/edl_point_mil/faster-rcnn_r50_fpn_pseudo_coco.py
```

Adjust `data_root`, `pseudo_train_ann_file`, and `val_ann_file` in the config if your dataset lives elsewhere.
