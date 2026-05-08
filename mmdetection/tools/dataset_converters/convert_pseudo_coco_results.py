import argparse
import os
import os.path as osp

import mmengine


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO-format prediction results to COCO train annotations')
    parser.add_argument(
        '--pred-json',
        required=True,
        help='Prediction json exported by TestResultSaverHook or CocoMetric')
    parser.add_argument(
        '--source-ann',
        required=True,
        help='Source COCO annotation file that provides images/categories')
    parser.add_argument(
        '--out-ann',
        required=True,
        help='Output pseudo annotation json file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='Drop predictions with score lower than this threshold')
    parser.add_argument(
        '--min-area',
        type=float,
        default=1.0,
        help='Drop predictions whose bbox area is smaller than this value')
    return parser.parse_args()


def _load_predictions(pred_json):
    data = mmengine.load(pred_json)
    if isinstance(data, dict):
        # Some dumpers may wrap payload in a dict.
        if 'results' in data:
            data = data['results']
        elif 'annotations' in data:
            data = data['annotations']
    if not isinstance(data, list):
        raise TypeError(
            f'Unsupported prediction json structure from {pred_json}: {type(data)}')
    return data


def _normalize_bbox(bbox):
    if len(bbox) != 4:
        raise ValueError(f'Expect bbox to have 4 values, but got {bbox}')
    x, y, w, h = bbox
    return [float(x), float(y), float(w), float(h)]


def convert_pseudo_labels(pred_json, source_ann, out_ann, score_thr=0.5, min_area=1.0):
    source = mmengine.load(source_ann)
    predictions = _load_predictions(pred_json)

    if 'images' not in source or 'categories' not in source:
        raise KeyError(
            f'Source annotation {source_ann} must contain images and categories')

    image_by_id = {img['id']: img for img in source['images']}

    # Build category mapping: COCO cat names -> COCO cat ids
    # This is needed because model outputs may use 0-indexed category indices
    cat_name_to_id = {cat['name']: cat['id'] for cat in source['categories']}

    pseudo = {
        'info': source.get('info', {}),
        'licenses': source.get('licenses', []),
        'images': source['images'],
        'categories': source['categories'],
        'annotations': []
    }

    if not isinstance(predictions, list):
        raise TypeError(
            'Prediction json must be a list of dictionaries with image_id/bbox/score/category_id')
    if predictions and not isinstance(predictions[0], dict):
        raise TypeError(
            'Prediction json must be a list of dictionaries with image_id/bbox/score/category_id')

    ann_id = 1
    kept = 0

    for pred in predictions:
        score = float(pred.get('score', 0.0))
        if score < score_thr:
            continue

        image_id = pred.get('image_id', None)
        if image_id is None:
            continue
        if image_id not in image_by_id:
            continue

        bbox = pred.get('bbox', None)
        category_id = pred.get('category_id', None)
        if bbox is None or category_id is None:
            continue

        # Convert 0-indexed category_id to COCO category id
        # Model outputs typically use 0-based indices, but COCO categories use 1-based ids
        pred_category_idx = int(category_id)
        if pred_category_idx >= len(source['categories']):
            continue
        # Map the prediction's category index to the actual COCO category id
        coco_category_id = source['categories'][pred_category_idx]['id']

        bbox = _normalize_bbox(bbox)
        area = float(bbox[2]) * float(bbox[3])
        if area < min_area:
            continue

        pseudo['annotations'].append(
            dict(
                id=ann_id,
                image_id=int(image_id),
                category_id=coco_category_id,
                bbox=bbox,
                area=area,
                iscrowd=0,
                score=score))
        ann_id += 1
        kept += 1

    out_dir = osp.dirname(out_ann)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mmengine.dump(pseudo, out_ann)

    print(f'Source images: {len(source["images"])}')
    print(f'Kept pseudo annotations: {kept}')
    print(f'Output annotation file: {out_ann}')


def main():
    args = parse_args()
    convert_pseudo_labels(
        pred_json=args.pred_json,
        source_ann=args.source_ann,
        out_ann=args.out_ann,
        score_thr=args.score_thr,
        min_area=args.min_area)


if __name__ == '__main__':
    main()
