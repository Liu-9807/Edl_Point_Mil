import os.path as osp
import hashlib
import numpy as np
from typing import List, Tuple
from mmengine.utils import scandir
from mmengine.fileio import load

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class Wind_turbine_generator_Dataset(BaseDetDataset):
    """Dataset for wind generator object detection with point annotations.
    
    Annotation format:
    - JSON files contain image info and point annotations
    - Optional TXT files in 'labels' subdirectory with additional info
    
    Args:
        point_to_bbox_size (int): Convert point to bbox with this size.
            Defaults to 10 (5 pixels radius).
        use_txt_labels (bool): Whether to load txt label files.
            Defaults to True.
    """
    
    METAINFO = {
        # Keep two classes to match model head num_classes=2 during visualization.
        'classes': ('wind_generator', 'aux_class'),
        'palette': [(220, 20, 60), (0, 180, 255)]
    }

    def __init__(self,
                 point_to_bbox_size: int = 10,
                 use_txt_labels: bool = True,
                 use_yolo_box_gt: bool = False,
                 yolo_label_dir: str = '',
                 *args,
                 **kwargs):
        self.point_to_bbox_size = point_to_bbox_size
        self.use_txt_labels = use_txt_labels
        self.use_yolo_box_gt = use_yolo_box_gt
        self.yolo_label_dir = yolo_label_dir
        super().__init__(*args, **kwargs)

    def _load_yolo_gt(self, label_path: str, img_w: int, img_h: int) -> Tuple[list, list]:
        """Load YOLO format labels and convert to xyxy boxes in image space."""
        if not osp.exists(label_path):
            return [], []

        bboxes = []
        labels = []
        with open(label_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                except ValueError:
                    continue

                x1 = max(0.0, cx - bw / 2.0)
                y1 = max(0.0, cy - bh / 2.0)
                x2 = min(float(img_w), cx + bw / 2.0)
                y2 = min(float(img_h), cy + bh / 2.0)

                if x2 <= x1 or y2 <= y1:
                    continue

                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        return bboxes, labels

    def load_data_list(self) -> List[dict]:
        """Load annotations from JSON files and optional TXT files.
        
        Returns:
            List[dict]: List of data info dicts with keys:
                - img_path (str): Path to image file
                - img_id (int): Unique image ID
                - height (int): Image height
                - width (int): Image width
                - instances (List[dict]): Instance annotations with keys:
                    - bbox (list): [x1, y1, x2, y2] converted from point
                    - bbox_label (int): Category label (0 for single class)
                    - point (list): [x, y] original point annotation
                - txt_content (List[str], optional): Content from txt file
        """
        # ann_file 是标注文件路径或目录
        if osp.isdir(self.ann_file):
            ann_dir = self.ann_file
        else:
            ann_dir = osp.dirname(self.ann_file)
        
        txt_dir = osp.join(ann_dir, 'labels')
        yolo_dir = self.yolo_label_dir
        if self.use_yolo_box_gt and yolo_dir and not osp.isabs(yolo_dir):
            yolo_dir = osp.join(self.data_root, yolo_dir)
        data_list = []

        # 扫描所有 JSON 文件
        json_files = list(scandir(ann_dir, suffix='.json', recursive=False))
        
        for json_file in sorted(json_files):  # 排序保证可重复性
            ann_list = load(osp.join(ann_dir, json_file))
            
            # 兼容单个 dict 或 list 格式
            if isinstance(ann_list, dict):
                ann_list = [ann_list]

            for ann in ann_list:
                img_filename = ann['image']
                
                # 构建完整图像路径
                img_path = osp.join(ann_dir, img_filename)
                
                # 生成稳定的图像 ID
                img_id = int(hashlib.md5(img_filename.encode()).hexdigest(), 16) % (1 << 31)
                
                # 基础数据信息
                data_info = dict(
                    img_path=img_path,
                    img_id=img_id,
                    height=ann.get('height', 2304),
                    width=ann.get('width', 2304),
                    instances=[]
                )

                # 处理点标注: 转换为小边界框
                points = ann.get('points', [])
                for pt in points:
                    x, y = pt['pixel_x'], pt['pixel_y']
                    
                    # 将点转换为小边界框 (x-r, y-r, x+r, y+r)
                    radius = self.point_to_bbox_size / 2
                    bbox = [
                        max(0, x - radius),
                        max(0, y - radius),
                        min(data_info['width'], x + radius),
                        min(data_info['height'], y + radius)
                    ]
                    
                    instance = dict(
                        bbox=bbox,
                        bbox_label=0,  # 单类别
                        point=[x, y],  # 保留原始点坐标供后续使用
                        ignore_flag=0
                    )
                    data_info['instances'].append(instance)

                # 可选: 加载 TXT 标签文件
                if self.use_txt_labels:
                    txt_path = osp.join(txt_dir, osp.splitext(img_filename)[0] + '.txt')
                    txt_content = []
                    if osp.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) > 1 and parts[0] == '0':
                                    txt_content.append(' '.join(parts[1:]))
                                elif parts:
                                    txt_content.append(' '.join(parts))
                    data_info['txt_content'] = txt_content

                # 可选: 加载用于检测评估的 YOLO 框标注，放入 metainfo 传给 metric。
                if self.use_yolo_box_gt:
                    yolo_path = osp.join(yolo_dir, osp.splitext(img_filename)[0] + '.txt') if yolo_dir else ''
                    eval_gt_bboxes, eval_gt_labels = self._load_yolo_gt(
                        yolo_path, data_info['width'], data_info['height'])
                    data_info['eval_gt_bboxes'] = np.array(eval_gt_bboxes, dtype=np.float32).reshape(-1, 4)
                    data_info['eval_gt_labels'] = np.array(eval_gt_labels, dtype=np.int64)
                    data_info['eval_gt_source'] = 'yolo'

                data_list.append(data_info)
        
        return data_list

    def filter_data(self) -> List[dict]:
        """Filter images without valid instances (optional)."""
        valid_data_infos = []
        for data_info in self.data_list:
            # 可选: 过滤掉没有标注的图像
            if len(data_info['instances']) > 0:
                valid_data_infos.append(data_info)
            # 或者保留所有图像用于背景训练
            # valid_data_infos.append(data_info)
        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.
        
        Override parent method to handle point annotations.
        """
        instances = self.get_data_info(idx).get('instances', [])
        return [instance['bbox_label'] for instance in instances]


# 使用示例
if __name__ == '__main__':
    # 测试数据集加载
    dataset = Wind_turbine_generator_Dataset(
        data_root='/home/user/Dataset/YouYu-JiangYong/ori',
        ann_file='result',  # 相对于 data_root 的路径
        data_prefix=dict(img=''),  # 图像路径前缀
        point_to_bbox_size=20,  # 点转边界框大小
        use_txt_labels=True,
        test_mode=False,
        pipeline=[],  # 空 pipeline 用于测试
        backend_args=None
    )
    
    print(f'Dataset size: {len(dataset)}')
    if len(dataset) > 0:
        data_info = dataset.get_data_info(0)
        print(f'First sample: {data_info.keys()}')
        print(f'Number of instances: {len(data_info["instances"])}')
        if data_info['instances']:
            print(f'First instance: {data_info["instances"][0]}')
