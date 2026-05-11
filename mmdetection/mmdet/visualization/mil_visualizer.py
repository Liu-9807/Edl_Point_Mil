import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Dict, Union, Tuple
from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
from mmdet.visualization.palette import get_palette
import io

@VISUALIZERS.register_module()
class MILVisualizer(DetLocalVisualizer):
    """继承标准检测可视化器，增加 MIL 专属的可视化方法。"""

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str, Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8,
                 draw_gt_pred_overlay: bool = False,
                 gt_overlay_color: str = 'deepskyblue',
                 pred_overlay_color: str = 'lime',
                 **kwargs):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha,
            **kwargs)
        self.draw_gt_pred_overlay = draw_gt_pred_overlay
        self.gt_overlay_color = gt_overlay_color
        self.pred_overlay_color = pred_overlay_color

    def _mil_class_names(self, class_names=None, include_background=True):
        if class_names is None:
            dataset_meta = getattr(self, 'dataset_meta', None)
            class_names = dataset_meta.get('classes', None) if isinstance(dataset_meta, dict) else None
        if class_names is None:
            return None
        class_names = list(class_names)
        if include_background and (len(class_names) == 0 or class_names[0] != 'bg'):
            class_names = ['bg'] + class_names
        return class_names

    def _label_to_name(self, label, class_names=None, background_label=0):
        label = int(label)
        if label == background_label:
            return 'bg'
        if class_names is not None:
            if 0 <= label < len(class_names):
                return str(class_names[label])
            shifted = label - 1
            if 0 <= shifted < len(class_names):
                return str(class_names[shifted])
        return f'C{label}'

    def _mil_colors_for_labels(self, labels_1d):
        """BGR tuples per instance label, consistent with DetLocalVisualizer."""
        labels_1d = np.asarray(labels_1d, dtype=np.int64).reshape(-1)
        if labels_1d.size == 0:
            return []
        meta = getattr(self, 'dataset_meta', None) or {}
        palette = meta.get('palette', None)
        if palette is None:
            palette = 'random'
        max_lab = int(labels_1d.max())
        if max_lab < 0:
            return [(128, 128, 128)] * int(labels_1d.size)
        n_cls = max_lab + 1
        try:
            pal = get_palette(palette, n_cls)
        except (AssertionError, TypeError, ValueError):
            pal = get_palette('random', n_cls)
        colors = []
        for lab in labels_1d:
            li = int(lab)
            if li < 0:
                colors.append((128, 128, 128))
            elif li >= len(pal):
                colors.append(tuple(pal[-1]))
            else:
                colors.append(tuple(pal[li]))
        return colors

    @staticmethod
    def _mil_mpl_rgb_tuple(c):
        """Matplotlib text bbox facecolor needs 0-1 RGB; palette uses 0-255 uint8."""
        if isinstance(c, str):
            return c
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            if max(r, g, b) <= 1.0:
                return (r, g, b)
            return (r / 255.0, g / 255.0, b / 255.0)
        return c

    def _mil_dataset_class_name(self, label: int) -> str:
        raw = (getattr(self, 'dataset_meta', None) or {}).get('classes')
        if raw is not None:
            raw = list(raw)
            li = int(label)
            if 0 <= li < len(raw):
                return str(raw[li])
        return f'C{int(label)}'

    def _fig_to_rgb(self, fig, dpi=120):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        result_img = plt.imread(buf)
        plt.close(fig)
        if result_img.dtype == np.float32 or result_img.dtype == np.float64:
            result_img = (result_img * 255).astype(np.uint8)
        if result_img.shape[-1] == 4:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
        return result_img

    def _draw_instances_overlay(self,
                                image: np.ndarray,
                                instances,
                                classes,
                                color: str,
                                text_prefix: str,
                                line_width: int = 2,
                                show_scores: bool = False) -> np.ndarray:
        """Draw instances with a unified style/color for overlay mode."""
        self.set_image(image)

        if 'bboxes' in instances and len(instances.bboxes) > 0:
            bboxes = instances.bboxes
            labels = instances.labels if 'labels' in instances else np.zeros(
                (len(bboxes), ), dtype=np.int64)

            self.draw_bboxes(
                bboxes,
                edge_colors=color,
                face_colors='none',
                alpha=0.95,
                line_widths=line_width)

            positions = bboxes[:, :2] + np.array([[2, 2]], dtype=np.float64)
            for i, (pos, label) in enumerate(zip(positions, labels)):
                if 'label_names' in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = classes[label] if classes is not None else f'class {label}'

                if show_scores and 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text = f'{text_prefix} {label_text}: {score}'
                else:
                    label_text = f'{text_prefix} {label_text}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors='white',
                    font_sizes=10,
                    bboxes=[{
                        'facecolor': color,
                        'alpha': 0.75,
                        'pad': 0.5,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional['DetDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw GT and predictions in one image when overlay mode is enabled."""
        if (not self.draw_gt_pred_overlay or data_sample is None or
                (not draw_gt) or (not draw_pred) or
                ('gt_sem_seg' in data_sample) or ('pred_sem_seg' in data_sample) or
                ('gt_panoptic_seg' in data_sample) or ('pred_panoptic_seg' in data_sample)):
            super().add_datasample(
                name=name,
                image=image,
                data_sample=data_sample,
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                pred_score_thr=pred_score_thr,
                step=step)
            return

        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)

        data_sample = data_sample.cpu()
        drawn_img = image

        if 'gt_instances' in data_sample and draw_gt:
            drawn_img = self._draw_instances_overlay(
                drawn_img,
                data_sample.gt_instances,
                classes,
                color=self.gt_overlay_color,
                text_prefix='GT',
                line_width=max(2, self.line_width),
                show_scores=False)

        if 'pred_instances' in data_sample and draw_pred:
            pred_instances = data_sample.pred_instances
            pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
            drawn_img = self._draw_instances_overlay(
                drawn_img,
                pred_instances,
                classes,
                color=self.pred_overlay_color,
                text_prefix='PD',
                line_width=max(1, self.line_width),
                show_scores=True)

        self.set_image(drawn_img)
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

    def draw_mil_proposals(self,
                           image,
                           pseudo_bboxes,
                           pseudo_labels,
                           gt_points=None,
                           gt_point_labels=None,
                           gt_bboxes=None,
                           gt_bbox_labels=None,
                           draw_gt_bboxes=True,
                           bag_label=None,
                           out_file=None,
                           class_names=None,
                           max_label_text=40):
        """
        Args:
            image (np.ndarray or str): 图像路径或图像数组 (H, W, C), RGB 格式。
            pseudo_bboxes (Tensor): [N, 4] 伪框。
            pseudo_labels (Tensor): [N, ] 框的标签（通常 -1 或 0 代表背景，正数代表前景）。
            gt_points (Tensor, optional): [M, 2] 标注点 (x, y)。
            gt_point_labels (Tensor, optional): [M] 与 gt_points 逐行对齐的类别 id。
            gt_bboxes (Tensor, optional): [K, 4] GT 水平框（与 image 同一坐标系）。
            gt_bbox_labels (Tensor, optional): [K] GT 框类别。
            draw_gt_bboxes (bool): 是否绘制 GT 框。默认 True。
            bag_label (int, optional): 当前包的标签。
        """
        # 初始化画布
        self.set_image(image)

        # 1. 准备数据：转为 CPU numpy
        if isinstance(pseudo_bboxes, torch.Tensor):
            pseudo_bboxes = pseudo_bboxes.detach().cpu().numpy()
        if isinstance(pseudo_labels, torch.Tensor):
            pseudo_labels = pseudo_labels.detach().cpu().numpy()
        class_names = self._mil_class_names(class_names, include_background=True)

        # 1b. GT 框（先画，便于与伪框叠加对比）
        if draw_gt_bboxes and gt_bboxes is not None:
            if isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.detach().cpu().numpy()
            if isinstance(gt_bbox_labels, torch.Tensor):
                gt_bbox_labels = gt_bbox_labels.detach().cpu().numpy()
            if gt_bboxes.size > 0 and gt_bboxes.shape[-1] == 4:
                if gt_bbox_labels is not None and len(gt_bbox_labels) == len(
                        gt_bboxes):
                    gt_cols = self._mil_colors_for_labels(gt_bbox_labels)
                elif gt_bbox_labels is not None and len(gt_bbox_labels) > 0:
                    m = min(len(gt_bbox_labels), len(gt_bboxes))
                    gt_bbox_labels = gt_bbox_labels[:m]
                    gt_cols = self._mil_colors_for_labels(gt_bbox_labels)
                    gt_bboxes = gt_bboxes[:m]
                else:
                    gt_cols = 'cyan'
                lw_gt = max(2, int(self.line_width))
                self.draw_bboxes(
                    gt_bboxes,
                    edge_colors=gt_cols,
                    face_colors='none',
                    alpha=0.85,
                    line_widths=lw_gt,
                    line_styles='--')
                if gt_bbox_labels is not None and len(gt_bbox_labels) > 0:
                    m = min(len(gt_bboxes), len(gt_bbox_labels), int(max_label_text))
                    if m > 0:
                        pos = gt_bboxes[:m, :2] + np.array([[2, 2]], dtype=np.float64)
                        texts = [
                            self._mil_dataset_class_name(int(gt_bbox_labels[i]))
                            for i in range(m)
                        ]
                        tcols = self._mil_colors_for_labels(gt_bbox_labels[:m])
                        self.draw_texts(
                            texts,
                            pos,
                            colors='white',
                            font_sizes=8,
                            bboxes=[{
                                'facecolor':
                                self._mil_mpl_rgb_tuple(tcols[i]),
                                'alpha': 0.65,
                                'pad': 0.3,
                                'edgecolor': 'none'
                            } for i in range(m)])

        # 2. 分离正负样本
        # 假设 MIL 中，实例标签与包标签一致或特定类别为正，其余为负/背景
        # 这里假设 pseudo_labels < 0 或者 == num_classes 是背景，视具体 generator 逻辑而定
        # 根据你的上下文，通常 pseudo_labels 对应具体类别
        
        # 简单的区分逻辑：假设我们只想看有没有生成框
        # 我们可以根据标签给不同的颜色
        
        # 绘制所有伪框 (默认使用一种颜色，或者根据 label 区分)
        # 这里为了强调正负逻辑，我们手动设置颜色
        # 比如：前景类用绿色，背景/负样本用红色
        
        # 假设 0 到 num_classes-1 是前景
        fg_mask = pseudo_labels > 0 
        bg_mask = ~fg_mask # 或者是 ignore label

        fg_bboxes = pseudo_bboxes[fg_mask]
        bg_bboxes = pseudo_bboxes[bg_mask]

        # 绘制背景框 (红色，虚线或半透明)
        if len(bg_bboxes) > 0:
            self.draw_bboxes(
                bg_bboxes,
                edge_colors='red',
                face_colors='red',
                alpha=0.1,  # 背景框画淡一点
                line_widths=1
            )

        # 绘制前景框 (绿色)
        if len(fg_bboxes) > 0:
            self.draw_bboxes(
                fg_bboxes,
                edge_colors='green',
                face_colors='none', # 空心
                line_widths=2
            )
            fg_labels = pseudo_labels[fg_mask]
            draw_count = min(len(fg_bboxes), int(max_label_text))
            if draw_count > 0:
                positions = fg_bboxes[:draw_count, :2] + np.array([[2, 2]])
                labels_str = [
                    self._label_to_name(l, class_names, background_label=0)
                    for l in fg_labels[:draw_count]
                ]
                self.draw_texts(
                    labels_str,
                    positions,
                    colors='white',
                    font_sizes=8,
                    bboxes=[{
                        'facecolor': 'green',
                        'alpha': 0.65,
                        'pad': 0.3,
                        'edgecolor': 'none'
                    }] * draw_count)

        # 3. 绘制 GT 点（按数据集 palette 区分类别）
        if gt_points is not None:
            if isinstance(gt_points, torch.Tensor):
                gt_points = gt_points.detach().cpu().numpy()
            if isinstance(gt_point_labels, torch.Tensor):
                gt_point_labels = gt_point_labels.detach().cpu().numpy()

            valid_mask = gt_points[:, 0] >= 0
            valid_points = gt_points[valid_mask]

            if len(valid_points) > 0:
                if (gt_point_labels is not None and len(gt_point_labels) >= len(gt_points)):
                    pl = np.asarray(gt_point_labels, dtype=np.int64).reshape(-1)
                    pl = pl[:len(gt_points)][valid_mask]
                    pt_colors = self._mil_colors_for_labels(pl)
                elif gt_point_labels is not None and len(gt_point_labels) == len(
                        valid_points):
                    pt_colors = self._mil_colors_for_labels(gt_point_labels)
                else:
                    pt_colors = 'blue'
                self.draw_points(
                    valid_points,
                    colors=pt_colors,
                    sizes=50,
                    marker='o')

        # 添加标题信息
        if bag_label is not None:
            bag_label_val = bag_label.item() if isinstance(bag_label, torch.Tensor) else bag_label
            # 在左上角绘制 Bag 标签
            self.draw_texts(
                str(f"Bag: {bag_label_val}"), 
                np.array([[5, 5]]), 
                colors='white', 
                font_sizes=10,
            )

        return self.get_image()

    def draw_instance_evidence(self,
                               image,
                               rois,
                               instance_scores,
                               instance_labels=None,
                               max_instances=10,
                               class_names=None,
                               topk_per_instance=5):
        """
        绘制实例图像块及其对应的证据值/分数。
        
        Args:
            image (np.ndarray): 原图 (H, W, C), BGR 或 RGB。
            rois (Tensor): [N, 4] 或 [N, 5], 坐标对应于传入的 image 尺度。
            instance_scores (Tensor): [N, NumClasses] 实例的证据或分数。
            instance_labels (Tensor, optional): [N] 实例的真实标签或伪标签。
            max_instances (int): 最多绘制多少个实例（通常选经过排序后最重要的）。
        """
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')
            
        if isinstance(rois, torch.Tensor):
            rois = rois.detach().cpu().numpy()
        if isinstance(instance_scores, torch.Tensor):
            instance_scores = instance_scores.detach().cpu().numpy()
        if isinstance(instance_labels, torch.Tensor):
            instance_labels = instance_labels.detach().cpu().numpy()
        class_names = self._mil_class_names(class_names, include_background=True)
        
        # 处理 RoI 格式，确保是 [N, 4] (x1, y1, x2, y2)
        if rois.shape[1] == 5:
            bboxes = rois[:, 1:] 
        else:
            bboxes = rois

        # 修改后的排序逻辑：
        # 基于最大的确定度（单类别证据/总证据）进行排序
        # 对各个类别分别排序，并各取 top max_instances
        total_evidence = np.sum(instance_scores, axis=1) # [N]
        # 避免除以零
        total_evidence = np.maximum(total_evidence, 1e-6)
        
        num_classes = instance_scores.shape[1]
        top_items = [] # 存储 (index, sorting_class_id) 用于绘图
        
        for c in range(num_classes):
            # 计算该类别的确定度 (Certainty)
            certainty = instance_scores[:, c] / total_evidence
            
            # 降序排列
            sorted_indices = np.argsort(-certainty)
            top_k = sorted_indices[:max_instances]
            
            for idx in top_k:
                top_items.append((idx, c))
        
        # 创建 Matplotlib 画布
        # 布局：一行显示原图（画框），下面几行显示 Top K 实例 Patch + 柱状图
        num_patches = len(top_items)
        cols = 5
        rows = (num_patches + cols - 1) // cols + 1 # +1 是为了留给原图
        
        fig = plt.figure(figsize=(15, 3 * rows))
        
        # 1. 绘制带有 bbox 的原图概览
        ax_main = plt.subplot2grid((rows, cols), (0, 0), colspan=cols)
        ax_main.imshow(image)
        ax_main.set_title(f"Top instances by Certainty (Per Class Top {max_instances})")
        ax_main.axis('off')
        
        h_img, w_img = image.shape[:2]
        # 定义一组颜色用于不同类别 (红, 绿, 蓝, 青, 品红, 黄)
        colors_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        
        for rank, (idx, sort_cls) in enumerate(top_items):
            bbox = bboxes[idx]
            x1, y1, x2, y2 = bbox.astype(int)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            # 颜色区分类别
            color = colors_list[sort_cls % len(colors_list)]
            
            # 在主图上画框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax_main.add_patch(rect)
            # 标注 Rank 和 归属类别
            ax_main.text(x1, y1, f"{rank}|C{sort_cls}", color='white', fontsize=8, backgroundcolor=color)

            # 2. 绘制各个 Instance Patch 和 Evidence 条形图
            # 如果框无效（面积为0），跳过
            if x2 <= x1 or y2 <= y1:
                continue

            patch = image[y1:y2, x1:x2]
            
            # 子图位置
            row_idx = (rank // cols) + 1
            col_idx = rank % cols
            ax_patch = plt.subplot2grid((rows, cols), (row_idx, col_idx))
            
            # 显示 Patch
            ax_patch.imshow(patch)
            ax_patch.axis('off')
            
            # 在 Patch 下方或标题显示 详情
            scores = instance_scores[idx]
            certainty_val = scores[sort_cls] / total_evidence[idx]
            # 格式化文本
            top_ids = np.argsort(-scores)[:max(1, min(int(topk_per_instance), len(scores)))]
            score_text = "\n".join([
                f"{self._label_to_name(c, class_names)}: {scores[c]:.2f}"
                for c in top_ids
            ])
            if instance_labels is not None:
                label_text = self._label_to_name(instance_labels[idx], class_names)
                label_text = f"L: {label_text}"
            else:
                label_text = ""
            sort_name = self._label_to_name(sort_cls, class_names)
            ax_patch.set_title(
                f"#{rank} ({sort_name} Cert:{certainty_val:.2f})\n{label_text}\n{score_text}",
                fontsize=8)

        plt.tight_layout()
        
        # 转为图像数组返回
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        result_img = plt.imread(buf)
        plt.close(fig)
        
        # Matplotlib 读取的是 RGBA (0-1 float) 或 RGB，需要标准化输出
        if result_img.dtype == np.float32 or result_img.dtype == np.float64:
             result_img = (result_img * 255).astype(np.uint8)
             
        # 如果是 RGBA，转 RGB
        if result_img.shape[-1] == 4:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
            
        return result_img

    def draw_multiclass_instance_analysis(self,
                                          instance_scores,
                                          instance_labels,
                                          class_names=None,
                                          topk=(1, 5),
                                          background_label=0,
                                          title='MIL multi-class analysis'):
        if isinstance(instance_scores, torch.Tensor):
            instance_scores = instance_scores.detach().cpu().numpy()
        if isinstance(instance_labels, torch.Tensor):
            instance_labels = instance_labels.detach().cpu().numpy()

        scores = np.asarray(instance_scores)
        labels = np.asarray(instance_labels).reshape(-1).astype(np.int64)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        n = min(scores.shape[0], labels.shape[0])
        scores = scores[:n]
        labels = labels[:n]
        num_classes = int(scores.shape[1]) if scores.ndim == 2 else 0
        names = self._mil_class_names(class_names, include_background=True)
        if names is None or len(names) < num_classes:
            names = [self._label_to_name(i, names, background_label) for i in range(num_classes)]
        else:
            names = names[:num_classes]

        preds = scores.argmax(axis=1) if n > 0 and num_classes > 0 else np.zeros((0,), dtype=np.int64)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for gt, pred in zip(labels, preds):
            if 0 <= gt < num_classes and 0 <= pred < num_classes:
                confusion[gt, pred] += 1

        exact_acc = float(np.mean(preds == labels) * 100) if n > 0 else 0.0
        pred_fg = preds != background_label
        label_fg = labels != background_label
        binary_acc = float(np.mean(pred_fg == label_fg) * 100) if n > 0 else 0.0
        tp = float(np.sum(pred_fg & label_fg))
        fp = float(np.sum(pred_fg & ~label_fg))
        fn = float(np.sum(~pred_fg & label_fg))
        precision = tp / max(tp + fp, 1e-12) * 100
        recall = tp / max(tp + fn, 1e-12) * 100

        topk_lines = []
        for k in topk:
            k = int(k)
            if k <= 1 or k > num_classes:
                continue
            top_preds = np.argsort(-scores, axis=1)[:, :k]
            top_acc = np.mean(np.any(top_preds == labels[:, None], axis=1)) * 100
            topk_lines.append(f'top{k}: {top_acc:.1f}%')

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.3, 1]})
        ax_cm, ax_bar = axes
        im = ax_cm.imshow(confusion, cmap='Blues')
        ax_cm.set_title(title)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Target')
        ax_cm.set_xticks(np.arange(num_classes))
        ax_cm.set_yticks(np.arange(num_classes))
        ax_cm.set_xticklabels(names, rotation=90, fontsize=7)
        ax_cm.set_yticklabels(names, fontsize=7)
        max_count = confusion.max() if confusion.size else 0
        if num_classes <= 25:
            for y in range(num_classes):
                for x in range(num_classes):
                    value = confusion[y, x]
                    if value <= 0:
                        continue
                    color = 'white' if value > max_count * 0.45 else 'black'
                    ax_cm.text(x, y, str(value), ha='center', va='center', fontsize=6, color=color)
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        support = confusion.sum(axis=1)
        pred_count = confusion.sum(axis=0)
        x = np.arange(num_classes)
        width = 0.42
        ax_bar.bar(x - width / 2, support, width, label='target', color='#4C78A8')
        ax_bar.bar(x + width / 2, pred_count, width, label='pred', color='#F58518')
        ax_bar.set_title('Class distribution')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, rotation=90, fontsize=7)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(axis='y', alpha=0.25)

        summary = [
            f'N: {n}',
            f'exact: {exact_acc:.1f}%',
            f'fg/bg: {binary_acc:.1f}%',
            f'fg precision: {precision:.1f}%',
            f'fg recall: {recall:.1f}%',
        ] + topk_lines
        ax_bar.text(
            0.02,
            0.98,
            '\n'.join(summary),
            transform=ax_bar.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        return self._fig_to_rgb(fig)

    def draw_evidence_scatter(self,
                              evidence_scores,
                              labels,
                              epoch_num,
                              num_classes=2,
                              max_points=10000):
        """
        绘制 Epoch 级别的实例证据分布散点图 (2D)。
        
        Args:
            evidence_scores (np.ndarray): [N, 2] 证据值 (alpha - 1)。
            labels (np.ndarray): [N] 真实标签 (0 or 1)。
            epoch_num (int): 当前 Epoch 数。
            num_classes (int): 类别数 (通常为 2)。
            max_points (int): 如果点太多，为了绘图速度进行降采样。
        """
        # 1. 数据降采样 (避免几十万个点把绘图卡死)
        total_points = evidence_scores.shape[0]
        if total_points > max_points:
            indices = np.random.choice(total_points, max_points, replace=False)
            evidence_scores = evidence_scores[indices]
            labels = labels[indices]

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 2. 分类绘制
        # 假设 Class 0 是背景 (Negative)，Class 1 是前景 (Positive)
        colors = ['red', 'green', 'blue', 'orange']
        labels_name = ['Background (GT)', 'Target (GT)']
        
        for c in range(min(num_classes, len(colors))):
            # 找到属于该类别的样本索引
            mask = (labels == c)
            if np.sum(mask) == 0:
                continue
                
            # 获取对应样本的 evidence
            # x轴: Class 0 Evidence, y轴: Class 1 Evidence
            data = evidence_scores[mask]
            
            # 绘制散点
            label_text = labels_name[c] if c < 2 else f'Class {c}'
            ax.scatter(data[:, 0], data[:, 1], 
                       c=colors[c], 
                       label=label_text,
                       s=10, 
                       alpha=0.5, # 半透明以观察重叠密度
                       edgecolors='none')

            # --- 新增功能: 绘制该类别的重心 ---
            center_x = np.mean(data[:, 0])
            center_y = np.mean(data[:, 1])
            
            ax.scatter(center_x, center_y,
                       c=colors[c],
                       marker='X',
                       s=200,  # 更大的尺寸
                       linewidths=1.5,
                       edgecolors='black', # 黑色轮廓突出显示
                       label=f'{label_text} Centroid',
                       zorder=5) # 保证显示在普通散点上方
            
            # 可选：在重心旁标记坐标数值
            ax.text(center_x, center_y, f"({center_x:.1f}, {center_y:.1f})",
                    fontsize=8, fontweight='bold', color='black',
                    ha='right', va='bottom')

        # 3. 装饰图表
        ax.set_title(f"Instance Evidence Distribution (Epoch {epoch_num})\nN={total_points}")
        ax.set_xlabel("Evidence for Class 0 (Background)")
        ax.set_ylabel("Evidence for Class 1 (Target)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 强制坐标轴从0开始
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # 添加对角辅助线 (y=x)，表示不确定区域
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.2, zorder=0)

        plt.tight_layout()

        # 4. 转为图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        result_img = plt.imread(buf)
        plt.close(fig)
        
        # 标准化为 uint8 RGB
        if result_img.dtype == np.float32 or result_img.dtype == np.float64:
             result_img = (result_img * 255).astype(np.uint8)
        if result_img.shape[-1] == 4:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
            
        return result_img

    def _extract_scale_factor(self, data_sample, scale_factor=None):
        """
        统一的 scale_factor 提取方法。
        支持多种格式：tuple, list, dict, 或直接值。
        
        Args:
            data_sample: DetDataSample，可能包含 metainfo['scale_factor']
            scale_factor: 预先提供的 scale_factor
            
        Returns:
            np.ndarray: [w_scale, h_scale] 格式，或 None
        """
        if scale_factor is not None:
            return scale_factor
            
        if data_sample is None or not hasattr(data_sample, 'metainfo'):
            return None
            
        sf = data_sample.metainfo.get('scale_factor', None)
        if sf is None:
            return None
            
        # 标准化为 np.ndarray
        if isinstance(sf, (tuple, list)):
            return np.array(sf, dtype=np.float32)
        elif isinstance(sf, dict):
            return np.array([sf.get('w_scale', 1.0), sf.get('h_scale', 1.0)], dtype=np.float32)
        elif isinstance(sf, torch.Tensor):
            return sf.detach().cpu().numpy().astype(np.float32)
        elif isinstance(sf, np.ndarray):
            return sf.astype(np.float32)
        else:
            return sf

    def draw_mil_inference_stages(self,
                                  image,
                                  gt_instances=None,
                                  points=None,
                                  proposals=None,
                                  refined_bboxes=None,
                                  final_bboxes=None,
                                  data_sample=None,
                                  scale_factor=None,
                                  max_proposals=200,
                                  max_refined=200,
                                  out_file=None):
        """Draw points, proposals, refined boxes, and final boxes on one image."""
        self.set_image(image)

        scale_factor = self._extract_scale_factor(data_sample, scale_factor)
        if scale_factor is not None:
            scale_arr = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
            scale_xy = scale_arr[:2] if scale_arr.size >= 2 else None
        else:
            scale_xy = None

        def _to_numpy(data):
            if data is None:
                return None
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            # MMDet box types (e.g., HorizontalBoxes) store raw data in `.tensor`.
            tensor_data = getattr(data, 'tensor', None)
            if isinstance(tensor_data, torch.Tensor):
                return tensor_data.detach().cpu().numpy()
            if hasattr(data, 'numpy') and callable(data.numpy):
                try:
                    return data.numpy()
                except Exception:
                    pass
            return data

        def _normalize_bboxes(bboxes):
            if bboxes is None:
                return None

            bboxes = _to_numpy(bboxes)

            if isinstance(bboxes, (list, tuple)):
                merged = []
                for item in bboxes:
                    item_np = _normalize_bboxes(item)
                    if item_np is None or item_np.size == 0:
                        continue
                    merged.append(item_np)
                if not merged:
                    return np.zeros((0, 4), dtype=np.float32)
                return np.concatenate(merged, axis=0)

            try:
                bboxes = np.asarray(bboxes, dtype=np.float32)
            except (TypeError, ValueError):
                return None

            if bboxes.size == 0:
                return bboxes.reshape(0, 4)

            if bboxes.ndim == 1:
                if bboxes.shape[0] != 4:
                    return None
                return bboxes.reshape(1, 4)

            if bboxes.shape[-1] != 4:
                return None

            return bboxes.reshape(-1, 4)

        def _scale_bboxes(bboxes):
            bboxes = _normalize_bboxes(bboxes)
            if bboxes is None:
                return None
            if scale_xy is None:
                return bboxes
            # Scale boxes back to image space with safe broadcasting.
            scale_tile = np.tile(scale_xy, 2).astype(np.float32)
            scale_tile[scale_tile == 0] = 1.0
            return bboxes / scale_tile

        points_np = _to_numpy(points)
        gt_points_np = None
        gt_bboxes_np = None
        if gt_instances is not None:
            if hasattr(gt_instances, 'points'):
                gt_points_np = _to_numpy(gt_instances.points)
            if hasattr(gt_instances, 'bboxes'):
                gt_bboxes_np = _scale_bboxes(_to_numpy(gt_instances.bboxes))
        proposals_np = _scale_bboxes(_to_numpy(proposals))
        refined_np = _scale_bboxes(_to_numpy(refined_bboxes))
        final_np = _scale_bboxes(_to_numpy(final_bboxes))

        if gt_bboxes_np is not None and len(gt_bboxes_np) > 0:
            self.draw_bboxes(gt_bboxes_np, edge_colors='deepskyblue', face_colors='none',
                             alpha=0.9, line_widths=2)

        if gt_points_np is not None and len(gt_points_np) > 0:
            valid_mask = (gt_points_np[:, 0] >= 0) & (gt_points_np[:, 1] >= 0)
            valid_points = gt_points_np[valid_mask]
            if len(valid_points) > 0:
                if scale_xy is not None:
                    valid_points = valid_points / scale_xy
                self.draw_points(valid_points, colors='deepskyblue', sizes=70, marker='o')

        if points_np is not None and len(points_np) > 0:
            valid_mask = (points_np[:, 0] >= 0) & (points_np[:, 1] >= 0)
            valid_points = points_np[valid_mask]
            if len(valid_points) > 0:
                if scale_xy is not None:
                    valid_points = valid_points / scale_xy
                self.draw_points(valid_points, colors='blue', sizes=80, marker='+')

        if proposals_np is not None and len(proposals_np) > 0:
            if max_proposals is not None and len(proposals_np) > max_proposals:
                proposals_np = proposals_np[:max_proposals]
            self.draw_bboxes(proposals_np, edge_colors='gray', face_colors='none',
                             alpha=0.25, line_widths=1)

        if refined_np is not None and len(refined_np) > 0:
            if max_refined is not None and len(refined_np) > max_refined:
                refined_np = refined_np[:max_refined]
            self.draw_bboxes(refined_np, edge_colors='orange', face_colors='none',
                             alpha=0.6, line_widths=1)

        if final_np is not None and len(final_np) > 0:
            self.draw_bboxes(final_np, edge_colors='green', face_colors='none',
                             alpha=0.9, line_widths=2)

        legend = [
            'GT-Box: deepskyblue',
            'GT-Point: deepskyblue',
            'Proposals: gray',
            'Refined: orange',
            'Final: green',
        ]
        legend_positions = np.array(
            [[5, 5 + 15 * i] for i in range(len(legend))], dtype=np.float64)
        self.draw_texts(legend,
                        legend_positions,
                        colors='white',
                        font_sizes=10,
                        bboxes=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        if out_file is not None:
            mmcv.imwrite(self.get_image()[..., ::-1], out_file)

        return self.get_image()

    def draw_mil_inference_result(self,
                                  image,
                                  pred_instances,
                                  gt_instances=None,
                                  data_sample=None,
                                  scale_factor=None,
                                  mode='point_box',
                                  out_file=None):
        """
        统一的 MIL 推理可视化方法。
        
        Args:
            image: 输入图像
            pred_instances: 预测实例（boxes）
            gt_instances: 真实实例（points 或 boxes）
            data_sample: DetDataSample，用于提取 metainfo
            scale_factor: 缩放因子，优先于从 data_sample 提取
            mode: 'point_box' 绘制 GT points+Pred boxes；'box_box' 仅绘制 boxes
            out_file: 可选的输出文件路径
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        self.set_image(image)
        
        # 统一提取 scale_factor
        scale_factor = self._extract_scale_factor(data_sample, scale_factor)

        valid_points = None
        
        # 1. 处理 GT Points（仅在 mode='point_box' 时绘制）
        if mode == 'point_box' and gt_instances is not None and hasattr(gt_instances, 'points'):
            points = gt_instances.points
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()

            # 过滤有效点: 使用 >= 0
            valid_mask = (points[:, 0] >= 0) & (points[:, 1] >= 0)
            valid_points = points[valid_mask]

            if len(valid_points) > 0:
                # 核心修复点：将点坐标除以缩放因子，恢复到原图尺度
                if scale_factor is not None:
                    valid_points_scaled = valid_points / scale_factor
                else:
                    valid_points_scaled = valid_points

                self.draw_points(valid_points_scaled, colors='blue', sizes=100, marker='+')
                # 标记 P0, P1...; 使用水平和垂直偏移保持对齐
                texts = [f"P{i}" for i in range(len(valid_points_scaled))]
                text_offset = np.array([[8, -5]], dtype=np.float64)
                self.draw_texts(texts, valid_points_scaled + text_offset,
                               colors='blue', font_sizes=10,
                               horizontal_alignments='left',
                               vertical_alignments='top')

        # 2. 绘制预测框 - 绿色实线
        if pred_instances is not None:
            bboxes = pred_instances.bboxes
            scores = pred_instances.scores
            labels = pred_instances.labels
            
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

            if len(bboxes) > 0:
                # 框坐标处理：除以 scale_factor 恢复到原图大小
                bboxes_scaled = bboxes.copy()
                if scale_factor is not None:
                    bboxes_scaled = bboxes / np.tile(scale_factor, 2)

                # 绘制边框
                self.draw_bboxes(bboxes_scaled, edge_colors='green', face_colors='none', line_widths=2)
                
                # 准备标签文本: Class|Score，绘制在框左上角（内侧偏移）
                texts = [f"C{l}|{s:.2f}" for l, s in zip(labels, scores)]
                positions = bboxes_scaled[:, :2] + np.array([[3, -20]], dtype=np.float64)  
                positions[:, 1] = np.maximum(positions[:, 1], 10) # 越界保护

                # 绘制标签背景和文字
                self.draw_texts(texts, positions, colors='white',
                                font_sizes=9,
                                horizontal_alignments='left',
                                vertical_alignments='top',
                                bboxes=dict(facecolor='green', alpha=0.6, edgecolor='none'))

                # 3. 仅在 mode='point_box' 且有有效点时绘制连接线
                if mode == 'point_box' and valid_points is not None and len(valid_points) > 0 and len(bboxes_scaled) > 0:
                    # 如果点数与框数相等，绘制连接线
                    if len(valid_points) == len(bboxes_scaled):
                        valid_points_scaled = valid_points / scale_factor if scale_factor is not None else valid_points
                        centers_x = (bboxes_scaled[:, 0] + bboxes_scaled[:, 2]) / 2
                        centers_y = (bboxes_scaled[:, 1] + bboxes_scaled[:, 3]) / 2
                        centers = np.stack([centers_x, centers_y], axis=1)
                        
                        self.draw_lines(
                            np.stack([valid_points_scaled[:, 0], centers[:, 0]], axis=1),
                            np.stack([valid_points_scaled[:, 1], centers[:, 1]], axis=1),
                            colors='yellow', line_styles='--', line_widths=1)

        return self.get_image()

    def draw_mil_inference(self,
                           image,
                           pred_instances,
                           gt_instances=None,
                           data_sample=None,
                           scale_factor=None,
                           out_file=None):
        """
        旧版接口：保持向后兼容性。
        内部调用新的统一方法。
        """
        return self.draw_mil_inference_result(
            image=image,
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            data_sample=data_sample,
            scale_factor=scale_factor,
            mode='point_box',
            out_file=out_file)

    def draw_instance_mask_strength(self,
                                    image,
                                    bboxes,
                                    mask_2d,
                                    instance_scores,
                                    sample_indices,
                                    sample_tags=None,
                                    epoch_num=0,
                                    iter_num=0):
        """Visualize spatial strength of 2D masks with original image and scores."""
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
        if isinstance(mask_2d, torch.Tensor):
            mask_2d = mask_2d.detach().cpu().numpy()
        if isinstance(instance_scores, torch.Tensor):
            instance_scores = instance_scores.detach().cpu().numpy()

        if bboxes.shape[0] == 0 or len(sample_indices) == 0:
            return image

        sample_indices = [int(i) for i in sample_indices if 0 <= int(i) < bboxes.shape[0]]
        if len(sample_indices) == 0:
            return image

        if sample_tags is None:
            sample_tags = [f"#{i}" for i in range(len(sample_indices))]
        else:
            sample_tags = [str(t) for t in sample_tags]
            if len(sample_tags) < len(sample_indices):
                sample_tags.extend([f"#{i}" for i in range(len(sample_tags), len(sample_indices))])
            elif len(sample_tags) > len(sample_indices):
                sample_tags = sample_tags[:len(sample_indices)]

        cols = max(1, len(sample_indices))
        fig = plt.figure(figsize=(4 * cols, 7))
        ax_main = plt.subplot2grid((2, cols), (0, 0), colspan=cols)
        ax_main.imshow(image)
        ax_main.set_title(f"Epoch {epoch_num} Iter {iter_num}: 2D Mask Strength")
        ax_main.axis('off')

        h_img, w_img = image.shape[:2]
        for rank, idx in enumerate(sample_indices):
            x1, y1, x2, y2 = bboxes[idx].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='yellow', facecolor='none')
            ax_main.add_patch(rect)

            probs = instance_scores[idx]
            score_txt = ", ".join([f"C{c}:{s:.2f}" for c, s in enumerate(probs)])
            ax_main.text(x1, max(10, y1 - 4), f"{sample_tags[rank]} {score_txt}",
                         color='white', fontsize=8, backgroundcolor='black')

            roi_mask = mask_2d[idx]
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Resize mask to the RoI patch size, then overlay on raw patch.
            mask_up = cv2.resize(
                roi_mask.astype(np.float32),
                (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_LINEAR)
            ax_sub = plt.subplot2grid((2, cols), (1, rank))
            ax_sub.imshow(patch)
            im = ax_sub.imshow(mask_up, cmap='inferno', vmin=0.0, vmax=1.0, alpha=0.55)
            ax_sub.set_title(f"{sample_tags[rank]} mean={mask_up.mean():.3f}")
            ax_sub.axis('off')
            fig.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        out = plt.imread(buf)
        plt.close(fig)

        if out.dtype == np.float32 or out.dtype == np.float64:
            out = (out * 255).astype(np.uint8)
        if out.shape[-1] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
        return out

    def draw_mask_refine_debug_item(self, image, debug_item):
        """Render a single ROI mask refinement debug item as a small composite image."""
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')

        if debug_item is None:
            return image

        def _to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        proposal_bbox = _to_numpy(debug_item.get('proposal_bbox'))
        refined_bbox = _to_numpy(debug_item.get('refined_bbox'))
        roi_mask = _to_numpy(debug_item.get('roi_mask'))
        bin_mask = _to_numpy(debug_item.get('bin_mask'))

        if proposal_bbox is None or roi_mask is None or bin_mask is None:
            return image

        proposal_bbox = proposal_bbox.reshape(-1)
        refined_bbox = refined_bbox.reshape(-1) if refined_bbox is not None else proposal_bbox

        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = proposal_bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return image

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return image

        mask_up = cv2.resize(
            roi_mask.astype(np.float32),
            (patch.shape[1], patch.shape[0]),
            interpolation=cv2.INTER_LINEAR)
        bin_up = cv2.resize(
            bin_mask.astype(np.uint8),
            (patch.shape[1], patch.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # Map refined box into patch coordinates.
        rx1, ry1, rx2, ry2 = refined_bbox.astype(int)
        rx1 = max(0, rx1 - x1)
        ry1 = max(0, ry1 - y1)
        rx2 = min(patch.shape[1], rx2 - x1)
        ry2 = min(patch.shape[0], ry2 - y1)

        thr_used = debug_item.get('mask_thr_used', None)
        is_valid = debug_item.get('is_valid_area', None)
        mask_area = debug_item.get('mask_area', None)

        fig = plt.figure(figsize=(9, 3))
        ax0 = plt.subplot(1, 3, 1)
        ax0.imshow(patch)
        ax0.set_title('ROI Patch')
        ax0.axis('off')

        ax1 = plt.subplot(1, 3, 2)
        ax1.imshow(patch)
        ax1.imshow(mask_up, cmap='inferno', vmin=0.0, vmax=1.0, alpha=0.6)
        ax1.set_title('Mask Heatmap')
        ax1.axis('off')

        ax2 = plt.subplot(1, 3, 3)
        ax2.imshow(bin_up, cmap='gray')
        if rx2 > rx1 and ry2 > ry1:
            rect = plt.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                 linewidth=2, edgecolor='orange', facecolor='none')
            ax2.add_patch(rect)
        title_bits = []
        if thr_used is not None:
            title_bits.append(f"thr={float(thr_used):.3f}")
        if mask_area is not None:
            title_bits.append(f"area={int(mask_area)}")
        if is_valid is not None:
            title_bits.append(f"valid={bool(is_valid)}")
        ax2.set_title('Binary Mask ' + ' '.join(title_bits))
        ax2.axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        out = plt.imread(buf)
        plt.close(fig)

        if out.dtype == np.float32 or out.dtype == np.float64:
            out = (out * 255).astype(np.uint8)
        if out.shape[-1] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
        return out

    def draw_mask_refine_debug(self, image, debug_items, max_items=None):
        """Render a list of ROI mask refinement debug items into small images."""
        if debug_items is None:
            return []
        limit = len(debug_items) if max_items is None else min(int(max_items), len(debug_items))
        outputs = []
        for idx in range(limit):
            outputs.append(self.draw_mask_refine_debug_item(image, debug_items[idx]))
        return outputs

    def draw_point_proposal_score_grid(self,
                                       image,
                                       debug_item,
                                       max_items=None,
                                       cols=4,
                                       patch_size=128):
        """Render proposal crops with scores for a single point."""
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')

        if debug_item is None:
            return image

        def _to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        bboxes = _to_numpy(debug_item.get('bboxes'))
        scores = _to_numpy(debug_item.get('scores'))
        point = _to_numpy(debug_item.get('point'))
        point_index = debug_item.get('point_index', None)

        if bboxes is None or scores is None or len(bboxes) == 0:
            return image

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)

        if max_items is not None and len(scores) > int(max_items):
            order = np.argsort(scores)[::-1][:int(max_items)]
            bboxes = bboxes[order]
            scores = scores[order]

        top_index = int(np.argmax(scores)) if scores.size > 0 else -1

        num_items = len(scores)
        cols = max(1, int(cols))
        rows = int(np.ceil(num_items / cols))
        fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))

        title_bits = []
        if point_index is not None:
            title_bits.append(f"P{int(point_index)}")
        if point is not None and point.size >= 2:
            title_bits.append(f"({float(point[0]):.1f},{float(point[1]):.1f})")
        if title_bits:
            fig.suptitle(' '.join(title_bits), fontsize=10)

        img_h, img_w = image.shape[:2]
        for idx in range(rows * cols):
            ax = plt.subplot(rows, cols, idx + 1)
            ax.axis('off')
            if idx >= num_items:
                continue

            x1, y1, x2, y2 = bboxes[idx].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                ax.set_title(f"s={scores[idx]:.3f}")
                continue

            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                ax.set_title(f"s={scores[idx]:.3f}")
                continue

            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            ax.imshow(patch)
            title = f"s={scores[idx]:.3f}"
            if idx == top_index:
                title = f"TOP {title}"
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            ax.set_title(title, fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        out = plt.imread(buf)
        plt.close(fig)

        if out.dtype == np.float32 or out.dtype == np.float64:
            out = (out * 255).astype(np.uint8)
        if out.shape[-1] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
        return out

    def draw_proposal_score_debug(self, image, debug_items, max_points=None, max_items=None):
        """Render per-point proposal score grids."""
        if debug_items is None:
            return []
        limit = len(debug_items) if max_points is None else min(int(max_points), len(debug_items))
        outputs = []
        for idx in range(limit):
            outputs.append(self.draw_point_proposal_score_grid(
                image=image,
                debug_item=debug_items[idx],
                max_items=max_items))
        return outputs

    def draw_point_proposal_mask_refine_grid(self,
                                             image,
                                             score_item,
                                             mask_items,
                                             max_items=None,
                                             cols=4,
                                             patch_size=160):
        """Render proposal score grid with mask refine overlay for one point."""
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')

        if score_item is None:
            return image

        def _to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        bboxes = _to_numpy(score_item.get('bboxes'))
        scores = _to_numpy(score_item.get('scores'))
        proposal_indices = _to_numpy(score_item.get('proposal_indices'))
        point = _to_numpy(score_item.get('point'))
        point_index = score_item.get('point_index', None)

        if bboxes is None or scores is None or len(bboxes) == 0:
            return image

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        if proposal_indices is not None:
            proposal_indices = proposal_indices.reshape(-1)

        if max_items is not None and len(scores) > int(max_items):
            order = np.argsort(scores)[::-1][:int(max_items)]
            bboxes = bboxes[order]
            scores = scores[order]
            if proposal_indices is not None:
                proposal_indices = proposal_indices[order]

        top_index = int(np.argmax(scores)) if scores.size > 0 else -1

        mask_map = {}
        if mask_items:
            for item in mask_items:
                idx = item.get('index', None)
                if idx is None:
                    continue
                mask_map[int(idx)] = item

        num_items = len(scores)
        cols = max(1, int(cols))
        rows = int(np.ceil(num_items / cols))
        fig = plt.figure(figsize=(cols * 2.8, rows * 2.8))

        title_bits = []
        if point_index is not None:
            title_bits.append(f"P{int(point_index)}")
        if point is not None and point.size >= 2:
            title_bits.append(f"({float(point[0]):.1f},{float(point[1]):.1f})")
        if title_bits:
            fig.suptitle(' '.join(title_bits), fontsize=10)

        img_h, img_w = image.shape[:2]
        for idx in range(rows * cols):
            ax = plt.subplot(rows, cols, idx + 1)
            ax.axis('off')
            if idx >= num_items:
                continue

            x1, y1, x2, y2 = bboxes[idx].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                ax.set_title(f"s={scores[idx]:.3f}")
                continue

            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                ax.set_title(f"s={scores[idx]:.3f}")
                continue

            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            ax.imshow(patch)

            mask_item = None
            if proposal_indices is not None and idx < proposal_indices.size:
                mask_item = mask_map.get(int(proposal_indices[idx]), None)

            if mask_item is not None:
                roi_mask = _to_numpy(mask_item.get('roi_mask'))
                bin_mask = _to_numpy(mask_item.get('bin_mask'))
                refined_bbox = _to_numpy(mask_item.get('refined_bbox'))
                if roi_mask is not None:
                    mask_up = cv2.resize(
                        roi_mask.astype(np.float32),
                        (patch.shape[1], patch.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
                    ax.imshow(mask_up, cmap='inferno', vmin=0.0, vmax=1.0, alpha=0.55)

                if refined_bbox is not None:
                    rx1, ry1, rx2, ry2 = refined_bbox.astype(int)
                    rx1 = max(0, rx1 - x1)
                    ry1 = max(0, ry1 - y1)
                    rx2 = min(patch.shape[1], rx2 - x1)
                    ry2 = min(patch.shape[0], ry2 - y1)
                    if rx2 > rx1 and ry2 > ry1:
                        rect = plt.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                             linewidth=2, edgecolor='orange', facecolor='none')
                        ax.add_patch(rect)

                if bin_mask is not None:
                    valid = mask_item.get('is_valid_area', None)
                    if valid is False:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('gray')
                            spine.set_linewidth(1.5)

            title = f"s={scores[idx]:.3f}"
            if idx == top_index:
                title = f"TOP {title}"
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            ax.set_title(title, fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        out = plt.imread(buf)
        plt.close(fig)

        if out.dtype == np.float32 or out.dtype == np.float64:
            out = (out * 255).astype(np.uint8)
        if out.shape[-1] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
        return out

    def draw_proposal_mask_refine_debug(self,
                                        image,
                                        score_items,
                                        mask_items,
                                        max_points=None,
                                        max_items=None):
        """Render per-point combined proposal score + mask refine grids."""
        if score_items is None:
            return []
        limit = len(score_items) if max_points is None else min(int(max_points), len(score_items))
        outputs = []
        for idx in range(limit):
            outputs.append(self.draw_point_proposal_mask_refine_grid(
                image=image,
                score_item=score_items[idx],
                mask_items=mask_items,
                max_items=max_items))
        return outputs
