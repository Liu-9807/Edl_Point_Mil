import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Dict, Union, Tuple
from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
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
                           bag_label=None,
                           out_file=None):
        """
        Args:
            image (np.ndarray or str): 图像路径或图像数组 (H, W, C), BGR 格式。
            pseudo_bboxes (Tensor): [N, 4] 伪框。
            pseudo_labels (Tensor): [N, ] 框的标签（通常 -1 或 0 代表背景，正数代表前景）。
            gt_points (Tensor, optional): [M, 2] 标注点 (x, y)。
            bag_label (int, optional): 当前包的标签。
        """
        # 初始化画布
        self.set_image(image)

        # 1. 准备数据：转为 CPU numpy
        if isinstance(pseudo_bboxes, torch.Tensor):
            pseudo_bboxes = pseudo_bboxes.detach().cpu().numpy()
        if isinstance(pseudo_labels, torch.Tensor):
            pseudo_labels = pseudo_labels.detach().cpu().numpy()
        
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
            # 可选: 绘制标签文字
            # positions = fg_bboxes[:, :2]  # 左上角
            # labels_str = [str(l) for l in pseudo_labels[fg_mask]]
            # self.draw_texts(labels_str, positions, colors='green')

        # 3. 绘制 GT 点 (蓝色圆点)
        if gt_points is not None:
            if isinstance(gt_points, torch.Tensor):
                gt_points = gt_points.detach().cpu().numpy()
            
            # 过滤掉无效点 (假设 padding 用 -1)
            valid_mask = gt_points[:, 0] >= 0
            valid_points = gt_points[valid_mask]

            if len(valid_points) > 0:
                self.draw_points(
                    valid_points,
                    colors='blue',
                    sizes=50, # 点的大小
                    marker='o'
                )

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
                               max_instances=10):
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
            score_text = "\n".join([f"C{c}: {s:.2f}" for c, s in enumerate(scores)])
            label_text = f"L: {instance_labels[idx]}" if instance_labels is not None else ""
            
            # 标题增加显示：这是作为 Class X 的 Top 选出来的，确定度是多少
            ax_patch.set_title(f"#{rank} (C{sort_cls} Cert:{certainty_val:.2f})\n{label_text}\n{score_text}", fontsize=8)

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

    def draw_mil_inference(self,
                           image,
                           pred_instances,
                           gt_instances=None,
                           data_sample=None, # 新增参数: 传入原始的 data_sample 以获取 metainfo
                           scale_factor=None, # 或者直接传入 scale_factor (如果是单独调用)
                           out_file=None):
        """
        绘制推理结果：展示从"不确定点"到"确定性框"的最终选择结果。
        """
        self.set_image(image)

        # 尝试从 data_sample 中提取 scale_factor [w_scale, h_scale]
        # MMDetection 的 scale_factor 通常表现为宽度和高度的缩放比例
        if scale_factor is None and data_sample is not None and 'scale_factor' in data_sample.metainfo:
            # 格式通常是 (w_scale, h_scale)
            # 有时是一个字典，有时是 tuple，需要安全提取
            sf = data_sample.metainfo['scale_factor']
            if isinstance(sf, tuple) or isinstance(sf, list):
                scale_factor = np.array(sf)
            elif isinstance(sf, dict):
                scale_factor = np.array([sf['w_scale'], sf['h_scale']])
            else:
                scale_factor = sf

        # 1. 绘制 GT Points (输入提示) - 蓝色加号
        valid_points = None
        if gt_instances is not None and hasattr(gt_instances, 'points'):
            points = gt_instances.points
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()

            # 过滤有效点: 使用 >= 0
            valid_mask = (points[:, 0] >= 0) & (points[:, 1] >= 0)
            valid_points = points[valid_mask]

            if len(valid_points) > 0:
                # 核心修复点：将点坐标除以缩放因子，恢复到原图尺度
                if scale_factor is not None:
                    valid_points = valid_points / scale_factor

                self.draw_points(valid_points, colors='blue', sizes=100, marker='+')
                # 标记 P0, P1...; 使用水平和垂直偏移保持对齐
                texts = [f"P{i}" for i in range(len(valid_points))]
                text_offset = np.array([[8, -5]], dtype=np.float64)
                self.draw_texts(texts, valid_points + text_offset,
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
                # 如果 bboxes 也未被 rescale，同样需要除以 scale_factor 恢复到原图大小
                # 注意：如果外部通过 MMDetection 标准流水线 postprocess_result 并设置了 rescale=True
                # 这边的 bboxes 可能已经是原图尺度了。如果框也不对齐，请取消下面这行的注释：
                if scale_factor is not None:
                    bboxes = bboxes / np.tile(scale_factor, 2) # [w, h, w, h] 取逆

                # 绘制边框
                self.draw_bboxes(bboxes, edge_colors='green', face_colors='none', line_widths=2)
                
                # 准备标签文本: Class|Score，绘制在框左上角（内侧偏移）
                texts = [f"C{l}|{s:.2f}" for l, s in zip(labels, scores)]
                positions = bboxes[:, :2] + np.array([[3, -20]], dtype=np.float64)  
                positions[:, 1] = np.maximum(positions[:, 1], 10) # 越界保护

                # 绘制标签背景和文字
                self.draw_texts(texts, positions, colors='white',
                                font_sizes=9,
                                horizontal_alignments='left',
                                vertical_alignments='top',
                                bboxes=dict(facecolor='green', alpha=0.6, edgecolor='none'))

                # 绘制连接线（点 → 框中心），增强可解释性
                if valid_points is not None and len(valid_points) == len(bboxes):
                    centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
                    centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
                    centers = np.stack([centers_x, centers_y], axis=1)
                    
                    self.draw_lines(
                        np.stack([valid_points[:, 0], centers[:, 0]], axis=1),
                        np.stack([valid_points[:, 1], centers[:, 1]], axis=1),
                        colors='yellow', line_styles='--', line_widths=1)

        return self.get_image()

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
