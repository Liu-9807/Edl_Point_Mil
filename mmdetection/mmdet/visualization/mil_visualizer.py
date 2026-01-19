import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import cv2
from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
import io

@VISUALIZERS.register_module()
class MILVisualizer(DetLocalVisualizer):
    """继承标准检测可视化器，增加 MIL 专属的可视化方法。"""

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

        # 简单的排序逻辑：选择 Evidence 总和最大的那些实例，或者最大置信度的
        # 这里假设 instance_scores 是 EDL 输出的 alpha 或 evidence
        # scores_sum: [N]
        scores_sum = np.sum(instance_scores, axis=1)
        # 获取排序索引 (降序)
        sorted_indices = np.argsort(-scores_sum)
        
        top_indices = sorted_indices[:max_instances]
        
        # 创建 Matplotlib 画布
        # 布局：一行显示原图（画框），下面几行显示 Top K 实例 Patch + 柱状图
        num_patches = len(top_indices)
        cols = 5
        rows = (num_patches + cols - 1) // cols + 1 # +1 是为了留给原图
        
        fig = plt.figure(figsize=(15, 3 * rows))
        
        # 1. 绘制带有 bbox 的原图概览
        ax_main = plt.subplot2grid((rows, cols), (0, 0), colspan=cols)
        ax_main.imshow(image)
        ax_main.set_title("Top instances visualization")
        ax_main.axis('off')
        
        h_img, w_img = image.shape[:2]
        
        for rank, idx in enumerate(top_indices):
            bbox = bboxes[idx]
            x1, y1, x2, y2 = bbox.astype(int)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            # 在主图上画框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax_main.add_patch(rect)
            ax_main.text(x1, y1, str(rank), color='white', fontsize=8, backgroundcolor='red')

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
            
            # 在 Patch 下方或标题显示 Evidence 值
            scores = instance_scores[idx]
            # 格式化文本
            score_text = "\n".join([f"C{c}: {s:.2f}" for c, s in enumerate(scores)])
            label_text = f"L: {instance_labels[idx]}" if instance_labels is not None else ""
            
            ax_patch.set_title(f"Rank {rank} {label_text}\n{score_text}", fontsize=8)

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
        labels_name = ['Background (GT)', 'Foreground (GT)']
        
        for c in range(min(num_classes, len(colors))):
            # 找到属于该类别的样本索引
            mask = (labels == c)
            if np.sum(mask) == 0:
                continue
                
            # 获取对应样本的 evidence
            # x轴: Class 0 Evidence, y轴: Class 1 Evidence
            data = evidence_scores[mask]
            
            ax.scatter(data[:, 0], data[:, 1], 
                       c=colors[c], 
                       label=labels_name[c] if c < 2 else f'Class {c}',
                       s=10, 
                       alpha=0.5, # 半透明以观察重叠密度
                       edgecolors='none')

        # 3. 装饰图表
        ax.set_title(f"Instance Evidence Distribution (Epoch {epoch_num})\nN={total_points}")
        ax.set_xlabel("Evidence for Class 0 (Background)")
        ax.set_ylabel("Evidence for Class 1 (Foreground)")
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
