import torch
import mmcv
import os.path as osp
import os
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class MILProposalHook(Hook):
    """
    可视化 Point-based MIL 过程中生成的 Pseudo Bboxes 和参考点。
    """
    def __init__(self,
                 interval=50,
                 draw_gt_points=True,
                 draw_gt_bboxes=True,
                 show=False,
                 out_dir=None):
        self.interval = interval
        self.draw_gt_points = draw_gt_points
        self.draw_gt_bboxes = draw_gt_bboxes
        self.show = show
        self._out_dir = out_dir  # 保存用户指定的路径(如果有)
        self.out_dir = None  # 实际使用的路径,在 before_run 中设置

    def before_run(self, runner):
        """在训练开始前设置输出目录"""
        if self._out_dir is None:
            # 使用 runner 的工作目录
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp, 'vis_data/pseudo_box_vis')
        else:
            self.out_dir = self._out_dir
        
        # 创建目录
        os.makedirs(self.out_dir, exist_ok=True)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """开启记录开关"""
        if self.every_n_train_iters(runner, self.interval):
            # 递归查找 model.roi_head (考虑到可能是 DDP 包装的)
            if hasattr(runner.model, 'module'):
                model = runner.model.module
            else:
                model = runner.model
            
            if hasattr(model, 'roi_head'):
                # 设置标志位，通知 forward_train 保存中间变量
                model.roi_head.debug_proposal_vis = True

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """获取数据并绘图"""
        if self.every_n_train_iters(runner, self.interval):
            if hasattr(runner.model, 'module'):
                model = runner.model.module
            else:
                model = runner.model

            roi_head = getattr(model, 'roi_head', None)
            
            # 1. 获取暂存的数据
            debug_data = getattr(roi_head, '_last_proposal_debug', None)
            
            if debug_data is None:
                return # 没取到数据，跳过

            # 2. 遍历 Batch (为了不影响速度，一般只画第一张)
            # visualizer 已经在 runner 中初始化好了
            visualizer = runner.visualizer
            
            # 确保 visualizer 是我们自定义的，或者具备相应方法（如果没用自定义类，需在下面手写逻辑，但这里假定我们换上了 MILVisualizer）
            if not hasattr(visualizer, 'draw_mil_proposals'):
                print("Warning: Current visualizer does not support 'draw_mil_proposals'. Please check config.")
                roi_head.debug_proposal_vis = False
                return

            img_metas = debug_data['img_metas']
            batch_bag_bboxes = debug_data['batch_bag_bboxes'] # list of tensors
            gt_points = debug_data.get('gt_points', None)
            gt_bboxes_list = debug_data.get('gt_bboxes', None)
            gt_labels_list = debug_data.get('gt_labels', None)
            batch_bag_labels = debug_data.get('batch_bag_labels', None)
            bag_to_img = debug_data.get('bag_to_img', None)
            batch_instance_labels = debug_data.get('batch_instance_labels', [])

            # 只画 batch 中的第一张物理图；多包时合并该图所有包的伪框
            idx = 0
            img_path = img_metas[idx].get('img_path', None)
            
            if img_path is not None:
                # 读取原始图片 (High Res)
                image = mmcv.imread(img_path, channel_order='rgb')
            else:
                return

            points = gt_points[idx] if gt_points else None
            gt_point_labels = None
            if (points is not None and gt_labels_list is not None
                    and idx < len(gt_labels_list)):
                gl0 = gt_labels_list[idx]
                if gl0 is not None and isinstance(gl0, torch.Tensor):
                    gl0 = gl0.to(points.device)
                    npt = int(points.shape[0])
                    if gl0.numel() >= npt:
                        gt_point_labels = gl0[:npt].clone()
                    elif gl0.numel() > 0:
                        gt_point_labels = gl0.clone()

            # 必须先得到 pseudo bboxes（设备/dtype），再克隆 GT 框；否则引用未赋值的 bboxes
            if bag_to_img is not None and len(bag_to_img) == len(batch_bag_bboxes):
                sel = [i for i, im in enumerate(bag_to_img) if im == idx]
                if sel:
                    bboxes = torch.cat([batch_bag_bboxes[i] for i in sel], dim=0)
                else:
                    bboxes = batch_bag_bboxes[0]
            else:
                bboxes = batch_bag_bboxes[idx]

            gt_bboxes_vis = None
            gt_bbox_labels_vis = None
            if (gt_bboxes_list is not None and idx < len(gt_bboxes_list)
                    and gt_bboxes_list[idx] is not None):
                gb0 = gt_bboxes_list[idx]
                if isinstance(gb0, torch.Tensor) and gb0.numel() > 0:
                    gt_bboxes_vis = gb0.clone().to(bboxes.device).type_as(bboxes)
                if (gt_labels_list is not None and idx < len(gt_labels_list)
                        and gt_labels_list[idx] is not None):
                    glb = gt_labels_list[idx]
                    if isinstance(glb, torch.Tensor) and glb.numel() > 0:
                        gt_bbox_labels_vis = glb.clone().to(bboxes.device)
            
            # --- [核心修复] 利用 scale_factor 反算坐标 ---
            # 获取缩放因子 (w_scale, h_scale, w_scale, h_scale)
            scale_factor = img_metas[idx].get('scale_factor', None)
            
            if scale_factor is not None:
                # 1. 确保 scale_factor 是 Tensor 且设备一致
                if not isinstance(scale_factor, torch.Tensor):
                    sf = torch.tensor(scale_factor, device=bboxes.device, dtype=bboxes.dtype)
                else:
                    sf = scale_factor.to(bboxes.device).type_as(bboxes)

                # 2. 如果是 [w_scale, h_scale] 只有两个数，补齐为 4 个用于 box 计算
                if sf.shape[0] == 2:
                    sf_bbox = sf.repeat(2) # [w, h, w, h]
                    sf_point = sf
                else:
                    sf_bbox = sf
                    sf_point = sf[:2]

                # 3. 反算 BBoxes (Resized -> Origin)
                # 注意：这里需要先处理 Pad 带来的偏移吗？
                # MMDetection 默认 Pad 是往右下角加黑边，左上角原点不变，所以直接除以 scale 即可。
                bboxes = bboxes / sf_bbox

                # 4. 反算 Points (Resized -> Origin)
                if points is not None:
                    points = points / sf_point
                if gt_bboxes_vis is not None and gt_bboxes_vis.numel() > 0:
                    gt_bboxes_vis = gt_bboxes_vis / sf_bbox

            # --- 处理 Flip (如果开启了Flip增强) ---
            if img_metas[idx].get('flip', False):
                img_h, img_w = img_metas[idx]['ori_shape'][:2]
                direction = img_metas[idx].get('flip_direction', 'horizontal')
                if direction == 'horizontal':
                    # 翻转 Bbox x 坐标
                    bboxes_x1 = img_w - bboxes[:, 2]
                    bboxes_x2 = img_w - bboxes[:, 0]
                    bboxes[:, 0] = bboxes_x1
                    bboxes[:, 2] = bboxes_x2
                    # 翻转 Point x 坐标
                    if points is not None:
                        points[:, 0] = img_w - points[:, 0]
                    if gt_bboxes_vis is not None and gt_bboxes_vis.numel() > 0:
                        gbx1 = img_w - gt_bboxes_vis[:, 2]
                        gbx2 = img_w - gt_bboxes_vis[:, 0]
                        gt_bboxes_vis[:, 0] = gbx1
                        gt_bboxes_vis[:, 2] = gbx2
            # ----------------------------------------

            if bag_to_img is not None and len(bag_to_img) == len(batch_instance_labels):
                sel = [i for i, im in enumerate(bag_to_img) if im == idx]
                if sel and batch_instance_labels:
                    labels = torch.cat([batch_instance_labels[i] for i in sel], dim=0)
                else:
                    labels = batch_instance_labels[idx] if batch_instance_labels else None
            else:
                labels = batch_instance_labels[idx] if batch_instance_labels else None
            if bag_to_img is not None and batch_bag_labels and len(bag_to_img) == len(
                    batch_bag_labels):
                sel_l = [i for i, im in enumerate(bag_to_img) if im == idx]
                if sel_l:
                    bag_label = batch_bag_labels[sel_l[0]]
                else:
                    bag_label = batch_bag_labels[0]
            else:
                bag_label = batch_bag_labels[idx] if batch_bag_labels else None

            # 3. 调用 Visualization 方法
            drawn_img = visualizer.draw_mil_proposals(
                image,
                bboxes,
                labels,
                gt_points=points if self.draw_gt_points else None,
                gt_point_labels=gt_point_labels
                if self.draw_gt_points else None,
                gt_bboxes=gt_bboxes_vis if self.draw_gt_bboxes else None,
                gt_bbox_labels=gt_bbox_labels_vis
                if self.draw_gt_bboxes else None,
                draw_gt_bboxes=self.draw_gt_bboxes,
                bag_label=bag_label)

            # 4. 保存或上传
            # 添加到 Tensorboard / WandB
            visualizer.add_image(f'mil_debug/bag_proposals', drawn_img, step=runner.iter)

            # 可选：保存到本地
            if self.out_dir:
                filename = osp.join(self.out_dir, f'iter_{runner.iter}_bag_{idx}.png')
                mmcv.imwrite(drawn_img[..., ::-1], filename) # RGB -> BGR for storage

            # 5. 清理
            roi_head.debug_proposal_vis = False
            del roi_head._last_proposal_debug