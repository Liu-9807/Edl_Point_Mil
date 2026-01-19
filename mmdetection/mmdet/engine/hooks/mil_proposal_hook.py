import torch
import mmcv
import os.path as osp
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class MILProposalHook(Hook):
    """
    可视化 Point-based MIL 过程中生成的 Pseudo Bboxes 和参考点。
    """
    def __init__(self, interval=50, draw_gt_points=True, show=False, out_dir=None):
        self.interval = interval
        self.draw_gt_points = draw_gt_points
        self.show = show
        self.out_dir = out_dir

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
            batch_bag_labels = debug_data.get('batch_bag_labels', None)

            # 只画 batch 中的第一张图
            idx = 0
            img_path = img_metas[idx].get('img_path', None)
            
            if img_path is not None:
                # 读取原始图片 (这比反归一化 tensor 效果更好)
                image = mmcv.imread(img_path, channel_order='rgb')
            else:
                # 如果没有 path (比如内存生成的数据)，则尝试从 input tensor 恢复
                # 这里暂略，通常 mmdet 数据集都有 img_path
                return

            points = gt_points[idx] if gt_points else None
            bboxes = batch_bag_bboxes[idx]
            # 这里需要对应的 instance labels，你在 mil_roi_head 中是 batch_instance_labels
            # 同样需要修改 mil_roi_head.py 将 batch_instance_labels 也存入 _last_proposal_debug
            # 假设你已经存了，key 为 'batch_instance_labels'
            labels = debug_data.get('batch_instance_labels', [None])[idx]
            bag_label = batch_bag_labels[idx] if batch_bag_labels else None

            # 3. 调用 Visualization 方法
            drawn_img = visualizer.draw_mil_proposals(
                image,
                bboxes,
                labels,
                gt_points=points,
                bag_label=bag_label
            )

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