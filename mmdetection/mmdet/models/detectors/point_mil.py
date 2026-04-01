from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from mmengine.logging import MessageHub


@MODELS.register_module()
class PointMIL(BaseDetector):
    """Point-based MIL Detector."""

    def __init__(self,
                 backbone,
                 neck=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # 统一使用 registry 构建，避免旧版 build_backbone/build_neck/build_head 引起的循环导入
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    # -------------------------
    # MMDetection 3.x 必需接口
    # -------------------------
    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Compute losses from a batch."""
        x = self.extract_feat(batch_inputs)
        epoch_num = kwargs.get('epoch_num', None)
        if epoch_num is None:
            message_hub = MessageHub.get_current_instance()
            if message_hub is not None:
                epoch_num = message_hub.get_info('epoch', 0)
        # 期望你的 MILRoIHead 已适配为 roi_head.loss(...)
        return self.roi_head.loss(x, batch_data_samples, epoch_num=epoch_num, **kwargs)

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True, **kwargs):
        """Predict results from a batch."""
        x = self.extract_feat(batch_inputs)
        # 期望你的 MILRoIHead 已适配为 roi_head.predict(...)
        return self.roi_head.predict(x, batch_data_samples, rescale=rescale, **kwargs)

    def _forward(self, batch_inputs, batch_data_samples=None, **kwargs):
        """Network forward (tensor mode)."""
        x = self.extract_feat(batch_inputs)
        # 给一个尽量通用的 forward：若 roi_head 支持 forward 就走它，否则返回特征
        if hasattr(self.roi_head, 'forward'):
            return self.roi_head.forward(x, batch_data_samples, **kwargs)
        return x

    # -------------------------
    # 旧版接口（可留作兼容/调试）
    # -------------------------
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = self.roi_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, img=img, **kwargs)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)
        return self.roi_head.simple_test(x, img_metas, proposals, rescale=rescale)

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        raise NotImplementedError('EDL does not support test-time augmentation')

