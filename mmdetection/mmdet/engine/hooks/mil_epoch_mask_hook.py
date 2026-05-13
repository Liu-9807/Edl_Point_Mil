import warnings

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MILEpochMaskHook(Hook):
    """Deprecated: mask debug visualization is merged into ``MILEvidenceHook``."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'MILEpochMaskHook is deprecated; bag + mask visualization is merged '
            'into MILEvidenceHook. Remove MILEpochMaskHook from custom_hooks.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def before_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        pass

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        pass

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        pass

    def after_train_epoch(self, runner):
        pass
