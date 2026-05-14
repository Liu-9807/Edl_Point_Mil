# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import pytest


def _point_pseudo_box_generator_cls():
    try:
        from mmdet.models.task_modules.prior_generators.point_pseudo_box_generator import (
            PointPseudoBoxGenerator,
        )
        return PointPseudoBoxGenerator
    except AssertionError as exc:
        pytest.skip(str(exc))


def test_point_pseudo_box_generator_deprecated_kwargs():
    PointPseudoBoxGenerator = _point_pseudo_box_generator_cls()
    with pytest.warns(DeprecationWarning):
        gen = PointPseudoBoxGenerator(
            box_sizes=[[32, 32]],
            box_offset=5,
            num_neg_samples=8,
            train_use_jitter=False,
        )
    assert gen.mil_bag_size_base == 8
    assert gen.use_jittered_positive_proposals is False


def test_point_pseudo_box_generator_new_kwargs_no_deprecation():
    PointPseudoBoxGenerator = _point_pseudo_box_generator_cls()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        gen = PointPseudoBoxGenerator(
            box_sizes=[[32, 32]],
            box_offset=5,
            mil_bag_size_base=8,
            use_jittered_positive_proposals=False,
        )
    assert gen.mil_bag_size_base == 8
