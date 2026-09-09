from __future__ import annotations

import numpy as np

from cnn_zone.semantic90.scripts.prepare_v3_training_selection import (
    proportional_joint_quota,
)
from cnn_zone.semantic90.scripts.run_v3_30000 import (
    SELECTED_MODEL_IDS,
    V3_TRAINING_SPECS,
)


def test_v3_uses_only_reviewed_models_and_epoch_limits() -> None:
    assert SELECTED_MODEL_IDS == (
        "logical10/mini_inception_v2",
        "semantic224/vgg16_v2",
    )
    assert V3_TRAINING_SPECS[SELECTED_MODEL_IDS[0]].max_epochs == 100
    assert V3_TRAINING_SPECS[SELECTED_MODEL_IDS[1]].max_epochs == 35
    assert all(spec.effective_batch_size == 64 for spec in V3_TRAINING_SPECS.values())


def test_proportional_joint_quota_respects_total_bounds_and_retained() -> None:
    available = np.asarray([[50, 30, 20], [40, 35, 25]], dtype=np.int64)
    retained = np.asarray([[4, 3, 2], [2, 3, 1]], dtype=np.int64)
    quota = proportional_joint_quota(available, retained, target_count=80)
    assert int(quota.sum()) == 80
    assert np.all(quota >= retained)
    assert np.all(quota <= available)

