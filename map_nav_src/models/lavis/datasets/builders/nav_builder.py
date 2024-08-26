"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.r2r_datasets import R2RNavgptDataset

from lavis.common.registry import registry


@registry.register_builder("r2r_navgpt")
class R2RNavgptBuilder(BaseDatasetBuilder):
    train_dataset_cls = R2RNavgptDataset
    eval_dataset_cls = R2RNavgptDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/r2r/defaults_r2r_navgpt.yaml"
    }