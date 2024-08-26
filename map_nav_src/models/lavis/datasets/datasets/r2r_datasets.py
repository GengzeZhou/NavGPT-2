import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset

import json
import copy
import torch

from PIL import Image
from PIL import ImageFile

import lmdb
import msgpack
import numpy as np

class R2RNavgptDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_processor (string): visual processor
        text_processor (string): textual processor
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): Root directory of annotations (e.g. coco/images/)
        """

        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()


    def __getitem__(self, index):

        ann = self.annotation[index]

        # Get viewpoint id
        viewpoint_id = ann["viewpoint"]
        scan = ann["scan"]

        # Read image features
        key = f"{scan}_{viewpoint_id}"
        # When batch size is larger than 1, disable the lock
        with lmdb.open(self.vis_root, readonly=True, lock=False).begin() as txn:
        # with lmdb.open(self.vis_root, readonly=True).begin() as txn:
            unpacked_data = msgpack.unpackb(txn.get(key.encode('ascii')))
            ft = np.frombuffer(unpacked_data[b'data'], dtype=unpacked_data[b'type'])
            ft = ft.reshape(unpacked_data[b'shape'])
        
        ft = torch.from_numpy(ft.copy())

        view_ix = set()
        for k, v in ann["candidate"].items():
            view_ix.add(v["viewID"])
            list_view_ix = list(view_ix)
            list_view_ix.sort()
        
        images = []
        for k, v in ann["candidate"].items():
            idx = list_view_ix.index(v["viewID"])
            image = ft[idx]
            images.append(image)
        images = torch.stack(images)
        
        # Get input text
        text = self.text_processor(ann)

        return {
            "sample_id": ann["sample_id"],
            "text_input": text,
            "qformer_text_input": ann["instruction"],
            "text_output": ann["llm_thought"],
            "images": images,
        }

    def collater(self, samples):

        text_input, qformer_text_input, text_output, images, sample_ids = [], [], [], [], []

        for i in samples:
            images.append(i['images'])
            # Get qformer text input
            n_candidates = i['images'].size(0)
            qformer_text_input.extend([i['qformer_text_input'] for _ in range(n_candidates)])
            text_input.append(i['text_input'])
            text_output.append(i['text_output'])
            sample_ids.append(i['sample_id'])
        
        images = torch.cat(images, dim=0)

        samples = {
            "sample_ids": sample_ids,
            "text_input": text_input,
            "qformer_text_input": qformer_text_input,
            "text_output": text_output,
            "images": images,
        }

        return samples