# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from seqlbtoolkit.training.dataset import Batch, unpack_instances
from transformers import DataCollatorForTokenClassification


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list, return_tensors=None):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])
        padded_inputs = self.tokenizer.pad({"input_ids": tk_ids, "attention_mask": attn_masks})
        tk_ids = torch.tensor(padded_inputs.input_ids, dtype=torch.int64)
        attn_masks = torch.tensor(padded_inputs.attention_mask, dtype=torch.int64)

        max_len = tk_ids.shape[1]

        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            lbs = torch.stack(
                [torch.cat((lb, torch.full((max_len - len(lb),), self.label_pad_token_id)), dim=0) for lb in lbs]
            )
        else:
            lbs = torch.stack(
                [torch.cat((torch.full((max_len - len(lb),), self.label_pad_token_id), lb), dim=0) for lb in lbs]
            )

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
