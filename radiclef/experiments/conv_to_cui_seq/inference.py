from radiclef import RESOURCES_DIR, ROCO_DATABASE_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare
from radiclef.networks import ConvEmbeddingToSec

from torchbase.utils.networks import load_network_from_state_dict_to_device

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from typing import Any, Dict, List, Tuple

import os
import json

CUI_ALPHABET_PATH = os.path.join(RESOURCES_DIR, "cui-alphabet.txt")

RUN_TAG = "2025-03-24_19-00-09_unige-poc"
DEVICE_NAME = "cpu"

RUN_DIR = os.path.join("./runs", RUN_TAG)

with open(os.path.join(RUN_DIR, "config.json"), "r") as f:
    config = json.load(f)

dataset_dict = load_from_disk(ROCO_DATABASE_PATH)

with open(CUI_ALPHABET_PATH, "r") as f:
    cui_alphabet = [v.strip() for v in f.readlines()]

cui_object = ConceptUniqueIdentifiers(alphabet=cui_alphabet)
image_prep = ImagePrepare(standard_image_size=(1024, 1024),
                          standard_image_mode=config["data"]["image_mode"],
                          concatenate_positional_embedding=config["data"]["image_positional_embedding"])


def map_fields(example):
    image_tensor = torch.cat([image_prep(_img).unsqueeze(0) for _img in example["image"]], dim=0)
    cui_seq = pad_sequence([torch.tensor(cui_object.encode_as_seq(_c)) for _c in example["cui_codes"]],
                           batch_first=True, padding_value=cui_object.c2i[cui_object.PAD_TOKEN])
    return {
        "image_tensor": image_tensor,
        "cui_seq": cui_seq
    }


dataset = dataset_dict["train"]
dataset.set_transform(lambda x: map_fields(x))

network = ConvEmbeddingToSec(config["network"])

network = load_network_from_state_dict_to_device(network, os.path.join(RUN_DIR, "network.pth"),
                                                 device=torch.device(DEVICE_NAME))

sample = dataset[1029]

im_tensor = sample["image_tensor"].to(torch.device(DEVICE_NAME)).unsqueeze(0)
seq = sample["cui_seq"].to(torch.device(DEVICE_NAME)).unsqueeze(0)

out = network(im_tensor, seq)
logits = torch.nn.functional.softmax(out, dim=-1)

gen = network.predict(im_tensor,
                      bos_token_idx=cui_object.c2i[cui_object.BOS_TOKEN],
                      eos_token_idx=cui_object.c2i[cui_object.EOS_TOKEN],
                      max_len=20)


print(seq)
print(gen)
