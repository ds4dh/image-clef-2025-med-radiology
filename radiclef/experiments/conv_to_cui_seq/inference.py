import datasets
import pandas as pd

from radiclef import RESOURCES_DIR, ROCO_DATABASE_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare
from radiclef.networks import ConvEmbeddingToSec
from radiclef.umls_api import CONCEPT_MAP_PATH
from radiclef.experiments.conv_to_cui_seq.main import F1Metric

from torchbase.utils.networks import load_network_from_state_dict_to_device

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk

import os
import json

RUN_TAG = "2025-04-09_07-29-28_unige-poc"
DEVICE_NAME = "cuda:0"
BATCH_SIZE = 50
BEAM_SEARCH_WIDTH: None | int = 3

run_dir = os.path.join("./runs", RUN_TAG)

with open(os.path.join(run_dir, "config.json"), "r") as f:
    config = json.load(f)

network = ConvEmbeddingToSec(config["network"])
network = load_network_from_state_dict_to_device(network, os.path.join(run_dir, "network.pth"),
                                                 device=torch.device(DEVICE_NAME))

network = network.eval()

dataset_dict = load_from_disk(ROCO_DATABASE_PATH)

with open(os.path.join(RESOURCES_DIR, "cui-alphabet.txt"), "r") as f:
    cui_alphabet = [v.strip() for v in f.readlines()]

with open(CONCEPT_MAP_PATH, "r") as f:
    concept_map = json.load(f)

cui_object = ConceptUniqueIdentifiers(alphabet=cui_alphabet, concept_map=concept_map)
image_prep = ImagePrepare(standard_image_size=(config["data"]["image_size"][0], config["data"]["image_size"][1]),
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


def eval_dataset(dataset: datasets.Dataset) -> pd.DataFrame:
    metric = F1Metric()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    header = ["ground-truth-codes", "predicted-codes", "ground-truth-concepts", "predicted-concepts", "f1-score"]
    data = []
    idx_minibatch = -1
    for mini_batch in dataloader:
        idx_minibatch += 1
        print("Processing batch {}/{}".format(idx_minibatch + 1, len(dataloader)))
        im_tensor = mini_batch["image_tensor"].to(torch.device(DEVICE_NAME))
        ground_truth_seq = mini_batch["cui_seq"].to(torch.device(DEVICE_NAME))

        with torch.no_grad():
            prediction_seq = network.predict(im_tensor,
                                             bos_token_idx=cui_object.c2i[cui_object.BOS_TOKEN],
                                             eos_token_idx=cui_object.c2i[cui_object.EOS_TOKEN],
                                             max_len=32,
                                             beam_search_width=BEAM_SEARCH_WIDTH)

        for idx in range(im_tensor.shape[0]):
            _gt_seq = ground_truth_seq[idx, :]
            _p_seq = prediction_seq[idx, :]

            _data = [
                cui_object.decode(_gt_seq.tolist()),
                cui_object.decode(_p_seq.tolist()),
                cui_object.decode_mapped(_gt_seq.tolist()),
                cui_object.decode_mapped(_p_seq.tolist()),
                metric.f1_score(ground_truth_seq=_gt_seq.unsqueeze(0), prediction_seq=_p_seq.unsqueeze(0))
            ]

            data.append(_data)

    return pd.DataFrame(data=data, columns=header, index=None)


if __name__ == "__main__":
    outputs = {}
    for split in ["test", "valid", "train"]:
        print("Processing {} split .. ".format(split))
        ds = dataset_dict[split]
        image_ids = ds["id"]
        ds.set_transform(lambda x: map_fields(x))
        df = eval_dataset(ds)
        df.insert(0, 'image-ID', pd.Series(image_ids))
        df_path = os.path.join(run_dir, "inference-{}_f1-score-{}.csv".format(
            split, df["f1-score"].mean().item()))

        df.to_csv(df_path, index=False)
