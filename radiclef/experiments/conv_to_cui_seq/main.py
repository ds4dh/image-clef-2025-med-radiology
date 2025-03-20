from radiclef import RESOURCES_DIR, ROCO_DATABASE_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare
from radiclef.networks import ConvEmbeddingToSec

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from torchbase import TrainingBaseSession
from torchbase.utils import BaseMetricsClass, ValidationDatasetsDict

from typing import Any, Dict, List, Tuple

import os

CUI_ALPHABET_PATH = os.path.join(RESOURCES_DIR, "cui-alphabet.txt")


class TrainingSession(TrainingBaseSession):
    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        return self.init_datasets_functional(self.config_data)

    @staticmethod
    def init_datasets_functional(config_data: Dict) -> Tuple[Dataset, ValidationDatasetsDict]:

        dataset_dict = load_from_disk(ROCO_DATABASE_PATH)

        with open(CUI_ALPHABET_PATH, "r") as f:
            cui_alphabet = [v.strip() for v in f.readlines()]

        cui_object = ConceptUniqueIdentifiers(alphabet=cui_alphabet)
        image_prep = ImagePrepare(standard_image_size=(1024, 1024),
                                  standard_image_mode=config_data["image_mode"],
                                  concatenate_positional_embedding=config_data["image_positional_embedding"])

        def map_fields(example):
            if len(example["image"]) == 1:
                image_tensor = image_prep(example["image"][0])
                cui_seq = torch.tensor(cui_object.encode_as_seq(example["cui_codes"][0]))
            else:
                image_tensor = torch.cat([image_prep(_img).unsqueeze(0) for _img in example["image"]], dim=0)
                cui_seq = pad_sequence([torch.tensor(cui_object.encode_as_seq(_c)) for _c in example["cui_codes"]],
                                       batch_first=True, padding_value=cui_object.c2i[cui_object.PAD_TOKEN])
            # TODO: Add some basic image augmentation

            return {
                "image_tensor": image_tensor,
                "cui_seq": cui_seq
            }

        dataset_train = dataset_dict["train"]
        dataset_train.set_transform(lambda x: map_fields(x))

        dataset_valid = dataset_dict["valid"]
        dataset_valid.set_transform(lambda x: map_fields(x))

        return dataset_train, ValidationDatasetsDict(
            datasets=(dataset_valid,),
            names=("valid",),
            only_for_demo=(False,)
        )

    @staticmethod
    def init_network_functional(config_network: Dict) -> torch.nn.Module:
        network = ConvEmbeddingToSec(config_network)

        return network

    def init_network(self) -> torch.nn.Module:
        return self.init_network_functional(self.config_network)

    def init_metrics(self) -> List[BaseMetricsClass] | None:
        # TODO: Add metrics from official challenge repo
        return

    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        image_tensor = mini_batch["image_tensor"].to(self.device)
        target_seq = mini_batch["cui_seq"].to(self.device)

        prediction_seq = self.network(image_tensor, target_seq)

        return {"target_seq": target_seq, "prediction_seq": prediction_seq}

    def loss_function(self, *, prediction_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(prediction_seq.transpose(-2, -1), target_seq)

        return loss


if __name__ == "__main__":

    if not os.path.exists(CUI_ALPHABET_PATH):

        dataset = load_from_disk(ROCO_DATABASE_PATH)["train"]
        dataset = dataset.remove_columns([col for col in dataset.features if col != "cui_codes"])
        cui_obj = ConceptUniqueIdentifiers()

        print("Generating the CUI alphabet for the first time ..")
        batch_size = 100
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            print("Analyzing range [{}, {}) from database of length {}".format(
                start_idx, end_idx, len(dataset)))

            items = [cui for item in dataset.select(range(start_idx, end_idx))["cui_codes"] for cui in item]
            cui_obj.integrate_items_into_alphabet(items)

        with open(CUI_ALPHABET_PATH, "w") as f:
            for vocab in cui_obj.alphabet:
                f.write(vocab + "\n")

    with open(CUI_ALPHABET_PATH, "r") as f:
        vocab_size = len([line for line in f.readlines()])

    config = {
        "session": {
            "device_name": "cuda:0",
            "num_epochs": 500,
            "mini_batch_size": 46,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "dataloader_num_workers": 0,
            "tag_postfix": None
        },
        "data": {
            "image_positional_embedding": True,
            "image_mode": "L"
        },
        "metrics": {

        },
        "network":
            {
                "architecture": "ConvEmbeddingToSec",
                "convolutional_embedding": {
                    "sampling_ratio_list": [
                        2,
                        2,
                        2,
                        2,
                        4
                    ],
                    "channels_list": [
                        3,
                        8,
                        16,
                        16,
                        16,
                        16
                    ],
                    "num_out_channels": 1,
                    "dropout": 0.1
                },
                "sequence_generator": {
                    "input_dim": 256,
                    "hidden_dim": 64,
                    "vocab_size": vocab_size,
                    "max_len": 8,
                    "num_layers": 4,
                    "num_heads": 2

                }
            }

    }

    session = TrainingSession(config)
    session()
