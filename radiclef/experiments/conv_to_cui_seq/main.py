from radiclef import RESOURCES_DIR, ROCO_DATABASE_DOWNLOAD_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare
from radiclef.networks import ConvEmbeddingToSec

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

from torchbase import TrainingBaseSession
from torchbase.utils import BaseMetricsClass, ValidationDatasetsDict

from typing import Any, Dict, List, Tuple

import os
import tempfile

CUI_ALPHABET_PATH = os.path.join(RESOURCES_DIR, "cui-alphabet.txt")


class TrainingSession(TrainingBaseSession):
    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        return self.init_datasets_functional(self.config_data)

    @staticmethod
    def init_datasets_functional(config_data: Dict) -> Tuple[Dataset, ValidationDatasetsDict]:

        download_only_num_elements_of_dataset: None | int = config_data["download_only_num_elements_of_dataset"]

        if download_only_num_elements_of_dataset is not None:
            if not os.path.exists(ROCO_DATABASE_DOWNLOAD_PATH):
                cache_dir = tempfile.TemporaryDirectory()
                dataset_dict = load_dataset("eltorio/ROCOv2-radiology",
                                            streaming=True,
                                            cache_dir=cache_dir.name)

                dataset_dict = DatasetDict(
                    {split: Dataset.from_list(list(dataset_split.take(download_only_num_elements_of_dataset)))
                     for split, dataset_split in dataset_dict.items()
                     })
                dataset_dict.save_to_disk(ROCO_DATABASE_DOWNLOAD_PATH)
                cache_dir.cleanup()
            dataset_dict = load_from_disk(ROCO_DATABASE_DOWNLOAD_PATH)

        else:
            dataset_dict = load_dataset("eltorio/ROCOv2-radiology",
                                        streaming=False,
                                        cache_dir=ROCO_DATABASE_DOWNLOAD_PATH)

        if not os.path.exists(CUI_ALPHABET_PATH):
            dataset = load_from_disk(ROCO_DATABASE_DOWNLOAD_PATH)["train"]

            dataset = dataset.remove_columns([col for col in dataset.features if col != "cui"])
            cui_obj = ConceptUniqueIdentifiers()

            batch_size = 100
            for start_idx in range(0, len(dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset))

                items = [cui for item in dataset.select(range(start_idx, end_idx))["cui"] for cui in item]
                cui_obj.integrate_items_into_alphabet(items)

            with open(CUI_ALPHABET_PATH, "w") as f:
                for vocab in cui_obj.alphabet:
                    f.write(vocab + "\n")

        with open(CUI_ALPHABET_PATH, "r") as f:
            cui_alphabet = [v.strip() for v in f.readlines()]

        cui_obj = ConceptUniqueIdentifiers(alphabet=cui_alphabet)
        image_prep = ImagePrepare(standard_image_size=(1024, 1024), standard_image_mode="L")

        def map_fields(example):
            if len(example["image"]) == 1:
                image_tensor = image_prep(example["image"][0])
                cui_seq = torch.tensor(cui_obj.encode_as_seq(example["cui"][0]))
            else:
                image_tensor = torch.cat([image_prep(_img).unsqueeze(0) for _img in example["image"]], dim=0)
                cui_seq = pad_sequence([torch.tensor(cui_obj.encode_as_seq(_c)) for _c in example["cui"]],
                                       batch_first=True, padding_value=cui_obj.c2i[cui_obj.PAD_TOKEN])
            # TODO: Add some basic image augmentation

            return {
                "image_tensor": image_tensor,
                "cui_seq": cui_seq
            }

        dataset_train = dataset_dict["train"]
        dataset_train.set_transform(lambda x: map_fields(x))

        dataset_valid = dataset_dict["validation"]
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






