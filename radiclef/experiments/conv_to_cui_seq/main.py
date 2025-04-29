from radiclef import RESOURCES_DIR, CLEF_2025_DATABASE_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare, ImageAugment
from radiclef.networks import ConvEmbeddingToSec

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_from_disk
from sklearn.metrics import f1_score

from torchbase import TrainingBaseSession
from torchbase.utils import BaseMetricsClass, ValidationDatasetsDict

from typing import Any, Dict, List, Tuple

import os
import json

EXP_DIR = os.path.join(os.path.dirname(__file__))

CUI_ALPHABET_PATH = os.path.join(RESOURCES_DIR, "cui-alphabet.txt")
if not os.path.exists(CUI_ALPHABET_PATH):

    dataset = load_from_disk(CLEF_2025_DATABASE_PATH)["train"]
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
    CUI_ALPHABET = [v.strip() for v in f.readlines()]

CUI_OBJ = ConceptUniqueIdentifiers(alphabet=CUI_ALPHABET)


class TrainingSession(TrainingBaseSession):
    def __init__(self, config: Dict,
                 cui_object: ConceptUniqueIdentifiers,
                 runs_parent_dir: str | None = None,
                 create_run_dir_afresh: bool = True,
                 source_run_dir_tag: str | None = None,
                 tag_postfix: str | None = None):

        self.cui_object = cui_object
        super().__init__(config,
                         runs_parent_dir=runs_parent_dir,
                         create_run_dir_afresh=create_run_dir_afresh,
                         source_run_dir_tag=source_run_dir_tag,
                         tag_postfix=tag_postfix
                         )

    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        return self.init_datasets_functional(self.config_data)

    @staticmethod
    def init_datasets_functional(config_data: Dict) -> Tuple[Dataset, ValidationDatasetsDict]:

        dataset_dict = load_from_disk(CLEF_2025_DATABASE_PATH)
        image_height, image_width = config_data["image_size"]
        image_prep = ImagePrepare(standard_image_size=(image_height, image_width),
                                  standard_image_mode=config_data["image_mode"],
                                  concatenate_positional_embedding=config_data["image_positional_embedding"])

        do_image_augment_transforms = False
        if "image_augment_transforms" in config_data.keys():
            if config_data["image_augment_transforms"]["do_transforms"]:
                do_image_augment_transforms = True
                image_augment = ImageAugment(config_data)

        def map_fields(example: Dict[str, List], do_transforms: bool):
            if len(example["image"]) == 1:
                image_tensor = image_prep(example["image"][0])
                if do_transforms:
                    image_tensor = image_augment(image_tensor)

                cui_seq = torch.tensor(CUI_OBJ.encode_as_seq(example["cui_codes"][0])).unsqueeze(0)
            else:
                if do_transforms:
                    batch = [image_augment(image_prep(_img)).unsqueeze(0) for _img in example["image"]]
                else:
                    batch = [image_prep(_img).unsqueeze(0) for _img in example["image"]]

                image_tensor = torch.cat(batch, dim=0)
                cui_seq = pad_sequence([torch.tensor(CUI_OBJ.encode_as_seq(_c)) for _c in example["cui_codes"]],
                                       batch_first=True, padding_value=CUI_OBJ.c2i[CUI_OBJ.PAD_TOKEN])

            return {
                "image_tensor": image_tensor,
                "cui_seq": cui_seq
            }

        dataset_train = dataset_dict["train"]
        dataset_train.set_transform(
            lambda x: map_fields(x, do_transforms=do_image_augment_transforms))

        dataset_valid = dataset_dict["valid"]
        dataset_valid.set_transform(lambda x: map_fields(x, do_transforms=False))

        return dataset_train, ValidationDatasetsDict(
            datasets=(dataset_valid,),
            names=("valid",),
            only_for_demo=(False,)
        )

    @staticmethod
    def init_network_functional(config_network: Dict) -> torch.nn.Module:
        network = ConvEmbeddingToSec(config_network)
        if config_network["sequence_generator"].get("use-pretrained-cui-embedding"):
            from radiclef.cui_embedding import MED_CPT_PRETRAINED_EMBEDDINGS_PATH
            embedding_dict = torch.load(MED_CPT_PRETRAINED_EMBEDDINGS_PATH, weights_only=True)

            if CUI_ALPHABET[4:] != embedding_dict["cui"]:
                raise RuntimeError("The alphabets do not match.")
            embeddings: torch.Tensor = embedding_dict["data"]
            random_projection_matrix = torch.randn(embeddings.shape[1],
                                                   config_network["sequence_generator"]["hidden_dim"])
            random_projection_matrix /= torch.sqrt(torch.tensor(random_projection_matrix.shape[1], dtype=torch.float32))

            embeddings = embeddings @ random_projection_matrix
            embeddings /= embeddings.std(dim=1).reshape(-1, 1).expand_as(embeddings)
            embeddings -= embeddings.mean()

            network.seq_generator.token_embedding.weight.data[4:, :] = embeddings

        return network

    def init_network(self) -> torch.nn.Module:
        return self.init_network_functional(self.config_network)

    def init_metrics(self) -> List[BaseMetricsClass] | None:
        metrics = F1Metric(keyword_maps={"target_seq": "ground_truth_seq", "prediction_seq": "prediction_seq"})

        return [metrics]

    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        image_tensor = mini_batch["image_tensor"].to(self.device)
        cui_seq = mini_batch["cui_seq"].to(self.device)
        input_seq = cui_seq[:, :-1]
        target_seq = cui_seq[:, 1:]

        output = self.network(image_tensor, input_seq)

        return {"target_seq": target_seq, "output": output, "prediction_seq": output.argmax(dim=-1)}

    def loss_function(self, *, output: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        token_weights = torch.ones(len(self.cui_object.alphabet))
        if self.config_session.loss_function_params is not None:
            eos_token_weight = self.config_session.loss_function_params.get("eos_token_weight")
            eos_index = self.cui_object.c2i[self.cui_object.EOS_TOKEN]
            token_weights[eos_index] = eos_token_weight

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.cui_object.c2i[self.cui_object.PAD_TOKEN],
                                              weight=token_weights.to(self.device))

        loss = criterion(output.transpose(-2, -1), target_seq)

        return loss


class F1Metric(BaseMetricsClass):
    CUI_OBJ: ConceptUniqueIdentifiers = CUI_OBJ

    def __init__(self, keyword_maps: Dict[str, str] | None = None):
        if not isinstance(self.CUI_OBJ, ConceptUniqueIdentifiers):
            raise ValueError
        super().__init__(keyword_maps)

    def _check_and_prepare_inputs(self, *, ground_truth_seq: torch.Tensor,
                                  prediction_seq: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(ground_truth_seq, torch.Tensor)
        assert isinstance(prediction_seq, torch.Tensor)

        assert ground_truth_seq.shape[0] == prediction_seq.shape[0]

        b, _ = ground_truth_seq.shape

        ground_truth_seq = ground_truth_seq.tolist()
        prediction_seq = prediction_seq.tolist()

        ground_truth_multi_hot = np.zeros((b, len(self.CUI_OBJ.alphabet)), dtype=np.int64)
        predictions_multi_hot = np.zeros((b, len(self.CUI_OBJ.alphabet)), dtype=np.int64)
        for ind, (gt_seq, p_seq) in enumerate(zip(ground_truth_seq, prediction_seq)):
            gt_seq = self.CUI_OBJ.decode_preprocess(gt_seq)
            p_seq = self.CUI_OBJ.decode_preprocess(p_seq)
            ground_truth_multi_hot[ind, gt_seq] = 1
            predictions_multi_hot[ind, p_seq] = 1

        return ground_truth_multi_hot, predictions_multi_hot

    def _calculate_f1_from_multi_hot(self, *, ground_truth_multi_hot: np.ndarray,
                                     prediction_multi_hot: np.ndarray) -> float:

        assert ground_truth_multi_hot.shape == prediction_multi_hot.shape
        assert ground_truth_multi_hot.shape[1] == self.CUI_OBJ.alphabet.__len__()

        scores = []
        for yt, yp in zip(ground_truth_multi_hot, prediction_multi_hot):
            if np.sum(yt) == 0:
                continue
            f1 = f1_score(yt, yp, average="binary", zero_division=0)
            scores.append(f1)

        return np.mean(scores) if scores else 0.0

    def f1_score(self, *, ground_truth_seq: torch.Tensor,
                 prediction_seq: torch.Tensor) -> float:

        ground_truth_multi_hot, predictions_multi_hot = self._check_and_prepare_inputs(
            ground_truth_seq=ground_truth_seq, prediction_seq=prediction_seq)

        score = self._calculate_f1_from_multi_hot(ground_truth_multi_hot=ground_truth_multi_hot,
                                                  prediction_multi_hot=predictions_multi_hot)

        return score


if __name__ == "__main__":
    vocab_size = len(CUI_ALPHABET)
    with open(os.path.join(os.getcwd(), "config.json"), "r") as f:
        config_dict = json.load(f)

    config_dict["network"]["sequence_generator"]["vocab_size"] = vocab_size

    session = TrainingSession(config_dict, CUI_OBJ)
    layout_dict = {
        'Loss': {
            'Loss (train vs val)': ['Multiline', ['training/loss/epochs',
                                                  'validation-valid/loss/epochs']],
        }
    }

    session.add_writer_custom_scalar_logging_layout(layout_dict)
    session()
