from radiclef.experiments.conv_to_cui_seq.main import TrainingSession
from radiclef.experiments.conv_to_cui_seq.main import CUI_ALPHABET_PATH

import unittest

import torch

import os


class InitDatasetsUnitTest(unittest.TestCase):
    def setUp(self):
        self.config_data = {
            "download_only_num_elements_of_dataset": 400
        }
        self.dataset_train, self.dataset_valid_dict = TrainingSession.init_datasets_functional(self.config_data)

    def test_getitem(self):
        idx = torch.randint(0, len(self.dataset_train), (1,))
        item = self.dataset_train[idx]
        self.assertIn("image_tensor", item.keys())
        self.assertIn("cui_seq", item.keys())
        print(item["image_tensor"].shape)
        print(item["cui_seq"].shape)

    def test_dataloader(self):
        def collate_fn(batch):
            examples = torch.utils.data.default_collate(batch)
            return examples

        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=5, collate_fn=collate_fn, shuffle=True)
        mini_batch = next(iter(dataloader))
        print(mini_batch.keys())
        print(mini_batch["image_tensor"].shape)
        print(mini_batch["cui_seq"].shape)
        print(mini_batch["cui_seq"])


class InitNetworkUnitTest(unittest.TestCase):
    def setUp(self):
        dataset, _ = TrainingSession.init_datasets_functional({"download_only_num_elements_of_dataset": 400})
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
        with open(CUI_ALPHABET_PATH, "r") as f:
            vocab_size = len([line for line in f.readlines()])
        self.config_network = {
            "architecture": "ConvEmbedding",
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

        self.network = TrainingSession.init_network_functional(config_network=self.config_network)

    def test(self):
        items = next(iter(self.dataloader))
        image_tensor = items["image_tensor"]
        cui_seq = items["cui_seq"]
        out = self.network(image_tensor, cui_seq)
        print(image_tensor.shape)
        print(out.shape)


if __name__ == "__main__":
    unittest.main()
