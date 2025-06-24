from radiclef.experiments.conv_to_cui_seq.main import TrainingSession
from radiclef.experiments.conv_to_cui_seq.main import CUI_ALPHABET_PATH
from radiclef.experiments.conv_to_cui_seq.main import F1Metric


import unittest

import torch


class InitDatasetsUnitTest(unittest.TestCase):
    def setUp(self):
        self.config_data = {

            "image_size": [
                1024,
                1024
            ],
            "image_positional_embedding": True,
            "image_mode": "RGB",
            "image_augment_transforms": {
                "do_transforms": True,
                "random_linear_illumination": {
                    "p": 0.3,
                    "gain": 0.1
                },
                "random_adjust_sharpness": {
                    "p": 0.1,
                    "sharpness_factor": 2
                },
                "jitter": {
                    "p": 0.1,
                    "brightness": 0.1,
                    "saturation": 0.0,
                    "contrast": 0.1
                },
                "random_resized_crop": {
                    "p": 0.1,
                    "ratio": [
                        0.8,
                        1.2
                    ],
                    "scale": [
                        0.25,
                        1.5
                    ]
                },
                "random_rotation": {
                    "p": 0.4,
                    "degrees": [
                        -45,
                        45
                    ]
                }
            }
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

        dataset = self.dataset_train.select(range(0, 100))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, collate_fn=collate_fn, shuffle=True)

        for mini_batch in dataloader:
            print(mini_batch.keys())
            print(mini_batch["image_tensor"].shape)
            print(mini_batch["cui_seq"].shape)
            print(mini_batch["cui_seq"])


class InitNetworkUnitTest(unittest.TestCase):
    def setUp(self):
        dataset, _ = TrainingSession.init_datasets_functional(
            {
                "image_size": [
                    512,
                    512
                ],
                "image_positional_embedding": True,
                "image_mode": "RGB",
            }
        )

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
                    4
                ],
                "channels_list": [
                    5,
                    8,
                    16,
                    16,
                    16
                ],
                "num_out_channels": 4,
                "proj_filter_size": 16,
                "dropout": 0.1
            },
            "sequence_generator": {
                "hidden_dim": 16,
                "vocab_size": vocab_size,
                "max_len": 32,
                "num_layers": 2,
                "dim_feedforward": 16,
                "num_heads": 1,
                "dropout": 0.1

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


class F1MetricUnitTest(unittest.TestCase):
    def setUp(self):
        dataset, _ = TrainingSession.init_datasets_functional(
            {
                "image_size": [
                    1024,
                    1024
                ],
                "image_positional_embedding": True,
                "image_mode": "RGB",
            }
        )

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
        self.metric = F1Metric()
        self.cui_obj = self.metric.CUI_OBJ

    def test(self):
        mini_batch = next(iter(self.dataloader))
        gt_seq = mini_batch["cui_seq"]
        b, _ = gt_seq.shape
        p_seq = torch.cat((
            self.cui_obj.c2i[self.cui_obj.BOS_TOKEN] * torch.ones(b, 1),
            torch.randint(0, len(self.cui_obj.alphabet), (b, 10)),
            self.cui_obj.c2i[self.cui_obj.EOS_TOKEN] * torch.ones(b, 1),
        ), dim=1).long()

        gt_1h, p_1h = self.metric._check_and_prepare_inputs(ground_truth_seq=gt_seq, prediction_seq=p_seq)

        print(gt_1h)

        self.assertEqual(self.metric.f1_score(ground_truth_seq=gt_seq, prediction_seq=gt_seq), 1.0)
        self.assertLess(self.metric.f1_score(ground_truth_seq=gt_seq, prediction_seq=p_seq), 0.01)


if __name__ == "__main__":
    unittest.main()
