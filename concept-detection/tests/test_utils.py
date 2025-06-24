from radiclef.utils import ConceptUniqueIdentifiers
from radiclef.utils import ImagePrepare

from radiclef import ROCO_DATABASE_PATH

import torch

from datasets import Dataset, load_from_disk

import unittest
import tempfile


class ConceptUniqueIdentifiersUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        dataset_ = load_from_disk(ROCO_DATABASE_PATH)["train"]

        cls.dataset = Dataset.from_list(list(dataset_.take(20)))
        cls.cui = ConceptUniqueIdentifiers()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()
        # os._exit(0)

    def test(self):
        items = [cui for item in self.dataset.select(range(0, 10))["cui_codes"] for cui in item]
        self.cui.integrate_items_into_alphabet(items)
        print(self.cui.alphabet)
        items = [cui for item in self.dataset.select(range(10, 20))["cui_codes"] for cui in item]
        self.cui.integrate_items_into_alphabet(items)
        print(self.cui.alphabet)
        decoded_encoded_items = [self.cui.i2c[self.cui.c2i[item]] for item in items]
        self.assertEqual(items, decoded_encoded_items)


class ImagePrepareUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        dataset_ = load_from_disk(ROCO_DATABASE_PATH)["train"]

        cls.dataset = Dataset.from_list(list(dataset_.take(10)))

        cls.standard_image_size = (300, 320)
        cls.standard_image_mode = "RGB"
        cls.image_prep = ImagePrepare(standard_image_size=cls.standard_image_size,
                                      standard_image_mode=cls.standard_image_mode)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()
        # os._exit(0)

    def test_coord_grid(self):
        image = torch.rand(1, 256, 256)
        image_coord_cat = self.image_prep.concatenate_with_coord_grid(image)
        print(image_coord_cat)

    def test_adjust_to_standard_size(self):
        image = torch.randn(3, 100, 400)
        image = self.image_prep.adjust_to_standard_size(image)

        self.assertEqual(image.shape[0], 3)
        self.assertEqual(image.shape[1], self.standard_image_size[0])
        self.assertEqual(image.shape[2], self.standard_image_size[1])

    def test_call(self):
        image_pil = self.dataset[0]["image"]
        image = self.image_prep(image_pil)

        if self.standard_image_mode == "RGB":
            self.assertEqual(image.shape[0], 5)
        else:
            self.assertEqual(image.shape[0], 3)

        self.assertEqual(image.shape[1], self.standard_image_size[0])
        self.assertEqual(image.shape[2], self.standard_image_size[1])


if __name__ == "__main__":
    unittest.main()
