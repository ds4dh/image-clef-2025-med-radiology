import torch
import torch.nn.functional as nn_functional
from torchvision.transforms.functional import to_tensor, pil_to_tensor

from PIL import Image

from typing import Dict, List, Tuple


class ConceptUniqueIdentifiers:
    PAD_TOKEN = "<PAD>"
    OOV_TOKEN = "<OOV>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, alphabet: List[str] | None = None):
        if alphabet is None:
            self.alphabet: List[str] = [self.PAD_TOKEN, self.OOV_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        else:
            self.alphabet = self._check_and_get_alphabet(alphabet)

        self.c2i = self.get_token_to_integer_mapping()
        self.i2c = self.get_integer_to_token_mapping()

    def _check_and_get_alphabet(self, alphabet: List[str]) -> List[str]:
        if not isinstance(alphabet, list) or not all(isinstance(element, str) for element in alphabet):
            raise TypeError("Alphabet must be a list of strings.")

        alphabet = self.get_alphabet(alphabet)

        return alphabet

    def integrate_items_into_alphabet(self, items: List[str]) -> None:
        alphabet = self.alphabet + self.get_alphabet(items)
        self.alphabet = self._check_and_get_alphabet(alphabet)
        self.c2i = self.get_token_to_integer_mapping()
        self.i2c = self.get_integer_to_token_mapping()

    def get_alphabet(self, items: List[str]) -> List[str]:
        items = sorted(set(items))
        items = [item for item in items if item not in {self.PAD_TOKEN, self.OOV_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}]
        return [self.PAD_TOKEN, self.OOV_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN] + items

    def get_token_to_integer_mapping(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.alphabet)}

    def get_integer_to_token_mapping(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self.alphabet)}

    def encode_as_seq(self, items: List[str]) -> List[int]:
        seq = [self.c2i[self.BOS_TOKEN]]
        seq += [self.c2i.get(c, self.c2i[self.OOV_TOKEN]) for c in items]
        seq += [self.c2i[self.EOS_TOKEN]]

        return seq

    def decode(self, seq: List[int]) -> List[str]:
        seq_decoded = []
        for idx in seq:
            seq_decoded.append(self.i2c[idx])
            if idx == self.c2i[self.EOS_TOKEN]:
                break

        return seq_decoded


class ImagePrepare:
    def __init__(self, standard_image_size: Tuple[int, int],
                 standard_image_mode: str = "RGB",
                 concatenate_positional_embedding: bool = True):
        if not isinstance(standard_image_size, tuple) or len(standard_image_size) != 2:
            raise TypeError("The target standard image size should be a tuple of length 2.")
        self.image_size = standard_image_size

        if not isinstance(standard_image_mode, str) or standard_image_mode not in ["RGB", "L"]:
            raise TypeError("The target standard image mode should be either of `RGB` or `L`.")
        self.image_mode = standard_image_mode

        if not isinstance(concatenate_positional_embedding, bool):
            raise TypeError("Indicate whether or not you want positional embedding to be concatenated to the image.")
        self.concatenate_positional_embedding = concatenate_positional_embedding

    @staticmethod
    def normalize(image: torch.Tensor) -> torch.Tensor:
        image = (image - image.min()) / (image.max() - image.min())
        return image

    @staticmethod
    def concatenate_with_coord_grid(image: torch.Tensor) -> torch.Tensor:
        """
        Concatenates an input tensor with a meshgrid between 0 and 1.

        This is due to `An intriguing failing of convolutional neural networks
        and the CoordConv solution`<https://arxiv.org/pdf/1807.03247.pdf>
        """
        assert isinstance(image, torch.Tensor) and len(image.shape) == 3, ("Pass a torch tensor of (channel, height, "
                                                                           "width).")

        _, height, width = image.shape
        device = image.device

        y = torch.linspace(0, 1, height)
        x = torch.linspace(0, 1, width)

        height_grid, width_grid = torch.meshgrid(y, x, indexing="ij")

        grid = torch.stack((height_grid, width_grid), dim=0)

        image = torch.cat((image, grid.to(device)), dim=0)

        return image

    def adjust_to_standard_size(self, image: torch.Tensor) -> torch.Tensor:

        assert len(image.shape) == 3
        _, h, w = image.shape

        image = nn_functional.interpolate(input=image.unsqueeze(0),
                                          size=(min(h, self.image_size[0]), min(w, self.image_size[1])),
                                          mode="bilinear").squeeze(0)
        _, h, w = image.shape

        pad_h = self.image_size[0] - h
        pad_w = self.image_size[1] - w

        image = nn_functional.pad(input=image,
                                  pad=(pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                                  mode="constant",
                                  value=0)

        return image

    def run(self, image: Image) -> torch.Tensor:
        if image.mode != self.image_mode:
            image = image.convert(self.image_mode)

        image = to_tensor(image)  # values in [0, 1]
        if self.concatenate_positional_embedding:
            image = self.concatenate_with_coord_grid(image)

        image = self.adjust_to_standard_size(image)

        return image

    def __call__(self, image: Image) -> torch.Tensor:
        return self.run(image)
