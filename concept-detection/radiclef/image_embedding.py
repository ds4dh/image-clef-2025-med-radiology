from external.MedSAM2.sam2.build_sam import build_sam2_video_predictor_npz
from radiclef import RESOURCES_DIR, ROCO_DATABASE_PATH, CLEF_2025_DATABASE_PATH
from radiclef.utils import ConceptUniqueIdentifiers, ImagePrepare

from datasets import load_from_disk
import os

import torch

DEVICE_NAME = "cpu"
BATCH_SIZE = 100


dataset_dict = load_from_disk(CLEF_2025_DATABASE_PATH)
image_prep = ImagePrepare(standard_image_size=(512, 512),
                          standard_image_mode="RGB",
                          concatenate_positional_embedding=False)


def map_fields(example):
    image_tensor = torch.cat([image_prep(_img).unsqueeze(0) for _img in example["image"]], dim=0)

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image_tensor = (image_tensor - mean) / std


    output = {
        "image_tensor": image_tensor,
    }

    return output



checkpoint = os.path.abspath("../external/MedSAM2/checkpoints/MedSAM2_latest.pt")
model_cfg = "configs/sam2.1_hiera_t512.yaml"

assert os.path.isfile(checkpoint)

predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

network = predictor.image_encoder.to(torch.device(DEVICE_NAME))




if __name__ == "__main__":
    for split in ["valid"]:
        dataset = dataset_dict[split]
        dataset.set_transform(lambda x: map_fields(x))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        for mini_batch in dataloader:
            images = mini_batch["image_tensor"].to(torch.device(DEVICE_NAME))
            print(images.shape)
            print(images.mean(dim=(0, 2, 3)), images.max())

            with torch.no_grad():
                outputs = network(images)

            print(outputs["vision_features"].shape)
            # print(outputs["vision_pos_enc"].shape)
            print(outputs["backbone_fpn"].shape)


            break


