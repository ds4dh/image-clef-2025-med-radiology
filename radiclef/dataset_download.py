import requests
from radiclef import CORPORA_DIR, ROCO_DATABASE_PATH

import os
import zipfile
from typing import Dict, List

import pandas as pd
import datasets

ZENODO_API_URL = "https://zenodo.org/api/records/10821435"
RAW_DOWNLOAD_DIR = os.path.join(CORPORA_DIR, "zenodo")

IMAGES_TRAIN_NAME = 'train_images.zip'
IMAGES_VALID_NAME = 'valid_images.zip'
IMAGES_TEST_NAME = 'test_images.zip'
CAPTIONS_TRAIN_NAME = 'train_captions.csv'
CAPTIONS_VALID_NAME = 'valid_captions.csv'
CAPTIONS_TEST_NAME = 'test_captions.csv'
CONCEPTS_TRAIN_NAME = 'train_concepts.csv'
CONCEPTS_VALID_NAME = 'valid_concepts.csv'
CONCEPTS_TEST_NAME = 'test_concepts.csv'


def download_from_zenodo(zenodo_url: str) -> None:
    response = requests.get(zenodo_url)
    if response.status_code != 200:
        print(f"Failed to fetch record: {response.status_code}")
        exit(1)

    data = response.json()
    files = data.get("files", [])

    os.makedirs(RAW_DOWNLOAD_DIR, exist_ok=True)

    for file in files:
        file_url = file["links"]["self"] + "?download=1"  # Ensure proper download URL
        file_name = file["key"]
        file_path = os.path.join(RAW_DOWNLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            file_response = requests.get(file_url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=1024):
                    f.write(chunk)

    print("All files downloaded successfully from Zenodo!")


def extract_images(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(extract_to)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str)


def process_cui_column(cui_series: pd.Series) -> List[List[str]]:
    return cui_series.fillna("").apply(lambda x: x.split(';') if x else []).tolist()


def create_dataset_from_split(
        image_dir: str, captions_file: str, concepts_file: str
) -> datasets.Dataset:
    captions_df = load_dataframe(captions_file)
    concepts_df = load_dataframe(concepts_file)

    merged_df = captions_df.merge(concepts_df, on="ID", how="inner")

    merged_df["CUIs"] = process_cui_column(merged_df["CUIs"])

    def generate_examples():
        for _, row in merged_df.iterrows():
            image_path = os.path.join(image_dir, f"{row['ID']}.jpg")
            if not os.path.exists(image_path):
                continue
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            yield {
                "id": row["ID"],
                "image": image_bytes,
                "caption": row["Caption"],
                "cui_codes": row["CUIs"],
            }

    dataset = datasets.Dataset.from_generator(
        generate_examples,
        features=datasets.Features({
            "id": datasets.Value("string"),
            "image": datasets.features.Image(),
            "caption": datasets.Value("string"),
            "cui_codes": datasets.Sequence(datasets.Value("string"))
        })
    )
    return dataset


def build_dataset_dict(data_root: str) -> datasets.DatasetDict:
    dataset_splits = {}
    file_paths = {
        "train": {
            "captions": os.path.join(data_root, CAPTIONS_TRAIN_NAME),
            "concepts": os.path.join(data_root, CONCEPTS_TRAIN_NAME),
            "images": os.path.join(data_root, IMAGES_TRAIN_NAME)
        },
        "valid": {
            "captions": os.path.join(data_root, CAPTIONS_VALID_NAME),
            "concepts": os.path.join(data_root, CONCEPTS_VALID_NAME),
            "images": os.path.join(data_root, IMAGES_VALID_NAME)
        },
        "test": {
            "captions": os.path.join(data_root, CAPTIONS_TEST_NAME),
            "concepts": os.path.join(data_root, CONCEPTS_TEST_NAME),
            "images": os.path.join(data_root, IMAGES_TEST_NAME)
        }
    }

    for split, paths in file_paths.items():
        image_zip = paths["images"]
        image_dir = os.path.join(data_root, split)
        if not os.path.exists(image_dir):
            extract_images(image_zip, data_root)

        dataset_splits[split] = create_dataset_from_split(
            image_dir, paths["captions"], paths["concepts"]
        )

    return datasets.DatasetDict(dataset_splits)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(ROCO_DATABASE_PATH, "dataset_dict.json")):
        if not os.path.exists(RAW_DOWNLOAD_DIR):
            download_from_zenodo(ZENODO_API_URL)

        dataset_dict = build_dataset_dict(RAW_DOWNLOAD_DIR)
        dataset_dict.save_to_disk(ROCO_DATABASE_PATH)
