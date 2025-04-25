import requests
from radiclef import ROCO_DATABASE_PATH, CLEF_2025_DATABASE_PATH

import os
import zipfile
from typing import List

import pandas as pd
import datasets

ROCO_API_URL_FROM_ZENODO = "https://zenodo.org/api/records/10821435"
RAW_ROCO_DOWNLOAD_DIR = os.path.join(ROCO_DATABASE_PATH, "zenodo")
RAW_CLEF_ZIP_FILE_PATH = os.path.join(CLEF_2025_DATABASE_PATH, "raw.zip")

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

    os.makedirs(RAW_ROCO_DOWNLOAD_DIR, exist_ok=True)

    for file in files:
        file_url = file["links"]["self"] + "?download=1"  # Ensure proper download URL
        file_name = file["key"]
        file_path = os.path.join(RAW_ROCO_DOWNLOAD_DIR, file_name)
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
    is_metadata_available: bool
    if os.path.exists(captions_file) and os.path.exists(concepts_file):
        is_metadata_available = True
        captions_df = load_dataframe(captions_file)
        concepts_df = load_dataframe(concepts_file)

        meta_df = captions_df.merge(concepts_df, on="ID", how="inner")

        meta_df["CUIs"] = process_cui_column(meta_df["CUIs"])
    else:
        is_metadata_available = False
        meta_df = pd.DataFrame(data=[_path.split(".")[0] for _path in os.listdir(image_dir) if _path.endswith(".jpg")],
                               columns=["ID"])

    def generate_examples():
        for _, row in meta_df.iterrows():
            image_path = os.path.join(image_dir, f"{row['ID']}.jpg")
            if not os.path.exists(image_path):
                continue
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                yield_dict = {
                    "id": row["ID"],
                    "image": image_bytes,
                }

                if is_metadata_available:
                    yield_dict = {**yield_dict, **{
                        "caption": row["Caption"],
                        "cui_codes": row["CUIs"],
                    }}

            yield yield_dict

    features_dict = {
        "id": datasets.Value("string"),
        "image": datasets.features.Image()
    }
    if is_metadata_available:
        features_dict = {**features_dict, **{
            "caption": datasets.Value("string"),
            "cui_codes": datasets.Sequence(datasets.Value("string"))
        }}
    dataset = datasets.Dataset.from_generator(
        generate_examples,
        features=datasets.Features(features_dict)
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


def main_roco():
    if not os.path.exists(os.path.join(ROCO_DATABASE_PATH, "dataset_dict.json")):
        if not os.path.exists(RAW_ROCO_DOWNLOAD_DIR):
            download_from_zenodo(ROCO_API_URL_FROM_ZENODO)

        dataset_dict = build_dataset_dict(RAW_ROCO_DOWNLOAD_DIR)
        dataset_dict.save_to_disk(ROCO_DATABASE_PATH)


def check_validity_of_clef_database():
    raw_dir = os.path.join(os.path.dirname(RAW_CLEF_ZIP_FILE_PATH), "raw")
    list_of_necessary_items = ["train_captions.csv", "train_concepts.csv",
                               "valid_captions.csv", "valid_concepts.csv",
                               "train", "valid", "test"]

    for _name in list_of_necessary_items:
        _path = os.path.join(raw_dir, _name)
        if not os.path.exists(_path):
            raise FileNotFoundError("You should manually organize the data as such.")


def main_clef():
    if not os.path.exists(os.path.join(CLEF_2025_DATABASE_PATH, "dataset_dict.json")):
        if not os.path.exists(RAW_CLEF_ZIP_FILE_PATH):
            raise FileNotFoundError(
                "You need to download the raw data from the challenge's website and do some manual preparations first.")

        with zipfile.ZipFile(RAW_CLEF_ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(RAW_CLEF_ZIP_FILE_PATH))

        check_validity_of_clef_database()
        os.remove(RAW_CLEF_ZIP_FILE_PATH)

        dataset_dict = build_dataset_dict(os.path.join(os.path.dirname(RAW_CLEF_ZIP_FILE_PATH), "raw"))
        dataset_dict.save_to_disk(CLEF_2025_DATABASE_PATH)


if __name__ == "__main__":
    main_roco()
    main_clef()
