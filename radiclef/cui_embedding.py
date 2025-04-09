from radiclef import CORPORA_DIR, RESOURCES_DIR
from radiclef.networks import ConvEmbeddingToSec
from radiclef.experiments.conv_to_cui_seq.main import EXP_DIR

from torchbase.utils.networks import load_network_from_state_dict_to_device

import torch

from typing import Dict, List

import zipfile
import requests
import os
import json

PRETRAINED_AND_ALIGNED_EMBEDDINGS_PATH = os.path.join(RESOURCES_DIR, "cui-embedding-500.pt")

with open(os.path.join(RESOURCES_DIR, "cui-alphabet.txt")) as f:
    ALPHABET = [_line.strip() for _line in f.readlines()][4:]


def download_pretrained_embeddings(pretrained_embeddings_path) -> None:
    os.makedirs(os.path.dirname(pretrained_embeddings_path), exist_ok=True)

    url = "https://figshare.com/ndownloader/files/10959626?private_link=00d69861786cd0156d81"
    zip_path = os.path.join(os.path.dirname(pretrained_embeddings_path), "cui2vec_pretrained.csv.zip")

    response = requests.get(url)

    with open(zip_path, 'wb') as f:
        f.write(response.content)

    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(pretrained_embeddings_path))

        print("Extraction successful.")
    if not os.path.exists(pretrained_embeddings_path):
        raise RuntimeError
    os.remove(zip_path)


def prepare_and_save_alphabet_aligned_pretrained_embeddings(pretrained_embeddings_path: str,
                                                            alphabet: List[str]) -> None:
    alphabet_embedding_dict: Dict[str, List[float] | None] = {vocab: None for vocab in alphabet}
    with (open(pretrained_embeddings_path, "r") as f):
        for line in f:
            cui = line.split(",")[0][1:-1]
            if cui in alphabet_embedding_dict.keys():
                vect = [float(val) for val in line.split(",")[1:]]
                alphabet_embedding_dict[cui] = vect

    data_dict = {
        "data": [],
        "embedding_available": [],
        "cui": []

    }

    for cui in alphabet:
        if alphabet_embedding_dict[cui] is None:
            vect = (0.2 * torch.randn(500)).tolist()
            is_available = False
        else:
            vect = alphabet_embedding_dict[cui]
            is_available = True

        data_dict["data"].append(vect)
        data_dict["embedding_available"].append(is_available)
        data_dict["cui"].append(cui)

    data_dict["data"] = torch.tensor(data_dict["data"])
    aligned_dictionary_path = os.path.join(RESOURCES_DIR, "cui-embedding-500.pt")
    torch.save(data_dict, aligned_dictionary_path)


def fetch_pretrained_implicit_embeddings(run_dir: str) -> torch.Tensor:
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        config = json.load(f)

    network = ConvEmbeddingToSec(config["network"])
    network = load_network_from_state_dict_to_device(network, os.path.join(run_dir, "network.pth"),
                                                     device=torch.device("cpu"))

    return network.seq_generator.token_embedding.weight.data


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if len(matrix.shape) != 2:
        raise ValueError

    num, dim = matrix.shape

    # matrix = matrix - matrix.mean(dim=0).reshape(1, dim).expand_as(matrix)
    matrix = matrix / matrix.std(dim=0).reshape(1, dim).expand_as(matrix)

    return matrix


def get_euclidian_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    matrix = normalize_matrix(matrix)

    return matrix @ matrix.t()


if not os.path.exists(PRETRAINED_AND_ALIGNED_EMBEDDINGS_PATH):
    the_pretrained_embeddings_path = os.path.join(CORPORA_DIR, "UMLS", "cui2vec_pretrained.csv")
    if not os.path.exists(the_pretrained_embeddings_path):
        download_pretrained_embeddings(the_pretrained_embeddings_path)

    prepare_and_save_alphabet_aligned_pretrained_embeddings(the_pretrained_embeddings_path, ALPHABET)

PRETRAINED_EMBEDDING_DICTIONARY = torch.load(PRETRAINED_AND_ALIGNED_EMBEDDINGS_PATH)
embeddings = PRETRAINED_EMBEDDING_DICTIONARY["data"]

if __name__ == "__main__":

    RUN_TAG = "2025-04-02_08-13-36_unige-poc"
    embeddings_implicit = fetch_pretrained_implicit_embeddings(os.path.join(EXP_DIR, "runs", RUN_TAG))
    embeddings_implicit = embeddings_implicit[4:, :]

    torch.save(
        {
            "data": embeddings_implicit,
            "cui": ALPHABET
        },
        os.path.join(RESOURCES_DIR, "cui-embedding-32-implicit.pt"))

    sim = get_euclidian_similarity_matrix(embeddings)
    sim_imp = get_euclidian_similarity_matrix(embeddings_implicit)

    K = 5

    intersection = []

    for idx in range(ALPHABET.__len__()):
        top_items = set(sim[idx, :].topk(k=K).indices[1:].tolist())
        top_imp_items = set(sim_imp[idx, :].topk(k=K).indices[1:].tolist())

        intersection.append(len(top_items.intersection(top_imp_items)))

    intersection = torch.tensor(intersection)
    print(intersection.float().mean())
