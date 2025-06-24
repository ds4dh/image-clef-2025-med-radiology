import os
import socket


REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
REPO_NAME = os.path.split(REPO_DIR)[-1]

datasets_root_dirs_dict = {
    "unige-poc": "/data/",
    "ordnance": "/data/bases/corpora/",
    "sf-macbook-pro.local": "/Users/sssohrab/data/corpora"
}

hostname = socket.gethostname().lower()
if hostname in datasets_root_dirs_dict.keys():
    CORPORA_DIR = datasets_root_dirs_dict[hostname]
else:
    raise ValueError("The root data directory not assigned for the hostname.")

ROCO_DATABASE_PATH = os.path.join(CORPORA_DIR, "ROCOv2-radiology")
CLEF_2025_DATABASE_PATH = os.path.join(CORPORA_DIR, "CLEF-2025-radiology")

RESOURCES_DIR = os.path.join(REPO_DIR, "resources")
