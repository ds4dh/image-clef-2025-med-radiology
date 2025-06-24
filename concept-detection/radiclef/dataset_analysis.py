import matplotlib.pyplot as plt

from radiclef import CLEF_2025_DATABASE_PATH, RESOURCES_DIR

from datasets import load_from_disk
from typing import Dict

from matplotlib.pyplot import plot
import os

dataset_dict = load_from_disk(CLEF_2025_DATABASE_PATH)

dataset_train = dataset_dict["train"].remove_columns(["image"])
dataset_valid = dataset_dict["valid"].remove_columns(["image"])
dataset_test = dataset_dict["test"].remove_columns(["image"])

with open(os.path.join(RESOURCES_DIR, "cui-alphabet.txt"), "r") as f:
    cui_alphabet = [v.strip() for v in f.readlines()][4:]

cui_frequency_dict_train: Dict[str, float] = {cui: 0 for cui in cui_alphabet}
for idx, cui_list in enumerate(dataset_train["cui_codes"]):
    for cui in cui_list:
        cui_frequency_dict_train[cui] += 1 / len(dataset_train)

cui_frequency_dict_valid = {cui: 0 for cui in cui_alphabet}
for idx, cui_list in enumerate(dataset_valid["cui_codes"]):
    for cui in cui_list:
        cui_frequency_dict_valid[cui] += 1 / len(dataset_valid)


print(cui_frequency_dict_train)
print(cui_frequency_dict_valid)

plt.figure()
plot([v for v in cui_frequency_dict_train.values()])
plot([v for v in cui_frequency_dict_valid.values()])
plt.show()
