from radiclef import CLEF_2025_DATABASE_PATH, RESOURCES_DIR

from datasets import load_from_disk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from typing import List, Dict
import os
import ast
import random

RUN_TAG = "2025-05-11_09-41-48_unige-poc"
# RUN_TAG = "2025-04-27_00-08-23_unige-poc"
SPLIT = "valid"
DATASET_NAME = "CLEF"


def get_annotated_figure(
        image: np.ndarray,
        ground_truth: List[str],
        predictions: List[str]) -> plt.Figure:

    figure = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    ax_img = plt.subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis('off')

    ax_text = plt.subplot(gs[1])
    ax_text.axis('off')

    max_len = max(len(ground_truth), len(predictions))
    lines = []
    for i in range(max_len):
        gt = ground_truth[i] if i < len(ground_truth) else ""
        pred = predictions[i] if i < len(predictions) else ""
        lines.append(f"GT: {gt}\nPR: {pred}\n")

    text_str = "\n".join(lines)
    ax_text.text(0, 1, text_str, va='top', ha='left', fontsize=12, wrap=True, family='monospace')

    return figure


with open(os.path.join(RESOURCES_DIR, "cui-alphabet.txt"), "r") as f:
    cui_alphabet = [v.strip() for v in f.readlines()][4:]

dataset_dict = load_from_disk(CLEF_2025_DATABASE_PATH)
ds = dataset_dict[SPLIT]

run_dir = os.path.join("runs", RUN_TAG)

inference_result_paths = [os.path.join(run_dir, _item) for _item in os.listdir(run_dir) if
                          _item.startswith("{}-inference-{}".format(DATASET_NAME, SPLIT))]

df = pd.read_csv(inference_result_paths[0])
df['ground-truth-codes'] = df['ground-truth-codes'].apply(ast.literal_eval)
df['predicted-codes'] = df['predicted-codes'].apply(ast.literal_eval)
df['ground-truth-concepts'] = df['ground-truth-concepts'].apply(ast.literal_eval)
df['predicted-concepts'] = df['predicted-concepts'].apply(ast.literal_eval)

df["len-ground-truth"] = df['ground-truth-codes'].apply(lambda x: len(x))
df["len-predicted"] = df['predicted-codes'].apply(lambda x: len(x))

print(df["len-ground-truth"].describe())
print(df["len-predicted"].describe())

# length_counts = df["len-ground-truth"].value_counts().sort_index()
#
# # Normalize to get the ratio
# length_ratios = length_counts / length_counts.sum()
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.bar(length_ratios.index, length_ratios.values)
# plt.xlabel("# ground-truth CUI's per image")
# plt.ylabel("ratio of images in the training set")
# # plt.title("Distribution of Ground-Truth Lengths in Training Set")
# plt.xticks(range(1, 1 + df["len-ground-truth"].max()))  # Ensure all lengths from 1 to 32 are shown
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

# Save as PDF
plt.savefig("ground_truth_length_distribution.pdf")
plt.close()  # Close the figure to free memory


cui_frequency_dict_gt: Dict[str, float] = {cui: 0 for cui in cui_alphabet}
cui_frequency_dict_p: Dict[str, float] = {cui: 0 for cui in cui_alphabet}
for idx, row in df.iterrows():
    for cui in row["ground-truth-codes"]:
        cui_frequency_dict_gt[cui] += 1 / len(df)
    for cui in row["predicted-codes"]:
        cui_frequency_dict_p[cui] += 1 / len(df)

print("Number of CUI's (ground-truth): ", len([_freq for _freq in cui_frequency_dict_gt.values() if _freq > 0]))
print("Number of CUI's learned (predicted): ", len([_freq for _freq in cui_frequency_dict_p.values() if _freq > 0]))

# plt.figure()
# plt.plot([v for v in cui_frequency_dict_gt.values()])
# plt.plot([v for v in cui_frequency_dict_p.values()])
# plt.show()

_df = df.sort_values(by="f1-score").reset_index()

# plt.figure()
# plt.plot(_df["len-ground-truth"])
# plt.plot(_df["len-predicted"])

idx = random.randint(0, len(ds))
item = ds[idx]
img = item["image"]
caption = item["caption"]
print(caption)
gt = df["ground-truth-concepts"][idx]
pred = df["predicted-concepts"][idx]
fig = get_annotated_figure(image=img,
                           ground_truth=gt,
                           predictions=pred)




# plt.show()
