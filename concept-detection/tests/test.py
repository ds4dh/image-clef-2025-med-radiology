from datasets import Dataset
from radiclef import CORPORA_DIR, RESOURCES_DIR

# dataset_iter = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True, cache_dir=RESOURCES_DIR)
# small_subset = dataset_iter.take(100)
# small_dataset = Dataset.from_list(list(small_subset))
# small_dataset.save_to_disk(RESOURCES_DIR)

ds = Dataset.load_from_disk(RESOURCES_DIR)


item = ds[73]
print(item)
print(item["caption"])

# item["image"].show()

min_h = float("inf")
min_w = float("inf")
max_h = float("-inf")
max_w = float("-inf")

for idx in range(len(ds)):
    image = ds[idx]["image"]
    size = image.size
    min_h = min(min_h, size[0])
    max_h = max(max_h, size[0])
    min_w = min(min_w, size[1])
    max_w = max(max_w, size[1])
    print(idx, image.size)

print(min_h, min_w)
print(max_h, max_w)
