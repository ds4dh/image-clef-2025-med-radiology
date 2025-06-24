#!/usr/bin/env python
import os
import json
import torch
import faiss
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


class ROCODataset(Dataset):
    def __init__(self, dataset_split, processor, mode='train'):
        """
        Args:
            dataset_split: HF dataset split (train, validation, test)
            processor: InstructBLIP processor
            mode: 'train', 'validation', or 'test'
        """
        self.dataset = dataset_split
        self.processor = processor
        self.mode = mode

        # Define image preprocessing - just get the processor's image transforms
        # This ensures the images are properly processed for the InstructBLIP model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image from the dataset
        image = item['image']
        caption = item['caption']
        cui_codes = item['cui_codes']

        return image, caption, cui_codes, item.get('id', str(idx))


def load_model_in_chunks(model, checkpoint_path, device, chunk_size=1000):
    """Load model weights in chunks to reduce memory usage"""
    print(f"Loading model weights in chunks from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get all keys
    keys = list(checkpoint.keys())
    num_chunks = (len(keys) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk_keys = keys[i * chunk_size: (i + 1) * chunk_size]
        chunk_dict = {k: checkpoint[k] for k in chunk_keys}

        # Update only this portion of parameters
        model_dict = model.state_dict()
        model_dict.update(chunk_dict)
        model.load_state_dict(model_dict, strict=False)

        # Clean up
        del chunk_dict
        torch.cuda.empty_cache()
        print(f"Loaded chunk {i + 1}/{num_chunks}")

    # Clean up full checkpoint
    del checkpoint
    torch.cuda.empty_cache()
    print("Model loading complete")
    return model


class TopicIndexBuilder:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.setup_args()
        args = self.parser.parse_args()

        self.device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

        # Create directories
        os.makedirs(args.index_dir, exist_ok=True)

        # Load cluster information
        self.cui_to_topic = {}
        self.topic_to_cui = {}
        self.load_cluster_mapping(args.cluster_mapping_file)

        # Load model and processor
        print(f"Loading InstructBLIP model...")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.bfloat16)
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

        # Load fine-tuned weights if specified
        if args.model_path and os.path.exists(args.model_path):
            self.model = load_model_in_chunks(self.model, args.model_path, self.device, chunk_size=200)
            self.model.to(self.device)
            print(f"Loading fine-tuned model from {args.model_path}")
            print("Load state_dict succeeded!")
            torch.cuda.empty_cache()

        self.model.eval()  # Set model to evaluation mode

    def setup_args(self):
        # Data directory settings
        self.parser.add_argument("--data_dir", type=str, default="/data/CLEF-2025-radiology",
                                 help="Path to the ROCOv2 data directory")
        self.parser.add_argument("--cluster_mapping_file", type=str, default="cluster_results/cluster_by_distance_dict.json",
                                 help="JSON file containing CUI code to topic mapping")

        # Model and index directories
        self.parser.add_argument("--model_path", type=str, default="/home/jiawei/pyproject/ImageCLEFcompetition/baseline/test_models/best-model.pt",
                                 help="Path to fine-tuned InstructBLIP model")
        self.parser.add_argument("--index_dir", type=str, default="./topic_indices_cluster_rag",
                                 help="Directory to save topic-specific FAISS indices")

        # GPU settings
        self.parser.add_argument('--cuda_device', type=int, default=0,
                                 help='CUDA device index to use')

        # Index settings
        self.parser.add_argument('--feature_dim', type=int, default=1408,
                                 help='Dimension of image features from InstructBLIP')
        self.parser.add_argument('--batch_size', type=int, default=12,
                                 help='Batch size for feature extraction')
        self.parser.add_argument('--rebuild_index', action='store_true',
                                 help='Force rebuilding the FAISS indices even if they exist')

    def load_cluster_mapping(self, mapping_file):
        """Load CUI code to topic mapping from JSON file"""
        print(f"Loading cluster mapping from {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.topic_to_cui = json.load(f)

        # Create reverse mapping from CUI code to topic
        for topic_id, cui_codes in self.topic_to_cui.items():
            for cui_code in cui_codes:
                self.cui_to_topic[cui_code] = topic_id

        print(f"Loaded {len(self.topic_to_cui)} topics and {len(self.cui_to_topic)} CUI codes")

    def load_dataset(self):
        """Load ROCOv2 dataset using HuggingFace datasets library"""
        args = self.parser.parse_args()
        data_dir = args.data_dir

        print(f"Loading dataset from {data_dir}")

        # Load dataset using the arrow files
        dataset = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_dir}/train/*.arrow",
                "validation": f"{data_dir}/valid/*.arrow",
                "test": f"{data_dir}/test/*.arrow"
            }
        )

        print("Dataset loaded successfully")
        print(dataset)

        return dataset

    def extract_image_features(self, image):
        """Extract image features from InstructBLIP's vision encoder"""
        # Prepare image for the model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Get image features from the model's vision encoder
        with torch.no_grad():
            image_features = self.model.vision_model(inputs.pixel_values)[0]
            # Use the pooled output (CLS token) as the image representation
            pooled_output = image_features[:, 0]

        return pooled_output.float().cpu().numpy()

    def create_topic_indices(self, train_dataset):
        """Build a separate FAISS index for each topic cluster"""
        args = self.parser.parse_args()

        # Prepare dictionary to hold index for each topic
        topic_indices = {}
        topic_metadata = {}

        # Initialize each topic's index
        for topic_id in self.topic_to_cui.keys():
            # Check if index already exists
            index_path = os.path.join(args.index_dir, f"topic_{topic_id}_index.faiss")
            metadata_path = os.path.join(args.index_dir, f"topic_{topic_id}_metadata.csv")

            if os.path.exists(index_path) and os.path.exists(metadata_path) and not args.rebuild_index:
                print(f"Loading existing FAISS index for topic {topic_id} from {index_path}")
                topic_indices[topic_id] = faiss.read_index(index_path)
                topic_metadata[topic_id] = pd.read_csv(metadata_path)
                continue

            print(f"Creating new FAISS index for topic {topic_id}")
            topic_indices[topic_id] = faiss.IndexFlatIP(args.feature_dim)  # Inner product for cosine similarity
            topic_metadata[topic_id] = {"id": [], "caption": [], "cui_codes": []}

        # Create a fallback index for images without CUI codes
        fallback_index_path = os.path.join(args.index_dir, "fallback_index.faiss")
        fallback_metadata_path = os.path.join(args.index_dir, "fallback_metadata.csv")

        if os.path.exists(fallback_index_path) and os.path.exists(fallback_metadata_path) and not args.rebuild_index:
            print(f"Loading existing fallback FAISS index from {fallback_index_path}")
            fallback_index = faiss.read_index(fallback_index_path)
            fallback_metadata = pd.read_csv(fallback_metadata_path)
        else:
            print("Creating new fallback FAISS index")
            fallback_index = faiss.IndexFlatIP(args.feature_dim)
            fallback_metadata = {"id": [], "caption": [], "cui_codes": []}

        # Prepare dataloader for feature extraction
        def collate_fn(batch):
            images, captions, cui_codes, ids = zip(*batch)
            return list(images), list(captions), list(cui_codes), list(ids)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        # Extract features and build indices
        for batch_idx, (images, captions, cui_codes_batch, ids) in enumerate(
                tqdm(train_loader, desc="Extracting features")):
            try:
                # Process each image in the batch
                for i, (img, caption, cui_codes, img_id) in enumerate(zip(images, captions, cui_codes_batch, ids)):
                    # Extract features
                    features = self.extract_image_features(img)
                    # Normalize features for cosine similarity
                    features = features / np.linalg.norm(features, axis=1, keepdims=True)

                    # Determine which topics this image belongs to
                    image_topics = set()
                    if len(cui_codes) > 0:
                        for cui in cui_codes:
                            if cui in self.cui_to_topic:
                                image_topics.add(self.cui_to_topic[cui])

                    if image_topics:
                        # Add to all relevant topic indices
                        for topic_id in image_topics:
                            topic_indices[topic_id].add(features)
                            topic_metadata[topic_id]["id"].append(img_id)
                            topic_metadata[topic_id]["caption"].append(caption)
                            topic_metadata[topic_id]["cui_codes"].append(json.dumps(cui_codes))
                    else:
                        # Add to fallback index if no CUI codes or no matching topics
                        fallback_index.add(features)
                        fallback_metadata["id"].append(img_id)
                        fallback_metadata["caption"].append(caption)
                        fallback_metadata["cui_codes"].append(json.dumps(cui_codes if cui_codes else []))

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * args.batch_size} images")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Save indices and metadata to disk
        for topic_id in topic_indices:
            if topic_indices[topic_id].ntotal > 0:  # Only save if the index contains vectors
                index_path = os.path.join(args.index_dir, f"topic_{topic_id}_index.faiss")
                metadata_path = os.path.join(args.index_dir, f"topic_{topic_id}_metadata.csv")

                faiss.write_index(topic_indices[topic_id], index_path)

                # Convert metadata to DataFrame and save
                metadata_df = pd.DataFrame(topic_metadata[topic_id])
                metadata_df.to_csv(metadata_path, index=False)

                print(f"Topic {topic_id} index with {topic_indices[topic_id].ntotal} vectors saved to {index_path}")

        # Save fallback index
        if fallback_index.ntotal > 0:
            fallback_index_path = os.path.join(args.index_dir, "fallback_index.faiss")
            fallback_metadata_path = os.path.join(args.index_dir, "fallback_metadata.csv")

            faiss.write_index(fallback_index, fallback_index_path)

            fallback_metadata_df = pd.DataFrame(fallback_metadata)
            fallback_metadata_df.to_csv(fallback_metadata_path, index=False)

            print(f"Fallback index with {fallback_index.ntotal} vectors saved to {fallback_index_path}")

        # Create a metadata file with topic statistics
        stats = {
            "topic_id": list(topic_indices.keys()) + ["fallback"],
            "vector_count": [idx.ntotal for idx in topic_indices.values()] + [fallback_index.ntotal]
        }
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(args.index_dir, "topic_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Topic statistics saved to {stats_path}")

        return topic_indices, topic_metadata, fallback_index, fallback_metadata

    def main(self):
        # Load dataset
        dataset = self.load_dataset()

        # Create train dataset
        train_dataset = ROCODataset(dataset['train'], self.processor, 'train')
        print(f"Train dataset size: {len(train_dataset)}")

        # Create topic-specific indices
        print("Building topic-specific indices...")
        topic_indices, topic_metadata, fallback_index, fallback_metadata = self.create_topic_indices(train_dataset)

        print("Index building completed successfully!")


if __name__ == "__main__":
    index_builder = TopicIndexBuilder()
    index_builder.main()