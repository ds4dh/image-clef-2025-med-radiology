from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import faiss
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


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
        self.image_processor = processor.image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        caption = item['caption']

        if isinstance(image, Image.Image):
            image = image.resize((512, 512))

            # For test mode, also return CUI codes if available
        if self.mode == 'test' and 'cui_codes' in item:
            concepts = item.get('cui_codes', [])
            return image, caption, concepts, item.get('id', str(idx))

        # For train/validation mode
        return image, caption, item.get('id', str(idx))


class InstructBLIPRAG:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.setup_args()
        args = self.parser.parse_args()

        self.device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

        # Create directories
        os.makedirs(args.index_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)

        # Load model and processor
        print(f"Loading InstructBLIP model...")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

        # Load fine-tuned weights if specified
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading fine-tuned model from {args.model_path}")
            state_dict = torch.load(args.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Load state_dict succeeded!")
            del state_dict  # Immediately delete state dict
            torch.cuda.empty_cache()

        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Base instruction
        self.base_instruction = 'You are an experienced radiologist. Generate a descriptive caption that highlights the location, nature and severity of the abnormality of the radiology image.'

    def setup_args(self):
        # Data directory settings
        self.parser.add_argument("--data_dir", type=str, default="/data/ROCOv2-radiology",
                                 help="Path to the ROCOv2 data directory")

        # Model and results directories
        self.parser.add_argument("--model_path", type=str, default="./models/best-model.pt",
                                 help="Path to fine-tuned InstructBLIP model")
        self.parser.add_argument("--index_dir", type=str, default="./index",
                                 help="Directory to save FAISS index")
        self.parser.add_argument("--results_dir", type=str, default="./results",
                                 help="Directory to save results")

        # GPU settings
        self.parser.add_argument('--cuda_device', type=int, default=1,
                                 help='CUDA device index to use')

        # RAG settings
        self.parser.add_argument('--top_k', type=int, default=3,
                                 help='Number of similar images to retrieve')
        self.parser.add_argument('--feature_dim', type=int, default=1408,
                                 help='Dimension of image features from InstructBLIP')
        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='Batch size for feature extraction')
        self.parser.add_argument('--rebuild_index', action='store_true',
                                 help='Force rebuilding the FAISS index even if it exists')

        # New argument: run mode selection
        self.parser.add_argument('--mode', type=str, choices=['base', 'rag'], default='base',
                                 help='Run mode: base for basic captioning, rag for RAG-enhanced captioning')

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

        return pooled_output.cpu().numpy()

    def build_index(self, train_dataset):
        """Build FAISS index from training images"""
        args = self.parser.parse_args()
        index_path = os.path.join(args.index_dir, "roco_image_index.faiss")
        metadata_path = os.path.join(args.index_dir, "roco_image_metadata.csv")

        # Check if index already exists and rebuild_index flag is not set
        if os.path.exists(index_path) and os.path.exists(metadata_path) and not args.rebuild_index:
            print(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            metadata_df = pd.read_csv(metadata_path)
            return index, metadata_df

        print("Building FAISS index from training images...")

        # Initialize FAISS index
        feature_dim = args.feature_dim
        index = faiss.IndexFlatIP(feature_dim)  # Inner product for cosine similarity

        # Prepare dataloader for feature extraction
        def collate_fn(batch):
            images, captions, ids = zip(*batch)
            return list(images), list(captions), list(ids)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        # Extract features and build metadata
        all_features = []
        all_ids = []
        all_captions = []

        for batch_idx, (images, captions, ids) in enumerate(tqdm(train_loader, desc="Extracting features")):
            try:
                batch_features = []

                # Process each image in the batch to get features
                for img in images:
                    features = self.extract_image_features(img)
                    # Normalize features for cosine similarity
                    features = features / np.linalg.norm(features, axis=1, keepdims=True)
                    batch_features.append(features.squeeze())

                # Add to collections
                all_features.extend(batch_features)
                all_ids.extend(ids)
                all_captions.extend(captions)

                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * args.batch_size} images")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Convert to numpy array and add to index
        all_features = np.vstack(all_features)
        index.add(all_features)

        # Save index to disk
        faiss.write_index(index, index_path)
        print(f"FAISS index saved to {index_path}")

        # Save metadata
        metadata_df = pd.DataFrame({
            'id': all_ids,
            'caption': all_captions
        })
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")

        return index, metadata_df

    def retrieve_similar_images(self, image, index, metadata_df, top_k=3):
        """Retrieve top_k similar images from the index"""
        # Extract features from the query image
        query_features = self.extract_image_features(image)
        # Normalize features
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)

        # Search in the index
        distances, indices = index.search(query_features, top_k)

        # Get metadata for the retrieved images
        retrieved_metadata = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(metadata_df):
                retrieved_metadata.append({
                    'id': metadata_df.iloc[idx]['id'],
                    'caption': metadata_df.iloc[idx]['caption'],
                    'similarity': float(distances[0][i])
                })

        return retrieved_metadata

    def generate_rag_prompt(self, retrieved_docs):
        """Generate a RAG-enhanced prompt using retrieved similar images"""
        context = "\n".join([f"Similar case {i + 1}: {doc['caption']}" for i, doc in enumerate(retrieved_docs)])

        # Enhanced instruction with retrieved context
        rag_instruction = f"{self.base_instruction}\n\nHere are similar cases for reference:\n{context}\n\nBased on these similar cases and what you see in the current image, generate a detailed caption."

        return rag_instruction

    def generate_caption(self, image, rag_instruction=None):
        """Generate caption for an image with optional RAG-enhanced prompt"""
        # If no RAG instruction is provided, use the base instruction
        instruction = rag_instruction if rag_instruction else self.base_instruction

        # Process the image and instruction
        inputs = self.processor(images=image, text=[instruction], return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=120,
                min_length=5,
            )

        # Decode generated text
        generated_caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return generated_caption

    def process_test_set_base(self, test_dataset):
        """Process the test set using base instruction only"""
        args = self.parser.parse_args()

        # Prepare dataloader for test set
        def collate_fn(batch):
            if len(batch[0]) == 4:  # If test dataset returns concepts
                images, captions, concepts, ids = zip(*batch)
                return list(images), list(captions), list(concepts), list(ids)
            else:
                images, captions, ids = zip(*batch)
                return list(images), list(captions), None, list(ids)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one by one for simplicity
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn
        )

        # Process each test image
        results = []

        for i, (images, captions, concepts, ids) in enumerate(
                tqdm(test_loader, desc="Processing test set (base mode)")):
            try:
                image = images[0]  # Get single image
                image_id = ids[0]

                # Generate caption with base instruction
                base_caption = self.generate_caption(image)

                # Store results
                results.append({
                    'id': image_id,
                    'actual_caption': captions[0],
                    'base_caption': base_caption
                })

                if i % 10 == 0:
                    print(f"\nProcessed {i} test images")
                    print(f"Example base caption: {base_caption}")

            except Exception as e:
                print(f"Error processing test image {i}: {e}")
                continue

        return results

    def process_test_set_rag(self, test_dataset, index, metadata_df):
        """Process the test set using RAG approach"""
        args = self.parser.parse_args()

        # Prepare dataloader for test set
        def collate_fn(batch):
            if len(batch[0]) == 4:  # If test dataset returns concepts
                images, captions, concepts, ids = zip(*batch)
                return list(images), list(captions), list(concepts), list(ids)
            else:
                images, captions, ids = zip(*batch)
                return list(images), list(captions), None, list(ids)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one by one for simplicity
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn
        )

        # Process each test image
        results = []

        for i, (images, captions, concepts, ids) in enumerate(tqdm(test_loader, desc="Processing test set (RAG mode)")):
            try:
                image = images[0]  # Get single image
                image_id = ids[0]

                # Retrieve similar images
                similar_docs = self.retrieve_similar_images(
                    image, index, metadata_df, top_k=args.top_k
                )

                # Generate RAG-enhanced prompt
                rag_instruction = self.generate_rag_prompt(similar_docs)

                # Generate caption with RAG prompt
                rag_caption = self.generate_caption(image, rag_instruction)

                # Store results
                results.append({
                    'id': image_id,
                    'actual_caption': captions[0],
                    'rag_caption': rag_caption,
                    'similar_docs': similar_docs
                })

                if i % 10 == 0:
                    print(f"\nProcessed {i} test images")
                    print(f"Example RAG caption: {rag_caption}")
                    print(f"Similar docs: {similar_docs}")

            except Exception as e:
                print(f"Error processing test image {i}: {e}")
                continue

        return results

    def load_index(self, custom_index_path=None, custom_metadata_path=None):
        """
        Load existing FAISS index and metadata

        Args:
            custom_index_path: Optional custom path to the FAISS index file
            custom_metadata_path: Optional custom path to the metadata CSV file

        Returns:
            tuple: (faiss_index, metadata_dataframe)
        """
        args = self.parser.parse_args()

        # Use custom paths if provided, otherwise use default paths
        index_path = custom_index_path if custom_index_path else os.path.join(args.index_dir,
                                                                              "roco_image_index.faiss")
        metadata_path = custom_metadata_path if custom_metadata_path else os.path.join(args.index_dir,
                                                                                       "roco_image_metadata.csv")

        # Check if index and metadata files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        try:
            print(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(index_path)

            print(f"Loading metadata from {metadata_path}")
            metadata_df = pd.read_csv(metadata_path)

            # Print some statistics about the loaded index
            print(f"Loaded index with {index.ntotal} vectors of dimension {index.d}")
            print(f"Loaded metadata with {len(metadata_df)} entries")

            # Verify that index size matches metadata size
            if index.ntotal != len(metadata_df):
                print(f"Warning: Index size ({index.ntotal}) does not match metadata size ({len(metadata_df)})")

            # Check if required columns exist in metadata
            required_columns = ['id', 'caption']
            missing_columns = [col for col in required_columns if col not in metadata_df.columns]
            if missing_columns:
                print(f"Warning: Metadata is missing required columns: {missing_columns}")

            return index, metadata_df

        except Exception as e:
            print(f"Error loading index or metadata: {e}")
            raise

    def save_results(self, results, mode='base'):
        """Save generated captions to files"""
        args = self.parser.parse_args()

        if mode == 'base':
            # Save base mode results
            results_df = pd.DataFrame([
                {
                    'id': r['id'],
                    'base_caption': r['base_caption'],
                    'actual_caption': r['actual_caption']
                }
                for r in results
            ])
            csv_path = os.path.join(args.results_dir, "base_results_test.csv")
            results_df.to_csv(csv_path, index=False)

            # Optional: Save in format compatible with original code
            # submission_path = os.path.join(args.results_dir, "submission_test_instructblip_base.csv")
            # with open(submission_path, 'w') as out_file:
            #     for r in results:
            #         out_file.write(f"{r['id']}|{r['base_caption']}\n")
            #
            # print(f"Base results saved to {csv_path} and {submission_path}")

        elif mode == 'rag':
            # Save RAG mode results
            results_df = pd.DataFrame([
                {
                    'id': r['id'],
                    'actual_caption': r['actual_caption'],
                    'rag_caption': r['rag_caption'],
                    # Convert similar_docs to string for CSV storage
                    'similar_docs': str(r['similar_docs'])
                }
                for r in results
            ])
            csv_path = os.path.join(args.results_dir, "rag_results_test.csv")
            results_df.to_csv(csv_path, index=False)

            # Save in format compatible with original code's output
            # submission_path = os.path.join(args.results_dir, "submission_test_instructblip_rag.csv")
            # with open(submission_path, 'w') as out_file:
            #     for r in results:
            #         out_file.write(f"{r['id']}|{r['rag_caption']}\n")
            #
            # print(f"RAG results saved to {csv_path} and {submission_path}")

    def main(self):
        args = self.parser.parse_args()

        # Load dataset
        dataset = self.load_dataset()

        # Create test dataset
        test_dataset = ROCODataset(dataset['test'], self.processor, 'test')
        print(f"Test dataset size: {len(test_dataset)}")

        # Run based on selected mode
        if args.mode == 'base':
            print("Running in BASE mode: generating captions with base instruction only")
            results = self.process_test_set_base(test_dataset)
            self.save_results(results, mode='base')
            print("Base captioning completed successfully!")

        elif args.mode == 'rag':
            print("Running in RAG mode: generating captions with RAG-enhanced prompts")
            # Load existing index (no need to build)
            index, metadata_df = self.load_index()
            # Process test set with RAG
            results = self.process_test_set_rag(test_dataset, index, metadata_df)
            # Save RAG results
            self.save_results(results, mode='rag')
            print("RAG processing completed successfully!")


if __name__ == "__main__":
    rag_system = InstructBLIPRAG()
    rag_system.main()