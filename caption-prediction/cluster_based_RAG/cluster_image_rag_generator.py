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
from sklearn.metrics.pairwise import cosine_similarity


class ROCODataset(Dataset):
    def __init__(self, dataset_split, processor, mode='test'):
        """
        Args:
            dataset_split: HF dataset split (train, validation, test)
            processor: InstructBLIP processor
            mode: 'test' is the primary mode for this script
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
        cui_codes = item.get('cui_codes', [])

        return image, caption, cui_codes, item.get('id', str(idx))


class CaptionGenerator:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.setup_args()
        args = self.parser.parse_args()

        self.device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

        # Create directories
        os.makedirs(args.results_dir, exist_ok=True)

        # Load cluster information
        self.cui_to_topic = {}
        self.topic_to_cui = {}
        self.load_cluster_mapping(args.cluster_mapping_file)

        # Load CUI embeddings
        self.cui_embeddings = None
        self.cui_to_embedding = {}
        self.load_cui_embeddings(args.cui_embeddings_file)

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
        self.parser.add_argument("--data_dir", type=str, default="/data/CLEF-2025-radiology",
                                 help="Path to the ROCOv2 data directory")
        self.parser.add_argument("--cluster_mapping_file", type=str,
                                 default="cluster_results/cluster_by_distance_dict.json",
                                 help="JSON file containing CUI code to topic mapping")
        self.parser.add_argument("--cui_embeddings_file", type=str, default="embeddings/cui_embeddings.pt",
                                 help="PyTorch file containing CUI embeddings")

        # Model and index directories
        self.parser.add_argument("--model_path", type=str,
                                 default="/home/jiawei/pyproject/ImageCLEFcompetition/baseline/test_models/best-model.pt",
                                 help="Path to fine-tuned InstructBLIP model")
        self.parser.add_argument("--index_dir", type=str, default="./topic_indices_cluster_rag",
                                 help="Directory where topic-specific FAISS indices are stored")
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
        self.parser.add_argument('--rrf_k', type=int, default=60,
                                 help='Constant k for Reciprocal Rank Fusion')
        self.parser.add_argument('--batch_size', type=int, default=10,
                                 help='Batch size for generation (recommended to keep at 1)')
        self.parser.add_argument('--include_cui_names', action='store_true', default=True,
                                 help='Include CUI code names in the RAG prompt')
        self.parser.add_argument('--cui_similarity_threshold', type=float, default=0.9,
                                 help='Threshold for CUI embedding similarity to filter retrieved documents')

    def load_cluster_mapping(self, mapping_file):
        """Load CUI code to topic mapping from JSON file"""
        print(f"Loading cluster mapping from {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.topic_to_cui = json.load(f)

        # Create reverse mapping from CUI code to topic
        for topic_id, cui_codes in self.topic_to_cui.items():
            for cui_code in cui_codes:
                self.cui_to_topic[cui_code] = topic_id

        # Load CUI code to name mapping if available
        print(f"Loaded {len(self.topic_to_cui)} topics and {len(self.cui_to_topic)} CUI codes")

    def load_cui_embeddings(self, embeddings_file):
        """Load CUI embeddings from a PyTorch file"""
        print(f"Loading CUI embeddings from {embeddings_file}")
        if not os.path.exists(embeddings_file):
            print(f"Warning: CUI embeddings file {embeddings_file} not found. Similarity filtering will be disabled.")
            return

        try:
            self.cui_embeddings = torch.load(embeddings_file, weights_only=False)
            cui_codes = self.cui_embeddings['cui']
            embeddings = self.cui_embeddings['data']

            # Create mapping from CUI code to embedding
            self.cui_to_embedding = {cui: emb for cui, emb in zip(cui_codes, embeddings)}
            print(f"Loaded {len(self.cui_to_embedding)} CUI embeddings")
        except Exception as e:
            print(f"Error loading CUI embeddings: {e}")
            self.cui_embeddings = None

    def get_embeddings_for_cuis(self, cui_list):
        """Get embeddings for a list of CUI codes"""
        if not self.cui_to_embedding:
            return None

        valid_embeddings = []
        for cui in cui_list:
            if cui in self.cui_to_embedding:
                emb = self.cui_to_embedding[cui]
                # Convert torch tensor to numpy if needed
                if isinstance(emb, torch.Tensor):
                    emb = emb.numpy()
                valid_embeddings.append(emb)

        if not valid_embeddings:
            return None

        return np.array(valid_embeddings)

    def calculate_cui_similarity(self, cui_set_A, cui_set_B):
        """Calculate similarity between two sets of CUI codes using their embeddings"""
        if not self.cui_to_embedding:
            return 0.0  # Default to high similarity if embeddings not available

        # Get embeddings for both sets
        embeddings_A = self.get_embeddings_for_cuis(cui_set_A)
        embeddings_B = self.get_embeddings_for_cuis(cui_set_B)

        # If embeddings not found for one or both sets, return default high similarity
        if embeddings_A is None or embeddings_B is None:
            return 0.0

        # Calculate centroids
        centroid_A = np.mean(embeddings_A, axis=0).reshape(1, -1)
        centroid_B = np.mean(embeddings_B, axis=0).reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(centroid_A, centroid_B)[0][0]

        return similarity

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

    def load_topic_indices(self):
        """Load all topic indices from disk"""
        args = self.parser.parse_args()

        # Dictionary to store indices and metadata for each topic
        topic_indices = {}
        topic_metadata = {}

        # Check if topic stats file exists
        stats_path = os.path.join(args.index_dir, "topic_stats.csv")
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path)
            topic_ids = [str(tid) for tid in stats_df[stats_df['topic_id'] != 'fallback']['topic_id']]
        else:
            # If no stats file, try to determine topics from the cluster mapping
            topic_ids = list(self.topic_to_cui.keys())

        # Load each topic index
        for topic_id in topic_ids:
            index_path = os.path.join(args.index_dir, f"topic_{topic_id}_index.faiss")
            metadata_path = os.path.join(args.index_dir, f"topic_{topic_id}_metadata.csv")

            if os.path.exists(index_path) and os.path.exists(metadata_path):
                try:
                    topic_indices[topic_id] = faiss.read_index(index_path)
                    topic_metadata[topic_id] = pd.read_csv(metadata_path)
                    print(f"Loaded topic {topic_id} index with {topic_indices[topic_id].ntotal} vectors")
                except Exception as e:
                    print(f"Error loading topic {topic_id} index: {e}")

        # Load fallback index
        fallback_index_path = os.path.join(args.index_dir, "fallback_index.faiss")
        fallback_metadata_path = os.path.join(args.index_dir, "fallback_metadata.csv")

        if os.path.exists(fallback_index_path) and os.path.exists(fallback_metadata_path):
            try:
                fallback_index = faiss.read_index(fallback_index_path)
                fallback_metadata = pd.read_csv(fallback_metadata_path)
                print(f"Loaded fallback index with {fallback_index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading fallback index: {e}")
                fallback_index = None
                fallback_metadata = None
        else:
            fallback_index = None
            fallback_metadata = None

        return topic_indices, topic_metadata, fallback_index, fallback_metadata

    def identify_relevant_topics(self, cui_codes):
        """Identify which topics an image belongs to based on its CUI codes"""
        if not cui_codes:
            return []

        relevant_topics = set()
        for cui in cui_codes:
            if cui in self.cui_to_topic:
                relevant_topics.add(self.cui_to_topic[cui])

        return list(relevant_topics)

    def reciprocal_rank_fusion(self, search_results, query_cui_codes, k=60):
        """
        Apply Reciprocal Rank Fusion to merge results from multiple indices,
        filtering out results with low CUI similarity

        Args:
            search_results: Dictionary mapping topic_id to (distances, indices)
            query_cui_codes: CUI codes of the query image
            k: Constant in RRF formula
            similarity_threshold: Threshold for CUI similarity filtering

        Returns:
            Merged list of (doc_id, score, metadata) tuples
        """
        # Track document scores across all result sets
        doc_scores = {}

        for topic_id, (distances, indices, metadata_df) in search_results.items():
            # Normalize each distance to get a similarity score (higher is better)
            # Since we're using inner product, higher values are already better
            similarities = distances[0]  # Get the first batch of results

            # Process each result
            for rank, (idx, sim) in enumerate(zip(indices[0], similarities)):
                # Get metadata for this result
                doc_id = metadata_df.iloc[idx]['id']
                caption = metadata_df.iloc[idx]['caption']

                # If CUI codes were stored as JSON strings, parse them
                if 'cui_codes' in metadata_df.columns:
                    try:
                        doc_cui_codes = json.loads(metadata_df.iloc[idx]['cui_codes'])
                    except:
                        doc_cui_codes = []
                else:
                    doc_cui_codes = []

                # Calculate CUI similarity between query and retrieved document
                cui_similarity = self.calculate_cui_similarity(query_cui_codes, doc_cui_codes)
                # Skip this document if CUI similarity is below threshold

                # Calculate RRF score for this document in this result set
                # RRF score = 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)

                # Initialize or update document score
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'rrf_score': rrf_score,
                        'caption': caption,
                        'cui_codes': doc_cui_codes,
                        'sim': sim,
                        'cui_similarity': cui_similarity,
                        'sources': [topic_id]
                    }
                else:
                    # Add RRF score from this result set
                    doc_scores[doc_id]['rrf_score'] += rrf_score
                    # Take max similarity across all sources
                    doc_scores[doc_id]['sim'] = max(doc_scores[doc_id]['sim'], sim)
                    doc_scores[doc_id]['sources'].append(topic_id)

        # Convert to list and sort by RRF score (descending)
        merged_results = [
            {
                'id': doc_id,
                'caption': info['caption'],
                'cui_codes': info['cui_codes'],
                'similarity': float(info['sim']),
                'cui_similarity': float(info['cui_similarity']),
                'rrf_score': info['rrf_score'],
                'sources': info['sources']
            }
            for doc_id, info in doc_scores.items()
        ]

        merged_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        return merged_results

    def search_topic_indices(self, query_features, query_cui_codes, relevant_topics, topic_indices,
                             topic_metadata, fallback_index, fallback_metadata, top_k=3):
        """
        Search for similar images across multiple topic indices

        Args:
            query_features: Features of the query image
            query_cui_codes: CUI codes of the query image
            relevant_topics: List of topic IDs relevant to the query
            topic_indices: Dictionary of topic indices
            topic_metadata: Dictionary of topic metadata
            fallback_index: Fallback index for when no relevant topics are found
            fallback_metadata: Metadata for fallback index
            top_k: Number of similar images to retrieve per index

        Returns:
            List of retrieved documents after fusion
        """
        args = self.parser.parse_args()

        # If no relevant topics found, search only in fallback index
        if not relevant_topics and fallback_index is not None:
            distances, indices = fallback_index.search(query_features, top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                # Get CUI codes for this result
                if 'cui_codes' in fallback_metadata.columns:
                    try:
                        doc_cui_codes = json.loads(fallback_metadata.iloc[idx]['cui_codes'])
                    except:
                        doc_cui_codes = []
                else:
                    doc_cui_codes = []

                # Calculate CUI similarity
                cui_similarity = self.calculate_cui_similarity(query_cui_codes, doc_cui_codes)

                # Skip if below threshold

                results.append({
                    'id': fallback_metadata.iloc[idx]['id'],
                    'caption': fallback_metadata.iloc[idx]['caption'],
                    'cui_codes': doc_cui_codes,
                    'similarity': float(distances[0][i]),
                    'cui_similarity': float(cui_similarity)
                })

            return results

        # Search in each relevant topic index
        search_results = {}

        for topic_id in relevant_topics:
            if topic_id in topic_indices and topic_indices[topic_id].ntotal > 0:
                # Search in this topic's index
                distances, indices = topic_indices[topic_id].search(query_features,
                                                                    min(top_k, topic_indices[topic_id].ntotal))
                search_results[topic_id] = (distances, indices, topic_metadata[topic_id])

        # Also search in fallback index if available
        if fallback_index is not None and fallback_index.ntotal > 0:
            distances, indices = fallback_index.search(query_features, min(top_k, fallback_index.ntotal))
            search_results["fallback"] = (distances, indices, fallback_metadata)

        # If no search results found
        if not search_results:
            return []

        # Apply reciprocal rank fusion with CUI similarity filtering
        merged_results = self.reciprocal_rank_fusion(
            search_results,
            query_cui_codes,
            k=args.rrf_k,
        )

        # Return top_k results
        return merged_results[:top_k]

    def generate_rag_prompt(self, retrieved_docs, cui_codes):
        """Generate a RAG-enhanced prompt using retrieved similar images and CUI codes"""
        # Create context from similar cases
        context = "\n".join([
            f"Similar case {i + 1} (CUI similarity: {doc.get('cui_similarity', 'N/A'):.2f}): {doc['caption']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        rag_instruction = f"{self.base_instruction}\n\nHere are similar cases for reference:\n{context}\n\nBased on these similar cases, the medical concepts, and what you see in the current image, generate a detailed caption."

        return rag_instruction

    def generate_caption(self, image, rag_instruction=None):
        """Generate caption for an image with optional RAG-enhanced prompt"""
        # If no RAG instruction is provided, use the base instruction
        instruction = rag_instruction if rag_instruction else self.base_instruction

        # Process the image and instruction
        # inputs = self.processor(images=image, text=[instruction], return_tensors="pt").to(self.device)

        inputs = self.processor(images=image, text=instruction, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=4,
                max_new_tokens=80,
                min_length=5,
                repetition_penalty=1.5,  # 添加重复惩罚
                no_repeat_ngram_size=3,  # 避免重复3-gram
                length_penalty=1.0,
            )

        # Decode generated text
        generated_caption = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return generated_caption
    def process_test_set(self):
        """Process the test set using the two-level RAG approach with CUI similarity filtering"""
        args = self.parser.parse_args()

        # Load dataset
        dataset = self.load_dataset()

        # Create test dataset
        test_dataset = ROCODataset(dataset['validation'], self.processor, 'test')
        print(f"Test dataset size: {len(test_dataset)}")

        # Load topic indices
        print("Loading topic indices...")
        topic_indices, topic_metadata, fallback_index, fallback_metadata = self.load_topic_indices()

        # Prepare dataloader for test set
        def collate_fn(batch):
            images, captions, cui_codes, ids = zip(*batch)
            return list(images), list(captions), list(cui_codes), list(ids)

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn
        )

        # Process each test image
        results = []

        for batch_idx, (images, captions, cui_codes_batch, ids) in enumerate(
                tqdm(test_loader, desc="Processing test set")):
            try:
                rag_instruction_list = []
                caption_type_list = []
                similar_docs_filter_list = []
                for i in range(len(images)):
                    image = images[i]  # Get single image
                    image_id = ids[i]
                    cui_codes = cui_codes_batch[i]

                    # Extract features
                    query_features = self.extract_image_features(image)
                    # Normalize features for cosine similarity
                    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)

                    # Find relevant topics based on CUI codes
                    relevant_topics = self.identify_relevant_topics(cui_codes)
                    similar_docs = self.search_topic_indices(
                        query_features,
                        cui_codes,
                        relevant_topics,
                        topic_indices,
                        topic_metadata,
                        fallback_index,
                        fallback_metadata,
                        top_k=args.top_k
                    )
                    similar_docs_filter = []
                    for doc in similar_docs:
                        if doc['similarity'] >= 0.97 and doc['cui_similarity'] >= 0.95 and doc['cui_similarity'] >= 0.95:
                            similar_docs_filter.append(doc)

                    # If similar documents found, use RAG approach
                    if similar_docs_filter:
                        # Generate RAG-enhanced prompt with CUI codes
                        rag_instruction = self.generate_rag_prompt(similar_docs_filter, cui_codes)
                        rag_instruction_list.append(rag_instruction)
                        caption_type_list.append("rag")
                        # Generate caption with RAG prompt

                    else:
                        rag_instruction_list.append(self.base_instruction)
                        caption_type_list.append("base")

                    similar_docs_filter_list.append(similar_docs_filter)

                batch_rag_captions = self.generate_caption(images, rag_instruction_list)

                for i in range(len(images)):
                    results.append({
                        'ID': ids[i],
                        'Caption': batch_rag_captions[i],
                        'actual_caption': captions[i],
                        'similar_docs': similar_docs_filter_list[i],
                        'caption_type': caption_type_list[i]
                    })
                # Store results

                if batch_idx % 10 == 0:
                    print(f"\nProcessed {batch_idx} test images")
                    print(f"Example rag prompt: {rag_instruction_list[0]}")
                    print(f"Example caption: {batch_rag_captions[0]}")

            except Exception as e:
                raise e
                print(f"Error processing test image {batch_idx}: {e}")
                continue

        return results

    def save_results(self, results):
        """Save generated captions to files"""
        args = self.parser.parse_args()

        # Create a custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        # Create basic results DataFrame
        results_df = pd.DataFrame([
            {
                'ID': r['ID'],
                'Caption': r['Caption'],
                'actual_caption': r['actual_caption'],
                'similar_docs': r['similar_docs'],
                'caption_type': r['caption_type'],
            }
            for r in results
        ])


        # Save both results
        basic_path = os.path.join(args.results_dir, "cluter_rag_results_valid.csv")

        results_df.to_csv(basic_path, index=False)

        # Save in format compatible with original code's output
        # submission_path = os.path.join(args.results_dir, "submission_test_instructblip_rag.csv")
        # with open(submission_path, 'w') as out_file:
        #     for r in results:
        #         out_file.write(f"{r['id']}|{r['generated_caption']}\n")
        #
        # print(f"Results saved to {basic_path}, {detailed_path}, and {submission_path}")

    def main(self):
        # Process test set
        print("Processing test set with two-level RAG approach...")
        results = self.process_test_set()

        # Save results
        self.save_results(results)

        print("Caption generation completed successfully!")


if __name__ == "__main__":
    generator = CaptionGenerator()
    generator.main()
