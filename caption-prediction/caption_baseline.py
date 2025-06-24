from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoTokenizer, \
    LogitsProcessorList, PhrasalConstraint
import torch
from PIL import Image
import requests

import numpy as np
import pandas as pd
import json
import copy
import argparse, os
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

from torch.cuda.amp import autocast, GradScaler

# Assuming DMM classes are still required
try:
    from dmm import DMM
    from dmm_logits import DMMLogits
except ImportError:
    print("Warning: DMM modules not found. Some functionality may be limited.")


# Create a custom dataset class compatible with ROCOv2 data format
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
        self.image_processor = processor.image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image from the dataset
        image = item['image']
        caption = item['caption']

        # No need to transform the image here, as the processor will handle it
        # The processor takes PIL images directly and converts them to tensors

        # For test mode, also return CUI codes if available
        # if self.mode == 'test' and 'cui_codes' in item:
        #     concepts = item.get('cui_codes', [])
        #     return image, caption, concepts

        # For train/validation mode
        return image, caption, item.get('id', str(idx))


class InstructBLIP:

    def __init__(self):
        # fetch user cmd selections
        self.parser = argparse.ArgumentParser()
        self.parse_agrs()

        n_gpu = self.parser.parse_args().n_gpu
        self.device = f"cuda:{self.parser.parse_args().cuda_device}" if torch.cuda.is_available() else "cpu"

    def parse_agrs(self) -> None:
        """ Parse all arguments selected in execution from the user
        """
        parser = argparse.ArgumentParser()

        # Data loader settings
        self.parser.add_argument("--dataset_name", type=str, default="roco", choices=["iu_xray", "imageclef", "roco"],
                                 help="the dataset to be used.")

        # Data directory settings
        # self.parser.add_argument("--data_dir", type=str, default="/data/ROCOv2-radiology",
        #                          help="Path to the ROCOv2 data directory")
        self.parser.add_argument("--data_dir", type=str, default="/data/CLEF-2025-radiology",
                                 help="Path to the ROCOv2 data directory")
        # Model and results directories
        self.parser.add_argument("--base_dir", type=str, default="./results",
                                 help="Directory to save results")
        self.parser.add_argument("--model_dir", type=str, default="./new_models",
                                 help="Directory to save models")

        # Add argument for pretrained model path
        self.parser.add_argument("--pretrained_model_path", type=str, default="",
                                 help="Path to pretrained model checkpoint to continue training from")

        # Concept mapper (if still needed)
        self.parser.add_argument("--dataset_concepts_mapper", type=str, default="",
                                 help="Path to concept mapper file")

        # GPU settings
        self.parser.add_argument('--n_gpu', type=int, default=1,
                                 help='the number of gpus to be used.')
        self.parser.add_argument('--cuda_device', type=int, default=0,
                                 help='CUDA device index to use')

        # Training settings
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='the number of workers for dataloader.')
        self.parser.add_argument('--batch_size', type=int, default=15,
                                 help='the number of samples for a batch')
        self.parser.add_argument('--max_length', type=int, default=80,
                                 help='the maximum sequence length of the reports.')
        self.parser.add_argument('--learning_rate', type=float, default=5e-6,
                                 help='learning rate')
        self.parser.add_argument('--epochs', type=int, default=8,
                                 help='number of training epochs')

        args = self.parser.parse_args()
        return args

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device(f'cuda:{self.parser.parse_args().cuda_device}' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def load_roco_dataset(self):
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

        print("Dataset loaded successfully:")
        print(dataset)

        # Load concept mapper if provided
        if args.dataset_concepts_mapper and os.path.exists(args.dataset_concepts_mapper):
            concepts_mapper = pd.read_csv(args.dataset_concepts_mapper, sep="\t", header=None,
                                          names=['cui', 'concept'])

            # Build a mapper
            self._concepts_dict = {}
            for row in concepts_mapper['concept']:
                mapper = concepts_mapper.loc[concepts_mapper['concept'] == row].values.flatten().tolist()
                self._concepts_dict[mapper[0]] = mapper[1]
        else:
            self._concepts_dict = {}

        return dataset

    # Creating the training function. This will be called in the main function. It is run depending on the epoch value.
    # The model is put into train mode and then we enumerate over the training loader and passed to the defined network
    def train(self, epoch, instruction_, model, processor, device, loader, optimizer):
        model.train()  # switch into training mode
        running_loss = 0  # define loss
        batch_counter = 0  # define batch counter

        # 获取数据加载器的总长度
        total_batches = len(loader)

        # 使用tqdm并指定total参数，显示总进度
        with tqdm(total=total_batches, desc=f"Epoch {epoch}") as pbar:
            # 训练循环开始
            for batch_idx, data in enumerate(loader, 0):
                try:
                    images = data[0]  # This is a list of PIL images
                    captions = data[1]
                    ids = data[2]

                    batch_counter += 1

                    # Create instruction for each image in batch
                    instructions = [instruction_ for _ in range(len(captions))]

                    # Process images and text together - processor handles PIL images
                    inputs = processor(images=images, text=instructions, return_tensors="pt")
                    inputs = inputs.to(device)


                    # Process captions as labels
                    labels = processor.tokenizer(captions, padding="max_length", max_length=40, truncation=True,
                                                 return_tensors="pt")

                    labels["input_ids"] = torch.tensor([
                        [-100 if x == processor.tokenizer.pad_token_id else x for x in labels["input_ids"][i].tolist()]
                        for i in range(len(captions))
                    ])

                    labels = labels["input_ids"].to(device)

                    # Forward pass
                    outputs = model(**inputs, labels=labels)

                    # ====== 检查 outputs 里有没有 NaN 或 Inf ======
                    for name, output in outputs.__dict__.items():
                        if isinstance(output, torch.Tensor):
                            if torch.isnan(output).any():
                                raise ValueError(f"Found NaN in outputs[{name}] at batch {batch_idx}")
                            if torch.isinf(output).any():
                                raise ValueError(f"Found Inf in outputs[{name}] at batch {batch_idx}")

                    # Calculate loss
                    loss = outputs.loss
                    running_loss += loss.item()

                    if batch_idx % 50 == 0:
                        print(f'Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item()}')

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    # Continue with next batch if there's an error
                    pbar.update(1)  # 即使出错也更新进度条
                    continue
        del inputs
        del labels

        epoch_loss = running_loss / max(1, batch_counter)  # Avoid division by zero
        print(f'Epoch: {epoch}, Loss: {epoch_loss}, Batch Counter: {batch_counter}')

    def validate(self, epoch, instruction_, model, processor, device, loader):
        model.eval()
        val_batch_counter = 0
        running_val_loss = 0

        # 获取验证数据加载器的总长度
        total_val_batches = len(loader)

        with torch.no_grad():
            # 使用tqdm并指定total参数，显示总进度
            with tqdm(total=total_val_batches, desc=f"Validation Epoch {epoch}") as pbar:
                for batch_idx, data in enumerate(loader, 0):
                    if batch_idx >= 100:
                        break
                    try:
                        val_batch_counter += 1
                        images = data[0]  # List of PIL images
                        captions = data[1]
                        ids = data[2]

                        # Create instruction for each image
                        instructions = [instruction_ for _ in range(len(captions))]

                        # Process images and text
                        inputs = processor(images=images, text=instructions, return_tensors="pt")
                        inputs = inputs.to(device)

                        # Process captions as labels
                        labels = processor.tokenizer(captions, padding="max_length", max_length=40, truncation=True,
                                                     return_tensors="pt")
                        labels["input_ids"] = torch.tensor([
                            [-100 if x == processor.tokenizer.pad_token_id else x for x in
                             labels["input_ids"][i].tolist()]
                            for i in range(len(captions))
                        ])
                        labels = labels["input_ids"].to(device)

                        # Forward pass
                        val_outputs = model(**inputs, labels=labels)

                        # Calculate loss
                        val_loss = val_outputs.loss
                        running_val_loss += val_loss.item()

                        if batch_idx % 50 == 0:
                            print(
                                f'Validation Epoch: {epoch}, Batch: {batch_idx}/{total_val_batches}, Loss: {val_loss.item()}')

                        # 更新进度条
                        pbar.update(1)
                        pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        # Continue with next batch if there's an error
                        pbar.update(1)  # 即使出错也更新进度条
                        continue
            del inputs
            del labels

            epoch_val_loss = running_val_loss / max(1, val_batch_counter)  # Avoid division by zero
            print(f'Epoch: {epoch}, Validation Loss: {epoch_val_loss}, Batch Counter: {val_batch_counter}')

            args = self.parser.parse_args()
            best_model_path = os.path.join(args.model_dir, 'best-model.pt')

            if (epoch_val_loss < self.best_loss):
                self.best_loss = epoch_val_loss
                self.early_stopping_counter = 0

                # save model in order to retrieve at the end...
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved to {best_model_path}")
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}")

    def is_subset(self, lst1, lst2):
        return set(lst1).issubset(set(lst2))

    def remove_subsets(self, list_of_lists):
        result = []
        for i, inner_list in enumerate(list_of_lists):
            is_subset_of_any = False
            for j, other_list in enumerate(list_of_lists):
                if i != j and self.is_subset(inner_list, other_list):
                    is_subset_of_any = True
                    break
            if not is_subset_of_any:
                result.append(inner_list)
        return result

    def test(self, epoch, instruction_, model, processor, device, loader, tokenizer, test_dataset):
        model.eval()
        predictions = []
        actuals = []
        ids = []  # To store image IDs

        # 获取验证数据加载器的总长度
        total_test_batches = len(loader)

        with torch.no_grad():
            # 使用tqdm并指定total参数，显示总进度
            with tqdm(total=total_test_batches, desc=f"test Epoch {epoch}") as pbar:
                for i, data in tqdm(enumerate(loader, 0)):
                    try:
                        image = data[0]  # This is a PIL image
                        caption = data[1]
                        # Get image ID from test dataset
                        if hasattr(test_dataset, 'dataset') and i < len(test_dataset.dataset):
                            try:
                                img_id = test_dataset.dataset[i].get('id', f"img_{i}")
                                ids.append(img_id)
                            except:
                                ids.append(f"img_{i}")
                        else:
                            ids.append(f"img_{i}")

                        # Create instruction for the image
                        instruction = [instruction_]  # Only need one for single image

                        # Process the image and instruction
                        inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)

                        # Generate text
                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=5,
                            max_length=120,
                            min_length=5,
                        )

                        # Decode the generated text
                        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

                        if i % 20 == 0:
                            print(f'Completed {i} test samples')
                            print(f'Example: {generated_text[0]}')
                        # 更新进度条
                        predictions.extend(generated_text)
                        actuals.extend([caption])  # Make sure caption is wrapped in a list
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error in test batch {i}: {e}")
                        # Continue with next batch if there's an error
                        predictions.append("Error generating caption")
                        actuals.append(caption if isinstance(caption, str) else "Unknown")
                        continue

        return predictions, actuals, ids

    def main(self):
        args = self.parser.parse_args()

        # Get parameters from args
        self.TRAIN_BATCH_SIZE = args.batch_size
        self.VALID_BATCH_SIZE = args.batch_size
        self.TEST_BATCH_SIZE = 1  # Test one by one for better generation
        TRAIN_EPOCHS = args.epochs
        LEARNING_RATE = args.learning_rate
        SEED = 42

        # Create directories if they don't exist
        os.makedirs(args.base_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

        # Results paths
        GENERATIONS_PATH = os.path.join(args.base_dir, 'generated_captions.csv')
        BEST_MODEL_PATH = os.path.join(args.model_dir, 'best-model.pt')

        # Set random seeds for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Load the dataset using new method
        dataset = self.load_roco_dataset()
        torch.cuda.empty_cache()
        # Load model and processor
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl",
                                                                     torch_dtype=torch.bfloat16)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        torch.cuda.empty_cache()  # 清理缓存

        # Define the instruction to be followed during training
        instruction = 'You are an experienced radiologist. You are being given radiology images along with a short medical diagnosis. Generate a descriptive caption that highlights the location, nature and severity of the abnormality of the radiology image.'

        # Move model to device
        self.model = model.to(self.device)
        self.processor = processor

        def load_model_in_chunks(model, checkpoint_path, device, chunk_size=1000):
            """分块加载模型权重以减少内存占用"""
            print(f"分块加载模型 {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 先加载到CPU

            # 获取所有键
            keys = list(checkpoint.keys())
            num_chunks = (len(keys) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                chunk_keys = keys[i * chunk_size: (i + 1) * chunk_size]
                chunk_dict = {k: checkpoint[k] for k in chunk_keys}

                # 只更新这一部分参数
                model_dict = model.state_dict()
                model_dict.update(chunk_dict)
                model.load_state_dict(model_dict, strict=False)

                # 立即清理
                del chunk_dict
                torch.cuda.empty_cache()
                print(f"已加载区块 {i + 1}/{num_chunks}")

            # 最后清理整个checkpoint
            del checkpoint
            torch.cuda.empty_cache()
            print("模型加载完成")
            return model

        # Load pretrained weights if specified
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            self.model = load_model_in_chunks(
                self.model,
                args.pretrained_model_path,
                self.device,
                chunk_size=200  # 可以调整这个值
            )

        # Freeze parts of the model
        for i, param in enumerate(self.model.vision_model.encoder.layers.parameters()):
            param.requires_grad = False

        for i, param in enumerate(self.model.language_model.encoder.parameters()):
            param.requires_grad = False

        c = 0
        for i, param in enumerate(self.model.language_model.decoder.parameters()):
            if i <= 300:
                param.requires_grad = False
            c += 1

        c2 = 0
        for i, param in enumerate(self.model.qformer.encoder.layer.parameters()):
            c2 += 1
            if i <= 150:
                param.requires_grad = False

        # Count trainable vs frozen parameters
        true_, false_ = 0.0, 0.0
        for i, param in enumerate(self.model.parameters()):
            g = param.requires_grad
            if (g):
                true_ += 1
            else:
                false_ += 1

        print("======================================")
        print("======================================")
        print('Trainable parameters:', true_)
        print('Frozen parameters:', false_)
        trainable_percentage = 100 * true_ / (true_+false_)
        frozen_percentage = 100 * false_ / (true_+false_)
        print(f"Trainable Parameters: {true_:,} ({trainable_percentage:.2f}%)")
        print(f"Frozen Parameters: {false_:,} ({frozen_percentage:.2f}%)")
        print("======================================")
        print("======================================")

        # Create datasets and add a custom collate function for the DataLoader
        self.train_dataset = ROCODataset(dataset['train'], processor, 'train')
        self.val_dataset = ROCODataset(dataset['validation'], processor, 'validation')
        self.test_dataset = ROCODataset(dataset['test'], processor, 'test')

        # Define a custom collate function to handle PIL images
        # We don't transform the images here as the processor will handle them
        def collate_fn(batch):
            images, captions, ids = zip(*batch)
            return list(images), list(captions), list(ids)

        # Dataloader parameters
        train_params = {
            'batch_size': self.TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': args.num_workers
        }

        val_params = {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': False,
            'num_workers': args.num_workers
        }

        test_params = {
            'batch_size': self.TEST_BATCH_SIZE,
            'shuffle': False,
            'num_workers': args.num_workers
        }

        # Create dataloaders with the custom collate function
        training_loader = DataLoader(self.train_dataset, collate_fn=collate_fn, **train_params)
        val_loader = DataLoader(self.val_dataset, collate_fn=collate_fn, **val_params)
        test_loader = DataLoader(self.test_dataset, collate_fn=collate_fn, **test_params)

        print(f'Running InstructBLIP-Flan-T5xl on: {self.device}')

        # Define optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)

        print('Initiating instruction-based fine-tuning for the model on our dataset')

        # Initialize best_loss based on pretrained model if available
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            # If continuing from a pretrained model, perform validation to get initial loss
            self.model.eval()
            val_batch_counter = 0
            running_val_loss = 0

            print("=======================run val_loader===========================")
            print(len(val_loader))
            # 获取验证数据加载器的总长度
            total_val_batches = len(val_loader)

            with torch.no_grad():
                with tqdm(total=total_val_batches) as pbar:
                    for batch_idx, data in tqdm(enumerate(val_loader, 0)):
                        if batch_idx > 5:
                            break
                        try:
                            val_batch_counter += 1
                            images = data[0]  # List of PIL images
                            captions = data[1]
                            ids = data[2]

                            # Create instruction for each image
                            instructions = [instruction for _ in range(len(captions))]

                            # Process images and text
                            inputs = processor(images=images, text=instructions, return_tensors="pt")
                            inputs = inputs.to(self.device)

                            # Process captions as labels
                            labels = processor.tokenizer(captions, padding="max_length", max_length=40, truncation=True,
                                                         return_tensors="pt")
                            labels["input_ids"] = torch.tensor([
                                [-100 if x == processor.tokenizer.pad_token_id else x for x in
                                 labels["input_ids"][i].tolist()]
                                for i in range(len(captions))
                            ])
                            labels = labels["input_ids"].to(self.device)

                            # Forward pass
                            val_outputs = self.model(**inputs, labels=labels)

                            # Calculate loss
                            val_loss = val_outputs.loss
                            running_val_loss += val_loss.item()

                            pbar.update(1)
                            pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

                        except Exception as e:
                            print(f"Error in validation batch {batch_idx}: {e}")
                            continue

                    initial_val_loss = running_val_loss / max(1, val_batch_counter)
                    self.best_loss = initial_val_loss
                    print(f'Initial validation loss from pretrained model: {initial_val_loss}')
        else:
            self.best_loss = 1000000  # Start with a high loss if no pretrained model

        self.early_stopping_counter = 0
        torch.cuda.empty_cache()

        for epoch in tqdm(range(TRAIN_EPOCHS)):
            print(f"Epoch [{epoch + 1}/{TRAIN_EPOCHS}]")
            self.train(epoch, instruction, self.model, self.processor, self.device, training_loader, optimizer)
            self.validate(epoch, instruction, self.model, self.processor, self.device, val_loader)

            # Check for early stopping
            if ((self.early_stopping_counter >= 8) or (epoch == (TRAIN_EPOCHS - 1))):
                print(f"Early stopping triggered or reached max epochs. Loading best model.")
                break
        print("congratulation! we complete the training")
        # best_model = self.model
        # # Testing phase
        # print('Now generating summaries on our fine tuned model for the test dataset')
        # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        # print("=======================run val_loader===========================")
        # predictions, actuals, img_ids = self.test(0, instruction, best_model, self.processor,
        #                                           self.device, test_loader, tokenizer, self.test_dataset)
        #
        # # Print some examples
        # for i in range(min(5, len(predictions))):
        #     print(f"\nExample {i}:")
        #     print(f"Image ID: {img_ids[i]}")
        #     print(f"Generated: {predictions[i]}")
        #     print(f"Actual: {actuals[i]}")
        #
        #
        # # Save to CSV as well
        # final_df = pd.DataFrame({
        #     'Image ID': img_ids,
        #     'Generated Text': predictions,
        # })
        # final_df.to_csv(GENERATIONS_PATH, index=False)
        # print(f'Output files generated for review at {GENERATIONS_PATH}')

    def inference_validation_set(self, model_path, output_path, batch_size=1,
                                 dataset_split='validation'):
        """
        Standalone inference method to generate captions for a dataset split and save results to CSV

        Args:
            model_path: Path to the trained model checkpoint
            output_path: Path to save the output CSV
            batch_size: Batch size for inference
            instruction: Custom instruction prompt (if None, uses default)
            dataset_split: Dataset split to use ('validation' or 'test')

        Returns:
            DataFrame with results
        """
        args = self.parser.parse_args()

        # Set default instruction if not provided
        instruction = ('You are an experienced radiologist. You are being given radiology '
                       'images along with a short medical diagnosis. Generate a descriptive '
                       'caption that highlights the location, nature and severity of the '
                       'abnormality of the radiology image.')

        # Initialize processor here rather than using self.processor
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

        # Load dataset
        dataset = self.load_roco_dataset()

        # Create datasets based on the selected split
        if dataset_split == 'validation':
            inference_dataset = ROCODataset(dataset['validation'], processor, 'validation')
        elif dataset_split == 'test':
            inference_dataset = ROCODataset(dataset['test'], processor, 'test')
        else:
            raise ValueError(f"Invalid dataset split: {dataset_split}. Use 'validation' or 'test'.")

        # Define a custom collate function
        def collate_fn(batch):
            images, captions, ids = zip(*batch)
            return list(images), list(captions), list(ids)

        # Create DataLoader
        loader_params = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': args.num_workers
        }

        inference_loader = DataLoader(inference_dataset, collate_fn=collate_fn, **loader_params)

        # Load model
        print(f"Loading model from {model_path}")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16
        )

        # Load weights using chunks to save memory
        def load_model_in_chunks(model, checkpoint_path, device, chunk_size=1000):
            """Load model weights in chunks to reduce memory usage"""
            print(f"Loading model weights in chunks from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

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

        # Load the model weights in chunks
        device = self.device  # Use the device from the class
        model = load_model_in_chunks(model, model_path, device, chunk_size=200)
        model = model.to(device)
        model.eval()

        # Prepare for inference
        predictions = []
        actuals = []
        ids = []  # To store image IDs
        total_batches = len(inference_loader)

        # Run inference
        with torch.no_grad():
            with tqdm(total=total_batches, desc=f"{dataset_split.capitalize()} Inference") as pbar:
                for i, data in enumerate(inference_loader):
                    try:
                        images = data[0]  # List of PIL images
                        captions = data[1]  # List of captions
                        image_ids = data[2]  # List of image IDs

                        # Save actual captions and image IDs
                        actuals.extend(captions)
                        ids.extend(image_ids)

                        # Create instruction for each image
                        instructions = [instruction for _ in range(len(images))]

                        # Process the images and instruction
                        inputs = processor(images=images, text=instructions, return_tensors="pt").to(device)

                        # Generate captions
                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=3,
                            max_length=80,
                            min_length=5,
                            repetition_penalty=1.5,  # 添加重复惩罚
                            no_repeat_ngram_size=3,  # 避免重复3-gram
                            length_penalty=1.0,  # 控制长度
                        )

                        # Decode the generated captions
                        generated_captions = processor.batch_decode(outputs, skip_special_tokens=True)
                        predictions.extend(generated_captions)

                        # Display progress and examples periodically
                        if i % 20 == 0:
                            print(f'\nCompleted {i}/{total_batches} samples')
                            print(f'Example: {generated_captions[0]}')

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({"completed": f"{i + 1}/{total_batches}"})

                    except Exception as e:
                        print(f"Error in batch {i}: {e}")
                        # Add placeholder for failed generations
                        predictions.extend(["Error generating caption"] * len(images))
                        continue

        # Create DataFrame with results
        result_df = pd.DataFrame({
            'ID': ids,
            'Caption': predictions,
        })

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f'{dataset_split.capitalize()} results saved to {output_path}')

        # Print some examples
        print(f"\n{dataset_split.capitalize()} Examples:")
        for i in range(min(5, len(result_df))):
            print(f"\nExample {i}:")
            print(f"Image ID: {result_df['ID'].iloc[i]}")
            print(f"Generated: {result_df['Caption'].iloc[i]}")

        return result_df


if __name__ == '__main__':
    instruct_blip = InstructBLIP()
    instruct_blip.main()
    # model_path = '/home/jiawei/pyproject/ImageCLEFcompetition/baseline/test_models/best-model.pt'
    # outpath = 'results/validation_results_v3.csv'
    # instruct_blip.inference_validation_set(model_path, outpath, batch_size=16)
