import mmsd_gpu  # noqa: F401  (must be first: pins CUDA_VISIBLE_DEVICES before torch import)

from collections import defaultdict
import copy
import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import wandb
import argparse

# MMSD shared config (also puts the repo root on sys.path for the utils imports below).
from mmsd_common import (
    MMSD_IMAGE_ROOT,
    MMSD_DATA_PATH,
    describe_task,
    split_mmsd_data,
)
from mmsd_metrics import SELECTION_METRIC, compute_metrics, format_metrics, wandb_metrics

# Import the modules
from utils.caption_selection import select_best_captions
from utils.loss_functions import calculate_loss_gs, FocalLoss

# Set seed for reproducibility
seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor using OpenCLIP
model_name = "xlm-roberta-large-ViT-H-14"
pretrained = "frozen_laion5b_s13b_b90k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
hf_tokenizer = tokenizer.tokenizer

# Caption field the dataset reads alongside the tweet's own text. Overridden by
# --caption-field when running the caption-specialization ablation (the unified/generic
# prompt variants written by mmsd_cap_gen.py).
CAPTION_FIELD = "ivl_8b_new_caption"


class MemeDatasetJSON(Dataset):
    def __init__(self, dataframe, preprocess_fn, tokenizer, caption_field=CAPTION_FIELD):
        self.data = dataframe.to_dict(orient='records')
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer
        self.caption_field = caption_field
        self.images = {}
        self.captions = defaultdict(list)
        self.best_captions = {}

        for row in tqdm(self.data, desc="Loading images and captions"):
            image_id = row['img']
            image_path = f'{MMSD_IMAGE_ROOT}/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image

                # MMSD uses a single generated caption (InternVL); Gemini is not part of this
                # flow, so captions are [tweet text, ivl caption] -- or just the tweet text
                # before captioning has run.
                captions = [
                    str(row.get('text', 'No caption')),
                    str(row.get(self.caption_field, 'No caption'))
                ]
                captions = [cap for cap in captions if cap.strip() and cap.strip().lower() != 'nan']
                # A tweet whose text is empty would, before captioning has run, leave the caption
                # list empty and make the tokenizer fail on []. Keep the sample (dropping it
                # would silently alter the official test split) and fall back to a placeholder.
                if not captions:
                    captions = ['No caption']
                self.captions[image_id] = captions
            else:
                print(f"Image file {image_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_id = row['img']
        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return None

        image = self.images[image_id]
        all_captions = self.captions[image_id]

        # Process image and text
        image_tensor = self.preprocess(image)

        label = torch.tensor(row['label'], dtype=torch.float)

        return {
            'image': image_tensor,
            'text': self.tokenizer(all_captions),
            'label': label,
            'image_id': image_id
        }


class CLIPClassifier(nn.Module):
    def __init__(self, base_model, projection_dim=1024, num_classes=1, print_params=False, fusion_type='cross_attn'):
        super(CLIPClassifier, self).__init__()
        self.clip_model = base_model
        self.fusion_type = fusion_type

        # Initialize learnable loss weights (log variances)
        self.log_vars = nn.Parameter(torch.zeros(3))  # One for each loss term: classification, relevance, contrastive

        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Freeze all text encoder layers by default
        for param in self.clip_model.text.parameters():
            param.requires_grad = False

        text_encoder = self.clip_model.text

        # Unfreeze last layer of text encoder
        num_layers = len(text_encoder.transformer.encoder.layer)
        print(f"Number of text encoder layers: {num_layers}")
        for idx in [num_layers-2, num_layers-5, num_layers-8, num_layers-11]:
            print(f"Unfreezing text encoder layer {idx}")
            for param in text_encoder.transformer.encoder.layer[idx].parameters():
                param.requires_grad = True

        caption_scorer_input_dim = self.clip_model.text.output_dim
        caption_scorer_hidden_dim = 1024
        caption_scorer_output_dim = 512
        caption_scorer_layers = []

        # First layer with higher dropout
        caption_scorer_layers.extend([
            nn.Linear(caption_scorer_input_dim, caption_scorer_hidden_dim),
            nn.LayerNorm(caption_scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        ])

        # Middle layers with moderate dropout
        for _ in range(2):  # Add 2 middle layers
            caption_scorer_layers.extend([
                nn.utils.parametrizations.weight_norm(nn.Linear(caption_scorer_hidden_dim, caption_scorer_hidden_dim)),
                nn.LayerNorm(caption_scorer_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.4)
            ])

        # Final reduction layer
        caption_scorer_layers.extend([
            nn.utils.parametrizations.weight_norm(nn.Linear(caption_scorer_hidden_dim, caption_scorer_output_dim)),
            nn.LayerNorm(caption_scorer_output_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(caption_scorer_output_dim, 1)
        ])

        self.caption_scorer = nn.Sequential(*caption_scorer_layers)

        # Initialize the weights for better training
        for layer in self.caption_scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.clip_model.text.output_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Add cross-attention layer for better fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )

        # Add a second cross-attention for bidirectional interaction
        self.cross_attn_reverse = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )

        # Pre-output layers with increased dropout
        self.pre_output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Linear(projection_dim, num_classes)

        if print_params:
            total_params = sum(p.numel() for p in base_model.parameters())
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%\n")

    def combine_features(self, image_features, text_features):
        # Project features
        image_proj = self.image_projection(image_features)  # [batch, dim]
        text_proj = self.text_projection(text_features)    # [batch, dim]

        # Cross attention from image to text
        attn_out_i2t, _ = self.cross_attn(
            query=image_proj.unsqueeze(1),     # [batch, 1, dim]
            key=text_proj.unsqueeze(1),        # [batch, 1, dim]
            value=text_proj.unsqueeze(1)       # [batch, 1, dim]
        )

        # Cross attention from text to image
        attn_out_t2i, _ = self.cross_attn_reverse(
            query=text_proj.unsqueeze(1),      # [batch, 1, dim]
            key=image_proj.unsqueeze(1),       # [batch, 1, dim]
            value=image_proj.unsqueeze(1)      # [batch, 1, dim]
        )

        # Combine attended features
        image_enhanced = image_proj + attn_out_i2t.squeeze(1)  # Add attended text features
        text_enhanced = text_proj + attn_out_t2i.squeeze(1)    # Add attended image features

        # Concatenate enhanced features
        combined = torch.cat([image_enhanced, text_enhanced], dim=1)

        return combined

    def forward(self, images, texts, return_embeddings=False):
        # Get embeddings
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)

        if return_embeddings:
            return image_features, text_features

        # Combine features using cross-attention
        combined = self.combine_features(image_features, text_features)

        # Process through pre-output layers
        combined = self.pre_output(combined)

        # Final classification
        logits = self.classifier(combined)
        return logits.squeeze(1)


def collate_fn(batch):
    # Instead of stacking text directly, pad the tensors to same first dimension
    texts = [item['text'] for item in batch]
    max_sequences = max([text.size(0) for text in texts])

    # Pad each text tensor to have the same first dimension
    padded_texts = []
    text_masks = []
    for text in texts:
        num_sequences = text.size(0)
        if num_sequences < max_sequences:
            # Create padding tensor with same second dimension (77)
            padding = torch.zeros((max_sequences - num_sequences, text.size(1)),
                                 dtype=text.dtype, device=text.device)
            padded_text = torch.cat([text, padding], dim=0)
            # Create mask to track which sequences are real vs padding
            mask = torch.cat([torch.ones(num_sequences), torch.zeros(max_sequences - num_sequences)])
        else:
            padded_text = text
            mask = torch.ones(max_sequences)

        padded_texts.append(padded_text)
        text_masks.append(mask)

    return {
        'image': torch.stack([item['image'] for item in batch]),
        'text': torch.stack(padded_texts),
        'text_mask': torch.stack(text_masks),  # Add mask to know which texts are valid
        'image_ids': [item['image_id'] for item in batch],
        'label': torch.stack([item['label'] for item in batch])  # Added label
    }


def evaluate_model(model, dataloaders, device):
    """Evaluate and return MMSD-official (acc + sarcastic-class F1 @0.5) + TRACE-style metrics."""
    model.eval()
    all_labels = []
    all_probs = []

    # Initialize Focal Loss for validation loss calculation
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    total_loss = 0
    total_samples = 0

    loss_config = {
        'classification': True,  # Always enabled
        'contrastive': False,     # Set to False to disable contrastive loss
        'relevance': True        # Set to False to disable relevance loss
    }

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                if batch is None:
                    continue

                labels = batch['label'].to(device)

                # Get logits using calculate_loss_gs in inference mode
                logits = calculate_loss_gs(model, batch, device, loss_config, is_training=False)
                probs = torch.sigmoid(logits)

                # Calculate Focal Loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    return compute_metrics(all_labels, all_probs, avg_loss=avg_loss)


def main(args=None):
    if args is None:
        args = parse_args()

    # Enable memory efficient attention and better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

    print(f"\n=== MMSD2.0 task: {describe_task()} ===")

    data = pd.read_json(args.data_path)
    # MMSD is a single binary task, so the skeleton already writes the generic `label`
    # column the shared TRACE machinery reads -- no per-task projection is needed.

    train_data, val_data, test_data = split_mmsd_data(data)
    if args.subset:
        train_data = train_data.sample(n=min(args.subset, len(train_data)), random_state=42)
        val_data = val_data.sample(n=min(max(args.subset // 5, 1), len(val_data)), random_state=42)
        test_data = test_data.sample(n=min(max(args.subset // 5, 1), len(test_data)), random_state=42)

    print(f"Splits -> train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")

    # Make dataset accessible globally for logging
    global dataset
    dataset = MemeDatasetJSON(train_data, preprocess, tokenizer, args.caption_field)
    val_datasets = [MemeDatasetJSON(val_data, preprocess, tokenizer, args.caption_field)]
    test_dataset = MemeDatasetJSON(test_data, preprocess, tokenizer, args.caption_field)

    # Define actual batch size and gradient accumulation steps
    actual_batch_size = 64
    target_batch_size = 512
    accumulation_steps = target_batch_size // actual_batch_size

    print(f"\nUsing batch size {actual_batch_size} with {accumulation_steps} accumulation steps "
          f"for effective batch size of {actual_batch_size * accumulation_steps}")

    learning_rate = 1e-4
    num_epochs = args.epochs

    if args.wandb:
        wandb.init(
            project="mmsd-classification",
            name="roberta",
            config={
                "learning_rate": learning_rate,
                "architecture": "CLIP-XLMR-Large with GS+CS (-1 layer)",
                "dataset": "MMSD2.0",
                "epochs": num_epochs,
                "batch_size": target_batch_size,
            },
        )

    train_dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloaders = [DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)
                      for val_dataset in val_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)

    # Create the base model with parameter printing
    base_model = CLIPClassifier(model, print_params=True)

    # Attach dataset to model for debug printing
    base_model.dataset = dataset

    # Move model to device
    base_model = base_model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Use smaller batch size per GPU for DataParallel
        actual_batch_size = actual_batch_size // torch.cuda.device_count()
        accumulation_steps = target_batch_size // (actual_batch_size * torch.cuda.device_count())
        print(f"Adjusted batch size per GPU: {actual_batch_size}, accumulation steps: {accumulation_steps}")

        base_model = nn.DataParallel(base_model)
        # Make sure dataset is still accessible through DataParallel
        base_model.module.dataset = dataset

    optimizer = optim.AdamW(base_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    scaler = torch.amp.GradScaler(device=device)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'mmsd_roberta_best_model.pth')

    start_epoch = 0
    # Model selection follows the OFFICIAL MMSD metric (sarcastic-class F1 @ 0.5), not AUROC.
    best_val_score = 0
    if os.path.exists(checkpoint_path) and not args.no_resume:
        print("Found existing checkpoint. Loading...")
        checkpoint = torch.load(checkpoint_path)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_score = checkpoint.get('best_val_score', 0)
        print(f"Resuming from epoch {start_epoch} with validation {SELECTION_METRIC}: {best_val_score:.4f}")
    else:
        print("No checkpoint found. Starting fresh training.")

    patience = 5
    epochs_without_improvement = 0
    best_model_state = copy.deepcopy(base_model.state_dict())

    # Define loss configuration for ablation experiments
    loss_config = {
        'classification': True,  # Always enabled
        'contrastive': False,     # Set to False to disable contrastive loss
        'relevance': True        # Set to False to disable relevance loss
    }

    print(f"\nLoss configuration: {loss_config}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Calculate temperature for Gumbel-Softmax
        current_temp = max(1.0 - ((epoch - start_epoch) / num_epochs) * 0.9, 0.1)  # Annealed from 1.0 to 0.1
        print(f"\nEpoch {epoch+1}, Temperature: {current_temp:.3f}")

        base_model.train()
        total_loss = 0
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if batch is None:
                continue

            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.amp.autocast(device_type=device.type):
                total_batch_loss = calculate_loss_gs(base_model, batch, device, loss_config, temp=current_temp)

            scaled_loss = scaler.scale(total_batch_loss / accumulation_steps)
            scaled_loss.backward()

            # Add gradient clipping
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in base_model.parameters() if p.requires_grad],
                    max_norm=2.0,
                    norm_type=2
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        if args.wandb:
            wandb.log({"Train Loss": avg_loss})

        # Clear memory before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation
        val_metrics, _, _ = evaluate_model(base_model, val_dataloaders, device)
        print("Validation:")
        print(format_metrics(val_metrics, prefix="  "))

        if args.wandb:
            wandb.log(wandb_metrics(val_metrics, "Validation"))

        current_val_score = float(val_metrics[SELECTION_METRIC])
        scheduler.step(current_val_score)

        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_model_state = copy.deepcopy(base_model.state_dict())
            epochs_without_improvement = 0
            print(f"New best model with validation {SELECTION_METRIC}: {best_val_score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

        # Save the BEST model state (not the latest), so a resumed or reloaded checkpoint
        # is the one selection actually chose.
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_score': best_val_score,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, checkpoint_path)

        # Clear memory at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the best model state
    base_model.load_state_dict(best_model_state)

    # Select best captions for test set
    print("\nSelecting best captions for test set...")
    test_best_captions = select_best_captions(base_model, test_dataset, device, loss_config)
    test_dataset.best_captions = test_best_captions

    # Final evaluation on the official MMSD2.0 test split.
    test_metrics, all_preds, all_labels = evaluate_model(base_model, [test_dataloader], device)
    print("\nFinal Test Metrics:")
    print(format_metrics(test_metrics, prefix="  "))

    if args.wandb:
        wandb.log(wandb_metrics(test_metrics, "Test"))

    # Save all predictions and labels for further analysis
    results = {
        'backbone': 'roberta',
        'metrics': test_metrics,
        'labels': [int(x) for x in all_labels],
        'predictions': [int(x) for x in all_preds],
    }
    preds_path = 'mmsd_roberta_preds.json'
    with open(preds_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Wrote predictions -> {preds_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-XLM-RoBERTa (TRACE) on MMSD2.0.")
    parser.add_argument('--data-path', dest='data_path', default=MMSD_DATA_PATH,
                        help="MMSD dataset JSON (records orient) from build_mmsd_skeleton.py.")
    parser.add_argument('--caption-field', dest='caption_field', default=CAPTION_FIELD,
                        help="JSON field holding the generated caption (default: ivl_8b_new_caption).")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--subset', type=int, default=None,
                        help="Train on N sampled train rows (val/test scaled down) for smoke tests.")
    parser.add_argument('--no-resume', action='store_true',
                        help="Ignore any existing checkpoint and start fresh.")
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
