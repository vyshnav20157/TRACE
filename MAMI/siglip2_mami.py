import mami_gpu  # noqa: F401  (must be first: pins CUDA_VISIBLE_DEVICES before torch import)

from collections import defaultdict
import copy
import os, json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torchvision.transforms as transforms
import wandb
import argparse

# MAMI shared config (also puts the repo root on sys.path for the utils imports below).
from mami_common import MAMI_IMAGE_ROOT, MAMI_DATA_PATH, split_mami_data
from mami_metrics import SELECTION_METRIC, compute_metrics, format_metrics, wandb_metrics

# Import the modules
from utils.caption_selection import select_best_captions
from utils.loss_functions import calculate_loss_gs, FocalLoss

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGLIP2_MODEL = "google/siglip2-large-patch16-384"


class _SigLIP2Processor:
    """Minimal image+text processor for SigLIP2, pairing its Gemma tokenizer by hand.

    SigLIP2 ships a Gemma tokenizer with only `tokenizer.json` (no `spiece.model`), but
    this transformers version maps the checkpoint to the *slow* `SiglipTokenizer`, which
    requires a sentencepiece vocab file -- so `AutoProcessor.from_pretrained(...)` dies with
    `TypeError: expected str, bytes or os.PathLike object, not NoneType`, and
    `SiglipProcessor` separately refuses a `GemmaTokenizerFast`. Loading the fast tokenizer
    and image processor independently and pairing them here sidesteps both checks.

    This is the environment-level bug documented in CONTRIBUTIONS.md; the shim is carried
    over verbatim from `Memotion/siglip2_memotion.py` / `MMSD/siglip2_mmsd.py`, where it is
    already verified. It tries `AutoProcessor` first, so it self-heals on a newer
    transformers.

    Exposes only the two call shapes the dataset uses: `(images=...)` and `(text=...)`.
    The Gemma tokenizer returns no `attention_mask`; `MemeDatasetJSON.__getitem__` already
    synthesizes one from the non-pad tokens.
    """

    def __init__(self, model_name=SIGLIP2_MODEL):
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            self.image_processor = processor.image_processor
            self.tokenizer = processor.tokenizer
        except Exception as e:  # noqa: BLE001
            print(f"AutoProcessor failed ({type(e).__name__}); falling back to the Gemma tokenizer pairing.")
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, tokenizer_type="gemma")

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if images is not None and text is None:
            return self.image_processor(images=images, return_tensors=return_tensors)
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)


# Use SigLIP2 model
siglip_processor = _SigLIP2Processor()

# Caption field the dataset reads alongside the meme's own OCR text. The misogyny prompt
# (the primary) writes this field; the --caption-field flag points training at the unified
# or generic caption sets instead, for the caption-specialization ablation.
CAPTION_FIELD = "ivl_8b_new_caption"


class MemeDatasetJSON(Dataset):
    def __init__(self, dataframe, processor, caption_field=CAPTION_FIELD):
        self.data = dataframe.to_dict(orient='records')
        self.processor = processor
        self.caption_field = caption_field
        self.images = {}
        self.captions = defaultdict(list)
        self.best_captions = {}

        for row in tqdm(self.data, desc="Loading images and captions"):
            image_id = row['img']
            image_path = f'{MAMI_IMAGE_ROOT}/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image

                captions = [
                    str(row.get('text', 'No caption')),
                    str(row.get(self.caption_field, 'No caption'))
                ]
                captions = [cap for cap in captions
                            if cap.strip() and cap.strip().lower() not in ('nan', 'none')]
                if not captions:
                    # A meme with neither usable OCR text nor a caption would otherwise
                    # hand the tokenizer an empty list and raise IndexError. Keep the
                    # sample (dropping it would silently alter the official split).
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
        
        # Process image
        image_input = self.processor(images=image, return_tensors="pt", padding=True)
        image_input = {k: v.squeeze(0) for k, v in image_input.items()}
        
        # Process all captions - SigLIP2 has max_position_embeddings of 64
        text_inputs = []
        for caption in all_captions:
            text_input = self.processor(text=caption, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
            text_inputs.append({k: v.squeeze(0) for k, v in text_input.items()})
        
        # Keep a list of input_ids - SigLIP2 may not have attention_mask
        input_ids = [inp['input_ids'] for inp in text_inputs]
        # Create attention masks if they don't exist (SigLIP2 handles padding differently)
        attention_masks = []
        for inp in text_inputs:
            if 'attention_mask' in inp:
                attention_masks.append(inp['attention_mask'])
            else:
                # Create attention mask based on non-zero tokens
                attention_mask = (inp['input_ids'] != 0).long()
                attention_masks.append(attention_mask)
        
        label = torch.tensor(row['label'], dtype=torch.float)
        
        return {
            'pixel_values': image_input['pixel_values'],
            'input_ids': input_ids,  # List of tensors
            'attention_mask': attention_masks,  # List of tensors
            'labels': label,
            'image_ids': image_id,
            'num_captions': len(all_captions)
        }

class SigLIP2Classifier(nn.Module):
    def __init__(self, projection_dim=1024, num_classes=1, fusion_type='cross_attn'):
        super(SigLIP2Classifier, self).__init__()
        # Use SigLIP2 model
        self.siglip_model = AutoModel.from_pretrained(SIGLIP2_MODEL)
        self.fusion_type = fusion_type
        
        # Initialize learnable loss weights
        self.log_vars = nn.Parameter(torch.zeros(3))  # One for each loss term
        
        # Freeze SigLIP encoders
        for param in self.siglip_model.parameters():
            param.requires_grad = False
            
        # Freeze all text encoder layers by default
        for param in self.siglip_model.text_model.parameters():
            param.requires_grad = False
            
        text_encoder = self.siglip_model.text_model
        image_encoder = self.siglip_model.vision_model

        # Unfreeze some vision encoder layers (SigLIP2 uses 'encoder.layers')
        # if hasattr(image_encoder, 'encoder') and hasattr(image_encoder.encoder, 'layers'):
        #     num_vision_layers = len(image_encoder.encoder.layers)
        #     print(f"Number of vision layers: {num_vision_layers}")
        #     layers_to_unfreeze = [num_vision_layers - 2]
        #     for idx in layers_to_unfreeze:
        #         if 0 <= idx < num_vision_layers:
        #             print(f"Unfreezing vision layer {idx}: {image_encoder.encoder.layers[idx]}")
        #             for param in image_encoder.encoder.layers[idx].parameters():
        #                 param.requires_grad = True
        # else:
        #     print("Vision encoder layers not found or different structure - keeping frozen")

        # Unfreeze some text encoder layers for SigLIP2
        num_layers = len(text_encoder.encoder.layers)
        print(f"Number of text layers: {num_layers}")
        # Unfreeze 2nd to last, 4th to last, and 6th to last layers
        layers_to_unfreeze = [num_layers - 2, num_layers - 5, num_layers - 8, num_layers - 11]
        for idx in layers_to_unfreeze:
            if 0 <= idx < num_layers:
                print(f"Unfreezing text layer {idx}: {text_encoder.encoder.layers[idx]}")
                for param in text_encoder.encoder.layers[idx].parameters():
                    param.requires_grad = True

        # Add hate-aware caption scorer
        caption_scorer_input_dim = self.siglip_model.config.text_config.hidden_size
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

        # Separate projection layers for image and text
        self.image_projection = nn.Sequential(
            nn.Linear(self.siglip_model.config.vision_config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.siglip_model.config.text_config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Add cross-attention layers for better fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )

        self.cross_attn_reverse = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        # Pre-output layers with increased dropout
        pre_output_layers = []
        current_dim = projection_dim*2
        
        # First dropout
        pre_output_layers.append(nn.Dropout(0.5))
        
        # Three reduction layers
        for _ in range(3):
            pre_output_layers.extend([
                nn.Linear(current_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            current_dim = projection_dim

        self.pre_output = nn.Sequential(*pre_output_layers)

        # Final classifier
        self.classifier = nn.Linear(projection_dim, num_classes)
        
        # Print trainable parameters count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%\n")

    def combine_features(self, image_features, text_features):
        # Project features
        image_proj = self.image_projection(image_features)  # [batch, projection_dim]
        text_proj = self.text_projection(text_features)    # [batch, projection_dim]
        
        # Cross attention from image to text
        attn_out_i2t, _ = self.cross_attn(
            query=image_proj.unsqueeze(1),     # [batch, 1, projection_dim]
            key=text_proj.unsqueeze(1),        # [batch, 1, projection_dim]
            value=text_proj.unsqueeze(1)       # [batch, 1, projection_dim]
        )
        
        # Cross attention from text to image
        attn_out_t2i, _ = self.cross_attn_reverse(
            query=text_proj.unsqueeze(1),      # [batch, 1, projection_dim]
            key=image_proj.unsqueeze(1),       # [batch, 1, projection_dim]
            value=image_proj.unsqueeze(1)      # [batch, 1, projection_dim]
        )
        
        # Combine attended features
        image_enhanced = image_proj + attn_out_i2t.squeeze(1)  # [batch, projection_dim]
        text_enhanced = text_proj + attn_out_t2i.squeeze(1)    # [batch, projection_dim]
        
        # Concatenate enhanced features
        combined = torch.cat([image_enhanced, text_enhanced], dim=1)  # [batch, projection_dim * 2]
        
        return combined

    def forward(self, pixel_values, input_ids, attention_mask, return_embeddings=False):
        # Get original pooler outputs and embeddings
        image_outputs = self.siglip_model.vision_model(pixel_values=pixel_values)
        text_outputs = self.siglip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        image_embeds = self.siglip_model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.siglip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        if return_embeddings:
            return image_embeds, text_embeds
        
        # Combine features using cross-attention
        combined = self.combine_features(image_outputs.pooler_output, text_outputs.pooler_output)  # [batch, projection_dim * 2]
        
        # Process through pre-output layers
        combined = self.pre_output(combined)  # [batch, projection_dim]
        
        # Final classification
        logits = self.classifier(combined)  # [batch, num_classes]
        return logits.squeeze(1), (image_embeds, text_embeds)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Get maximum number of captions in this batch
    max_captions = max([item['num_captions'] for item in batch])
    
    # Prepare lists for stacking
    pixel_values_list = []
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    image_ids_list = []
    
    for item in batch:
        pixel_values_list.append(item['pixel_values'])
        labels_list.append(item['labels'])
        image_ids_list.append(item['image_ids'])
        
        # Pad input_ids and attention_mask if needed
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        # If this item has fewer captions than max, pad with zeros
        if len(input_ids) < max_captions:
            # Get shape of the first caption tensor
            seq_len = input_ids[0].size(0)
            
            # Create padding tensors
            pad_input_ids = torch.zeros((max_captions - len(input_ids), seq_len), dtype=input_ids[0].dtype)
            pad_attention_mask = torch.zeros((max_captions - len(attention_mask), seq_len), dtype=attention_mask[0].dtype)
            
            # Stack original tensors with padding
            input_ids = torch.stack(input_ids + [pad_input_ids[i] for i in range(pad_input_ids.size(0))])
            attention_mask = torch.stack(attention_mask + [pad_attention_mask[i] for i in range(pad_attention_mask.size(0))])
        else:
            # If we have exactly max_captions or more, just stack
            input_ids = torch.stack(input_ids[:max_captions])
            attention_mask = torch.stack(attention_mask[:max_captions])
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    
    return {
        'pixel_values': torch.stack(pixel_values_list),
        'input_ids': torch.stack(input_ids_list),  # [batch, max_captions, seq_len]
        'attention_mask': torch.stack(attention_mask_list),  # [batch, max_captions, seq_len]
        'labels': torch.stack(labels_list),
        'image_ids': image_ids_list,
        'num_captions': max_captions  # Store max_captions for reference
    }

def evaluate_model(model, dataloaders, device, loss_config):
    model.eval()
    all_probs = []
    all_labels = []

    criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for dataloader in dataloaders:
            for batch in dataloader:
                if batch is None:
                    continue
                    
                # Get logits using calculate_loss_gs in inference mode
                logits = calculate_loss_gs(model, batch, device, loss_config, is_training=False)
                labels = batch['labels'].to(device)

                # Calculate Focal Loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
                
                preds = torch.sigmoid(logits)
                all_probs.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Clear memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Calculate average loss
    avg_loss = total_loss / total_samples

    # MAMI-official macro-F1 @ 0.5 (the SemEval-2022 Task 5A ranking metric) plus the
    # TRACE-style tuned-threshold metrics and AUROC. See MAMI/mami_metrics.py.
    metrics, all_labels, preds_binary = compute_metrics(all_labels, all_probs, avg_loss=avg_loss)
    # This backbone's callers unpack (metrics, preds, labels) -- the opposite order to
    # clip_xlm_roberta_mami.evaluate_model -- so keep that contract.
    return metrics, preds_binary, all_labels

def train_epoch(model, train_dataloader, optimizer, device, accumulation_steps, loss_config, current_temp=1.0):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler()
    
    for batch_idx, batch in enumerate(train_dataloader):
        if batch is None:
            continue
            
        with torch.amp.autocast(device_type=device.type):
            loss = calculate_loss_gs(model, batch, device, loss_config, temp=current_temp)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_dataloader)

def main(args=None):
    if args is None:
        args = parse_args()

    data_path = args.data_path
    data = pd.read_json(data_path)

    # MAMI uses plain train/val/test splits (no FHM seen/unseen scheme).
    train_data, val_data, test_data = split_mami_data(data)
    if args.subset:
        train_data = train_data.sample(n=min(args.subset, len(train_data)), random_state=42)
        val_data = val_data.sample(n=min(max(args.subset // 5, 1), len(val_data)), random_state=42)
        test_data = test_data.sample(n=min(max(args.subset // 5, 1), len(test_data)), random_state=42)

    train_dataset = MemeDatasetJSON(train_data, siglip_processor, args.caption_field)
    val_datasets = [MemeDatasetJSON(val_data, siglip_processor, args.caption_field)]
    test_dataset = MemeDatasetJSON(test_data, siglip_processor, args.caption_field)

    # Enable memory efficient attention
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Define actual batch size and gradient accumulation steps
    learning_rate = 1e-4
    num_epochs = args.epochs
    actual_batch_size = 64
    target_batch_size = 512
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"\nUsing batch size {actual_batch_size} with {accumulation_steps} accumulation steps "
          f"for effective batch size {actual_batch_size * accumulation_steps}")

    # wandb.init(
    #     project="hate-memes-classification",
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": "SigLIP2-L/14-384 with GS and CS (-5 layer)",
    #         "dataset": "Hateful Memes",
    #         "epochs": num_epochs,
    #         "batch_size": actual_batch_size,
    #     },
    # )

    train_dataloader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloaders = [DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn) 
                      for val_dataset in val_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)

    model = SigLIP2Classifier()
    # Attach dataset for debug printing
    model.dataset = train_dataset
    model.to(device)

    # Enable gradient checkpointing only for the text encoder layers that are unfrozen
    if hasattr(model.siglip_model.text_model, 'gradient_checkpointing_enable'):
        try:
            # Only enable for text model since that's what we're training
            model.siglip_model.text_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for text encoder")
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

    # Force use specific GPUs for better utilization
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Use specific GPU devices
        model = nn.DataParallel(model, device_ids=[0, 1])  # Explicitly specify GPU IDs
        model.module.dataset = train_dataset

    # Initialize optimizer with initial learning rate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize scheduler with proper parameters
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.1,
        patience=2,
        min_lr=1e-7
    )
    
    # Check for existing checkpoints
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'mami_siglip2_best_model.pth')
    
    start_epoch = 0
    # Model selection follows the MAMI-official metric (macro-F1 @ 0.5), not AUROC.
    best_val_score = 0
    
    if os.path.exists(checkpoint_path) and not args.no_resume:
        print("Found existing checkpoint. Loading...")
        try:
            checkpoint = torch.load(checkpoint_path)
            # Use strict=False to handle potential architecture mismatches
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys:
                print(f"Missing keys in checkpoint (will be randomly initialized): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint (will be ignored): {len(unexpected_keys)} keys")
                
            # Only load optimizer state if the architecture hasn't changed significantly
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_score = checkpoint.get('best_val_score', checkpoint.get('best_val_auc', 0))
                print(f"Resuming from epoch {start_epoch} with validation {SELECTION_METRIC}: {best_val_score:.4f}")
            else:
                print("Architecture mismatch detected. Starting fresh training with new model architecture.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training.")
    else:
        print("No checkpoint found. Starting fresh training.")

    patience = 5
    epochs_without_improvement = 0
    best_epoch = start_epoch

    # Define loss configuration for ablation experiments
    loss_config = {
        'classification': True,  # Always enabled
        'contrastive': False,      # Enable sigmoid loss for SigLIP2
        'relevance': True        # Enable relevance loss
    }
    
    print(f"\nLoss configuration: {loss_config}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        relative_epoch = epoch - start_epoch
        total_epochs = num_epochs
        current_temp = max(1.0 - (relative_epoch / total_epochs) * 0.9, 0.1)  # Annealed from 1.0 to 0.1
        print(f"\nEpoch {epoch+1}, Temperature: {current_temp:.3f}")
        
        # Clear memory at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Select best captions for both train and val sets
        # print(f"Epoch {epoch+1}: Selecting best captions...")
        # train_best_captions = select_best_captions(model, train_dataset, device, loss_config, batch_size=512)
        # train_dataset.best_captions = train_best_captions
        
        # Clear memory after caption selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Training with gradient accumulation and mixed precision
        model.train()
        optimizer.zero_grad()
        
        # Call train_epoch with accumulation_steps and current temperature
        avg_loss = train_epoch(model, train_dataloader, optimizer, device, accumulation_steps, loss_config, current_temp)
        
        # Print progress
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # wandb.log({"Training Loss": avg_loss})
        
        # Clear memory before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Select best captions for validation
        # print("Selecting best captions for validation...")
        # for val_dataset in val_datasets:
        #     val_best_captions = select_best_captions(model, val_dataset, device, loss_config, batch_size=512)
        #     val_dataset.best_captions = val_best_captions
            
        # Clear memory after validation caption selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Validation
        print("Evaluating on Validation Set...")
        val_metrics, _, _ = evaluate_model(
            model.module if isinstance(model, nn.DataParallel) else model, 
            val_dataloaders, device, loss_config
        )
        print("Validation Metrics:")
        print(format_metrics(val_metrics, prefix="  "))
        
        # wandb.log(wandb_metrics(val_metrics, "Validation"))

        # Select on the MAMI-official metric (macro-F1 @ 0.5) rather than AUROC.
        current_val_score = float(val_metrics[SELECTION_METRIC])
        scheduler.step(current_val_score)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        # wandb.log({"Learning Rate": current_lr})
        
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            print(f"New best model with validation {SELECTION_METRIC}: {best_val_score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement. Best model was from epoch {best_epoch}")
                break
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_score': best_val_score,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, checkpoint_path)

    print(f"Training completed. Using model from epoch {best_epoch}")  

    # Dynamic Caption Selection for Test Set using the final model
    print("Selecting best captions for Test Set with the final model...")
    select_best_captions(model, test_dataset, device, loss_config, batch_size=512)

    # Evaluate on test set
    print("Evaluating on Test Set...")
    test_metrics, test_preds, test_labels = evaluate_model(model, [test_dataloader], device, loss_config)
    print("Test Metrics:")
    print(format_metrics(test_metrics, prefix="  "))

    # wandb.log(wandb_metrics(test_metrics, "Test"))

    # Save all predictions and labels for test set
    test_results = {
        'predictions': [int(pred) for pred in test_preds],
        'labels': [int(label) for label in test_labels],
    }
    with open('mami_siglip2_preds.json', 'w') as f:
        json.dump(test_results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP2 (TRACE) on MAMI.")
    parser.add_argument('--data-path', dest='data_path', default=MAMI_DATA_PATH,
                        help="MAMI dataset JSON (records orient) from build_mami_skeleton.py.")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--subset', type=int, default=None,
                        help="Train on the first N train rows (val/test scaled down) for smoke tests.")
    parser.add_argument('--wandb', action='store_true',
                        help="(Accepted for parity; SigLIP2 wandb logging is disabled in-script.)")
    parser.add_argument('--no-resume', dest='no_resume', action='store_true',
                        help="Ignore an existing checkpoint and start training from scratch.")
    parser.add_argument('--caption-field', dest='caption_field', default=CAPTION_FIELD,
                        help="JSON field holding the generated caption (default: ivl_8b_new_caption). "
                             "Use ivl_caption_unified / ivl_caption_generic for the prompt ablation.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
