import memotion_gpu  # noqa: F401  (must be first: pins CUDA_VISIBLE_DEVICES before torch import)

from collections import defaultdict
import copy
import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import wandb
import argparse

# Memotion shared config (also puts the repo root on sys.path for the utils imports below).
from memotion_common import (
    MEMOTION_IMAGE_ROOT,
    MEMOTION_DATA_PATH,
    TASK_ORDER,
    apply_task_labels,
    describe_task,
    split_memotion_data,
)
from memotion_metrics import SELECTION_METRIC, compute_metrics, format_metrics, wandb_metrics

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

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Caption field the dataset reads alongside the meme's own text.
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
            image_path = f'{MEMOTION_IMAGE_ROOT}/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image

                # Memotion uses a single generated caption (InternVL); Gemini is not part
                # of this flow.
                captions = [
                    str(row.get('text', 'No caption')),
                    str(row.get(self.caption_field, 'No caption'))
                ]
                captions = [cap for cap in captions if cap.strip() and cap.strip().lower() != 'nan']
                # 35 Memotion memes carry no usable OCR text, so before captioning has run
                # their caption list would be empty and the processor would fail on []. Keep
                # the sample (dropping it would silently alter the official test set) and
                # fall back to a neutral placeholder.
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

        # Process image
        image_input = self.processor(images=image, return_tensors="pt", padding=True)
        image_input = {k: v.squeeze(0) for k, v in image_input.items()}

        # Process all captions
        text_inputs = []
        for caption in all_captions:
            text_input = self.processor(text=caption, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
            text_inputs.append({k: v.squeeze(0) for k, v in text_input.items()})

        # Instead of stacking, we keep a list of input_ids and attention_masks
        input_ids = [inp['input_ids'] for inp in text_inputs]
        attention_masks = [inp['attention_mask'] for inp in text_inputs]

        label = torch.tensor(row['label'], dtype=torch.float)

        return {
            'pixel_values': image_input['pixel_values'],
            'input_ids': input_ids,  # List of tensors
            'attention_mask': attention_masks,  # List of tensors
            'labels': label,
            'image_ids': image_id,
            'num_captions': len(all_captions)
        }


class CLIPClassifier(nn.Module):
    def __init__(self, projection_dim=1024, num_classes=1, fusion_type='cross_attn'):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.fusion_type = fusion_type

        # Initialize learnable loss weights
        # We use log(sigma^2) for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(3))  # One for each loss term

        # Freeze CLIP encoders
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Freeze all text encoder layers by default
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False

        text_encoder = self.clip_model.text_model

        # Unfreeze last 4 transformer encoder layers
        num_layers = len(text_encoder.encoder.layers)
        print(f"Number of layers: {num_layers}")
        for idx in [num_layers - 2, num_layers - 5, num_layers - 8]:
            print(f"Unfreezing layer {idx}")
            for param in text_encoder.encoder.layers[idx].parameters():
                param.requires_grad = True

        # Caption scorer with larger capacity and multiple pre-output layers
        caption_scorer_input_dim = self.clip_model.config.text_config.hidden_size
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
            nn.Linear(self.clip_model.config.vision_config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.clip_model.config.text_config.hidden_size, projection_dim),
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
        image_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        text_outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)

        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

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
            # If we have exactly max_captions or more (should be exactly), just stack
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
    """Evaluate and return Memotion-official (macro-F1 @0.5) + TRACE-style metrics."""
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
                all_probs.extend(preds.float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Clear memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    metrics, preds_binary, labels_arr = compute_metrics(all_labels, all_probs, avg_loss=avg_loss)
    return metrics, preds_binary, labels_arr


def train_epoch(model, train_dataloader, optimizer, device, accumulation_steps, loss_config, current_temp=1.0):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler()

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
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

    task = args.task
    print(f"\n=== Memotion task: {describe_task(task)} ===")

    data = pd.read_json(args.data_path)
    # Project the chosen task's label onto the generic `label` column.
    data = apply_task_labels(data, task)

    train_data, val_data, test_data = split_memotion_data(data)
    if args.subset:
        train_data = train_data.sample(n=min(args.subset, len(train_data)), random_state=42)
        val_data = val_data.sample(n=min(max(args.subset // 5, 1), len(val_data)), random_state=42)
        test_data = test_data.sample(n=min(max(args.subset // 5, 1), len(test_data)), random_state=42)

    print(f"Splits -> train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")

    train_dataset = MemeDatasetJSON(train_data, clip_processor, args.caption_field)
    val_datasets = [MemeDatasetJSON(val_data, clip_processor, args.caption_field)]
    test_dataset = MemeDatasetJSON(test_data, clip_processor, args.caption_field)

    # Enable memory efficient attention
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Define actual batch size and gradient accumulation steps
    learning_rate = 1e-4
    num_epochs = args.epochs
    actual_batch_size = 128
    target_batch_size = 512
    accumulation_steps = target_batch_size // actual_batch_size

    print(f"\nUsing batch size {actual_batch_size} with {accumulation_steps} accumulation steps "
          f"for effective batch size {actual_batch_size * accumulation_steps}")

    if args.wandb:
        wandb.init(
            project="memotion-classification",
            name=f"vitl14-{task}",
            config={
                "learning_rate": learning_rate,
                "architecture": "CLIP-ViT-L/14 with GS and CS (-1 layer)",
                "dataset": "Memotion 1.0",
                "task": task,
                "epochs": num_epochs,
                "batch_size": actual_batch_size,
            },
        )

    train_dataloader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloaders = [DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)
                      for val_dataset in val_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)

    zeroshot_model = CLIPClassifier()
    # Attach dataset for debug printing
    zeroshot_model.dataset = train_dataset
    zeroshot_model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        zeroshot_model = nn.DataParallel(zeroshot_model)
        # Make sure dataset is still accessible through DataParallel
        zeroshot_model.module.dataset = train_dataset

    # Initialize optimizer with initial learning rate
    optimizer = optim.AdamW(zeroshot_model.parameters(), lr=learning_rate)

    # Initialize scheduler with proper parameters
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,  # Reduce LR by factor of 0.1
        patience=2,   # Wait for 2 epochs without improvement
        min_lr=1e-7  # Minimum LR to prevent it from becoming too small
    )

    # Checkpoints are task-scoped: humour / offensive / sarcasm each train their own model.
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'memotion_{task}_vitl14_best_model.pth')

    start_epoch = 0
    # Model selection follows the OFFICIAL Memotion metric (macro-F1 @ 0.5), not AUROC.
    best_val_score = 0

    if os.path.exists(checkpoint_path) and not args.no_resume:
        print("Found existing checkpoint. Loading...")
        checkpoint = torch.load(checkpoint_path)
        zeroshot_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_score = checkpoint.get('best_val_score', 0)
        print(f"Resuming from epoch {start_epoch} with validation {SELECTION_METRIC}: {best_val_score:.4f}")
    else:
        print("No checkpoint found. Starting fresh training.")

    patience = 5
    epochs_without_improvement = 0
    best_epoch = start_epoch
    best_model_state = copy.deepcopy(zeroshot_model.state_dict())

    # Define loss configuration for ablation experiments
    loss_config = {
        'classification': True,  # Always enabled
        'contrastive': False,     # Set to False to disable contrastive loss
        'relevance': True        # Set to False to disable relevance loss
    }

    print(f"\nLoss configuration: {loss_config}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        relative_epoch = epoch - start_epoch
        current_temp = max(1.0 - (relative_epoch / num_epochs) * 0.9, 0.1)  # Annealed from 1.0 to 0.1
        print(f"\nEpoch {epoch+1}, Temperature: {current_temp:.3f}")

        # Clear memory at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training with gradient accumulation and mixed precision
        zeroshot_model.train()
        optimizer.zero_grad()

        # Call train_epoch with accumulation_steps and current temperature
        avg_loss = train_epoch(zeroshot_model, train_dataloader, optimizer, device, accumulation_steps, loss_config, current_temp)

        # Print progress
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        if args.wandb:
            wandb.log({"Training Loss": avg_loss})

        # Clear memory before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation
        print("Evaluating on Validation Set...")
        val_metrics, _, _ = evaluate_model(
            zeroshot_model.module if isinstance(zeroshot_model, nn.DataParallel) else zeroshot_model,
            val_dataloaders, device, loss_config
        )
        print(format_metrics(val_metrics, prefix="  "))

        if args.wandb:
            wandb.log(wandb_metrics(val_metrics, "Validation"))

        current_val_score = float(val_metrics[SELECTION_METRIC])
        scheduler.step(current_val_score)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        if args.wandb:
            wandb.log({"Learning Rate": current_lr})

        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(zeroshot_model.state_dict())
            print(f"New best model with validation {SELECTION_METRIC}: {best_val_score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement. Best model was from epoch {best_epoch}")
                break

        # Save the BEST model state (not the latest).
        checkpoint = {
            'epoch': epoch + 1,
            'task': task,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_score': best_val_score,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, checkpoint_path)

    print(f"Training completed. Using model from epoch {best_epoch}")
    zeroshot_model.load_state_dict(best_model_state)

    # Dynamic Caption Selection for Test Set using the final model
    print("Selecting best captions for Test Set with the final model...")
    select_best_captions(zeroshot_model, test_dataset, device, loss_config, batch_size=512)

    # Evaluate on test set
    print("Evaluating on Test Set...")
    test_metrics, all_preds, all_labels = evaluate_model(zeroshot_model, [test_dataloader], device, loss_config)
    print(f"\nFinal Test Metrics ({task}):")
    print(format_metrics(test_metrics, prefix="  "))

    if args.wandb:
        wandb.log(wandb_metrics(test_metrics, "Test"))

    # Save all predictions and labels for test set
    test_results = {
        'task': task,
        'backbone': 'vitl14',
        'metrics': test_metrics,
        'predictions': [int(pred) for pred in all_preds],
        'labels': [int(label) for label in all_labels],
    }
    preds_path = f'memotion_{task}_vitl14_preds.json'
    with open(preds_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Wrote predictions -> {preds_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP-ViT-L/14 (TRACE) on Memotion.")
    parser.add_argument('--task', choices=TASK_ORDER, default='humour',
                        help="Which Memotion Task B binary problem to train (default: humour).")
    parser.add_argument('--data-path', dest='data_path', default=MEMOTION_DATA_PATH,
                        help="Memotion dataset JSON (records orient) from build_memotion_skeleton.py.")
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
