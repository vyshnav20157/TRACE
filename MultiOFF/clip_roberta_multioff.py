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
import open_clip
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torchvision.transforms as transforms
import wandb

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor using OpenCLIP
# model_name = "xlm-roberta-large-ViT-H-14"
# pretrained = "frozen_laion5b_s13b_b90k"
model_name = "roberta-ViT-B-32"
pretrained = "laion2b_s12b_b32k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

def augment_image(image):
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    return augmentation(image)

class MemeDatasetCSV(Dataset):
    def __init__(self, dataframe, preprocess_fn, tokenizer):
        self.data = dataframe.to_dict(orient='records')
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer
        self.images = {}
        self.captions = defaultdict(list)
        self.context_length = model.context_length  # Get context length from model

        for row in tqdm(self.data, desc="Loading images and captions"):
            image_id = row['image_name']
            image_path = f'/backup/girish_datasets/MultiOFF/Labelled_Images/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image

                captions = [
                    str(row['sentence']),
                    str(row['ivl_8b_new_caption']),
                    str(row['gemini_caption'])
                ]
                captions = [cap for cap in captions if cap.strip()]
                self.captions[image_id] = captions
            else:
                print(f"Image file {image_path} not found.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        image_id = row['image_name']
        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return None
        
        image = self.images[image_id]
        all_captions = self.captions[image_id]
        
        # Get the best caption (selected by cosine similarity)
        best_caption = getattr(self, 'best_captions', {}).get(image_id, all_captions[0])
        
        # Process image and text
        image_tensor = self.preprocess(image)
        text_tensor = self.tokenizer([best_caption]).squeeze(0)  # Remove batch dimension
        
        # Process other captions for contrastive learning
        other_captions = [cap for cap in all_captions if cap != best_caption]
        if other_captions:
            other_tokens = self.tokenizer(other_captions).squeeze(0)  # Remove batch dimension
        else:
            # If no other captions, use the best caption again
            other_tokens = text_tensor.clone()
        
        # Convert label to binary (0 for "Non-offensiv", 1 for "offensive")
        label = 1 if str(row['label']) == "offensive" else 0
        label = torch.tensor(label, dtype=torch.float)
        
        return {
            'image': image_tensor,
            'text': self.tokenizer(all_captions),
            'label': label,
            'image_id': image_id
        }

def wise_ft(zeroshot_model, finetuned_model, alpha):
    """Apply WiSE-FT interpolation between zero-shot and fine-tuned models."""
    # Handle DataParallel wrapped models
    zeroshot_state = zeroshot_model.module.state_dict() if isinstance(zeroshot_model, nn.DataParallel) else zeroshot_model.state_dict()
    finetuned_state = finetuned_model.module.state_dict() if isinstance(finetuned_model, nn.DataParallel) else finetuned_model.state_dict()

    # Ensure both models have the same keys
    assert set(zeroshot_state.keys()) == set(finetuned_state.keys()), "Model states have different keys"

    # Interpolate between the models
    theta = {}
    for key in zeroshot_state.keys():
        theta[key] = (1 - alpha) * zeroshot_state[key] + alpha * finetuned_state[key]

    # Create a new model for the interpolated weights
    interpolated_model = copy.deepcopy(zeroshot_model)
    if isinstance(interpolated_model, nn.DataParallel):
        interpolated_model.module.load_state_dict(theta)
    else:
        interpolated_model.load_state_dict(theta)

    return interpolated_model

def select_best_captions(model, dataset, device, batch_size=128):
    """Select best captions using cosine similarity with current encoder state."""
    model.eval()
    best_captions = {}
    
    # Get the actual model from DataParallel if needed
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for batch in tqdm(dataloader, desc="Selecting best captions"):
            if batch is None:
                continue
                
            # Clear cache before processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            images = batch['image'].to(device)
            texts = batch['text'].to(device)  # [batch, 5, seq_len]
            image_ids = batch['image_ids']
            
            try:
                # Process in smaller chunks for text
                chunk_size = 8  # Process 8 captions at a time
                batch_size = texts.size(0)
                num_captions = texts.size(1)
                feature_dim = actual_model.clip_model.text.output_dim
                
                # Get image features first
                image_features = actual_model.clip_model.encode_image(images)  # [batch, dim]
                
                # Process text features in chunks
                text_features = torch.zeros(batch_size, num_captions, feature_dim, device=device)
                texts = texts.view(-1, texts.size(-1))  # [batch*5, seq_len]
                
                for i in range(0, len(texts), chunk_size):
                    end_idx = min(i + chunk_size, len(texts))
                    chunk_features = actual_model.clip_model.encode_text(texts[i:end_idx])
                    text_features.view(-1, feature_dim)[i:end_idx] = chunk_features
                    
                    # Clear memory after each chunk
                    torch.cuda.empty_cache()
                
                # Score captions using the caption scorer
                text_features_flat = text_features.view(-1, feature_dim)
                caption_scores = actual_model.caption_scorer(text_features_flat).squeeze()
                caption_scores = caption_scores.view(batch_size, num_captions)
                caption_probs = F.softmax(caption_scores, dim=-1)
                
                # Select best caption indices based on probabilities
                best_caption_idx = caption_probs.argmax(dim=1)
                
                # Store best captions
                for idx, image_id in enumerate(image_ids):
                    caption_idx = best_caption_idx[idx].item()
                    if caption_idx < len(dataset.captions[image_id]):
                        best_captions[image_id] = dataset.captions[image_id][caption_idx]
                    else:
                        best_captions[image_id] = dataset.captions[image_id][0]
                        
            except RuntimeError as e:
                print(f"Error processing batch: {e}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return best_captions

class CLIPClassifier(nn.Module):
    def __init__(self, base_model, projection_dim=1024, num_classes=1, print_params=False, fusion_type='cross_attn'):
        super(CLIPClassifier, self).__init__()
        self.clip_model = base_model
        self.fusion_type = fusion_type

        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Freeze all text encoder layers by default
        for param in self.clip_model.text.parameters():
            param.requires_grad = False
            
        image_encoder = self.clip_model.visual
        text_encoder = self.clip_model.text

        # Unfreeze last layer of image encoder
        # if hasattr(image_encoder, 'transformer'):
        #     # For transformer-based image encoders
        #     for name, param in image_encoder.transformer.named_parameters():
        #         if 'resblocks.23.' in name:  # Last transformer block
        #             print(f"Unfreezing image encoder parameter: {name}")
        #             param.requires_grad = True
        # elif hasattr(image_encoder, 'layers'):
        #     # For layer-based image encoders
        #     last_layer = image_encoder.layers[-1]
        #     print(f"Unfreezing last image encoder layer")
        #     for param in last_layer.parameters():
        #         param.requires_grad = True

        # Unfreeze last layer of text encoder
        num_layers = len(text_encoder.transformer.encoder.layer)
        print(f"Number of text encoder layers: {num_layers}")
        for layer in text_encoder.transformer.encoder.layer[num_layers-1:]:
            print(f"Unfreezing text encoder layer {layer}")
            for param in layer.parameters():
                param.requires_grad = True

        # Add hate-aware caption scorer with larger capacity and increased dropout
        self.caption_scorer = nn.Sequential(
            nn.Linear(self.clip_model.text.output_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.utils.parametrizations.weight_norm(nn.Linear(1024, 512)),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

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

        # In CLIPClassifier __init__ after unfreezing layers:
        self.text_encoder_lr = 1e-6  # Very low LR for text encoder
        self.projection_lr = 1e-4     # Higher LR for new layers

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

def train_contrastive(model, batch, device, temperature=0.07):
    """Contrastive training step using pre-selected best caption pairs."""
    images = batch['image'].to(device)
    texts = batch['text'].to(device)  # [batch, num_captions, seq_len]
    
    batch_size = images.size(0)
    num_captions = texts.size(1)  # Get number of captions dynamically
    
    # Get the actual model from DataParallel if needed
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Get image features
    image_features = actual_model.clip_model.encode_image(images)  # [batch, dim]
    
    # Get text features for all captions
    all_texts = texts.view(-1, texts.size(-1))  # [batch*num_captions, seq_len]
    text_features = actual_model.clip_model.encode_text(all_texts)  # [batch*num_captions, dim]
    text_features = text_features.view(batch_size, num_captions, -1)  # [batch, num_captions, dim]
    
    # Score captions using the caption scorer
    caption_scores = actual_model.caption_scorer(text_features.view(-1, text_features.size(-1))).squeeze()
    caption_scores = caption_scores.view(batch_size, num_captions)  # [batch, num_captions]
    
    # Get indices of best captions
    best_caption_idx = caption_scores.argmax(dim=1)  # [batch]
    
    # Select best text features for each image
    batch_indices = torch.arange(batch_size, device=device)
    best_text_features = text_features[batch_indices, best_caption_idx]  # [batch, dim]
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    best_text_features = F.normalize(best_text_features, dim=-1)
    
    # Compute similarity matrix between best pairs
    logits = torch.matmul(image_features, best_text_features.t()) / temperature
    
    # Labels for contrastive loss (diagonal is positive pairs)
    labels = torch.arange(batch_size, device=device)
    
    # Compute contrastive loss in both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smooth=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smooth = label_smooth
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
            
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute alpha factor
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute modulating factor
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        # Add label smoothing
        focal_loss = focal_loss * (1 - self.label_smooth) + self.label_smooth * bce_loss
        
        return focal_loss.mean()

def calculate_loss_gs(model, batch, device, temp=0.1, hard=True, print_every=100, is_training=True):
    """Calculate loss with Gumbel-Softmax during training and regular softmax during inference."""
    # Initialize call_count if not exists
    if is_training and not hasattr(calculate_loss_gs, 'call_count'):
        calculate_loss_gs.call_count = 0

    images = batch['image'].to(device)
    all_texts = batch['text'].to(device)  # [batch, 5, seq_len]
    labels = batch['label'].to(device)
    image_ids = batch['image_ids']
    
    batch_size = images.size(0)
    num_captions = all_texts.size(1)

    # Get the actual model from DataParallel if needed
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    # Get image features
    image_features = actual_model.clip_model.encode_image(images)  # [batch, dim]
    
    # Get text features for all captions
    all_texts = all_texts.view(-1, all_texts.size(-1))  # [batch*5, seq_len]
    text_features = actual_model.clip_model.encode_text(all_texts)  # [batch*5, dim]
    feature_dim = text_features.size(-1)
    
    # Score captions for hate relevance
    caption_scores_logits = actual_model.caption_scorer(text_features).squeeze()  # [batch*5]
    caption_scores = torch.sigmoid(caption_scores_logits)  # Apply sigmoid for probabilities
    caption_logits = caption_scores.view(batch_size, num_captions)  # [batch, 5]
    
    # Use Gumbel-Softmax during training and regular softmax during inference
    if is_training:
        selected_mask = F.gumbel_softmax(caption_logits, tau=temp, hard=hard)  # [batch, 5]
    else:
        selected_mask = F.softmax(caption_logits, dim=-1)  # [batch, 5]
    
    # Print example outputs periodically during training
    if is_training and calculate_loss_gs.call_count % print_every == 0:
        print("\n=== Caption Scorer & Selection Example Outputs ===")
        num_examples = min(3, batch_size)
        
        # Get raw probabilities
        raw_probs = F.softmax(caption_logits, dim=-1)
        
        for i in range(num_examples):
            print(f"\nImage {image_ids[i]}:")
            print("Label:", labels[i].item())
            
            # Get all captions for this image
            img_captions = [dataset.captions[image_ids[i]][j] for j in range(num_captions)]
            
            print("\nCaption Scores (Hate Relevance):")
            for j, (cap, score, raw_prob, sel_prob) in enumerate(zip(
                img_captions,
                caption_scores.view(batch_size, num_captions)[i].cpu().tolist(),
                raw_probs[i].cpu().tolist(),
                selected_mask[i].cpu().tolist()
            )):
                print(f"{j+1}. Hate Score: {score:.3f} | Raw Prob: {raw_prob:.3f} | {'Gumbel' if is_training else 'Softmax'} Prob: {sel_prob:.3f}")
                print(f"   Caption: {cap}")
                
            selected_idx = selected_mask[i].argmax().item()
            print(f"\nSelected Caption ({selected_idx+1}): {img_captions[selected_idx]}")
            print("-" * 80)
    
    # Update call count for training
    if is_training:
        calculate_loss_gs.call_count += 1
    
    # Reshape text features and apply mask
    text_features = text_features.view(batch_size, num_captions, feature_dim)  # [batch, 5, dim]
    text_features = torch.bmm(
        selected_mask.unsqueeze(1),  # [batch, 1, 5]
        text_features  # [batch, 5, dim]
    ).squeeze(1)  # [batch, dim]

    # Combine features and get logits
    combined = actual_model.combine_features(image_features, text_features)
    combined = actual_model.pre_output(combined)
    logits = actual_model.classifier(combined).squeeze()
    
    if not is_training:
        return logits
    
    # Calculate losses only during training
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smooth=0.1)
    cls_loss = criterion(logits, labels)
    contrastive_loss = train_contrastive(model, batch, device)
    
    # Add caption relevance loss using logits version of BCE
    caption_scores_logits = caption_scores_logits.view(batch_size, num_captions)
    relevance_loss = FocalLoss(alpha=0.25, gamma=2.0)(caption_scores_logits.mean(dim=1), labels.float())
    
    # Combine losses with weights
    total_loss = cls_loss + 0.3 * contrastive_loss + 0.7 * relevance_loss
    
    return total_loss

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'text': torch.stack([item['text'] for item in batch]), 
        'label': torch.stack([item['label'] for item in batch]),
        'image_ids': [item['image_id'] for item in batch]
    }

def evaluate_model(model, dataloaders, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Initialize Focal Loss for validation loss calculation
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                if batch is None:
                    continue

                labels = batch['label'].to(device)
                
                # Get logits using calculate_loss_gs in inference mode
                logits = calculate_loss_gs(model, batch, device, is_training=False)
                probs = torch.sigmoid(logits)
                
                # Calculate Focal Loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
                
                # Store probabilities for AUC calculation (aggregated over batches)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Use the current batch's probabilities (probs) for threshold calculation,
                # not the aggregated all_probs.
                precision, recall, thresholds = precision_recall_curve(labels.cpu().numpy(), probs.cpu().numpy())
                f1_scores = 2 * precision * recall / (precision + recall)
                threshold = thresholds[np.argmax(f1_scores)]

                preds = (probs >= threshold).float()
                all_preds.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate average loss
    avg_loss = total_loss / total_samples

    # Calculate metrics
    metrics = {
        'loss': f"{avg_loss:.4f}",
        'accuracy': f"{accuracy_score(all_labels, all_preds):.4f}",
        'precision': f"{precision_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}",
        'recall': f"{recall_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}",
        'f1': f"{f1_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}",
        'auc': f"{roc_auc_score(all_labels, all_probs):.4f}"
    }
    return metrics

def main():
    # Enable memory efficient attention
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    data_path = "/backup/girish_datasets/MultiOFF/multioff_updated_captions_final.csv"
    data = pd.read_csv(data_path)

    train_data = data[data['split'] == 'train']
    dev_data = data[data['split'] == 'val']
    test_data = data[data['split'] == 'test']

    # Make dataset accessible globally for logging
    global dataset
    dataset = MemeDatasetCSV(train_data, preprocess, tokenizer)
    val_datasets = [MemeDatasetCSV(dev_data, preprocess, tokenizer)]
    test_dataset = MemeDatasetCSV(test_data, preprocess, tokenizer)
    
    # Define actual batch size and gradient accumulation steps
    actual_batch_size = 32
    target_batch_size = 128
    accumulation_steps = target_batch_size // actual_batch_size
    
    print(f"\nUsing batch size {actual_batch_size} with {accumulation_steps} accumulation steps "
          f"for effective batch size of {actual_batch_size * accumulation_steps}")
    
    learning_rate = 1e-4
    num_epochs = 20

    # wandb.init(
    #     project="hate-memes-classification",
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": "CLIP-RoBERTa-Base with Gumbel-Softmax + Caption Scorer",
    #         "dataset": "Hateful Memes",
    #         "epochs": num_epochs,
    #         "batch_size": target_batch_size,
    #     },
    # )

    train_dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloaders = [DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn) 
                      for val_dataset in val_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, collate_fn=collate_fn)

    # Create the base model with parameter printing
    base_model = CLIPClassifier(model, print_params=True)
    
    # Move model to device
    base_model = base_model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        base_model = nn.DataParallel(base_model)

    optimizer = optim.AdamW(base_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    scaler = torch.amp.GradScaler(device=device)

    best_val_f1 = 0
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Calculate temperature for Gumbel-Softmax
        current_temp = max(1.0 - (epoch / num_epochs) * 0.9, 0.1)  # Annealed from 1.0 to 0.1
        print(f"\nEpoch {epoch+1}, Temperature: {current_temp:.3f}")
        
        base_model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\nEpoch {epoch+1}: Selecting best captions using current encoder state...")
        train_best_captions = select_best_captions(base_model, dataset, device)
        dataset.best_captions = train_best_captions
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if batch is None:
                continue
            
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.amp.autocast(device_type=device.type):
                total_batch_loss = calculate_loss_gs(base_model, batch, device, temp=current_temp)
            
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
        # wandb.log({"Train Loss": avg_loss})
        
        # Clear memory before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Select best captions for each validation dataset separately
        print("Selecting best captions for validation...")
        for val_dataset in val_datasets:
            val_best_captions = select_best_captions(base_model, val_dataset, device)
            val_dataset.best_captions = val_best_captions
        
        # Validation
        val_metrics = evaluate_model(base_model, val_dataloaders, device)
        print(f"Validation Metrics: {val_metrics}")
        
        # wandb.log({
        #     "Validation Accuracy": float(val_metrics['accuracy']),
        #     "Validation Precision": float(val_metrics['precision']),
        #     "Validation Recall": float(val_metrics['recall']),
        #     "Validation F1": float(val_metrics['f1']),
        #     "Validation ROC AUC": float(val_metrics['auc'])
        # })
        
        current_val_f1 = float(val_metrics['f1'])
        scheduler.step(current_val_f1)
        
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_model_state = copy.deepcopy(base_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
                
        # Clear memory at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the best model state
    base_model.load_state_dict(best_model_state)

    # Select best captions for test set
    print("\nSelecting best captions for test set...")
    test_best_captions = select_best_captions(base_model, test_dataset, device)
    test_dataset.best_captions = test_best_captions

    # Final evaluation
    test_metrics = evaluate_model(base_model, [test_dataloader], device)
    print(f"Final Test Metrics: {test_metrics}")

    # wandb.log({
    #     "Test Accuracy": float(test_metrics['accuracy']),
    #     "Test Precision": float(test_metrics['precision']),
    #     "Test Recall": float(test_metrics['recall']),
    #     "Test F1": float(test_metrics['f1']),
    #     "Test ROC AUC": float(test_metrics['auc'])
    # })

if __name__ == "__main__":
    main()
