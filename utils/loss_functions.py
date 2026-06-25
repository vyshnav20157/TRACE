import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.caption_selection import print_caption_selection_debug

try:
    from __main__ import hf_tokenizer
except ImportError:
    hf_tokenizer = None

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smooth=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smooth = label_smooth
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape and are on same device
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
            
        # Ensure both tensors are on the same device
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        # Compute BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute alpha factor and ensure it's on the right device
        alpha_tensor = torch.tensor([1-self.alpha, self.alpha], device=inputs.device)
        alpha_factor = alpha_tensor[targets.long()]
        
        # Compute modulating factor
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = alpha_factor * modulating_factor * BCE_loss
        
        # Add label smoothing
        if self.label_smooth > 0:
            focal_loss = focal_loss * (1 - self.label_smooth) + self.label_smooth * BCE_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_classification_loss(model, batch, device):
    """Calculate only classification loss."""
    # Handle different batch formats (CLIP vs OpenCLIP)
    if 'pixel_values' in batch:
        # CLIP format
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Get first caption for each image (or the best caption if available)
        first_input_ids = input_ids[:, 0]
        first_attention_mask = attention_mask[:, 0]
        
        # Forward pass
        logits, _ = model(pixel_values, first_input_ids, first_attention_mask)
    else:
        # OpenCLIP format
        images = batch['image'].to(device)
        texts = batch['text'].to(device)  # [batch, num_captions, seq_len]
        labels = batch['label'].to(device)

        if texts.dim() == 3:
            selected_texts = texts[:, 0, :]  # Use first caption
            
            # Forward pass with selected texts
            logits = model(images, selected_texts)
        else:
            # If texts is already the right shape, use it directly
            logits = model(images, texts)
    
    # Calculate classification loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    cls_loss = criterion(logits, labels)
    
    return cls_loss

def calculate_contrastive_loss(model, batch, device, temperature=0.07):
    """Calculate contrastive loss between image and text embeddings."""
    # Handle different batch formats (CLIP vs OpenCLIP vs SigLIP2)
    if 'pixel_values' in batch:
        # CLIP/SigLIP2 format
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        batch_size = pixel_values.size(0)
        
        # Get the actual model from DataParallel if needed
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Check if it's SigLIP2 or CLIP model
        if hasattr(actual_model, 'siglip_model'):
            # SigLIP2 model - use sigmoid loss
            image_embeds = actual_model.siglip_model.get_image_features(pixel_values=pixel_values)
            text_embeds = actual_model.siglip_model.get_text_features(
                input_ids=input_ids[:, 0],  # Use first caption
                attention_mask=attention_mask[:, 0]
            )
            
            # Normalize features
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            # Compute similarity matrix
            logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
            
            # SigLIP uses sigmoid loss instead of softmax
            # Positive pairs (diagonal) should have high similarity, negative pairs should have low similarity
            labels = torch.eye(batch_size, device=device)  # Identity matrix for positive pairs
            
            # Apply sigmoid and compute binary cross entropy
            sigmoid_logits = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(sigmoid_logits, labels, reduction='mean')
            
        else:
            # CLIP model - use traditional contrastive loss
            image_embeds = actual_model.clip_model.get_image_features(pixel_values=pixel_values)
            text_embeds = actual_model.clip_model.get_text_features(
                input_ids=input_ids[:, 0],  # Use first caption
                attention_mask=attention_mask[:, 0]
            )
            
            # Normalize features
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            # Compute similarity matrix
            logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
            
            # Labels for contrastive loss (diagonal is positive pairs)
            labels = torch.arange(batch_size, device=device)
            
            # Compute contrastive loss in both directions
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            loss = (loss_i2t + loss_t2i) / 2.0
    else:
        # OpenCLIP format - always use traditional contrastive loss
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        
        batch_size = images.size(0)
        
        # Handle different text tensor shapes
        if texts.dim() == 3:
            # For multiple captions, use only the first caption
            first_caption = texts[:, 0, :]
            
            # Get the actual model from DataParallel if needed
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            text_embeds = actual_model.clip_model.encode_text(first_caption)
        else:
            # Normal 2D case
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            text_embeds = actual_model.clip_model.encode_text(texts)
        
        image_embeds = actual_model.clip_model.encode_image(images)
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
        
        # Labels for contrastive loss (diagonal is positive pairs)
        labels = torch.arange(batch_size, device=device)
        
        # Compute contrastive loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss

def calculate_relevance_loss(model, batch, device):
    """Calculate relevance loss using caption scorer."""
    # Handle different batch formats (CLIP vs OpenCLIP vs SigLIP2)
    if 'pixel_values' in batch:
        # CLIP/SigLIP2 format
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Get the actual model from DataParallel if needed
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        batch_size = input_ids.size(0)
        num_captions = input_ids.size(1)
        
        # Process each caption
        text_features_list = []
        valid_mask = torch.zeros(batch_size, num_captions, device=device)
        
        for i in range(num_captions):
            # Use attention mask to determine if this is a real caption or padding
            is_valid_caption = attention_mask[:, i].sum(dim=1) > 0
            valid_mask[:, i] = is_valid_caption.float()
            
            if not is_valid_caption.any():
                continue  # Skip if this caption position is all padding
            
            # Check if it's SigLIP2 or CLIP model
            if hasattr(actual_model, 'siglip_model'):
                # SigLIP2 model
                text_outputs = actual_model.siglip_model.text_model(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i]
                )
            else:
                # CLIP model
                text_outputs = actual_model.clip_model.text_model(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i]
                )
            text_features = text_outputs.pooler_output
            text_features_list.append(text_features)
        
        if not text_features_list:
            # This shouldn't happen, but just in case
            return torch.tensor(0.0, device=device)
            
        text_features = torch.stack(text_features_list, dim=1)  # [batch, valid_captions, dim]
        valid_captions = text_features.size(1)
        
    else:
        # OpenCLIP format
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Get the actual model from DataParallel if needed
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Make sure text is 3D: [batch, num_captions, seq_len]
        if texts.dim() == 2:
            # If only 2D, add a caption dimension
            texts = texts.unsqueeze(1)
        
        batch_size = texts.size(0)
        num_captions = texts.size(1)
        
        # Get text features for all captions
        all_texts = texts.view(-1, texts.size(-1))  # [batch*num_captions, seq_len]
        text_features = actual_model.clip_model.encode_text(all_texts)  # [batch*num_captions, dim]
        text_features = text_features.view(batch_size, num_captions, -1)  # [batch, num_captions, dim]
        
        # Create valid mask (assuming all captions are valid in OpenCLIP format)
        valid_mask = torch.ones(batch_size, num_captions, device=device)
        valid_captions = num_captions
    
    # Score captions using the caption scorer
    text_features_flat = text_features.view(-1, text_features.size(-1))
    caption_scores = actual_model.caption_scorer(text_features_flat).squeeze()
    caption_scores = caption_scores.view(batch_size, valid_captions)
    
    # Apply valid mask
    caption_scores = caption_scores * valid_mask[:, :valid_captions]
    
    # Calculate mean score for each image, only considering valid captions
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    mean_scores = (caption_scores * valid_mask[:, :valid_captions]).sum(dim=1) / valid_counts
    
    # Calculate relevance loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    relevance_loss = criterion(mean_scores, labels)
    
    return relevance_loss

def calculate_loss_gs(model, batch, device, loss_config, temp=0.1, hard=True, print_every=100, is_training=True, temperature=0.07):
    """Calculate combined losses based on configuration."""
    # Initialize call_count if not exists
    if is_training and not hasattr(calculate_loss_gs, 'call_count'):
        calculate_loss_gs.call_count = 0
    
    # Get active losses from config
    use_classification = loss_config.get('classification', True)  # Classification is enabled by default
    use_contrastive = loss_config.get('contrastive', False)
    use_relevance = loss_config.get('relevance', False)
    
    # Clear cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Handle different batch formats (CLIP vs OpenCLIP vs SigLIP2)
    if 'pixel_values' in batch:
        # CLIP/SigLIP2 format
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Get model (handle DataParallel)
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Check if it's SigLIP2 or CLIP model and get image features accordingly
        if hasattr(actual_model, 'siglip_model'):
            # SigLIP2 model
            image_outputs = actual_model.siglip_model.vision_model(pixel_values=pixel_values)
            image_embeds = actual_model.siglip_model.get_image_features(pixel_values=pixel_values)
            image_features = image_outputs.pooler_output
        else:
            # CLIP model
            image_outputs = actual_model.clip_model.vision_model(pixel_values=pixel_values)
            image_embeds = actual_model.clip_model.get_image_features(pixel_values=pixel_values)
            image_features = image_outputs.pooler_output
        
        batch_size = pixel_values.size(0)
        max_captions = input_ids.size(1)
        
        # Process each caption separately
        text_features_list = []  # For classification
        text_embeds_list = []    # For contrastive learning
        valid_caption_mask = []  # To track which captions are valid (not padding)
        
        for i in range(max_captions):
            # Check if this caption position has any valid captions
            is_valid = attention_mask[:, i].sum(dim=1) > 0
            valid_caption_mask.append(is_valid)
            
            # Only process if at least one example in the batch has a valid caption
            if is_valid.any():
                # Get text outputs for classification
                if hasattr(actual_model, 'siglip_model'):
                    # SigLIP2 model
                    text_outputs = actual_model.siglip_model.text_model(
                        input_ids=input_ids[:, i],
                        attention_mask=attention_mask[:, i]
                    )
                    text_embeds = actual_model.siglip_model.get_text_features(
                        input_ids=input_ids[:, i],
                        attention_mask=attention_mask[:, i]
                    )
                else:
                    # CLIP model
                    text_outputs = actual_model.clip_model.text_model(
                        input_ids=input_ids[:, i],
                        attention_mask=attention_mask[:, i]
                    )
                    text_embeds = actual_model.clip_model.get_text_features(
                        input_ids=input_ids[:, i],
                        attention_mask=attention_mask[:, i]
                    )
                
                text_features = text_outputs.pooler_output
                text_features_list.append(text_features)
                text_embeds_list.append(text_embeds)
        
        # Handle case where we might have fewer than max_captions valid caption positions
        valid_captions = len(text_features_list)
        if valid_captions == 0:
            # This shouldn't happen, but just in case
            print("Warning: No valid captions found in batch")
            return torch.tensor(0.0, requires_grad=True, device=device) if is_training else torch.zeros(batch_size, device=device)
        
        text_features = torch.stack(text_features_list, dim=1)  # [batch, valid_captions, dim]
        text_embeds = torch.stack(text_embeds_list, dim=1)     # [batch, valid_captions, dim]
        
        # Create a mask tensor for valid captions
        valid_mask = torch.stack([mask for mask in valid_caption_mask if mask.any()], dim=1).float()
        
        # Score captions using caption scorer if needed for selection
        if use_relevance or (is_training and not use_contrastive):
            caption_scores = actual_model.caption_scorer(text_features.view(-1, text_features.size(-1))).squeeze()
            caption_scores = caption_scores.view(batch_size, valid_captions)
            caption_scores = caption_scores * valid_mask  # Apply valid_mask
        else:
            # Use cosine similarity for caption selection
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds_flat = text_embeds.view(-1, text_embeds.size(-1))
            text_embeds_norm = F.normalize(text_embeds_flat, p=2, dim=-1)
            text_embeds_norm = text_embeds_norm.view(batch_size, valid_captions, -1)
            
            # Calculate similarities [batch, valid_captions]
            similarities = torch.bmm(
                image_embeds_norm.unsqueeze(1),
                text_embeds_norm.transpose(1, 2)
            ).squeeze(1)
            caption_scores = similarities * valid_mask
    else:
        # OpenCLIP format
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Get model (handle DataParallel)
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Get image features
        image_features = actual_model.clip_model.encode_image(images)
        image_embeds = image_features  # For OpenCLIP, embed and features are the same
        
        # Make sure text is 3D: [batch, num_captions, seq_len]
        if texts.dim() == 2:
            # If only 2D, add a caption dimension
            texts = texts.unsqueeze(1)
            
        batch_size = images.size(0)
        num_captions = texts.size(1)
        
        # Get text features for all captions
        text_features_list = []
        text_embeds_list = []
        valid_mask = torch.ones(batch_size, num_captions, device=device)
        
        # Process text in chunks to avoid memory issues
        for i in range(num_captions):
            if 'text_mask' in batch:
                is_valid = batch['text_mask'][:, i].bool()
                valid_mask[:, i] = is_valid.float()
                if not is_valid.any():
                    continue
            
            # Clear cache before each text encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get text features for this caption position with memory optimization
            with torch.amp.autocast(device_type=device.type, enabled=False):  # Disable autocast for text encoding
                text_feats = actual_model.clip_model.encode_text(texts[:, i])
            text_features_list.append(text_feats)
            text_embeds_list.append(text_feats)  # For OpenCLIP, embed and features are the same
            
        # Stack features
        if not text_features_list:
            # This shouldn't happen, but just in case
            print("Warning: No valid captions found in batch")
            return torch.tensor(0.0, requires_grad=True, device=device) if is_training else torch.zeros(batch_size, device=device)
            
        text_features = torch.stack(text_features_list, dim=1)  # [batch, valid_captions, dim]
        text_embeds = torch.stack(text_embeds_list, dim=1)  # [batch, valid_captions, dim]
        valid_captions = text_features.size(1)
        
        # Score captions
        if use_relevance:
            # Process caption scoring in smaller chunks
            caption_scores_list = []
            chunk_size = min(batch_size, 32)  # Process in smaller chunks
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_features = text_features[i:end_idx].view(-1, text_features.size(-1))
                chunk_scores = actual_model.caption_scorer(chunk_features).squeeze()
                chunk_scores = chunk_scores.view(end_idx - i, valid_captions)
                caption_scores_list.append(chunk_scores)
            
            caption_scores = torch.cat(caption_scores_list, dim=0)
            caption_scores = caption_scores * valid_mask[:, :valid_captions]
        else:
            # Use cosine similarity for caption selection
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds_flat = text_embeds.view(-1, text_embeds.size(-1))
            text_embeds_norm = F.normalize(text_embeds_flat, p=2, dim=-1)
            text_embeds_norm = text_embeds_norm.view(batch_size, valid_captions, -1)
            
            # Calculate similarities [batch, valid_captions]
            similarities = torch.bmm(
                image_embeds_norm.unsqueeze(1),
                text_embeds_norm.transpose(1, 2)
            ).squeeze(1)
            caption_scores = similarities * valid_mask[:, :valid_captions]
    
    # Caption selection with Gumbel-Softmax for training or Softmax for inference
    if is_training:
        selected_mask = F.gumbel_softmax(caption_scores, tau=temp, hard=hard)
    else:
        selected_mask = F.softmax(caption_scores, dim=-1)
        
    # Apply valid_mask again to ensure we don't select invalid captions
    selected_mask = selected_mask * valid_mask[:, :valid_captions]
    
    # Normalize the mask
    row_sums = selected_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    selected_mask = selected_mask / row_sums
    
    # Select features based on the mask
    selected_text_features = torch.bmm(
        selected_mask.unsqueeze(1),  # [batch, 1, valid_captions]
        text_features  # [batch, valid_captions, dim]
    ).squeeze(1)  # [batch, dim]
    
    selected_text_embeds = torch.bmm(
        selected_mask.unsqueeze(1), 
        text_embeds
    ).squeeze(1)
    
    # Clear intermediate tensors to save memory
    del text_features_list, text_embeds_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Debug printing with improved caption selection details
    if is_training and hasattr(calculate_loss_gs, 'call_count'):
        if calculate_loss_gs.call_count % print_every == 0:
            # Get raw probabilities before applying softmax/gumbel_softmax
            raw_probs = F.softmax(caption_scores, dim=-1)
            
            # Extract the necessary data for debug printing
            if 'pixel_values' in batch:
                image_ids = batch['image_ids']
                labels_tensor = batch['labels']
            else:
                image_ids = batch['image_ids']
                labels_tensor = batch['label']
            
            # Get dataset to access captions
            if hasattr(model, 'dataset'):
                dataset = model.dataset
            elif hasattr(model, 'module') and hasattr(model.module, 'dataset'):
                dataset = model.module.dataset
            else:
                # Try to access global dataset if defined
                try:
                    from __main__ import dataset
                except ImportError:
                    dataset = None
            
            # Call our debug printing function if we have access to the captions
            if dataset is not None and hasattr(dataset, 'captions'):
                print_caption_selection_debug(
                    image_ids=image_ids,
                    captions=dataset.captions,
                    caption_scores=torch.sigmoid(caption_scores),
                    selected_mask=selected_mask,
                    labels=labels_tensor,
                    raw_probs=raw_probs
                )
            else:
                # Fallback to simpler debug output if we can't access captions
                print("\n=== Caption Selection Debug (Simple) ===")
                for i in range(min(3, len(image_ids))):
                    print(f"Example {i}:")
                    print(f"- Image ID: {image_ids[i]}")
                    print(f"- Valid captions: {valid_mask[i].sum().item()}")
                    print(f"- Selected mask: {selected_mask[i].detach().cpu().numpy()}")
                    print(f"- Scores: {torch.sigmoid(caption_scores[i]).detach().cpu().numpy()}")
                    if labels_tensor is not None:
                        print(f"- Label: {labels_tensor[i].item()}")
                    print("")
    
    # Initialize total loss
    total_loss = 0
    
    # Calculate classification loss if enabled
    if use_classification:
        # Get logits using actual model's combine features and classifier
        if 'pixel_values' in batch:
            # CLIP format
            combined = actual_model.combine_features(image_features, selected_text_features)
            combined = actual_model.pre_output(combined)
            logits = actual_model.classifier(combined).squeeze(-1)
        else:
            # OpenCLIP format
            combined = actual_model.combine_features(image_features, selected_text_features)
            combined = actual_model.pre_output(combined)
            logits = actual_model.classifier(combined).squeeze(-1)
        
        # Return logits directly for inference
        if not is_training:
            return logits
        
        # Classification loss - ensure labels are on the correct device
        labels = labels.to(device)  # Explicitly move labels to the device
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = criterion(logits, labels)
        
        # Add to total loss - keep using the learnable weight for classification
        precision_cls = torch.exp(-actual_model.log_vars[0])
        total_loss += precision_cls * cls_loss + 0.5 * actual_model.log_vars[0]
    
    # Calculate contrastive loss if enabled
    if use_contrastive and is_training:
        # Check if it's SigLIP2 or CLIP model for different loss calculations
        if hasattr(actual_model, 'siglip_model'):
            # SigLIP2 model - use sigmoid loss
            # Normalize embeddings
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds_norm = F.normalize(selected_text_embeds, p=2, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(image_embeds_norm, text_embeds_norm.t()) / temperature
            
            # SigLIP uses sigmoid loss - positive pairs should have high similarity
            labels = torch.eye(batch_size, device=device)  # Identity matrix for positive pairs
            # Use binary_cross_entropy_with_logits instead of applying sigmoid + binary_cross_entropy
            contrastive_loss = F.binary_cross_entropy_with_logits(similarity, labels, reduction='mean')
        else:
            # CLIP model - use traditional contrastive loss
            # Normalize embeddings
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds_norm = F.normalize(selected_text_embeds, p=2, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(image_embeds_norm, text_embeds_norm.t()) / temperature
            contrastive_labels = torch.arange(batch_size, device=device)
            
            # Compute contrastive loss in both directions
            loss_i2t = F.cross_entropy(similarity, contrastive_labels)
            loss_t2i = F.cross_entropy(similarity.t(), contrastive_labels)
            contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        # Add to total loss using learnable uncertainty weight
        precision_cont = torch.exp(-actual_model.log_vars[2])
        total_loss += precision_cont * contrastive_loss + 0.5 * actual_model.log_vars[2]
    
    # Calculate relevance loss if enabled
    if use_relevance and is_training:
        # Apply valid_mask to get mean caption scores
        # Ensure we're only using the valid captions dimension
        valid_mask_subset = valid_mask[:, :valid_captions]
        caption_score_mean = (caption_scores * valid_mask_subset).sum(dim=1) / valid_mask_subset.sum(dim=1).clamp(min=1)
        
        # Relevance loss - ensure labels are on the correct device and have correct shape
        # Make sure labels is 1D and matches caption_score_mean shape
        if 'pixel_values' in batch:
            labels_for_relevance = batch['labels'].to(device).squeeze()  # Ensure 1D
        else:
            labels_for_relevance = batch['label'].to(device).squeeze()   # Ensure 1D
            
        # Verify shapes match
        if caption_score_mean.shape != labels_for_relevance.shape:
            print(f"Warning: Shape mismatch in relevance loss - caption_score_mean: {caption_score_mean.shape}, labels: {labels_for_relevance.shape}")
            # Take only the batch dimension if labels somehow got duplicated
            if labels_for_relevance.numel() > caption_score_mean.numel():
                labels_for_relevance = labels_for_relevance[:caption_score_mean.size(0)]
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        relevance_loss = criterion(caption_score_mean, labels_for_relevance.float())
        
        # Add to total loss using learnable uncertainty weight
        precision_rel = torch.exp(-actual_model.log_vars[1])
        total_loss += precision_rel * relevance_loss + 0.5 * actual_model.log_vars[1]
    
    # Print debugging information
    if is_training and hasattr(calculate_loss_gs, 'call_count'):
        if calculate_loss_gs.call_count % print_every == 0:
            print("\n=== Loss Components ===")
            if use_classification:
                cls_weight = torch.exp(-actual_model.log_vars[0])
                print(f"Classification loss: {cls_loss.item():.4f} (learnable weight: {cls_weight.item():.4f})")
            if use_contrastive:
                cont_learnable_weight = torch.exp(-actual_model.log_vars[2])
                print(f"Contrastive loss: {contrastive_loss.item():.4f} (learnable weight: {cont_learnable_weight.item():.4f})")
            if use_relevance:
                rel_learnable_weight = torch.exp(-actual_model.log_vars[1])
                print(f"Relevance loss: {relevance_loss.item():.4f} (learnable weight: {rel_learnable_weight.item():.4f})")
            print(f"Total loss: {total_loss.item():.4f}")
    
    if is_training:
        calculate_loss_gs.call_count += 1
        
    return total_loss if is_training else logits