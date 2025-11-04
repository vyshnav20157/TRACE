import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

def collate_fn_generic(batch):
    """Generic collate function that can handle different dataset formats"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Check if this is the CLIP processor format or OpenCLIP format
    if 'pixel_values' in batch[0]:
        # CLIP processor format
        return collate_fn_clip(batch)
    else:
        # OpenCLIP format
        return collate_fn_open_clip(batch)

def collate_fn_clip(batch):
    """Collate function for HuggingFace CLIP processor format"""
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

def collate_fn_open_clip(batch):
    """Collate function for OpenCLIP format"""
    # Instead of stacking text directly, pad the tensors to same first dimension
    texts = [item['text'] for item in batch]
    max_sequences = max([text.size(0) for text in texts])
    
    # Pad each text tensor to have the same first dimension
    padded_texts = []
    text_masks = []
    for text in texts:
        num_sequences = text.size(0)
        if num_sequences < max_sequences:
            # Create padding tensor with same second dimension (77 for CLIP, 64 for SigLIP2)
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
        'label': torch.stack([item['label'] for item in batch])
    }

def select_best_captions_by_caption_scorer(model, dataset, device, batch_size=256):
    """
    Selects the best caption for each image based on caption scorer predictions.
    This is used when relevance loss is present.
    """
    model.eval()
    best_captions = {}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_generic)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Selecting best captions using caption scorer"):
            if batch is None:
                continue
            
            # Clear cache before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Get model (handle DataParallel)
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # Handle different batch formats (CLIP vs OpenCLIP)
            if 'pixel_values' in batch:
                # CLIP/SigLIP2 format
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)  # [batch, max_captions, seq_len]
                attention_mask = batch['attention_mask'].to(device)  # [batch, max_captions, seq_len]
                image_ids = batch['image_ids']
                
                batch_size = pixel_values.size(0)
                num_captions = input_ids.size(1)  # This is max_captions for the batch
                
                # Process each caption
                text_features_list = []
                
                for i in range(num_captions):
                    # Use attention mask to determine if this is a real caption or padding
                    is_valid_caption = attention_mask[:, i].sum(dim=1) > 0
                    
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
            else:
                # OpenCLIP format
                images = batch['image'].to(device)
                texts = batch['text'].to(device)  # [batch, num_captions, seq_len]
                image_ids = batch['image_ids']
                
                batch_size = images.size(0)
                num_captions = texts.size(1)
                feature_dim = actual_model.clip_model.text.output_dim
                
                # Process all captions
                text_features_list = []
                for i in range(num_captions):
                    # Check if this is valid or padding
                    if 'text_mask' in batch:
                        is_valid = batch['text_mask'][:, i].bool()
                        if not is_valid.any():
                            continue
                    
                    # Use autocast for memory efficiency
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        text_features = actual_model.clip_model.encode_text(texts[:, i])
                    text_features_list.append(text_features)
                    
                    # Clear cache after each caption
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if not text_features_list:  # In case all captions were padding
                continue
                
            text_features = torch.stack(text_features_list, dim=1)  # [batch, valid_captions, dim]
            valid_captions = text_features.size(1)
            
            # Score captions using caption scorer with chunking
            chunk_size = min(batch_size, 16)  # Process in smaller chunks
            caption_scores_list = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_features = text_features[i:end_idx].view(-1, text_features.size(-1))
                chunk_scores = actual_model.caption_scorer(chunk_features).squeeze()
                chunk_scores = torch.sigmoid(chunk_scores).view(end_idx - i, valid_captions)
                caption_scores_list.append(chunk_scores)
            
            caption_scores = torch.cat(caption_scores_list, dim=0)
            
            # Select best caption indices based on scores
            best_caption_idx = caption_scores.argmax(dim=1)
            
            # Store best captions
            for idx, image_id in enumerate(image_ids):
                caption_idx = best_caption_idx[idx].item()
                valid_captions_for_image = min(len(dataset.captions[image_id]), valid_captions)
                if caption_idx < valid_captions_for_image:
                    best_captions[image_id] = dataset.captions[image_id][caption_idx]
                else:
                    # Default to first caption if selected index is out of bounds
                    best_captions[image_id] = dataset.captions[image_id][0]
                    
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return best_captions

def select_best_captions_by_cosine_similarity(model, dataset, device, batch_size=512):
    """
    Selects the best caption for each image based on cosine similarity between image and text features.
    This is used when contrastive loss is present but relevance loss is not.
    """
    model.eval()
    best_captions = {}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_generic)
    
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for batch in tqdm(dataloader, desc="Selecting best captions using cosine similarity"):
            if batch is None:
                continue
                
            # Get model (handle DataParallel)
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # Handle different batch formats (CLIP vs OpenCLIP vs SigLIP2)
            if 'pixel_values' in batch:
                # CLIP/SigLIP2 format
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)  # [batch, max_captions, seq_len]
                attention_mask = batch['attention_mask'].to(device)  # [batch, max_captions, seq_len]
                image_ids = batch['image_ids']
                
                batch_size = pixel_values.size(0)
                num_captions = input_ids.size(1)  # This is max_captions for the batch
                
                # Get image embeddings - check if SigLIP2 or CLIP
                if hasattr(actual_model, 'siglip_model'):
                    # SigLIP2 model
                    image_embeddings = actual_model.siglip_model.get_image_features(pixel_values=pixel_values)
                else:
                    # CLIP model
                    image_embeddings = actual_model.clip_model.get_image_features(pixel_values=pixel_values)
                
                # Process each caption
                text_embeddings_list = []
                
                for i in range(num_captions):
                    # Use attention mask to determine if this is a real caption or padding
                    is_valid_caption = attention_mask[:, i].sum(dim=1) > 0
                    
                    if not is_valid_caption.any():
                        continue  # Skip if this caption position is all padding
                    
                    # Get text embeddings - check if SigLIP2 or CLIP
                    if hasattr(actual_model, 'siglip_model'):
                        # SigLIP2 model
                        text_embeddings = actual_model.siglip_model.get_text_features(
                            input_ids=input_ids[:, i],
                            attention_mask=attention_mask[:, i]
                        )
                    else:
                        # CLIP model
                        text_embeddings = actual_model.clip_model.get_text_features(
                            input_ids=input_ids[:, i],
                            attention_mask=attention_mask[:, i]
                        )
                    text_embeddings_list.append(text_embeddings)
            else:
                # OpenCLIP format
                images = batch['image'].to(device)
                texts = batch['text'].to(device)  # [batch, num_captions, seq_len]
                image_ids = batch['image_ids']
                
                batch_size = images.size(0)
                num_captions = texts.size(1)
                
                # Get image embeddings
                image_embeddings = actual_model.clip_model.encode_image(images)
                
                # Process all captions
                text_embeddings_list = []
                for i in range(num_captions):
                    # Check if this is valid or padding
                    if 'text_mask' in batch:
                        is_valid = batch['text_mask'][:, i].bool()
                        if not is_valid.any():
                            continue
                    
                    text_embeddings = actual_model.clip_model.encode_text(texts[:, i])
                    text_embeddings_list.append(text_embeddings)
            
            if not text_embeddings_list:  # In case all captions were padding
                continue
                
            text_embeddings = torch.stack(text_embeddings_list, dim=1)  # [batch, valid_captions, dim]
            valid_captions = text_embeddings.size(1)
            
            # Normalize embeddings for cosine similarity
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings_flat = text_embeddings.view(-1, text_embeddings.size(-1))
            text_embeddings_norm = F.normalize(text_embeddings_flat, p=2, dim=-1)
            text_embeddings_norm = text_embeddings_norm.view(batch_size, valid_captions, -1)
            
            # Calculate cosine similarity between each image and its captions
            similarities = torch.bmm(
                image_embeddings.unsqueeze(1),  # [batch, 1, dim]
                text_embeddings_norm.transpose(1, 2)  # [batch, dim, valid_captions]
            ).squeeze(1)  # [batch, valid_captions]
            
            # Select best caption indices based on similarity
            best_caption_idx = similarities.argmax(dim=1)
            
            # Store best captions
            for idx, image_id in enumerate(image_ids):
                caption_idx = best_caption_idx[idx].item()
                valid_captions_for_image = min(len(dataset.captions[image_id]), valid_captions)
                if caption_idx < valid_captions_for_image:
                    best_captions[image_id] = dataset.captions[image_id][caption_idx]
                else:
                    # Default to first caption if selected index is out of bounds
                    best_captions[image_id] = dataset.captions[image_id][0]
                    
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return best_captions

def print_caption_selection_debug(image_ids, captions, caption_scores, selected_mask, labels=None, raw_probs=None):
    """
    Print detailed debug information about caption selection.
    
    Args:
        image_ids: List of image IDs
        captions: Dict mapping image IDs to list of captions
        caption_scores: Tensor with caption scores [batch, num_captions]
        selected_mask: Tensor with selected mask [batch, num_captions]
        labels: Optional tensor with labels [batch]
        raw_probs: Optional tensor with raw probabilities [batch, num_captions]
    """
    batch_size = min(3, len(image_ids))  # Show at most 3 examples
    
    print("\n=== Caption Selection Debug ===")
    
    for i in range(batch_size):
        image_id = image_ids[i]
        print(f"\nImage {image_id}:")
        
        if labels is not None:
            print(f"Label: {labels[i].item():.1f}")
        
        print("\nCaption Scores:")
        
        # Get available captions for this image
        image_captions = captions[image_id]
        caption_count = len(image_captions)
        
        for j in range(caption_count):
            if j < len(caption_scores[i]):
                score = caption_scores[i][j].item()
                prob_gumbel = selected_mask[i][j].item() if selected_mask is not None else None
                prob_raw = raw_probs[i][j].item() if raw_probs is not None else None
                
                caption_type = "Text" if j == 0 else "InternVL Caption" if j == 1 else "Gemini Caption"
                
                print(f"\n{j+1}. {caption_type}: {image_captions[j]}")
                
                probs_str = f"Score: {score:.3f}"
                if prob_raw is not None:
                    probs_str += f" | Raw Prob: {prob_raw:.3f}"
                if prob_gumbel is not None:
                    probs_str += f" | Gumbel Prob: {prob_gumbel:.3f}"
                    
                print(f"   - {probs_str}")
                
        # Print selected caption index
        if selected_mask is not None:
            selected_idx = selected_mask[i].argmax().item()
            print(f"\nSelected Caption Index: {selected_idx + 1}")
            
        print("-" * 50)

def select_best_captions(model, dataset, device, loss_config, batch_size=512, verbose=False):
    """
    Select best captions based on the loss configuration.
    
    Args:
        model: The model to use for selection
        dataset: Dataset containing images and captions
        device: Device to use for computation
        loss_config: Dict with keys 'classification', 'contrastive', 'relevance' indicating which losses are active
        batch_size: Batch size for processing
        verbose: Whether to print debug information
    """
    has_relevance = loss_config.get('relevance', False)
    has_contrastive = loss_config.get('contrastive', False)
    
    if has_relevance:
        # If relevance loss is enabled, use caption scorer for selection
        return select_best_captions_by_caption_scorer(model, dataset, device, batch_size)
    elif has_contrastive:
        # If only contrastive loss is enabled, use cosine similarity for selection
        return select_best_captions_by_cosine_similarity(model, dataset, device, batch_size)
    else:
        # If only classification loss is used, just use the caption scorer as default
        return select_best_captions_by_caption_scorer(model, dataset, device, batch_size)

# Explicitly make functions available for import
__all__ = [
    'select_best_captions', 
    'print_caption_selection_debug', 
    'collate_fn_generic'
]
