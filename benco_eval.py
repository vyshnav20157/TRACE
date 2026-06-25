import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import torch.nn as nn
import open_clip

from clip_encoders_finetune import CLIPClassifier as CLIPClassifierCLIP
from clip_encoders_finetune import MemeDatasetCSV as MemeDatasetCSV_CLIP

from clip_xlm_roberta_ft import CLIPClassifier as CLIPClassifierXLM
from clip_xlm_roberta_ft import MemeDatasetCSV as MemeDatasetCSV_XLM

class MemeDatasetCSV(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.to_dict(orient='records')
        self.processor = processor
        self.images = {}
        self.captions = defaultdict(list)
        
        for row in tqdm(self.data, desc="Loading images"):
            image_id = row['img']
            image_path = f'/backup/girish_datasets/Hateful_Memes_Extended/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image
                
                # Store all available captions
                self.captions[image_id].extend([
                    str(row['text']),
                    str(row['ivl_8b_new_caption']),
                    str(row['gemini_caption']),
                ])
            else:
                print(f"Image file {image_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_id = row['img']

        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return self.__getitem__((idx + 1) % len(self.data))
        
        image = self.images[image_id]
        caption = self.best_captions[image_id] if hasattr(self, 'best_captions') else self.captions[image_id][0]
        
        # Process image
        image = self.processor(image)
        
        # Process text
        text_tokens = self.processor.tokenizer([caption])
        
        return {
            'image': image,
            'text': text_tokens,
            'label': torch.tensor(row['label'], dtype=torch.float),
            'image_id': image_id
        }

class SimpleCLIPClassifier(nn.Module):
    def __init__(self, projection_dim=1024, num_classes=1, fusion_type='product'):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.fusion_type = fusion_type
        
        # Separate projection layers for image and text
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_model.config.vision_config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.clip_model.config.text_config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Residual connections
        self.image_residual = nn.Linear(self.clip_model.config.vision_config.hidden_size, projection_dim)
        self.text_residual = nn.Linear(self.clip_model.config.text_config.hidden_size, projection_dim)
        
        pre_output_layers = [nn.Dropout(0.2)]
        pre_output_input_dim = projection_dim
        for _ in range(3):
            pre_output_layers.extend([
                nn.Linear(pre_output_input_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            pre_output_input_dim = projection_dim

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, num_classes)
        )
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Add batch dimension if not present
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # Get original pooler outputs (for residual connections)
        orig_img_pooler = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        orig_text_pooler = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # Get updated embeddings from CLIP encoders
        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # Get updated pooler outputs
        image_pooler = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        text_pooler = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # Project features through separate pathways
        image_proj = self.image_projection(image_pooler)
        text_proj = self.text_projection(text_pooler)
        
        # Add residual connections
        image_resid = self.image_residual(orig_img_pooler)
        text_resid = self.text_residual(orig_text_pooler)
        
        image_features = image_proj + image_resid
        text_features = text_proj + text_resid

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Product fusion
        combined = torch.mul(image_features, text_features)
        combined = self.pre_output(combined)
        
        # Classification
        logits = self.classifier(combined)
        logits = logits.squeeze(1)
        
        return logits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model_name = "xlm-roberta-large-ViT-H-14"
pretrained = "frozen_laion5b_s13b_b90k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
hf_tokenizer = tokenizer.tokenizer 

# Load data
data_path = "/backup/girish_datasets/Hateful_Memes_Extended/ivl_plus_gemini_captions.csv"
data = pd.read_csv(data_path)

test_seen_data = data[data['split'] == 'test_seen']
test_dataset = MemeDatasetCSV_XLM(test_seen_data, preprocess, tokenizer)

model_path = "checkpoints/clip_roberta_large_txt_4.pth"

# Load the finetuned model
model = CLIPClassifierXLM(model, print_params=False)
checkpoint = torch.load(model_path, weights_only=True)
state_dict = checkpoint['model_state_dict']

# Remove 'module.' prefix from state dict keys if present
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load state dict and move to device
model.load_state_dict(state_dict)
model = model.to(device)

# Wrap in DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.eval()  # Set to eval mode

# Choose among the samples having similar pseudo_text_idx or pseudo_img_idx
similar_samples = []
pseudo_text_groups = defaultdict(list)
pseudo_img_groups = defaultdict(list)

# Create a mapping of image paths to dataset indices
img_to_idx = {row['img']: idx for idx, row in enumerate(test_dataset.data)}

# Group samples by pseudo indices
for idx, row in test_seen_data.iterrows():
    pseudo_text_groups[row['pseudo_text_idx']].append(row)
    pseudo_img_groups[row['pseudo_img_idx']].append(row)

# Add samples that share pseudo indices
for group in pseudo_text_groups.values():
    if len(group) > 1:  # More than one sample shares this pseudo_text_idx
        similar_samples.extend(group)
        
for group in pseudo_img_groups.values():
    if len(group) > 1:  # More than one sample shares this pseudo_img_idx
        similar_samples.extend(group)

# Remove duplicates while preserving order
similar_samples = list({str(row.to_dict()): row for row in similar_samples}.values())

print(f"Number of similar samples: {len(similar_samples)}")

# Create a new dataframe with the similar samples and reset the index
similar_samples_df = pd.DataFrame(similar_samples).reset_index(drop=True)

# Collect the predictions and labels
predictions = []
labels = []
probabilities = []  # Add list for storing raw probabilities
caption_scores = {}  # Store caption scores for analysis

with torch.no_grad():
    for idx, row in tqdm(similar_samples_df.iterrows(), desc="Evaluating"):
        # Get the dataset index using the image path
        dataset_idx = img_to_idx[row['img']]
        batch_item = test_dataset[dataset_idx]
        
        # Move inputs to device and ensure correct shapes
        image = batch_item['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
        text = batch_item['text'].to(device)  # [num_captions, seq_len]
        
        # Get image features
        image_features = model.module.clip_model.encode_image(image) if isinstance(model, nn.DataParallel) else model.clip_model.encode_image(image)
        
        # Get text features for all captions
        text_features = model.module.clip_model.encode_text(text) if isinstance(model, nn.DataParallel) else model.clip_model.encode_text(text)
        
        # Score captions using the caption scorer
        scores = model.module.caption_scorer(text_features).squeeze() if isinstance(model, nn.DataParallel) else model.caption_scorer(text_features).squeeze()
        caption_probs = F.softmax(scores, dim=0)
        
        # Select best caption
        best_idx = caption_probs.argmax().item()
        best_text = text[best_idx].unsqueeze(0)  # Add batch dimension back
        
        # Store caption scores for analysis
        caption_scores[row['img']] = {
            'scores': scores.cpu().numpy(),
            'probs': caption_probs.cpu().numpy(),
            'best_idx': best_idx
        }
        
        inputs = {
            'images': image,
            'texts': best_text
        }
        label = batch_item['label']
        
        try:
            # Get model output
            logits = model(**inputs)
            
            # Get raw probability
            prob = F.sigmoid(logits).squeeze().cpu().numpy()
            probabilities.append(prob)  # Store raw probability
            labels.append(label.item())
            
        except RuntimeError as e:
            print(f"Error processing sample {idx}:")
            print(f"Image shape: {image.shape}")
            print(f"Text shape: {best_text.shape}")
            raise e

# Convert lists to numpy arrays for metric calculation
labels = np.array(labels)
probabilities = np.array(probabilities)

# Calculate optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(labels, probabilities)
f1_scores = 2 * precision * recall / (precision + recall + 1e-7)  # Add small epsilon to prevent division by zero
threshold = thresholds[np.argmax(f1_scores)]

# Get binary predictions using optimal threshold
predictions = (probabilities > threshold).astype(float)

# Calculate metrics
auroc = roc_auc_score(labels, probabilities)  # Use raw probabilities for AUROC
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"\nOptimal threshold: {threshold:.4f}")
print(f"AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Print detailed analysis of caption selection
# print("\nDetailed Caption Selection Analysis:")
# for idx, row in similar_samples_df.iterrows():
#     image_id = row['img']
#     scores = caption_scores[image_id]
#     captions = test_dataset.captions[image_id]
    
#     print(f"\n{'='*80}")
#     print(f"Image {idx+1}: {image_id}")
#     print(f"True Label: {'Hateful' if labels[idx] == 1 else 'Non-hateful'}")
#     print(f"Prediction: {'Hateful' if predictions[idx] == 1 else 'Non-hateful'} (Probability: {probabilities[idx]:.4f})")
    
#     print("\nAll Captions:")
#     for j, (caption, score, prob) in enumerate(zip(captions, scores['scores'], scores['probs'])):
#         print(f"\nCaption {j+1}:")
#         print(f"Text: {caption}")
#         print(f"Score: {score:.4f}")
#         print(f"Probability: {prob:.4f}")
#         if j == scores['best_idx']:
#             print("*** Selected as Best Caption ***")
    
#     print('-' * 80)
