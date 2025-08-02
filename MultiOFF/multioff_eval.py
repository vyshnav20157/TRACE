import os
import time
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import io
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Dataset paths
folder = "/backup/girish_datasets/MultiOFF/Labelled_Images/"
input_file = "/backup/girish_datasets/MultiOFF/Testing_meme_dataset.csv"

class MemeDatasetCSV(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.to_dict(orient='records')
        self.processor = processor
        self.images = {}
        
        for row in tqdm(self.data, desc="Loading images"):
            image_id = row['image_name']
            image_path = f'/backup/girish_datasets/MultiOFF/Labelled_Images/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image
            else:
                print(f"Image file {image_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_id = row['image_name']
        caption = str(row['sentence'])

        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return self.__getitem__((idx + 1) % len(self.data))
        
        image = self.images[image_id]
        
        # Process image
        image_processor_output = self.processor.image_processor(
            image, 
            return_tensors="pt"
        )
        pixel_values = image_processor_output.pixel_values.squeeze(0)  # Remove batch dim
        
        # Process text with truncation
        text_processor_output = self.processor.tokenizer(
            caption,
            padding='max_length',
            max_length=77,  # CLIP's max length
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_processor_output.input_ids.squeeze(0)  # Remove batch dim
        attention_mask = text_processor_output.attention_mask.squeeze(0)  # Remove batch dim
        
        inputs = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Convert label to binary (0 for "Non-offensiv", 1 for "offensive")
        label = 1 if str(row['label']) == "offensive" else 0
        label = torch.tensor(label, dtype=torch.float)
        return inputs, label

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

test_data = pd.read_csv(input_file)
test_dataset = MemeDatasetCSV(test_data, clip_processor)
model_path = "checkpoints/best_model.pth"

model = SimpleCLIPClassifier()
checkpoint = torch.load(model_path, weights_only=True)  # Add weights_only=True to avoid pickle warning
state_dict = checkpoint['model_state_dict']

# Remove 'module.' prefix from state dict keys
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Set to eval mode

predictions = []
true_labels = []

# Perform evaluation using the test dataset and 
# calculate auroc, accuracy, precision, recall, f1 score

with torch.no_grad():
    for inputs, label in test_dataset:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        prob = F.sigmoid(logits).squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(float)

        predictions.append(pred)
        true_labels.append(label.item())

print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
print(f"AUROC: {roc_auc_score(true_labels, predictions):.4f}")
print(f"Precision: {precision_score(true_labels, predictions):.4f}")
print(f"Recall: {recall_score(true_labels, predictions):.4f}")
print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")
