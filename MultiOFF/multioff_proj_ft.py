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
from transformers import AutoModel, CLIPImageProcessor
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import random
import torchvision.transforms as transforms

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

class MultiOFFDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.to_dict(orient='records')
        self.processor = processor
        self.images = {}
        
        for row in tqdm(self.data, desc="Loading images"):
            image_id = row['image_name']
            image_path = f'/backup/girish_datasets/MultiOFF/Labelled Images/{image_id}'
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

        if not isinstance(caption, str):
            caption = str(caption)
        
        inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Convert label to binary (0 for "Non-offensiv", 1 for "offensive")
        label = 1 if str(row['label']) == "offensive" else 0
        label = torch.tensor(label, dtype=torch.float)
        return inputs, label

class CLIPClassifier(nn.Module):
    def __init__(self, model_path="llava_clip_ohd_caps_lora/clip", map_dim=1024, num_pre_output_layers=3, 
                 fusion_type='align', drop_probs=[0.2, 0.3, 0.2], weight_image_loss=0.5, weight_text_loss=0.5):
        super(CLIPClassifier, self).__init__()
        self.clip_model = AutoModel.from_pretrained(model_path)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.map_dim = map_dim
        self.num_pre_output_layers = num_pre_output_layers
        self.fusion_type = fusion_type
        self.weight_image_loss = weight_image_loss
        self.weight_text_loss = weight_text_loss

        self.image_map = nn.Sequential(
            nn.Linear(self.clip_model.config.vision_config.hidden_size, self.map_dim),
            nn.Dropout(p=drop_probs[0])
        )
        self.text_map = nn.Sequential(
            nn.Linear(self.clip_model.config.text_config.hidden_size, self.map_dim),
            nn.Dropout(p=drop_probs[0])
        )

        if fusion_type == 'concat':
            pre_output_input_dim = self.map_dim * 2
        elif fusion_type == 'cross':
            pre_output_input_dim = self.map_dim ** 2
        else:
            pre_output_input_dim = self.map_dim

        pre_output_layers = [nn.Dropout(p=drop_probs[1])]
        for _ in range(self.num_pre_output_layers):
            pre_output_layers.extend([
                nn.Linear(pre_output_input_dim, self.map_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_probs[2])
            ])
            pre_output_input_dim = self.map_dim
        self.pre_output = nn.Sequential(*pre_output_layers)

        self.output = nn.Linear(self.map_dim, 1)

        if self.weight_image_loss > 0:
            self.pre_output_image = copy.deepcopy(self.pre_output)
            self.output_image = nn.Linear(self.map_dim, 1)

        if self.weight_text_loss > 0:
            self.pre_output_text = copy.deepcopy(self.pre_output)
            self.output_text = nn.Linear(self.map_dim, 1)

    def forward(self, input_ids, pixel_values, attention_mask=None):
        text_features = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output

        text_features = self.text_map(text_features)
        image_features = self.image_map(image_features)

        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        if self.fusion_type == 'align':
            features = torch.mul(image_features, text_features)
        elif self.fusion_type == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion_type == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))
            features = features.reshape(features.shape[0], -1)

        features = self.pre_output(features)
        logits = self.output(features).squeeze(1)

        output = {'logits': logits}

        if self.weight_image_loss > 0:
            image_features = self.pre_output_image(image_features)
            image_logits = self.output_image(image_features).squeeze(1)
            output['image_logits'] = image_logits

        if self.weight_text_loss > 0:
            text_features = self.pre_output_text(text_features)
            text_logits = self.output_text(text_features).squeeze(1)
            output['text_logits'] = text_logits

        return output

    def calculate_loss(self, outputs, labels):
        # loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
        loss = FocalLoss()(outputs['logits'], labels)
        
        if self.weight_image_loss > 0:
            # image_loss = F.binary_cross_entropy_with_logits(outputs['image_logits'], labels)
            image_loss = FocalLoss()(outputs['image_logits'], labels)
            loss += self.weight_image_loss * image_loss

        if self.weight_text_loss > 0:
            # text_loss = F.binary_cross_entropy_with_logits(outputs['text_logits'], labels)
            text_loss = FocalLoss()(outputs['text_logits'], labels)
            loss += self.weight_text_loss * text_loss

        return loss

    def calculate_metrics(self, outputs, labels, threshold=0.5):
        if isinstance(outputs, dict):
            preds = torch.sigmoid(outputs['logits'])
        else:
            preds = torch.sigmoid(outputs)
        
        preds_binary = (preds >= threshold).float()  # Convert to binary predictions
        
        accuracy = accuracy_score(labels.cpu(), preds_binary.cpu())
        precision = precision_score(labels.cpu(), preds_binary.cpu(), zero_division=0, average='macro')
        recall = recall_score(labels.cpu(), preds_binary.cpu(), zero_division=0, average='macro')
        f1 = f1_score(labels.cpu(), preds_binary.cpu(), zero_division=0, average='macro')
        auc = roc_auc_score(labels.cpu(), preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan')
        
        metrics = {
            'accuracy': format_metric(accuracy),
            'precision': format_metric(precision),
            'recall': format_metric(recall),
            'f1': format_metric(f1),
            'auc': format_metric(auc)
        }

        if isinstance(outputs, dict) and 'image_logits' in outputs:
            image_preds = torch.sigmoid(outputs['image_logits'])
            image_preds_binary = (image_preds >= threshold).float()
            metrics['image_accuracy'] = format_metric(accuracy_score(labels.cpu(), image_preds_binary.cpu()))
            metrics['image_f1'] = format_metric(f1_score(labels.cpu(), image_preds_binary.cpu(), zero_division=0, average='macro'))

        if isinstance(outputs, dict) and 'text_logits' in outputs:
            text_preds = torch.sigmoid(outputs['text_logits'])
            text_preds_binary = (text_preds >= threshold).float()
            metrics['text_accuracy'] = format_metric(accuracy_score(labels.cpu(), text_preds_binary.cpu()))
            metrics['text_f1'] = format_metric(f1_score(labels.cpu(), text_preds_binary.cpu(), zero_division=0, average='macro'))

        return metrics

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None samples
    if not batch:
        return None

    inputs = {
        'input_ids': [],
        'pixel_values': [],
        'attention_mask': []
    }
    labels = []

    for item in batch:
        for key in inputs:
            if key in item[0]:
                inputs[key].append(item[0][key])
        labels.append(item[1])

    # Stack or pad tensors
    for key in inputs:
        if inputs[key]:
            if key == 'pixel_values':
                inputs[key] = torch.stack(inputs[key])
            else:
                inputs[key] = torch.nn.utils.rnn.pad_sequence(inputs[key], batch_first=True, padding_value=0)

    return {
        'text_inputs': inputs['input_ids'],
        'image_inputs': inputs['pixel_values'],
        'attention_mask': inputs['attention_mask'],
        'labels': torch.tensor(labels),
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# Add the wise_ft function after the existing functions
def wise_ft(zeroshot_model, finetuned_model, alpha, fisher_floor=1e-8):
    theta_0 = zeroshot_model.state_dict()
    theta_1 = finetuned_model.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # Initialize an empty dictionary to store the interpolated weights
    theta = {}
    
    # Iterate over each key in the state dictionary of the zeroshot model
    for key in theta_0.keys():
        # Create a tensor of ones with the same shape as the current weight tensor
        ones = torch.ones_like(theta_0[key])
        
        # Calculate the Fisher information for the zeroshot model, ensuring a minimum value defined by fisher_floor
        f_0 = torch.maximum(ones, fisher_floor * ones)
        
        # Calculate the Fisher information for the finetuned model, ensuring a minimum value defined by fisher_floor
        f_1 = torch.maximum(ones, fisher_floor * ones)
        
        # Compute the coefficients for the interpolation based on the alpha value
        c_0 = (1 - alpha) * f_0  # Coefficient for the zeroshot model
        c_1 = alpha * f_1        # Coefficient for the finetuned model
        
        # Interpolate the weights using the computed coefficients
        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    # create a new model with the interpolated weights
    interpolated_model = copy.deepcopy(finetuned_model)
    interpolated_model.load_state_dict(theta)

    return interpolated_model

def train_model(model, train_dataloader, validation_dataloader, num_epochs, learning_rate, device):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    
    best_val_f1 = 0.0
    best_model = None
    epochs_without_improvement = 0
    early_stopping_patience = 5

    if isinstance(model, nn.DataParallel):
        model = model.module
    else:
        model = model

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            input_ids = batch['text_inputs'].to(device)
            pixel_values = batch['image_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, pixel_values, attention_mask)
            classification_loss = model.calculate_loss(outputs, labels)

            loss = classification_loss
            if torch.isnan(loss).any():
                print(f"NaN loss detected. Skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        val_metrics = evaluate_model(model, validation_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_dataloader):.4f}") 
        print(f"Val F1: {val_metrics['f1']}, Val Accuracy: {val_metrics['accuracy']}, Val AUC: {val_metrics['auc']}, Val Precision: {val_metrics['precision']}, Val Recall: {val_metrics['recall']}")

        if 'text_f1' and 'text_accuracy' in val_metrics:
            print(f"Text F1: {val_metrics['text_f1']}, Text Accuracy: {val_metrics['text_accuracy']}")
        if 'image_f1' and 'image_accuracy' in val_metrics:
            print(f"Image F1: {val_metrics['image_f1']}, Image Accuracy: {val_metrics['image_accuracy']}")
        
        # Convert val_metrics['auc'] to float for comparison
        current_val_f1 = float(val_metrics['f1'])
        scheduler.step(current_val_f1)
        
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break

    return best_model

def evaluate_model(model, dataloaders, device):
    model.eval()
    all_outputs = {'logits': [], 'image_logits': [], 'text_logits': []}
    all_labels = []

    with torch.no_grad():
        for batch in dataloaders:
            if batch is None:
                continue
            input_ids = batch['text_inputs'].to(device)
            pixel_values = batch['image_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, pixel_values, attention_mask)
            
            for key in all_outputs.keys():
                if key in outputs:
                    all_outputs[key].append(outputs[key].cpu())
            all_labels.extend(labels.cpu().numpy())

    combined_outputs = {k: torch.cat(v) for k, v in all_outputs.items() if v}
    
    # Check if model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        metrics = model.module.calculate_metrics(combined_outputs, torch.tensor(all_labels))
    else:
        metrics = model.calculate_metrics(combined_outputs, torch.tensor(all_labels))
    
    return metrics

def format_metric(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else value

def main():
    train_data_path = "/backup/girish_datasets/MultiOFF/Training_meme_dataset.csv"
    validation_data_path = "/backup/girish_datasets/MultiOFF/Validation_meme_dataset.csv"
    test_data_path = "/backup/girish_datasets/MultiOFF/Testing_meme_dataset.csv"
    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)
    test_data = pd.read_csv(test_data_path)

    train_dataset = MultiOFFDataset(train_data, clip_processor)
    validation_dataset = MultiOFFDataset(validation_data, clip_processor)
    test_dataset = MultiOFFDataset(test_data, clip_processor)

    batch_size = 512
    learning_rate = 1e-4

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Instantiate model, optimizer, and loss function
    zeroshot_model = CLIPClassifier(model_path="openai/clip-vit-large-patch14", fusion_type='align', 
                                   weight_image_loss=0.3, weight_text_loss=0.7)
    zeroshot_model.to(device)

    # Calculate the number of parameters being fine-tuned
    num_finetune_params = sum(p.numel() for p in zeroshot_model.parameters() if p.requires_grad)
    print(f"Number of parameters being fine-tuned: {num_finetune_params / 1e6:.2f}M")

    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # model_name = type(zeroshot_model).__name__  # Get the model class name

    # Parallelize model to multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     zeroshot_model = nn.DataParallel(zeroshot_model)

    # Train the model
    num_epochs = 20
    learning_rate = 1e-4
    finetuned_model = train_model(zeroshot_model, train_dataloader, validation_dataloader, num_epochs, learning_rate, device)

    # Apply WiSE-FT
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_val_f1 = 0.0
    best_model = None

    for alpha in alphas:
        wise_ft_model = wise_ft(zeroshot_model, finetuned_model, alpha)
        wise_ft_model = wise_ft_model.to(device)

        # Evaluate the WiSE-FT model
        val_metrics = evaluate_model(wise_ft_model, validation_dataloader, device)
        
        print(f"Alpha: {alpha}, Validation F1: {val_metrics['f1']}")
        
        current_val_f1 = float(val_metrics['f1'])
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_alpha = alpha
            best_model = copy.deepcopy(wise_ft_model)

    print(f"Best alpha: {best_alpha}")

    # Use the best model for final evaluation
    final_model = best_model.to(device)

    # Evaluate on test set
    test_metrics = evaluate_model(final_model, test_dataloader, device)
    print(f"Test F1: {test_metrics['f1']}, Test Accuracy: {test_metrics['accuracy']}, Test AUC: {test_metrics['auc']}, Test Precision: {test_metrics['precision']}, Test Recall: {test_metrics['recall']}")
    
    if 'text_f1' and 'text_accuracy' in test_metrics:
        print(f"Text F1: {test_metrics['text_f1']}, Text Accuracy: {test_metrics['text_accuracy']}")
    if 'image_f1' and 'image_accuracy' in test_metrics:
        print(f"Image F1: {test_metrics['image_f1']}, Image Accuracy: {test_metrics['image_accuracy']}")

    # Save the best model
    # torch.save(final_model.state_dict(), f'{model_name}_best_wise_ft_model_{timestamp}.pth')

if __name__ == "__main__":
    main()
