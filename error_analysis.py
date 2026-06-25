import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import open_clip
import numpy as np

from clip_xlm_roberta_ft import CLIPClassifier

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor using OpenCLIP
model_name = "xlm-roberta-large-ViT-H-14"
pretrained = "frozen_laion5b_s13b_b90k"
base_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# Load data
data_path = "/backup/girish_datasets/Hateful_Memes_Extended/ivl_plus_gemini_captions.csv"
data = pd.read_csv(data_path)

test_seen_data = data[data['split'] == 'test_seen']
model_path = "checkpoints/clip_roberta_large_txt_4.pth"

# Create and load model
model = CLIPClassifier(base_model)
checkpoint = torch.load(model_path, weights_only=True)
state_dict = checkpoint['model_state_dict']

# Remove 'module.' prefix from state dict keys
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Set to eval mode

# Define test memes
test_memes = ["45139.png", "07193.png", "10476.png", "10938.png", "18406.png", "26891.png", "28579.png", "32907.png", 
              "35198.png", "36128.png", "50643.png", "63502.png", "73842.png", "80463.png", "80473.png", "85630.png", "91260.png"]

# Add 'img/' prefix to test memes to match DataFrame format
test_memes_with_prefix = [f"img/{meme}" for meme in test_memes]
similar_samples_df = test_seen_data[test_seen_data['img'].isin(test_memes_with_prefix)]

if len(similar_samples_df) == len(test_memes):
    print("All chosen memes are present in the test set.")
else:
    print(f"Only {len(similar_samples_df)} out of {len(test_memes)} chosen memes are present in the test set.")

# print(f"\nFound {len(similar_samples_df)} matching samples in the test set")
# print("Sample image paths from the dataset:")
# print(similar_samples_df['img'].head().tolist())

# Evaluate
all_preds = []
all_labels = []
all_probs = []
caption_scores = {}

model.eval()
with torch.no_grad():
    for _, row in tqdm(similar_samples_df.iterrows(), desc="Evaluating"):
        # Load and process image
        image_path = f'/backup/girish_datasets/Hateful_Memes_Extended/{row["img"]}'
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Get all captions
        captions = [
            str(row['text']),
            str(row['ivl_8b_new_caption']),
            str(row['gemini_caption'])
        ]
        captions = [cap for cap in captions if cap.strip()]  # Remove empty captions
        
        # Process all captions
        text_tensors = tokenizer(captions).to(device)  # [num_captions, seq_len]
        
        # Get text features for all captions
        text_features = model.clip_model.encode_text(text_tensors)
        
        # Score captions using the caption scorer
        scores = model.caption_scorer(text_features).squeeze()
        caption_probs = F.softmax(scores, dim=0)
        
        # Select best caption
        best_idx = caption_probs.argmax().item()
        best_caption = captions[best_idx]
        
        # Store caption scores for later display
        caption_scores[row['img']] = {
            'captions': captions,
            'scores': scores.cpu().numpy(),
            'probs': caption_probs.cpu().numpy(),
            'best_idx': best_idx
        }
        
        # Process best caption for prediction
        best_text_tensor = tokenizer([best_caption]).to(device)
        
        # Get label
        label = torch.tensor([row['label']], dtype=torch.float).to(device)
        
        # Forward pass with best caption
        logits = model(image_tensor, best_text_tensor)
        probs = torch.sigmoid(logits)
        
        # Store probabilities and labels
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
        # Get binary predictions
        # preds = (probs >= 0.5).float()
        # all_preds.extend(preds.cpu().numpy())

# Convert to numpy arrays for threshold calculation
temp_probs = np.array(all_probs)
temp_labels = np.array(all_labels)

# Calculate optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(temp_labels, temp_probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-7)  # Add small epsilon to prevent division by zero
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nOptimal threshold from PR curve: {optimal_threshold:.4f}")

# Get binary predictions using optimal threshold
all_preds = (temp_probs >= optimal_threshold).astype(float)

# Calculate and print metrics
print("\nResults for test memes:")
print(f"AUROC: {roc_auc_score(temp_labels, temp_probs):.4f}")
print(f"Accuracy: {accuracy_score(temp_labels, all_preds):.4f}")
print(f"Precision: {precision_score(temp_labels, all_preds, zero_division=0):.4f}")
print(f"Recall: {recall_score(temp_labels, all_preds, zero_division=0):.4f}")
print(f"F1: {f1_score(temp_labels, all_preds, zero_division=0):.4f}")

# Print predictions and caption details for each meme
print("\nPredictions for each meme:")
for i, (image_id, pred, prob, label) in enumerate(zip(similar_samples_df['img'].tolist(), all_preds, temp_probs, temp_labels)):
    print(f"\n{'='*80}")
    print(f"Meme {i+1}: {image_id}")
    print(f"Prediction: {'Hateful' if pred == 1 else 'Non-hateful'} (Probability: {prob:.4f})")
    print(f"True Label: {'Hateful' if label == 1 else 'Non-hateful'}")
    
    # Print all captions with their scores
    scores = caption_scores[image_id]
    print("\nAll Captions:")
    for j, (caption, score, prob) in enumerate(zip(scores['captions'], scores['scores'], scores['probs'])):
        print(f"\nCaption {j+1}:")
        print(f"Text: {caption}")
        print(f"Score: {score:.4f}")
        print(f"Probability: {prob:.4f}")
        if j == scores['best_idx']:
            print("*** Selected as Best Caption ***")
    
    print('-' * 80)
