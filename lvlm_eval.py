### Evaluate InternVL2_5-8B model on Hateful Memes test set

import os
import random
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from typing import List, Tuple, Dict
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

# Use multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")

folder = "/backup/girish_datasets/Hateful_Memes_Extended/"
input_file = "/backup/girish_datasets/Hateful_Memes_Extended/hateful_memes_expanded_new.csv"

def load_model_and_tokenizer():
    """
    Load InternVL2_5-8B model and tokenizer with multi-GPU support
    """
    model = AutoModel.from_pretrained(
        'OpenGVLab/InternVL2_5-8B',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.eval().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        'OpenGVLab/InternVL2_5-8B', 
        trust_remote_code=True,
        use_fast=False
    )
    
    return model, tokenizer

def load_image(image_path, input_size=448):
    """
    Load and preprocess image for the model
    """
    try:
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(device)
        return pixel_values
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def predict_hatefulness(model, tokenizer, image_path, text):
    """
    Predict if a meme is hateful or not using InternVL2_5-8B with both image and text input
    Returns prediction and explanation
    """
    pixel_values = load_image(image_path)
    if pixel_values is None:
        return None
        
    pixel_values = pixel_values.to(torch.bfloat16)
    
    # First, get the model's analysis of the content
    analysis_prompt = f"""<image>
    The image is accompanied by the following text: {text}

    Task: Analyze this meme by considering both the visual elements and the text content.
    Provide a brief analysis focusing on:
    1. What is shown in the image
    2. How the text relates to the image
    3. The overall message or intent
    4. Whether this combination promotes bias, discrimination, or harmful stereotypes
    
    Format your response as a brief paragraph."""
    
    try:
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                analysis = model.module.chat(tokenizer, pixel_values, analysis_prompt, dict(
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                ))
            else:
                analysis = model.chat(tokenizer, pixel_values, analysis_prompt, dict(
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                ))
        
        # Now, use this analysis to make the final decision
        decision_prompt = f"""Based on the following analysis of a meme:
        {analysis}
        
        Determine if this meme contains hateful content by considering:
        - Does it promote discrimination or prejudice?
        - Does it target specific groups based on protected characteristics?
        - Does it use harmful stereotypes or derogatory language?
        - Is the combination of image and text intended to cause harm or offense?

        Respond with ONLY 'hateful' or 'not hateful', followed by a brief explanation after a semicolon."""
        
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                response = model.module.chat(tokenizer, pixel_values, decision_prompt, dict(
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                ))
            else:
                response = model.chat(tokenizer, pixel_values, decision_prompt, dict(
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                ))
        
        # Split response into decision and explanation
        response = response.strip().lower()
        parts = response.split(';', 1)
        decision = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        
        # For debugging
        # print(f"\nImage: {os.path.basename(image_path)}")
        # print(f"Text: {text}")
        # print(f"Analysis: {analysis}")
        # print(f"Decision: {decision}")
        # print(f"Explanation: {explanation}")
        # print("-" * 80)
        
        return 0 if 'not hateful' in decision else 1, explanation
        
    except Exception as e:
        print(f"Error predicting hatefulness for {image_path}: {str(e)}")
        return None

def eval_lvlm(model, tokenizer, test_df):
    """
    Evaluate InternVL2_5-8B model on Hateful Memes test set using both image and text
    """
    if len(test_df) == 0:
        raise ValueError("Test dataset is empty!")
        
    predictions = []
    true_labels = []
    failed_images = []
    explanations = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        image_path = os.path.join(folder, row['img'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            failed_images.append(image_path)
            continue
            
        # Get the text caption for the image
        text = row['ivl_8b_short_caption']
        if pd.isna(text):
            print(f"No caption found for {image_path}")
            failed_images.append(image_path)
            continue
            
        result = predict_hatefulness(model, tokenizer, image_path, text)
        if result is not None:
            pred, explanation = result
            predictions.append(pred)
            true_labels.append(row['label'])
            explanations.append(explanation)
    
    if len(predictions) == 0:
        raise ValueError("No valid predictions were made!")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics with zero_division parameter
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'auroc': roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.0
    }
    
    # Add additional information
    metrics['total_samples'] = len(test_df)
    metrics['processed_samples'] = len(predictions)
    metrics['failed_samples'] = len(failed_images)
    
    # Save predictions and explanations
    results_df = pd.DataFrame({
        'image': test_df['img'].values[:len(predictions)],
        'text': test_df['text'].values[:len(predictions)],
        'true_label': true_labels,
        'predicted_label': predictions,
        'explanation': explanations
    })
    results_df.to_csv('prediction_results.csv', index=False)
    
    return metrics, failed_images

if __name__ == "__main__":
    try:
        # Load test data
        print("Loading dataset...")
        df = pd.read_csv(input_file)
        test_df = df[df['split'] == 'test_seen'].reset_index(drop=True)
        print(f"Found {len(test_df)} test samples")
        
        if len(test_df) == 0:
            raise ValueError("No test samples found in the dataset!")
        
        # Check if required column exists
        if 'ivl_8b_short_caption' not in test_df.columns:
            raise ValueError("Column 'ivl_8b_short_caption' not found in the dataset!")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        
        # Evaluate model
        print("Starting evaluation...")
        metrics, failed_images = eval_lvlm(model, tokenizer, test_df)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Processed samples: {metrics['processed_samples']}")
        print(f"Failed samples: {metrics['failed_samples']}")
        print(f"\nMetrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        
        if failed_images:
            print("\nFailed images:")
            for img in failed_images:
                print(f"- {img}")
                
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")