### Evaluate Gemini 2.0 Flash model on MultiOFF test set

import os
import time
import pandas as pd
import google.generativeai as genai
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import io
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Dataset paths
folder = "/backup/girish_datasets/MultiOFF/Labelled Images/"
input_file = "/backup/girish_datasets/MultiOFF/Testing_meme_dataset.csv"
processed_file = "processed_images.txt"

def load_processed_images():
    """
    Load the list of already processed image IDs
    """
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_image(image_id):
    """
    Save an image ID to the processed images file
    """
    with open(processed_file, 'a') as f:
        f.write(f"{image_id}\n")

def load_model():
    """
    Load Gemini 2.0 Flash model
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_image(image_path):
    """
    Load and prepare image for Gemini model
    """
    try:
        # Open and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Convert to JPEG format in memory to avoid WebP encoding issues
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return Image.open(img_byte_arr)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def predict_with_retry(model, prompt, image=None, max_retries=3, initial_delay=1):
    """
    Make prediction with exponential backoff retry logic
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Configure safety settings to allow all content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            if image is not None:
                response = model.generate_content([prompt, image],
                                                  safety_settings=safety_settings)
            else:
                response = model.generate_content(prompt,
                                                  safety_settings=safety_settings)
            return response
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            raise e
    return None

def predict_hatefulness(model, image_path, text):
    """
    Predict if a meme is offensive or not using Gemini model
    Returns prediction and explanation
    """
    image = load_image(image_path)
    if image is None:
        return None
        
    # First, get the model's analysis of the content
    analysis_prompt = f"""Analyze this meme by considering both the visual elements and the text content: "{text}"

    Focus on:
    1. What is shown in the image
    2. How the text relates to the image
    3. The overall message or intent
    4. Whether this combination promotes bias, discrimination, or harmful stereotypes
    
    Provide your analysis in a brief paragraph."""
    
    try:
        # Generate content analysis with retry logic
        response = predict_with_retry(model, analysis_prompt, image)
        if response is None or not hasattr(response, 'text'):
            return 1, "Content blocked due to safety concerns"  # Assume offensive if blocked
        analysis = response.text
        
        # Now, use this analysis to make the final decision
        decision_prompt = f"""Based on the following analysis of a meme:
        {analysis}
        
        Determine if this meme contains offensive content by considering:
        - Does it promote discrimination or prejudice?
        - Does it target specific groups based on protected characteristics?
        - Does it use harmful stereotypes or derogatory language?
        - Is the combination of image and text intended to cause harm or offense?

        Respond with ONLY 'offensive' or 'not offensive', followed by a brief explanation after a semicolon."""
        
        response = predict_with_retry(model, decision_prompt)
        if response is None or not hasattr(response, 'text'):
            return 1, "Content blocked due to safety concerns"  # Assume offensive if blocked
            
        response_text = response.text.strip().lower()
        
        # Split response into decision and explanation
        parts = response_text.split(';', 1)
        decision = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        
        return 0 if 'not offensive' in decision else 1, explanation
        
    except Exception as e:
        print(f"Error predicting offensiveness for {image_path}: {str(e)}")
        return None

def eval_gemini(model, test_df, batch_size=10):
    """
    Evaluate Gemini model on MultiOFF test set
    """
    if len(test_df) == 0:
        raise ValueError("Test dataset is empty!")
        
    # Load already processed images
    processed_images = load_processed_images()
    print(f"Found {len(processed_images)} already processed images")
    
    predictions = []
    true_labels = []
    failed_images = []
    explanations = []
    processed_indices = []
    
    # Filter out already processed images
    test_df['img_id'] = test_df['image_name'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    remaining_df = test_df[~test_df['img_id'].isin(processed_images)].reset_index(drop=True)
    print(f"Remaining images to process: {len(remaining_df)}")
    
    # Process in smaller batches to handle rate limits better
    for i in tqdm(range(0, len(remaining_df), batch_size), desc="Evaluating batches"):
        batch_df = remaining_df.iloc[i:i+batch_size]
        batch_predictions = []
        batch_labels = []
        batch_explanations = []
        batch_images = []
        
        for _, row in batch_df.iterrows():
            image_path = os.path.join(folder, row['image_name'])
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                failed_images.append(image_path)
                continue
                
            text = str(row['sentence'])
            if pd.isna(text):
                print(f"No text found for {image_path}")
                failed_images.append(image_path)
                continue
                
            result = predict_hatefulness(model, image_path, text)
            if result is not None:
                pred, explanation = result
                batch_predictions.append(pred)
                # Convert label to binary (0 for "Non-offensiv", 1 for "offensive")
                label = 1 if str(row['label']) == "offensive" else 0
                batch_labels.append(label)
                batch_explanations.append(explanation)
                batch_images.append(row['img_id'])
                
                # Save each processed image ID immediately
                save_processed_image(row['img_id'])
            
            # Small delay between items in batch
            time.sleep(5)
        
        # Extend the main lists with batch results
        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)
        explanations.extend(batch_explanations)
        processed_indices.extend(batch_images)
        
        # Large delay between batches (60 seconds for 10 requests per minute limit)
        print(f"Waiting 60 seconds before next batch...")
        time.sleep(60)
    
    if len(predictions) == 0:
        raise ValueError("No valid predictions were made!")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
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
        'image': test_df['image_name'].values[:len(predictions)],
        'text': test_df['sentence'].values[:len(predictions)],
        'true_label': true_labels,
        'predicted_label': predictions,
        'explanation': explanations
    })
    
    # If results file exists, append to it, otherwise create new
    if os.path.exists('gemini_multioff_prediction_results.csv'):
        existing_results = pd.read_csv('gemini_multioff_prediction_results.csv')
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    
    results_df.to_csv('gemini_multioff_prediction_results.csv', index=False)
    
    return metrics, failed_images

if __name__ == "__main__":
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set!")
            
        # Load test data
        print("Loading dataset...")
        test_df = pd.read_csv(input_file)
        print(f"Found {len(test_df)} test samples")
        
        if len(test_df) == 0:
            raise ValueError("No test samples found in the dataset!")
        
        # Load model
        print("Loading Gemini model...")
        model = load_model()
        if model is None:
            raise ValueError("Failed to load Gemini model!")
        
        # Evaluate model
        print("Starting evaluation...")
        metrics, failed_images = eval_gemini(model, test_df)
        
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
