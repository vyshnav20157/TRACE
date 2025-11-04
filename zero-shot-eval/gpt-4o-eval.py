### Evaluate GPT-4o-mini and o1-mini models on Hateful Memes test set

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import base64
from openai import OpenAI
from PIL import Image
import io

# Configure OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Dataset paths
folder = "/backup/girish_datasets/Hateful_Memes_Extended/"
input_file = "/backup/girish_datasets/Hateful_Memes_Extended/hateful_memes_expanded_new.csv"
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

def encode_image(image_path):
    """
    Encode image to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def predict_with_retry_gpt4o(image_path, text, max_retries=3, initial_delay=1):
    """
    Make prediction using GPT-4o-mini model with exponential backoff retry logic
    """
    delay = initial_delay
    base64_image = encode_image(image_path)
    if base64_image is None:
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in understanding and identifying hateful memes."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The meme has the text '{text}'. Is this meme hateful or not? Return '1' if hateful, '0' if not, followed by a brief explanation after a semicolon."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            parts = response_text.split(';', 1)
            prediction = int(parts[0].strip())
            explanation = parts[1].strip() if len(parts) > 1 else ""
            
            return prediction, explanation

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            print(f"Error in GPT-4o prediction for {image_path}: {str(e)}")
            return None

def predict_with_retry_o1mini(image_path, text, max_retries=3, initial_delay=1):
    """
    Make prediction using o1-mini model with exponential backoff retry logic
    """
    delay = initial_delay
    base64_image = encode_image(image_path)
    if base64_image is None:
        return None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o1-mini-2024-09-12",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"As an expert in understanding and identifying hateful memes, analyze this meme. The meme has the text '{text}'. Is this meme hateful or not? Return '1' if hateful, '0' if not, followed by a brief explanation after a semicolon."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            parts = response_text.split(';', 1)
            prediction = int(parts[0].strip())
            explanation = parts[1].strip() if len(parts) > 1 else ""
            
            return prediction, explanation

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            print(f"Error in o1-mini prediction for {image_path}: {str(e)}")
            return None

def evaluate_model(model_name, predict_func, test_df, batch_size=10):
    """
    Evaluate model on Hateful Memes test set
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
    test_df['img_id'] = test_df['img'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    remaining_df = test_df[~test_df['img_id'].isin(processed_images)].reset_index(drop=True)
    print(f"Remaining images to process: {len(remaining_df)}")
    
    # Process in smaller batches to handle rate limits better
    for i in tqdm(range(0, len(remaining_df), batch_size), desc=f"Evaluating {model_name}"):
        batch_df = remaining_df.iloc[i:i+batch_size]
        batch_predictions = []
        batch_labels = []
        batch_explanations = []
        batch_images = []
        
        for _, row in batch_df.iterrows():
            image_path = os.path.join(folder, row['img'])
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                failed_images.append(image_path)
                continue
                
            text = row['text']
            if pd.isna(text):
                print(f"No text found for {image_path}")
                failed_images.append(image_path)
                continue
                
            result = predict_func(image_path, text)
            if result is not None:
                pred, explanation = result
                batch_predictions.append(pred)
                batch_labels.append(row['label'])
                batch_explanations.append(explanation)
                batch_images.append(row['img_id'])
                
                # Save each processed image ID immediately
                save_processed_image(row['img_id'])
            
            # Small delay between items in batch
            time.sleep(0.5)
        
        # Extend the main lists with batch results
        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)
        explanations.extend(batch_explanations)
        processed_indices.extend(batch_images)
        
        # Large delay between batches
        print(f"Waiting 10 seconds before next batch...")
        time.sleep(10)
    
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
        'image': test_df['img'].values[:len(predictions)],
        'text': test_df['text'].values[:len(predictions)],
        'true_label': true_labels,
        'predicted_label': predictions,
        'explanation': explanations
    })
    
    # Save to model-specific results file
    results_file = f'{model_name}_prediction_results.csv'
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    
    results_df.to_csv(results_file, index=False)
    
    return metrics, failed_images

if __name__ == "__main__":
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
            
        # Load test data
        print("Loading dataset...")
        df = pd.read_csv(input_file)
        test_df = df[df['split'] == 'test_seen'].reset_index(drop=True)
        print(f"Found {len(test_df)} test samples")
        
        if len(test_df) == 0:
            raise ValueError("No test samples found in the dataset!")
        
        # Evaluate GPT-4o-mini model
        print("\nStarting GPT-4o-mini evaluation...")
        gpt4o_metrics, gpt4o_failed = evaluate_model(
            "gpt4o_mini",
            predict_with_retry_gpt4o,
            test_df
        )
        
        # Print GPT-4o-mini results
        print("\nGPT-4o-mini Evaluation Results:")
        print(f"Total samples: {gpt4o_metrics['total_samples']}")
        print(f"Processed samples: {gpt4o_metrics['processed_samples']}")
        print(f"Failed samples: {gpt4o_metrics['failed_samples']}")
        print(f"\nMetrics:")
        print(f"Accuracy: {gpt4o_metrics['accuracy']:.4f}")
        print(f"F1 Score: {gpt4o_metrics['f1']:.4f}")
        print(f"Precision: {gpt4o_metrics['precision']:.4f}")
        print(f"Recall: {gpt4o_metrics['recall']:.4f}")
        print(f"AUROC: {gpt4o_metrics['auroc']:.4f}")
        
        # # Evaluate o1-mini model
        # print("\nStarting o1-mini evaluation...")
        # o1_metrics, o1_failed = evaluate_model(
        #     "o1_mini",
        #     predict_with_retry_o1mini,
        #     test_df
        # )
        
        # # Print o1-mini results
        # print("\no1-mini Evaluation Results:")
        # print(f"Total samples: {o1_metrics['total_samples']}")
        # print(f"Processed samples: {o1_metrics['processed_samples']}")
        # print(f"Failed samples: {o1_metrics['failed_samples']}")
        # print(f"\nMetrics:")
        # print(f"Accuracy: {o1_metrics['accuracy']:.4f}")
        # print(f"F1 Score: {o1_metrics['f1']:.4f}")
        # print(f"Precision: {o1_metrics['precision']:.4f}")
        # print(f"Recall: {o1_metrics['recall']:.4f}")
        # print(f"AUROC: {o1_metrics['auroc']:.4f}")
        
        if gpt4o_failed:
            print("\nGPT-4o-mini failed images:")
            for img in gpt4o_failed:
                print(f"- {img}")
                
        # if o1_failed:
        #     print("\no1-mini failed images:")
        #     for img in o1_failed:
        #         print(f"- {img}")
                
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")