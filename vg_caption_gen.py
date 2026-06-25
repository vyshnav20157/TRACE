import io
import os
import random
import time
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer, AutoModel
from typing import List, Tuple, Dict
import torchvision.transforms as T

from ram.models import ram_plus
# from ram import inference_ram_openset as inference
from ram import inference_ram as inference
from ram import get_transform
from ram.utils import build_openset_llm_label_embedding
from torch import nn
import json
import numpy as np
from PIL import ImageDraw, ImageFont

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)  # Set the default CUDA device

folder = "/backup/girish_datasets/Hateful_Memes_Extended/"
input_file = "/backup/girish_datasets/Hateful_Memes_Extended/ivl_plus_gemini_captions.csv"

def load_random_images(directory: str, num_images: int = 10) -> List[str]:
    """
    Load random images from the specified directory.
    
    Args:
        directory (str): Path to the directory containing images.
        num_images (int): Number of random images to load.
    
    Returns:
        List[str]: List of file paths for the randomly selected images.
    """
    all_images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(directory, f) for f in random.sample(all_images, min(num_images, len(all_images)))]

def load_ram_model(pretrained_path, image_size, llm_tag_des_path):
    model = ram_plus(pretrained=pretrained_path, image_size=image_size, vit='swin_l')
    
    # print('Building tag embedding:')
    # with open(llm_tag_des_path, 'rb') as fo:
    #     llm_tag_des = json.load(fo)
    # openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    # model.tag_list = np.array(openset_categories)
    # model.label_embed = nn.Parameter(openset_label_embedding.float())
    # model.num_class = len(openset_categories)
    # model.class_threshold = torch.ones(model.num_class, device=device) * 0.5

    model.eval()
    return model.to(device)

def recognize_tags(image_path: str, model: ram_plus, transform) -> List[str]:
    """
    Recognize tags from the image using the provided RAM Plus model.
    
    Args:
        image_path (str): Path to the input image.
        model (ram_plus): Pre-loaded RAM Plus model.
        transform: Image transform function.
    
    Returns:
        List[str]: List of recognized tags.
    """
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        tags = inference(image, model)

    return tags[0]

def extract_grounding_info(image_path: str, tags: str, processor, model) -> Dict[str, List[Tuple[float, float, float, float, float]]]:
    """
    Extract boundary boxes for text and other entities in the image using GroundingDINO.
    
    Args:
        image_path (str): Path to the input image.
        tags (str): String of tags separated by ' | '.
        processor: Pre-loaded GroundingDINO processor.
        model: Pre-loaded GroundingDINO model.
    
    Returns:
        Dict[str, List[Tuple[float, float, float, float, float]]]: Dictionary mapping entity labels to their bounding boxes.
    """
    image = Image.open(image_path)
    # Format tags: lowercase, replace ' | ' with '. ', and add a final dot
    text = tags.lower().replace(' | ', '. ') + '.' if tags else "no tags."
    
    inputs = processor(
        images=image, 
        text=text, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    
    grounding_info = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i.item(), 2) for i in box]
        score = round(score.item(), 2)
        label_text = label
        if label_text not in grounding_info:
            grounding_info[label_text] = []
        grounding_info[label_text].append(tuple(box + [score]))
    
    if not grounding_info:
        print(f"No objects detected for {image_path}")
    # else:
    #     print(f"Extracted grounding information for {image_path}: {grounding_info}")
    return grounding_info

def extract_bbox_image(image_path: str, tags: str, processor, model) -> Image.Image:
    """
    Extract boundary boxes for text and other entities in the image using GroundingDINO.
    
    Args:
        image_path (str): Path to the input image.
        tags (str): String of tags separated by ' | '.
        processor: Pre-loaded GroundingDINO processor.
        model: Pre-loaded GroundingDINO model.
    
    Returns:
        Image.Image: Image with boundary boxes and labels drawn on it.
    """
    image = Image.open(image_path)
    # Format tags: lowercase, replace ' | ' with '. ', and add a final dot
    tags_list = [tag.strip() for tag in tags.split('|') if tag.strip().lower() != 'close-up']
    tags_list.append('text')
    text = '. '.join(tags_list).lower() + '.' if tags_list else "no tags."
    
    inputs = processor(
        images=image, 
        text=text, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i.item(), 2) for i in box]
        score = round(score.item(), 2)
        
        # Draw bounding box
        draw.rectangle(box, outline="red", width=2)
        
        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1] - 10), label_text, fill="red", font=font)
    
    if not results["labels"]:
        print(f"No objects detected for {image_path}")
    
    return image

def load_image(image_path, input_size=448):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0).to(device)
    return pixel_values

def generate_captions(image_path: str, grounding_info: Dict[str, List[Tuple[float, float, float, float, float]]], model, tokenizer) -> str:
    """
    Generate short and long captions using InternVL2-8B model, incorporating grounding information.
    
    Args:
        image_path (str): Path to the input image.
        grounding_info (Dict[str, List[Tuple[float, float, float, float, float]]]): Grounding information from GroundingDINO.
        model: Pre-loaded InternVL2-8B model.
        tokenizer: Pre-loaded InternVL2-8B tokenizer.
    
    Returns:
        str: Caption.
    """
    # Load and preprocess the image
    pixel_values = load_image(image_path).to(torch.bfloat16).to(device)

    # Prepare the prompt with grounding information
    grounding_prompt = "Key elements in the image:"
    if grounding_info:
        for obj, boxes in grounding_info.items():
            confidence = max(box[4] for box in boxes)
            if confidence > 0.7:
                certainty = "high certainty"
            elif confidence > 0.5:
                certainty = "moderate certainty"
            else:
                certainty = "low certainty"
            grounding_prompt += f" {obj} (detected with {certainty}, count: {len(boxes)}),"
        grounding_prompt = grounding_prompt.rstrip(',') + "."
    else:
        grounding_prompt += " No specific object information available."

    # prompt = f"""<image>
    # {grounding_prompt}
    # Task: Analyze this meme image and generate two captions suitable for CLIP fine-tuning:

    # 1. A short, concise caption (10-15 words) that captures:
    # - The main visual elements in the image
    # - The key message conveyed by the text in the meme
    # - The overall theme or context of the meme

    # 2. A detailed caption (20-30 words) that provides:
    # - A comprehensive description of the visual elements
    # - The exact text present in the meme (if any)
    # - An explanation of how the visual and textual elements interact to create the meme's meaning
    # - Any cultural references or context necessary to understand the meme

    # Important guidelines:
    # - Describe the content and tone objectively without using labels like 'hateful' or 'non-hateful'
    # - If the meme contains sensitive content, use neutral language to describe the theme or message
    # - Focus on factual descriptions of visual and textual elements
    # - Avoid subjective interpretations or judgments about the meme's intent
    # - If there are strong emotions or attitudes expressed, describe them without endorsement

    # Format the response as:
    # Short: [Short caption here]
    # Long: [Long caption here]"""

    prompt = f"""<image>
    {grounding_prompt}
    Task: Analyze this meme image using the above grounding information and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.

    The caption should:
    - Describe the main visual elements (people, objects, and setting)
    - Mention key identifying features (e.g., race, gender, religion, disability, nationality) if relevant
    - Summarize the text overlay (if short) or explain its meaning concisely
    - Describe any societal implications or stereotypes referenced in the image
    - Avoid subjective interpretations or judgmental labels (e.g., 'hateful', 'offensive')

    Additionally, if the meme references a specific group (race, religion, sex, etc.), ensure it is explicitly mentioned.
    If the meme implies attitudes like exclusion, mocking, contempt, or inferiority, briefly describe how the text and image interact to convey this.

    Format the response as:
    Caption: [Generated caption here]
    """

    # Generate captions
    generation_config = dict(max_new_tokens=4096, do_sample=False, pad_token_id = tokenizer.pad_token_id)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    # Extract short and long captions from the response
    caption = response.split("Caption:")[1].strip()
    # short_caption = ""
    # long_caption = ""
    # for line in response.split('\n'):
    #     if line.startswith("Short:"):
    #         short_caption = line.replace("Short:", "").strip()
    #     elif line.startswith("Long:"):
    #         long_caption = line.replace("Long:", "").strip()

    # print(f"Short caption: {short_caption}")
    # print(f"Long caption: {long_caption}")
    # return short_caption, long_caption
    print(f"Caption: {caption}")
    return caption

def generate_captions_without_grounding(image_path: str, model, tokenizer) -> Tuple[str, str]:
    """
    Generate CLIP-suitable captions for meme images using InternVL2-1B model, without using grounding information.
    
    Args:
        image_path (str): Path to the input meme image.
        model: Pre-loaded InternVL2-8B model.
        tokenizer: Pre-loaded InternVL2-8B tokenizer.
    
    Returns:
        Tuple[str, str]: Short caption and long caption suitable for CLIP fine-tuning, capturing both visual and textual elements of the meme.
    """
    pixel_values = load_image(image_path).to(torch.bfloat16).to(device)

    prompt = """<image>
    Task: Analyze this meme image and generate two captions suitable for CLIP fine-tuning:

    1. A short, concise caption (10-15 words) that captures:
    - The main visual elements in the image
    - The key message conveyed by the text in the meme (if any)
    - The overall theme or context of the meme

    2. A detailed caption (20-30 words) that provides:
    - A comprehensive description of the visual elements
    - The exact text present in the meme (if any)
    - An explanation of how the visual and textual elements interact to create the meme's meaning
    - Any cultural references or context necessary to understand the meme

    Important guidelines:
    - Describe the content and tone objectively without using labels like 'hateful' or 'non-hateful'
    - If the meme contains sensitive content, use neutral language to describe the theme or message
    - Focus on factual descriptions of visual and textual elements
    - Avoid subjective interpretations or judgments about the meme's intent
    - If there are strong emotions or attitudes expressed, describe them without endorsement

    Format the response as:
    Short: [Short caption here]
    Long: [Long caption here]"""

    generation_config = dict(max_new_tokens=4096, pad_token_id = tokenizer.pad_token_id)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    # Extract short and long captions from the response
    short_caption = ""
    long_caption = ""
    for line in response.split('\n'):
        if line.startswith("Short:"):
            short_caption = line.replace("Short:", "").strip()
        elif line.startswith("Long:"):
            long_caption = line.replace("Long:", "").strip()

    # print(f"Short caption: {short_caption}")
    # print(f"Long caption: {long_caption}")
    return short_caption, long_caption

def predict_with_retry(model, prompt, image=None, max_retries=3, initial_delay=1):
    """
    Make prediction with exponential backoff retry logic
    
    Args:
        model: The Gemini model to use for prediction
        prompt: The text prompt to use
        image: Optional image to include with the prompt
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying
        
    Returns:
        The model response or None if all retries fail
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Configure safety settings to allow all content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Add a small random delay to avoid synchronized requests
            jitter = random.uniform(0, 1)
            time.sleep(jitter)
            
            if image is not None:
                response = model.generate_content([prompt, image],
                                                  safety_settings=safety_settings)
            else:
                response = model.generate_content(prompt,
                                                  safety_settings=safety_settings)
            return response
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Handle different types of rate limit errors
            if "429" in error_str or "resource exhausted" in error_str or "quota" in error_str:
                wait_time = delay + random.uniform(0, delay * 0.1)  # Add jitter
                print(f"Rate limit error on attempt {attempt+1}/{max_retries}. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60)  # Exponential backoff, cap at 60 seconds
                continue
            elif "500" in error_str or "503" in error_str or "backend error" in error_str:
                # Server errors might need some time to resolve
                wait_time = delay * 1.5 + random.uniform(0, delay * 0.2)
                print(f"Server error on attempt {attempt+1}/{max_retries}. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                delay = min(delay * 2, 45)  # Exponential backoff, cap at 45 seconds
                continue
            else:
                # For other errors, retry with shorter backoff
                if attempt < max_retries - 1:
                    wait_time = delay * 0.5 + random.uniform(0, delay * 0.1)
                    print(f"Error on attempt {attempt+1}/{max_retries}: {str(e)}. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    delay = min(delay * 1.5, 30)  # Gentler backoff, cap at 30 seconds
                    continue
                else:
                    # On last attempt, raise the error
                    raise e
    
    print(f"All {max_retries} retry attempts failed. Last error: {str(last_error)}")
    return None

def load_image_for_gemini(image_path):
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

def generate_captions_using_gemini(image_path: str, grounding_info: Dict[str, List[Tuple[float, float, float, float, float]]]) -> str:
    """
    Generate a caption for an image using the Gemini 2.0 Flash model, explicitly incorporating grounding information.

    Args:
        image_path (str): Path to the input image.
        grounding_info (Dict[str, List[Tuple[float, float, float, float, float]]]): Grounding information from GroundingDINO.
    
    Returns:
        str: The generated caption.
    """
    # Add delay to avoid API quota exhaustion
    time.sleep(3)  # Add a 3-second delay between API calls
    
    # model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    # Load and preprocess the image
    image = load_image_for_gemini(image_path)

    # Prepare the grounding text
    grounding_prompt = "Key elements in the image:"
    if grounding_info:
        for obj, boxes in grounding_info.items():
            confidence = max(box[4] for box in boxes)
            if confidence > 0.7:
                certainty = "high certainty"
            elif confidence > 0.5:
                certainty = "moderate certainty"
            else:
                certainty = "low certainty"
            grounding_prompt += f" {obj} (detected with {certainty}, count: {len(boxes)}),"
        grounding_prompt = grounding_prompt.rstrip(',') + "."
    else:
        grounding_prompt += " No specific object information available."

    prompt = f"""<image>
    {grounding_prompt}
    Task: Analyze this meme image using the above grounding information and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.

    The caption should:
    - Describe the main visual elements (people, objects, and setting)
    - Mention key identifying features (e.g., race, gender, religion, disability, nationality) if relevant
    - Summarize the text overlay (if short) or explain its meaning concisely
    - Describe any societal implications or stereotypes referenced in the image
    - Avoid subjective interpretations or judgmental labels (e.g., 'hateful', 'offensive')

    Additionally, if the meme references a specific group (race, religion, sex, etc.), ensure it is explicitly mentioned.
    If the meme implies attitudes like exclusion, mocking, contempt, or inferiority, briefly describe how the text and image interact to convey this.

    Format the response as:
    Caption: [Generated caption here]
    """

    try:
        # Generate caption with retry logic
        response = predict_with_retry(model, prompt, image, max_retries=5, initial_delay=5)  # Increase retries and delay
        response_text = response.text.strip()
        if "Caption:" in response_text:
            caption = response_text.split("Caption:", 1)[1].strip()
        else:
            caption = response_text
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {str(e)}")
        return None


def process_meme_dataset(output_dir: str):
    """
    Process the meme dataset by loading images, recognizing tags, extracting grounding information, and generating captions.
    
    Args:
        directory (str): Path to the directory containing meme images.
        output_dir (str): Path to the directory where outputs will be saved.
    """
    # os.makedirs(output_dir, exist_ok=True)
    # test_memes = ["24098.png", "27614.png", "31208.png", "43275.png", "45139.png", "47103.png", "52894.png", "78962.png"]
    # image_paths = [os.path.join(directory, f) for f in test_memes]
    
    # image_paths = load_random_images(directory)

    # Read input file for processing
    df_input = pd.read_csv(input_file)

    # Prepare image paths along with their corresponding row indices
    image_paths = [(idx, os.path.join(folder, row['img'])) for idx, row in df_input.iterrows()]

    # Load the output file (which may have existing 'gemini_caption' entries) for appending captions;
    # if it doesn't exist, initialize it using the input file data.
    if os.path.exists(output_dir):
        df_out = pd.read_csv(output_dir)
    else:
        df_out = df_input.copy()
        if 'gemini_caption' not in df_out.columns:
            df_out['gemini_caption'] = ''

    # Load the RAM Plus model once
    pretrained_path = 'ram_plus_swin_large_14m.pth'
    image_size = 384
    llm_tag_des_path = 'openimages_rare_200_llm_tag_descriptions.json'
    ram_model = load_ram_model(pretrained_path, image_size, llm_tag_des_path)
    transform = get_transform(image_size=image_size)

    # Load models outside the loop
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # caption_model = AutoModel.from_pretrained(
    #     'OpenGVLab/InternVL2_5-8B',
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # ).eval().to(device)
    # tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-8B', trust_remote_code=True, use_fast=False)

    # Load processed image paths from a text file
    processed_images_file = os.path.join('processed_images_gemini.txt')
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
    else:
        processed_images = set()

    for idx, image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        if image_path in processed_images:
            continue
        try:
            # Recognize tags
            tags = recognize_tags(image_path, ram_model, transform)
            # tags_output_path = os.path.join(output_dir, f"tags_{i}.txt")
            # with open(tags_output_path, "w") as f:
            #     f.write(tags)
            
            # Extract grounding information
            grounding_info = extract_grounding_info(image_path, tags, processor, grounding_model)
            # grounding_output_path = os.path.join(output_dir, f"grounding_info_{i}.txt")
            # with open(grounding_output_path, "w") as f:
            #     for label, boxes in grounding_info.items():
            #         f.write(f"{label}: {boxes}\n")
            
            # Generate captions with grounding info
            caption = generate_captions_using_gemini(image_path, grounding_info)
            # short_caption, long_caption = generate_captions(image_path, grounding_info, caption_model, tokenizer)
            # caption = generate_captions(image_path, grounding_info, caption_model, tokenizer)
            # Append the new caption text onto the existing 'gemini_caption' in the output file
            current_caption = df_out.loc[idx, 'gemini_caption'] if pd.notna(df_out.loc[idx, 'gemini_caption']) else ""
            df_out.loc[idx, 'gemini_caption'] = (current_caption + " " + str(caption)).strip()
    
            df_out.to_csv(output_dir, index=False)
    
            processed_images.add(image_path)
            with open(processed_images_file, 'a') as f:
                f.write(image_path + '\n')
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

def process_missing_gemini_captions(input_csv_path: str, output_json_path: str):
    """
    Process only the rows in the CSV that have missing or empty Gemini captions,
    and save the result as a JSON file.
    
    Args:
        input_csv_path (str): Path to the input CSV file.
        output_json_path (str): Path to save the output JSON file.
    """
    # Read input file
    df_input = pd.read_csv(input_csv_path)
    
    # Filter rows with missing or empty Gemini captions
    missing_captions_mask = (df_input['gemini_caption'].isna()) | (df_input['gemini_caption'] == 'None') | (df_input['gemini_caption'].str.strip() == '')
    rows_to_process = df_input[missing_captions_mask].index.tolist()
    
    print(f"Found {len(rows_to_process)} rows with missing Gemini captions out of {len(df_input)} total rows")
    
    if len(rows_to_process) == 0:
        print("No rows with missing captions found. Saving current data to JSON.")
        # Remove 'appended_caption' if it exists
        if 'appended_caption' in df_input.columns:
            df_input = df_input.drop(columns=['appended_caption'])
        df_input.to_json(output_json_path, orient='records', indent=2)
        return
    
    # Create a copy of the input dataframe for the output
    df_out = df_input.copy()
    
    # Load the RAM Plus model once
    pretrained_path = 'ram_plus_swin_large_14m.pth'
    image_size = 384
    llm_tag_des_path = 'openimages_rare_200_llm_tag_descriptions.json'
    ram_model = load_ram_model(pretrained_path, image_size, llm_tag_des_path)
    transform = get_transform(image_size=image_size)

    # Load models outside the loop
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Process each row with missing captions
    for idx in tqdm(rows_to_process, desc="Processing images with missing captions", unit="image"):
        try:
            row = df_input.iloc[idx]
            image_path = os.path.join(folder, row['img'])
            
            # Recognize tags
            tags = recognize_tags(image_path, ram_model, transform)
            
            # Extract grounding information
            grounding_info = extract_grounding_info(image_path, tags, processor, grounding_model)
            
            # Generate captions with grounding info
            caption = generate_captions_using_gemini(image_path, grounding_info)
            
            # Update the caption in the output dataframe
            df_out.loc[idx, 'gemini_caption'] = str(caption).strip() if caption else ""
            
            # Periodically save progress
            if idx % 10 == 0:
                # Remove 'appended_caption' if it exists
                temp_df = df_out.copy()
                if 'appended_caption' in temp_df.columns:
                    temp_df = temp_df.drop(columns=['appended_caption'])
                temp_df.to_json(output_json_path, orient='records', indent=2)
                print(f"Progress saved after processing {idx} rows")
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Remove 'appended_caption' column if it exists
    if 'appended_caption' in df_out.columns:
        df_out = df_out.drop(columns=['appended_caption'])
    
    # Save the final result as JSON
    df_out.to_json(output_json_path, orient='records', indent=2)
    print(f"Processing complete. Results saved to {output_json_path}")

def process_json_with_missing_captions(json_path: str):
    """
    Process JSON file with missing Gemini captions and update the same file.
    Only processes records with empty or missing gemini_caption values.
    
    Args:
        json_path (str): Path to the JSON file with missing captions.
    """
    # Read the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {str(e)}")
        return
    
    # Filter rows with missing or empty Gemini captions
    missing_captions_mask = (df_input['gemini_caption'].isna()) | (df_input['gemini_caption'] == 'None') | (df_input['gemini_caption'].str.strip() == '')
    rows_to_process = df_input[missing_captions_mask].index.tolist()
    
    print(f"Found {len(rows_to_process)} rows with missing Gemini captions out of {len(df_input)} total rows")
    
    if len(rows_to_process) == 0:
        print("No rows with missing captions found. No changes needed.")
        return
    
    # Create a copy of the input dataframe for output
    df_out = df_input.copy()
    
    # Load the RAM Plus model once
    pretrained_path = 'ram_plus_swin_large_14m.pth'
    image_size = 384
    llm_tag_des_path = 'openimages_rare_200_llm_tag_descriptions.json'
    ram_model = load_ram_model(pretrained_path, image_size, llm_tag_des_path)
    transform = get_transform(image_size=image_size)

    # Load models outside the loop
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Create a file to track processed images
    processed_images_file = os.path.join('processed_images_gemini_json.txt')
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
    else:
        processed_images = set()
    
    # Process each row with missing captions
    for idx in tqdm(rows_to_process, desc="Processing images with missing captions", unit="image"):
        try:
            row = df_input.iloc[idx]
            image_path = os.path.join(folder, row['img'])
            
            if image_path in processed_images:
                continue
                
            # Recognize tags
            tags = recognize_tags(image_path, ram_model, transform)
            
            # Extract grounding information
            grounding_info = extract_grounding_info(image_path, tags, processor, grounding_model)
            
            # Generate captions with grounding info (with added delay)
            caption = generate_captions_using_gemini(image_path, grounding_info)
            
            # Update the caption in the output dataframe
            if caption:
                df_out.loc[idx, 'gemini_caption'] = str(caption).strip()
                
                # Add to processed images
                processed_images.add(image_path)
                with open(processed_images_file, 'a') as f:
                    f.write(image_path + '\n')
            
            # Periodically save progress
            if idx % 5 == 0:  # Save more frequently
                # Save the current state without modifying columns
                df_out.to_json(json_path, orient='records', indent=2)
                print(f"Progress saved after processing {idx} rows")
                
            # Add additional delay between processing images to avoid rate limiting
            time.sleep(1)
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            # Add a longer delay after errors to handle potential rate limiting
            time.sleep(10)
            continue
    
    # Save the final result back to the same JSON file
    df_out.to_json(json_path, orient='records', indent=2)
    print(f"Processing complete. Results saved to {json_path}")

if __name__ == "__main__":
    # output_directory = "./meme_analysis_output/"
    # output_directory = "/backup/girish_datasets/Hateful_Memes_Extended/img_bbox/"
    # process_meme_dataset(input_directory, output_directory)
    # output_file = "updated_captions_gemini.csv"
    # process_meme_dataset(output_file)
    
    # Process rows with missing Gemini captions and save as JSON
    # input_csv_path = input_file  # Use the existing input file path
    # output_json_path = "/backup/girish_datasets/Hateful_Memes_Extended/ivl_plus_gemini_captions_complete.json"
    # process_missing_gemini_captions(input_csv_path, output_json_path)
    
    # Process JSON file with missing captions
    json_path = "/backup/girish_datasets/Hateful_Memes_Extended/ivl_plus_gemini_captions_complete.json"
    process_json_with_missing_captions(json_path)
