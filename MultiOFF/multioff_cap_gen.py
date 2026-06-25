import io
import os
import random
import time
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Union
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)  # Set the default CUDA device

folder = "/backup/girish_datasets/MultiOFF/Labelled_Images/"
train_data_path = "/backup/girish_datasets/MultiOFF/Training_meme_dataset.csv"
validation_data_path = "/backup/girish_datasets/MultiOFF/Validation_meme_dataset.csv"
test_data_path = "/backup/girish_datasets/MultiOFF/Testing_meme_dataset.csv"

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

def recognize_tags(image: Union[str, Image.Image], model: ram_plus, transform) -> List[str]:
    """
    Recognize tags from the image using the provided RAM Plus model.
    
    Args:
        image (Union[str, Image.Image]): Path to the input image or PIL Image object.
        model (ram_plus): Pre-loaded RAM Plus model.
        transform: Image transform function.
    
    Returns:
        List[str]: List of recognized tags.
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # If it's already a PIL Image, just use it directly
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            tags = inference(image_tensor, model)

        return tags[0]
    except Exception as e:
        print(f"Error in recognize_tags: {str(e)}")
        raise

def extract_grounding_info(image: Union[str, Image.Image], tags: str, processor, model) -> Dict[str, List[Tuple[float, float, float, float, float]]]:
    """
    Extract boundary boxes for text and other entities in the image using GroundingDINO.
    
    Args:
        image (Union[str, Image.Image]): Path to the input image or PIL Image object.
        tags (str): String of tags separated by ' | '.
        processor: Pre-loaded GroundingDINO processor.
        model: Pre-loaded GroundingDINO model.
    
    Returns:
        Dict[str, List[Tuple[float, float, float, float, float]]]: Dictionary mapping entity labels to their bounding boxes.
    """
    try:
        # If image is a file path, open it
        if isinstance(image, str):
            image = Image.open(image)
        # If it's already a PIL Image, use it directly
        # Ensure it's in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
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
            print(f"No objects detected for image")
        
        return grounding_info
    except Exception as e:
        print(f"Error in extract_grounding_info: {str(e)}")
        raise

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

def load_image(image: Union[str, Image.Image], input_size=448):
    """
    Load and preprocess an image for the caption model.
    
    Args:
        image (Union[str, Image.Image]): Path to the image file or PIL Image object
        input_size (int): Size to resize the image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # If it's already a PIL Image, ensure it's in RGB mode
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(device)
        return pixel_values
    except Exception as e:
        print(f"Error in load_image: {str(e)}")
        raise

def generate_captions(image: Union[str, Image.Image], grounding_info: Dict[str, List[Tuple[float, float, float, float, float]]], model, tokenizer) -> str:
    """
    Generate captions using InternVL2-8B model, incorporating grounding information.
    
    Args:
        image (Union[str, Image.Image]): Path to the input image or PIL Image object.
        grounding_info (Dict[str, List[Tuple[float, float, float, float, float]]]): Grounding information from GroundingDINO.
        model: Pre-loaded InternVL2-8B model.
        tokenizer: Pre-loaded InternVL2-8B tokenizer.
    
    Returns:
        str: Generated caption.
    """
    try:
        # Load and preprocess the image
        pixel_values = load_image(image).to(torch.bfloat16).to(device)

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

        prompt = f"""<image>
        {grounding_prompt}
        Task: Analyze this meme image and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.  

        The caption should:
        - **Describe the main visual elements** (people, objects, and setting)  
        - **Mention key identifying features** (race, gender, nationality, disability, etc.) if relevant  
        - **Summarize the text overlay** (if short) or **explain its meaning** concisely  
        - **Describe any implied humor, stereotypes, sarcasm, or exaggerated statements**  
        - **Identify if the meme references a specific group or topic that may be controversial**  
        - **Avoid subjective labels like 'offensive' or 'non-offensive'**  

        Additionally, **if the meme conveys strong emotions (mockery, sarcasm, contempt, etc.), describe how the text and image interact to create this impression.**  

        Format the response as:  
        Caption: [Generated caption here]
        """

        # Generate caption
        generation_config = dict(max_new_tokens=4096, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)

        # Extract caption from response
        caption = response.split("Caption:")[1].strip()
        print(f"Caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error in generate_captions: {str(e)}")
        raise

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
    Task: Analyze this meme image and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.  

    The caption should:
    - **Describe the main visual elements** (people, objects, and setting)  
    - **Mention key identifying features** (race, gender, nationality, disability, etc.) if relevant  
    - **Summarize the text overlay** (if short) or **explain its meaning** concisely  
    - **Describe any implied humor, stereotypes, sarcasm, or exaggerated statements**  
    - **Identify if the meme references a specific group or topic that may be controversial**  
    - **Avoid subjective labels like 'offensive' or 'non-offensive'**  

    Additionally, **if the meme conveys strong emotions (mockery, sarcasm, contempt, etc.), describe how the text and image interact to create this impression.**  

    Format the response as:  
    Caption: [Generated caption here]
    """

    generation_config = dict(max_new_tokens=4096, pad_token_id = tokenizer.pad_token_id)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    caption = response.split("Caption:")[1].strip()
    return caption

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

def load_image_for_gemini(image: Union[str, Image.Image]) -> Image.Image:
    """
    Load and prepare image for Gemini model.
    
    Args:
        image (Union[str, Image.Image]): Path to the image file or PIL Image object
        
    Returns:
        Image.Image: Processed PIL Image ready for Gemini
    """
    try:
        # If it's a file path, open it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to JPEG format in memory to avoid WebP encoding issues
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return Image.open(img_byte_arr)
    except Exception as e:
        print(f"Error in load_image_for_gemini: {str(e)}")
        raise

def generate_captions_using_gemini(image: Union[str, Image.Image], grounding_info: Dict[str, List[Tuple[float, float, float, float, float]]]) -> str:
    """
    Generate a caption for an image using the Gemini 2.0 Flash model.

    Args:
        image (Union[str, Image.Image]): Path to the input image or PIL Image object.
        grounding_info (Dict[str, List[Tuple[float, float, float, float, float]]]): Grounding information from GroundingDINO.
    
    Returns:
        str: The generated caption.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Load and preprocess the image
        processed_image = load_image_for_gemini(image)
        if processed_image is None:
            raise ValueError("Failed to process image")

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
        Task: Analyze this meme image and generate a **single caption** (under 77 tokens) suitable for CLIP fine-tuning.  

        The caption should:
        - **Describe the main visual elements** (people, objects, and setting)  
        - **Mention key identifying features** (race, gender, nationality, disability, etc.) if relevant  
        - **Summarize the text overlay** (if short) or **explain its meaning** concisely  
        - **Describe any implied humor, stereotypes, sarcasm, or exaggerated statements**  
        - **Identify if the meme references a specific group or topic that may be controversial**  
        - **Avoid subjective labels like 'offensive' or 'non-offensive'**  

        Additionally, **if the meme conveys strong emotions (mockery, sarcasm, contempt, etc.), describe how the text and image interact to create this impression.**  

        Format the response as:  
        Caption: [Generated caption here]
        """

        # Generate caption with retry logic
        response = predict_with_retry(model, prompt, processed_image)
        if response is None:
            raise ValueError("Failed to generate caption")
            
        response_text = response.text.strip()
        if "Caption:" in response_text:
            caption = response_text.split("Caption:", 1)[1].strip()
        else:
            caption = response_text
        return caption
    except Exception as e:
        print(f"Error in generate_captions_using_gemini: {str(e)}")
        raise

def load_and_convert_image(image_path: str) -> Image.Image:
    """
    Load an image and convert it to RGB format, handling different image formats and color modes.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Image.Image: PIL Image in RGB format
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert grayscale to RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        # Convert RGBA to RGB
        elif img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        # Convert indexed (P) to RGB
        elif img.mode == 'P':
            img = img.convert('RGB')
        # Convert any other mode to RGB
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise

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

    df_train = pd.read_csv(train_data_path)
    df_train['split'] = 'train'
    df_val = pd.read_csv(validation_data_path)
    df_val['split'] = 'val'
    df_test = pd.read_csv(test_data_path)
    df_test['split'] = 'test'
    df = pd.concat([df_train, df_val, df_test])
    image_paths = [os.path.join(folder, f) for f in df['image_name']]
    
    # Load models
    pretrained_path = 'ram_plus_swin_large_14m.pth'
    image_size = 384
    llm_tag_des_path = 'openimages_rare_200_llm_tag_descriptions.json'
    ram_model = load_ram_model(pretrained_path, image_size, llm_tag_des_path)
    transform = get_transform(image_size=image_size)
    
    # Initialize columns for captions if they don't exist
    # if 'ivl_8b_new_caption' not in df.columns:
    #     df['ivl_8b_new_caption'] = ''
    if 'gemini_caption' not in df.columns:
        df['gemini_caption'] = ''

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
    processed_images_file = os.path.join('processed_multioff_gemini_images.txt')
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
    else:
        processed_images = set()

    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images", unit="image")):
        if image_path in processed_images:
            continue
        try:
            # Load and convert image
            img_input = load_and_convert_image(image_path)
            
            try:
                # Pass the PIL Image directly
                tags = recognize_tags(img_input, ram_model, transform)
            except Exception as e:
                print(f"Error recognizing tags for {image_path}: {str(e)}")
                continue

            # Extract grounding information using the PIL Image directly
            try:
                grounding_info = extract_grounding_info(img_input, tags, processor, grounding_model)
            except Exception as e:
                print(f"Error extracting grounding info for {image_path}: {str(e)}")
                continue
                
            # Generate captions with grounding info using the same image input
            try:
                # caption = generate_captions(img_input, grounding_info, caption_model, tokenizer)
                caption = generate_captions_using_gemini(image_path, grounding_info)  # Pass the file path instead of PIL Image
                if caption is None:
                    print(f"Failed to generate caption for {image_path}")
                    continue
            except Exception as e:
                print(f"Error generating caption for {image_path}: {str(e)}")
                continue
            
            # Update the caption in the dataframe
            df.loc[i, 'gemini_caption'] = str(caption)
            
            # Save after each successful caption generation
            df.to_csv(output_dir, index=False)

            # Save the processed image path to the text file
            processed_images.add(image_path)
            with open(processed_images_file, 'a') as f:
                f.write(image_path + '\n')
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

if __name__ == "__main__":
    # output_directory = "./meme_analysis_output/"
    # output_directory = "/backup/girish_datasets/Hateful_Memes_Extended/img_bbox/"
    # process_meme_dataset(input_directory, output_directory)
    output_file = "multioff_updated_gemini_captions.csv"
    process_meme_dataset(output_file)
