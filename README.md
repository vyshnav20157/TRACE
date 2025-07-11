# CAMU: Context Augmentation for Meme Understanding

A modular implementation for fine-tuning CLIP models for multimodal classification tasks, specifically for the Hateful Memes dataset with extended captions. This repository includes scripts for fine-tuning both `CLIP-ViT-L/14` and `CLIP-XLM-RoBERTa-Large`.

## Project Structure

- `clip_vitL_14_ft.py` - Main training script for CLIP-ViT-L/14.
- `clip_xlm_roberta_ft.py` - Main training script for CLIP-XLM-RoBERTa-Large.
- `caption_selection.py` - Functions for selecting best captions during training.
- `loss_functions.py` - Loss functions for different ablation settings.
- `vg_caption_gen.py` - Generates rich captions for the dataset using models like RAM++, GroundingDINO, and Gemini.
- `benco_eval.py` - Evaluates the fine-tuned model on benign confounders in the test set.
- `error_analysis.py` - Performs error analysis on specific mis-predicted samples from the test set.
- `lvlm_eval.py` - Zero-shot evaluation on the test set using the InternVL2 model.
- `gemini_eval.py` - Zero-shot evaluation on the test set using the Gemini model.
- `gpt-4o-eval.py` - Zero-shot evaluation on the test set using the GPT-4o model.

## Setup

### Requirements

Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

For zero-shot evaluations using proprietary models, you will need to set your API keys as environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Caption Generation

The `vg_caption_gen.py` script is used to generate descriptive captions for the dataset. It uses a pipeline of models:
1.  **RAM++** for tag recognition.
2.  **GroundingDINO** for object detection based on tags.
3.  **Gemini** or **InternVL** to generate a final caption based on the image and grounding information.

### Running Caption Generation
Configuration, such as input/output file paths, is handled inside the script.

```bash
python vg_caption_gen.py
```

## Running Training

Training configurations (e.g., learning rate, batch size, loss configuration) are set directly within the training scripts (`clip_vitL_14_ft.py` and `clip_xlm_roberta_ft.py`).

To run training, execute one of the scripts:

**For CLIP-ViT-L/14:**
```bash
python clip_vitL_14_ft.py
```

**For CLIP-XLM-RoBERTa-Large:**
```bash
python clip_xlm_roberta_ft.py
```

### Configuration

Before running, you may want to adjust the following inside the `main()` function of the desired script:
- **Data Path**: Update the `data_path` variable to point to your dataset JSON file.
- **Batch Size & Epochs**: Modify `actual_batch_size`, `target_batch_size`, and `num_epochs`.
- **Loss Configuration**: Change the `loss_config` dictionary to enable or disable different loss components for ablation studies.

## Ablation Experiments

The code supports three main ablation settings, which can be configured by modifying the `loss_config` dictionary in the main training scripts.

Example `loss_config` in the script:
```python
loss_config = {
    'classification': True,  # Classification loss (always enabled)
    'contrastive': False,    # Contrastive loss between image and text embeddings
    'relevance': True        # Relevance loss from the caption scorer
}
```

1. **Classification + Relevance (Default)**:
   - `contrastive: False`, `relevance: True`
   - Uses classification and relevance losses. Captions are selected using the caption scorer.

2. **Classification + Contrastive**:
   - `contrastive: True`, `relevance: False`
   - Uses classification and contrastive losses. Captions are selected based on cosine similarity between image and caption embeddings.

3. **All Losses**:
   - `contrastive: True`, `relevance: True`
   - Uses classification, contrastive, and relevance losses with dynamic weighting. Caption selection is based on a combination of relevance scores and cosine similarity.

## Evaluation

This project includes scripts for evaluating both the fine-tuned models and zero-shot performance of various LVLMs.

### Fine-tuned Model Evaluation
These scripts evaluate the models trained by `clip_vitL_14_ft.py` or `clip_xlm_roberta_ft.py`. You will need to update the model path and data paths inside the scripts.

- **Benign Confounders Evaluation (`benco_eval.py`)**: Evaluates model performance on samples with benign confounders.
  ```bash
  python benco_eval.py
  ```
- **Error Analysis (`error_analysis.py`)**: Performs a detailed analysis of mis-predicted samples from the test set.
  ```bash
  python error_analysis.py
  ```

### Zero-shot LVLM Evaluation
These scripts evaluate the zero-shot capabilities of various large vision-language models on the Hateful Memes dataset. Ensure your API keys are configured.

- **InternVL (`lvlm_eval.py`)**:
  ```bash
  python lvlm_eval.py
  ```
- **Gemini (`gemini_eval.py`)**:
  ```bash
  python gemini_eval.py
  ```
- **GPT-4o (`gpt-4o-eval.py`)**:
  ```bash
  python gpt-4o-eval.py
  ```

## Dataset Format

The dataset should be a JSON file with the following structure:

```json
[
  {
    "img": "path/to/image.jpg",
    "text": "text on the meme",
    "label": 0,
    "ivl_8b_new_caption": "caption from IVL model",
    "gemini_caption": "caption from Gemini model",
    "split": "train"
  },
  ...
]
```

## Example Workflow

1. **Generate Captions**:
   - Modify and run `vg_caption_gen.py` to populate the dataset with rich captions.

2. **Configure Training**:
   - Open `clip_xlm_roberta_ft.py` or `clip_vitL_14_ft.py`.
   - Set the `data_path` to your JSON dataset.
   - Adjust `loss_config` for the desired ablation experiment.

3. **Run Training**:
   ```bash
   python clip_xlm_roberta_ft.py
   ```

4. **Evaluate the Fine-tuned Model**:
   - Modify `benco_eval.py` or `error_analysis.py` to point to your saved model checkpoint.
   - Run the evaluation script: `python benco_eval.py`.
