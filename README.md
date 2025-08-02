# TRACE: Textual Relevance Augmentation and Contextual Encoding for Multimodal Hate Detection

This repository contains the implementation for **TRACE** (Textual Relevance Augmentation and Contextual Encoding), a methodology for fine-tuning CLIP models for multimodal hate speech detection. It includes scripts for two primary tasks: the Facebook Hateful Memes (FHM) challenge and the MultiOFF (Multimodal Offensive Language) dataset.

## Project Structure

All files in the main directory are implementations for the Facebook Hateful Memes (FHM) dataset.

- `clip_vitL_14_ft.py`: Main training script for CLIP-ViT-L/14 on FHM.
- `clip_xlm_roberta_ft.py`: Main training script for CLIP-XLM-RoBERTa-Large on FHM.
- `caption_selection.py`: Functions for selecting best captions during training.
- `loss_functions.py`: Loss functions for different ablation settings.
- `vg_caption_gen.py`: Generates rich captions for the FHM dataset.
- `benco_eval.py`: Evaluates the fine-tuned model on benign confounders in the FHM test set.
- `error_analysis.py`: Performs error analysis on specific mis-predicted samples from the FHM test set.
- `lvlm_eval.py`: Zero-shot evaluation on the FHM test set using the InternVL2 model.
- `gemini_eval.py`: Zero-shot evaluation on the FHM test set using the Gemini model.
- `gpt-4o-eval.py`: Zero-shot evaluation on the FHM test set using the GPT-4o model.
- `stats_significance.py`: Performs statistical significance tests (e.g., McNemar's test) between model predictions.
- `MultiOFF/`: Contains scripts for the MultiOFF (offensive meme identification) dataset.
    - `clip_roberta_multioff.py`: Training script for CLIP+RoBERTa on MultiOFF.
    - `multioff_proj_ft.py`: Training script with a projection layer for MultiOFF.
    - `multioff_cap_gen.py`: Caption generation for the MultiOFF dataset.
    - `multioff_eval.py`: Evaluation script for models trained on MultiOFF.

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

### FHM Dataset
The `vg_caption_gen.py` script is used to generate descriptive captions for the FHM dataset. It uses a pipeline of models:
1.  **RAM++** for tag recognition.
2.  **GroundingDINO** for object detection based on tags.
3.  **Gemini** or **InternVL** to generate a final caption based on the image and grounding information.

#### Running Caption Generation
Configuration, such as input/output file paths, is handled inside the script.

```bash
python vg_caption_gen.py
```

### MultiOFF Dataset
Caption generation for the MultiOFF dataset is handled by `MultiOFF/multioff_cap_gen.py`.

## Running Training

Training configurations are set directly within the training scripts.

### Facebook Hateful Memes (FHM)
To run training on the FHM dataset, execute one of the scripts:

**For CLIP-ViT-L/14:**
```bash
python clip_vitL_14_ft.py
```

**For CLIP-XLM-RoBERTa-Large:**
```bash
python clip_xlm_roberta_ft.py
```

### MultiOFF Dataset
To train a model on the MultiOFF dataset, use the scripts within the `MultiOFF/` directory.

**For CLIP-RoBERTa:**
```bash
python MultiOFF/clip_roberta_multioff.py
```

### Configuration

Before running, you may want to adjust the following inside the `main()` function of the desired script:
- **Data Path**: Update the `data_path` variable to point to your dataset JSON file.
- **Batch Size & Epochs**: Modify `actual_batch_size`, `target_batch_size`, and `num_epochs`.
- **Loss Configuration**: Change the `loss_config` dictionary to enable or disable different loss components for ablation studies.

## Ablation Experiments

The code for the FHM task supports three main ablation settings, which can be configured by modifying the `loss_config` dictionary in the main training scripts.

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

### Fine-tuned Model Evaluation
Update the model path and data paths inside the scripts before running.

- **FHM Benign Confounders (`benco_eval.py`)**: Evaluates model performance on samples with benign confounders from the FHM dataset.
  ```bash
  python benco_eval.py
  ```
- **FHM Error Analysis (`error_analysis.py`)**: Performs a detailed analysis of mis-predicted samples from the FHM test set.
  ```bash
  python error_analysis.py
  ```
- **MultiOFF Evaluation (`multioff_eval.py`)**: Evaluates models trained on the MultiOFF dataset.
  ```bash
  python MultiOFF/multioff_eval.py
  ```

### Zero-shot LVLM Evaluation (FHM)
These scripts evaluate the zero-shot capabilities of various large vision-language models on the FHM dataset. Ensure your API keys are configured.

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

### Statistical Significance
To run statistical significance tests between two models' prediction files, use `stats_significance.py`. You will need to provide paths to the prediction files inside the script.
```bash
python stats_significance.py
```

## Dataset Format

### FHM Dataset
The FHM dataset should be a JSON file with the following structure:

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

### MultiOFF Dataset
The data format for the MultiOFF dataset is handled by the scripts within the `MultiOFF/` directory.

## Example Workflow (FHM Task)

1. **Generate Captions**:
   - Modify and run `vg_caption_gen.py` to populate the FHM dataset with rich captions.

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
