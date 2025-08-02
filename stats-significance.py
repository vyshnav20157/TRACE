import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
import json

# Load predictions from JSON files
def load_predictions_from_json(filepath):
    """Load labels and predictions from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data['labels']), np.array(data['predictions'])

# Load actual model predictions from JSON files
labels, preds_xlm = load_predictions_from_json('clip_xlm_preds.json')
_, preds_vit = load_predictions_from_json('clip_vitl14_preds.json')
_, preds_siglip = load_predictions_from_json('siglip2_preds.json')

# Ensure all prediction arrays have the same length as labels
assert len(labels) == len(preds_xlm) == len(preds_vit) == len(preds_siglip), "All arrays must have the same length"

# --- Create a dictionary to hold predictions ---
model_preds = {
    'CLIP-XLM-T': preds_xlm,
    'CLIP-ViT-L': preds_vit,
    'SigLIP2': preds_siglip
}

model_names = list(model_preds.keys())

# --- Perform pairwise McNemar's tests ---
for model1_name, model2_name in combinations(model_names, 2):
    print(f"--- Comparing {model1_name} and {model2_name} ---")
    
    preds1 = model_preds[model1_name]
    preds2 = model_preds[model2_name]

    # Create the 2x2 contingency table for McNemar's test
    # Table format:
    # [[model1_correct_model2_correct, model1_correct_model2_incorrect],
    #  [model1_incorrect_model2_correct, model1_incorrect_model2_incorrect]]
    
    n11 = np.sum((preds1 == labels) & (preds2 == labels))
    n10 = np.sum((preds1 == labels) & (preds2 != labels))
    n01 = np.sum((preds1 != labels) & (preds2 == labels))
    n00 = np.sum((preds1 != labels) & (preds2 != labels))
    
    contingency_table = [[n11, n10],
                         [n01, n00]]

    # Perform the test
    result = mcnemar(contingency_table, exact=False, correction=True)

    print(f"Contingency Table (Correct/Incorrect):\n{np.array(contingency_table)}")
    print(f"Statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")

    # Interpretation
    alpha = 0.05
    if result.pvalue < alpha:
        print(f"The difference between {model1_name} and {model2_name} is statistically significant.\n")
    else:
        print(f"The difference between {model1_name} and {model2_name} is not statistically significant.\n")