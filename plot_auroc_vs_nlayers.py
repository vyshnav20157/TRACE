import matplotlib.pyplot as plt
import numpy as np

# Data
layers = [1, 2, 3, 4]
clip_vit = [0.787, 0.801, 0.808, 0.812]
clip_xlm = [0.795, 0.807, 0.791, 0.819]

# Set style
plt.style.use('seaborn-v0_8')  # Use updated seaborn style name

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
line1 = ax.plot(layers, clip_vit, marker='o', linewidth=2, markersize=8, 
                label='CLIP-ViT-L/14', color='#8884d8')
line2 = ax.plot(layers, clip_xlm, marker='o', linewidth=2, markersize=8, 
                label='CLIP-xlm-roberta-large-ViT-H-14', color='#82ca9d')

# Customize grid
ax.grid(True, linestyle='--', alpha=0.7)

# Set axis labels
ax.set_xlabel('Number of Text Encoder Layers Fine-tuned', fontsize=12, labelpad=10)
ax.set_ylabel('AUROC Score', fontsize=12, labelpad=10)

# Set y-axis limits
ax.set_ylim([0.78, 0.83])

# Set x-axis ticks
ax.set_xticks(layers)

# Format y-axis ticks
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# Add legend
ax.legend(bbox_to_anchor=(1.05, 0.5), loc='best', frameon=True, 
          fancybox=True, shadow=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Optional: Add title
# plt.title('CLIP Models AUROC Comparison', pad=20, fontsize=14)

# Save plot (optional)
plt.savefig('auroc_comparison.png', dpi=300, bbox_inches='tight')

plt.show()