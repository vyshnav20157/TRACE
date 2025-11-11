import matplotlib.pyplot as plt
import numpy as np

# Data
layers = [1, 2, 3, 4, 5]

clip_vit = [0.740, 0.738, 0.745, 0.764, 0.750]
clip_xlm = [0.771, 0.764, 0.788, 0.806, 0.768]
siglip2 = [0.700, 0.666, 0.663, 0.721, 0.692]

# Set style
plt.style.use('seaborn-v0_8')  # Use updated seaborn style name

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
line1 = ax.plot(layers, clip_vit, marker='o', linewidth=2, markersize=8, 
                label='CLIP-ViT-L/14', color='#8884d8')
line2 = ax.plot(layers, clip_xlm, marker='o', linewidth=2, markersize=8, 
                label='CLIP-XLM-R-ViT-H-14', color='#82ca9d')
line3 = ax.plot(layers, siglip2, marker='o', linewidth=2, markersize=8, 
                label='SigLIP2-L/14-384', color='#ff7f0e')

# Customize grid
ax.grid(True, linestyle='--', alpha=0.7)

# Set axis labels
ax.set_xlabel('Number of Text Encoder Layers Fine-tuned', fontsize=18, labelpad=10)
ax.set_ylabel('F1 Score', fontsize=18, labelpad=10)

# Set y-axis limits
ax.set_ylim([0.65, 0.81])

# Set x-axis ticks
ax.set_xticks(layers)

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=14)

# Format y-axis ticks
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# Add legend with larger font
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=15)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot (optional)
plt.savefig('f1_comparison.png', dpi=300, bbox_inches='tight')

plt.show()
