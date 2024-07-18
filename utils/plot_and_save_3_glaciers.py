import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_and_save_figure(labels_image, labels_image_global_IA, labels_image_GLIA, hh_band):
    # Define colors
    colors = {
        -1: [1, 1, 1],  # White
        0: [1.00, 0.50, 0.00],  # Orange
        1: [0.5, 0, 0.5],  # Purple
        2: [0, 0, 1],  # Green
        3: [0, 1, 0]   # Blue
    }

    colors_IA = {
        -1: [1, 1, 1],  # White
        1: [1.00, 0.50, 0.00],  # Orange
        2: [0.50, 0.0, 0.5],  # Purple
        3: [0, 0, 1],  # Green
        4: [0, 1, 0]   # Blue
    }

    color_array = np.zeros((labels_image.shape[0], labels_image.shape[1], 3))
    color_array_global = np.zeros((labels_image_global_IA.shape[0], labels_image_global_IA.shape[1], 3))
    color_array_IA = np.zeros((labels_image_GLIA.shape[0], labels_image_GLIA.shape[1], 3))

    for label, color in colors.items():
        color_array[labels_image == label] = color

    for label, color in colors_IA.items():
        color_array_IA[labels_image_GLIA == label] = color

    for label, color in colors_IA.items():
        color_array_global[labels_image_global_IA == label] = color

    # Create subplot to display images side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Titles for subplots
    titles = ["Gaussian classifier", "Global IA", "Local IA per class"]

    # Display images
    axes[0].imshow(color_array)
    axes[0].set_title(titles[0])
    axes[0].axis('off')  # Hide axes ticks

    axes[1].imshow(color_array_global)
    axes[1].set_title(titles[1])
    axes[1].axis('off')  # Hide axes ticks

    axes[2].imshow(color_array_IA)
    axes[2].set_title(titles[2])
    axes[2].axis('off')  # Hide axes ticks

    # Create legend
    labels = ["Masked", "Firn", "Superimposed Ice", "Glacier Ice texture1", "Glacier Ice texture2"]
    patches = [Patch(color=np.array(color), label=labels[i]) for i, color in enumerate(colors.values())]
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure)
    fig.suptitle('Classifier comparison - IW scene of 01/02/2021', fontsize=16)

    # Update layout and show plot
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig('classified_glacier_scene.png')
    plt.show()
