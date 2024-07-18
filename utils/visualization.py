import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from datetime import datetime



def save_confusion_matrix(cm, target_names, title, accuracy, filename):
    cm_percentage = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(9, 6))
    sns.heatmap(cm_percentage, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.text(0.5, -0.1, f'Training accuracy: {accuracy}', size=12, ha='center', transform=plt.gca().transAxes)
    plt.savefig(filename)
    plt.close()

def compare_confusion_matrices(y_test, y_pred, y_test_IA, y_pred_global_IA, y_pred_GLIA, target_names, accuracies, save_dir=None, std=False):
    # Compute confusion matrices
    cm_Gaussian = confusion_matrix(y_test, y_pred)
    cm_GLIA = confusion_matrix(y_test_IA, y_pred_GLIA)
    cm_global = confusion_matrix(y_test_IA, y_pred_global_IA)

    # Save individual confusion matrices
    # Create directory based on target names
    target_names_str = '_'.join(target_names)
    save_dir = os.path.join(save_dir, target_names_str)
    if std: 
        save_dir = save_dir + "_std"
    os.makedirs(save_dir, exist_ok=True)
    
    if save_dir:
        save_confusion_matrix(cm_Gaussian, target_names, 'Gaussian Classifier Confusion Matrix', accuracies[0], f'{save_dir}/cm_gaussian.png')
        save_confusion_matrix(cm_global, target_names, 'Global IA Confusion Matrix', accuracies[1],f'{save_dir}/cm_global.png')
        save_confusion_matrix(cm_GLIA, target_names, 'Local IA Classifier Confusion Matrix', accuracies[2],f'{save_dir}/cm_glia.png')

    # Normalize confusion matrices
    cm_Gaussian_percentage = cm_Gaussian / cm_Gaussian.sum(axis=1, keepdims=True)
    cm_GLIA_percentage = cm_GLIA / cm_GLIA.sum(axis=1, keepdims=True)
    cm_global_percentage = cm_global / cm_global.sum(axis=1, keepdims=True)

    # Plot confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    sns.heatmap(cm_Gaussian_percentage, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax[0], vmin=0, vmax=1, cbar=False)
    ax[0].set_title('Gaussian Classifier')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    sns.heatmap(cm_global_percentage, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax[1], vmin=0, vmax=1, cbar=False)
    ax[1].set_title('Global IA')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')

    sns.heatmap(cm_GLIA_percentage, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax[2], vmin=0, vmax=1)
    ax[2].set_title('Local IA Classifier')
    ax[2].set_xlabel('Predicted')
    ax[2].set_ylabel('True')

    plt.suptitle('Confusion matrices - values are predicted scenes/true scenes', size=16)
    plt.tight_layout()

    # Include accuracy in the plots
    ax[0].text(0.5, -0.1, f'Training Accuracy: {round(accuracy_score(y_test, y_pred), 2)}', size=12, ha='center', transform=ax[0].transAxes)
    ax[1].text(0.5, -0.1, f'Training Accuracy: {round(accuracy_score(y_test_IA, y_pred_global_IA), 2)}', size=12, ha='center', transform=ax[1].transAxes)
    ax[2].text(0.5, -0.1, f'Training Accuracy: {round(accuracy_score(y_test_IA, y_pred_GLIA), 2)}', size=12, ha='center', transform=ax[2].transAxes)

    if save_dir:
        combined_path = f'{save_dir}/combined_confusion_matrices.png'
        plt.savefig(combined_path)

    plt.show()

def plot_classification_results(labels_image, labels_image_global, labels_image_IA, titles, target_names, save_dir, scene_path, std):
    color_array = np.zeros((labels_image.shape[0], labels_image.shape[1], 3))
    color_array_IA = np.zeros((labels_image_IA.shape[0], labels_image_IA.shape[1], 3))
    color_array_global = np.zeros((labels_image_global.shape[0], labels_image_global.shape[1], 3))
    zone_colors = {
        "masked": [1, 1, 1],  # White
        "crevasse": [1.00, 0, 0.00],  # Orange
        "firn": [1, 0.5, 0],  # Purple
        "superimposedice": [0.5, 0, 0.5],  # Blue
        "glacierice_textured1": [0, 0, 1],  # Green
        "glacierice_textured2": [0, 1, 0]  # Red
    }

    # Define label mappings for images
    label_mappings = {
        0: "firn",
        1: "superimposedice",
        2: "glacierice_textured1",
        3: "glacierice_textured2",
        4: "crevasse",
        -1: "masked"
    }

    label_mappings_IA = {
        1: "firn",
        2: "superimposedice",
        3: "glacierice_textured1",
        4: "glacierice_textured2",
        5: "crevasse",
        -1: "masked"
    }

    for label, zone in label_mappings.items():
        color_array[labels_image == label] = zone_colors[zone]

    for label, zone in label_mappings_IA.items():
        color_array_IA[labels_image_IA == label] = zone_colors[zone]
        color_array_global[labels_image_global == label] = zone_colors[zone]

    labels = ["Masked", "Crevasse", "Firn", "Superimposed Ice", "Glacier Ice texture1", "Glacier Ice texture2"]
    patches = [Patch(color=np.array(zone_colors[label]), label=label.title().replace("_", " ")) for label in zone_colors.keys()]
    images = [color_array, color_array_global, color_array_IA]

    target_names_str = '_'.join(target_names)
    save_dir = os.path.join(save_dir, target_names_str)
    if std: 
        save_dir = save_dir + "_std"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = scene_path.split('_')[5]
    parsed_date = datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
    formatted_date = parsed_date.strftime("%d-%m-%Y")
    for idx, (img, title) in enumerate(zip(images, titles)):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        fig.suptitle(f'Classifier comparison - {formatted_date}', fontsize=16)
        
        if save_dir:
            image_path = f'{save_dir}/{title.replace(" ", "_").lower()}.png'
            plt.savefig(image_path)
            print(f"Figure saved as {image_path}")
            plt.close()
        
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(color_array)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    axes[1].imshow(color_array_global)
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    axes[2].imshow(color_array_IA)
    axes[2].set_title(titles[2])
    axes[2].axis('off')

    
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure)
    fig.suptitle(f'Classifier comparison - {formatted_date}', fontsize=16)
    
    if save_dir:
        combined_path = f'{save_dir}/glacier_classification_comparison.png'
        plt.savefig(combined_path)
        print("Figure saved")
    
    plt.tight_layout()
    plt.show()
