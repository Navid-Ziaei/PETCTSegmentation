import nibabel as nib
import matplotlib.pyplot as plt
import os
import seaborn as sns

import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.colors import Normalize

import pandas as pd


def plot_results(df_results, save_dir='results/'):
    """
    Generates boxplots with scatter points for each metric and saves the plots.
    Also calculates and saves the mean and standard deviation of the metrics.

    Parameters:
        df_results (pd.DataFrame): DataFrame containing evaluation metrics.
        save_dir (str): Directory to save the plots and metrics.
    """


    # Calculate mean and standard deviation
    metrics = ['false_neg_vol', 'false_pos_vol', 'dice_sc']
    summary_stats = {
        'metric': [],
        'mean': [],
        'std': []
    }

    for metric in metrics:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        summary_stats['metric'].append(metric)
        summary_stats['mean'].append(mean_val)
        summary_stats['std'].append(std_val)
        print(f"{metric} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_results, y=metric, color='lightblue')
        sns.stripplot(data=df_results, y=metric, color='black', jitter=True)
        plt.title(f'Boxplot of {metric}')
        plt.ylabel(metric)
        plt.xlabel('Patients')
        plt.grid(True)

        # Save the plot
        plot_save_path = os.path.join(save_dir, f'{metric}_boxplot.png')
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Convert summary stats to DataFrame and save as CSV
    df_summary = pd.DataFrame(summary_stats)
    summary_save_path = os.path.join(save_dir, 'summary_stats.csv')
    df_summary.to_csv(summary_save_path, index=False)

    print(f"Plots and summary statistics saved in {save_dir}")






def visualize_slices(file_id, slice_index, file_names_img, file_names_label, img_path, label_path):
    """
    Visualize specific slices of CTres, SUV, and Label images.

    Parameters:
    file_id (str): The identifier of the file to visualize.
    slice_index (int): The index of the slice to visualize.
    file_names_img (list): List of image file names.
    file_names_label (list): List of label file names.
    img_path (str): Path to the image files.
    label_path (str): Path to the label files.

    Returns:
    tuple: A tuple containing the CTres data, SUV data, and label data.
    """
    ctres_file = None
    suv_file = None
    label_file = None

    # Identify image files containing the specified file_id
    for file_name in file_names_img:
        if file_id in file_name:
            if '_0000' in file_name:
                ctres_file = file_name
            elif '_0001' in file_name:
                suv_file = file_name

    # Identify label file containing the specified file_id
    for file_name in file_names_label:
        if file_id in file_name:
            label_file = file_name
            break

    if ctres_file and suv_file and label_file:
        # Load the files using nibabel
        ctres_img = nib.load(os.path.join(img_path, ctres_file))
        suv_img = nib.load(os.path.join(img_path, suv_file))
        label_img = nib.load(os.path.join(label_path, label_file))

        # Get the data from the images
        ctres_data = ctres_img.get_fdata()
        suv_data = suv_img.get_fdata()
        label_data = label_img.get_fdata()

        # Plot the CTres slice
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(ctres_data[:, :, slice_index], cmap='gray')
        plt.title('CTres Slice')
        plt.axis('off')

        # Plot the SUV slice
        plt.subplot(1, 3, 2)
        plt.imshow(suv_data[:, :, slice_index], cmap='hot')
        plt.title('SUV Slice')
        plt.axis('off')

        # Plot the Label slice
        plt.subplot(1, 3, 3)
        plt.imshow(label_data[:, :, slice_index], cmap='binary', alpha=1)
        plt.title('Label Mask')
        plt.axis('off')

        plt.show()
    else:
        print(f"Files containing '{file_id}' not found.")

    return ctres_data, suv_data, label_data

def animate_slices(file_id, file_names_img, file_names_label, img_path, label_path, save_path=None):
    """
    Animate slices of CTres, SUV, and Label images.

    Parameters:
    file_id (str): The identifier of the file to animate.
    file_names_img (list): List of image file names.
    file_names_label (list): List of label file names.
    img_path (str): Path to the image files.
    label_path (str): Path to the label files.
    save_path (str, optional): Path to save the animation. Defaults to None.

    Returns:
    HTML: HTML object to display the animation in Jupyter Notebook.
    """
    ctres_file = None
    suv_file = None
    label_file = None

    # Identify image files containing the specified file_id
    for file_name in file_names_img:
        if file_id in file_name:
            if '_0000' in file_name:
                ctres_file = os.path.join(img_path, file_name)
            elif '_0001' in file_name:
                suv_file = os.path.join(img_path, file_name)

    # Identify label file containing the specified file_id
    for file_name in file_names_label:
        if file_id in file_name:
            label_file = os.path.join(label_path, file_name)
            break

    if ctres_file and suv_file and label_file:
        # Load the files using nibabel
        ctres_img = nib.load(ctres_file)
        suv_img = nib.load(suv_file)
        label_img = nib.load(label_file)

        # Get the data from the images
        ctres_data = ctres_img.get_fdata()
        suv_data = suv_img.get_fdata()
        label_data = label_img.get_fdata()

        # Define the number of slices
        num_slices = ctres_data.shape[2]

        # Create a figure and axis for the animation
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

        # Initialize the images
        ctres_img_plot = axes[0].imshow(ctres_data[:, :, 50], cmap='gray', norm=Normalize())
        suv_img_plot = axes[1].imshow(suv_data[:, :, 50], cmap='hot', norm=Normalize())
        label_img_plot = axes[2].imshow(label_data[:, :, 50], cmap='binary', alpha=0.5, norm=Normalize())

        # Set titles for the subplots
        axes[0].set_title('CTres Slice')
        axes[1].set_title('SUV Slice')
        axes[2].set_title('Label Mask')

        plt.tight_layout()

        # Turn off axis for all subplots
        for ax in axes:
            ax.axis('off')

        # Update function for the animation
        def update(slice_index):
            # Update the color range for each slice
            ctres_img_plot.set_array(ctres_data[:, :, slice_index])
            ctres_img_plot.set_norm(Normalize(vmin=np.min(ctres_data[:, :, slice_index]),
                                              vmax=np.max(ctres_data[:, :, slice_index])))

            suv_img_plot.set_array(suv_data[:, :, slice_index])
            suv_img_plot.set_norm(Normalize(vmin=np.min(suv_data[:, :, slice_index]),
                                            vmax=np.max(suv_data[:, :, slice_index])))

            label_img_plot.set_array(label_data[:, :, slice_index])
            label_img_plot.set_norm(Normalize(vmin=np.min(label_data[:, :, slice_index]),
                                              vmax=np.max(label_data[:, :, slice_index])))

            return ctres_img_plot, suv_img_plot, label_img_plot

        # Create the animation
        ani = FuncAnimation(fig, update, frames=num_slices, interval=100, blit=True)

        if save_path:
            # Save the animation
            ani.save(save_path + f'{file_id}.gif', writer='imagemagick')

        # Display the animation in Jupyter Notebook
        return HTML(ani.to_jshtml())

    else:
        print(f"Files containing '{file_id}' not found.")
        return None

def plot_training_history(history, save_path='training_history.png'):
    """
    Plots the training and validation metrics stored in the history dictionary.
    The plots are saved as a PNG image.

    Parameters:
        history (dict): A dictionary containing the training and validation metrics.
        save_path (str): Path to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Loss
    axs[0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axs[0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    axs[1].plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Dice Score
    axs[2].plot(epochs, history['train_dice'], 'bo-', label='Training Dice Score')
    axs[2].plot(epochs, history['val_dice'], 'ro-', label='Validation Dice Score')
    axs[2].set_title('Dice Score')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Dice Score')
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_histogram_of_statistics(data, data_type, save_path):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Histogram 1: min of data
    sns.histplot(data['min'], bins=40, kde=True, color='skyblue', ax=axs[0])
    axs[0].set_title(f'Histogram of Min Values for {data_type} Images', fontsize=18, weight='bold')
    axs[0].set_xlabel('Min Value', fontsize=14)
    axs[0].set_ylabel('Number of Patients', fontsize=14)

    # Histogram 2: mean of data
    sns.histplot(data['mean'], bins=40, kde=True, color='lightgreen', ax=axs[1])
    axs[1].set_title(f'Histogram of Mean Values for {data_type} Images', fontsize=18, weight='bold')
    axs[1].set_xlabel('Mean Value', fontsize=14)
    axs[1].set_ylabel('Number of Patients', fontsize=14)

    # Histogram 3: max of data
    sns.histplot(data['max'], bins=40, kde=True, color='salmon', ax=axs[2])
    axs[2].set_title(f'Histogram of Max Values for {data_type} Images', fontsize=18, weight='bold')
    axs[2].set_xlabel('Max Value', fontsize=14)
    axs[2].set_ylabel('Number of Patients', fontsize=14)

    # Adjust layout for better spacing and aesthetics
    plt.tight_layout()

    # Save the figure for scientific presentation with a higher DPI
    plt.savefig(save_path + f'histograms_of_{data_type}.png', dpi=300)