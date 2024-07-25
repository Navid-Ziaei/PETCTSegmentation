import nibabel as nib
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


def visualize_slices(file_id, slice_index, file_names_img, file_names_label, img_path, label_path):
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
        plt.imshow(label_data[:, :, slice_index], cmap='binary', alpha=0.5)
        plt.title('Label Mask')
        plt.axis('off')

        plt.show()
    else:
        print(f"Files containing '{file_id}' not found.")

    return ctres_data, suv_data, label_data


def animate_slices(file_id, file_names_img, file_names_label, img_path, label_path, save_path=None):
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Initialize the images
        ctres_img_plot = axes[0].imshow(ctres_data[:, :, 0], cmap='gray')
        suv_img_plot = axes[1].imshow(suv_data[:, :, 0], cmap='hot')
        label_img_plot = axes[2].imshow(label_data[:, :, 0], cmap='viridis', alpha=0.5)

        # Set titles for the subplots
        axes[0].set_title('CTres Slice')
        axes[1].set_title('SUV Slice')
        axes[2].set_title('Label Mask')

        # Turn off axis for all subplots
        for ax in axes:
            ax.axis('off')

        # Update function for the animation
        def update(slice_index):
            ctres_img_plot.set_array(ctres_data[:, :, slice_index])
            suv_img_plot.set_array(suv_data[:, :, slice_index])
            label_img_plot.set_array(label_data[:, :, slice_index])
            return ctres_img_plot, suv_img_plot, label_img_plot

        # Create the animation
        ani = FuncAnimation(fig, update, frames=num_slices, interval=100, blit=True)

        if save_path:
            # Save the animation
            ani.save(save_path, writer='imagemagick')

        plt.show()
    else:
        print(f"Files containing '{file_id}' not found.")