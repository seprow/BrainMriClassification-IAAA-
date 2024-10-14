import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from typing import Optional
import pydicom
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider


def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  def fn(SLICE):
    plt.figure(figsize=(7,7))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
  """
  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  def fn(SLICE):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))

    ax1.set_title('Before', fontsize=15)
    ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

    ax2.set_title('After', fontsize=15)
    ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

    plt.tight_layout()

  interact(fn, SLICE=(0, arr_before.shape[0]-1))


def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')



def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape

  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)

  interact(fn, SLICE=(0, arr.shape[0]-1))


#############################################################################################

def create_animation(directory_path, output_file='dicom_animation.mp4'):
    """
    Creates an animation from a DICOM series stored in the specified directory.
    
    HTML(create_animation(directory_path).to_jshtml()) --> Usage
    """
    # Load and sort the DICOM series
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory_path))
    volume = sitk.ReadImage(series_file_names, sitk.sitkInt32)
    volume = np.array(sitk.GetArrayFromImage(volume), dtype=np.float32)
    image_slices = volume

    # Setup the figure and axes
    fig, ax = plt.subplots()
    
    def update(frame_index):
        """Update function for each frame in the animation."""
        ax.clear()
        ax.imshow(image_slices[frame_index], cmap='gray')
        ax.set_title(f"Slice {frame_index + 1}")
        ax.axis('off')

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=image_slices.shape[0], repeat=False)

    # Save the animation if required (output_file is provided)
    if output_file:
        ani.save(output_file, writer='ffmpeg')

    return ani



def visualize_dicom_3d(pixel_volume, title=None):
    
    nb_frames = len(pixel_volume)  # Number of frames
    r, c = pixel_volume[0].shape   # Shape of each frame (rows, cols)

    # Create frames for each slice in the volume
    frames = [
        go.Frame(
            data=go.Surface(
                z=(k + 1) * np.ones((r, c)),  # Set a constant z value for each frame
                surfacecolor=pixel_volume[k],  # Surface color based on pixel data
                colorscale='gray',  # Grayscale color mapping
                cmin=pixel_volume.min(), cmax=pixel_volume.max()    # Color scale limits
            ),
            name=str(k)
        )
        for k in range(nb_frames)
    ]

    # Initial surface setup (first frame)
    initial_surface = go.Surface(
        z=np.ones((r, c)), surfacecolor=pixel_volume[0],
        colorscale='gray', cmin=pixel_volume.min(), cmax=pixel_volume.max()
    )

    # Create the figure and add the initial surface
    fig = go.Figure(data=[initial_surface], frames=frames)

    # Update function for the slider
    def update_slider(index):
        fig.data[0].surfacecolor = pixel_volume[index]

    # Create the slider for interacting with frames
    interact(update_slider, index=IntSlider(min=0, max=nb_frames - 1, step=1, value=0))

    # Layout settings for the figure
    fig.update_layout(
        title=title, width=800, height=800,
        scene=dict(
            zaxis=dict(range=[0, nb_frames], autorange=False)
        ),
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True, fromcurrent=True))]
            )]
        )]
    )

    # Display the figure
    fig.show()

def pydicom_dicom_reader(study_path) -> Optional[dict]:
        
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(study_path))

        headers = [pydicom.dcmread(str(fn), stop_before_pixels=True) for fn in series_file_names]
        volume = sitk.ReadImage(series_file_names, sitk.sitkInt32)
        volume_array = np.array(sitk.GetArrayFromImage(volume), dtype=np.float32)

        slice_number_tag = 'InstanceNumber' if all(i.get('InstanceNumber') is not None for i in headers) else 'ImageNumber' if all(i.get('ImageNumber') is not None for i in headers) else None

        if slice_number_tag:
            sorted_indices = np.argsort([int(header.get(slice_number_tag, 0)) for header in headers])
            sorted_volume_array = volume_array[sorted_indices]
            sorted_headers = [headers[i] for i in sorted_indices]
            sorted_file_names = [series_file_names[i] for i in sorted_indices]
        else:
            sorted_volume_array = volume_array
            sorted_headers = headers
            sorted_file_names = series_file_names

        return {
            'array': sorted_volume_array,
            'headers': sorted_headers,
            'dcm_paths': sorted_file_names
        }

def sitk_dicom_reader(path: str) -> sitk.Image:
    """
    Reads a DICOM series from the specified directory and returns it as a SimpleITK image.

    Parameters:
    - path (str): The directory containing the DICOM files.

    Returns:
    - sitk.Image: A 3D medical image created by reading the DICOM series.
    
    Raises:
    - ValueError: If no DICOM files are found in the directory.
    """
    
    reader = sitk.ImageSeriesReader()
    
    # Get all DICOM file names in the directory
    image_filenames = reader.GetGDCMSeriesFileNames(path)
    
    if len(image_filenames) == 0:
        raise ValueError(f"No DICOM files found in the directory: {path}")
    
    # Set the file names for the reader
    reader.SetFileNames(image_filenames)
    
    # Execute the reader to get the 3D image
    image = reader.Execute()
    
    return image


def estimate_class_weights(df, method='mfb'):
    labels = list(df.prediction.unique())
    class_counts = np.zeros_like(labels, dtype=np.float32)

    for i in range(len(labels)):
        class_counts[i] = df.prediction.value_counts()[i]

    class_counts = class_counts.astype(np.float32)

    # Compute the median and mode of the class frequencies
    median_frequency = np.median(class_counts)
    mode_frequency = np.max(class_counts)

    # Select the appropriate weighting method
    if method == 'mfb':
        class_weights = median_frequency / class_counts
    elif method == 'mode':
        class_weights = mode_frequency / class_counts

    weights = [class_weights[i] for i in range(2)]
    return weights


def print_trainable_parameters(model):
    """Print trainable parameters of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    print("\nTrainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")

  