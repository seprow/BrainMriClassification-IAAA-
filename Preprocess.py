import numpy as np
import pydicom
import SimpleITK as sitk
import ants
import torch

def apply_windowing(series: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    Applies windowing to a series of images.

    Args:
        series (np.ndarray): Input array of images.
        window_center (int): Center of the window.
        window_width (int): Width of the window.

    Returns:
        np.ndarray: Windowed series of images.

    Raises:
        ValueError: If the window center or width is invalid.
    """
    if window_center < 0 or window_width < 0:
        raise ValueError("Invalid window center or width")

    w_min = int(window_center - (window_width / 2))
    w_max = int(window_center + (window_width / 2))
    windowed_series = np.clip(series, w_min, w_max)
    return windowed_series


def windowing(series: np.ndarray, headers: list[pydicom.FileDataset]) -> np.ndarray:
    """
    Applies windowing to a series of images based on the provided headers.

    Args:
        series (np.ndarray): Input array of images.
        headers (list[pydicom.FileDataset]): List of headers for the images.

    Returns:
        np.ndarray: Windowed series of images.
    """
    windowed_series = []
    for i, header in enumerate(headers):
        window_center = header.get('WindowCenter')
        window_width = header.get('WindowWidth')
        windowed_series.append(apply_windowing(series[i], window_center, window_width))
    windowed_series = np.array(windowed_series)
    return windowed_series


def histogram_matcher(source_image: sitk.Image, reference_image_paths: dict, headers: list) -> sitk.Image:
    """
    Applies histogram matching to a source image based on the provided reference image path and headers.

    Args:
        source_image (sitk.Image): Input source image.
        reference_image_paths (dict): Dictionary of reference image paths.
        headers (list): List of DICOM headers.

    Returns:
        sitk.Image: Histogram matched source image.

    Raises:
        ValueError: If the reference image path or headers are invalid.
    """
    if not isinstance(reference_image_paths, dict):
        raise ValueError("Invalid reference image path")
    if not isinstance(headers, list):
        raise ValueError("Invalid headers")

    match_series_description = None

    for header in headers:
        match_series_description = getattr(header, 'SeriesDescription', None)
        if match_series_description:
            break

    if match_series_description is None or match_series_description not in reference_image_paths:
        raise ValueError(f"No matching series description found in reference image path for: {match_series_description}")

    reference_image_path = reference_image_paths[match_series_description]

    # Read the reference image
    reader = sitk.ImageSeriesReader()
    reference_image_filenames = reader.GetGDCMSeriesFileNames(reference_image_path)
    reader.SetFileNames(reference_image_filenames)
    reference_image = reader.Execute()

    # Cast the reference image to float32
    reference_image = sitk.Cast(reference_image, sitk.sitkFloat32)

    # Cast the source image to float32
    source_image = sitk.Cast(source_image, sitk.sitkFloat32)

    # Initialize histogram matcher
    histogram_matcher = sitk.HistogramMatchingImageFilter()
    histogram_matcher.SetNumberOfHistogramLevels(256)
    histogram_matcher.SetNumberOfMatchPoints(10)
    histogram_matcher.ThresholdAtMeanIntensityOn()

    output_image = histogram_matcher.Execute(source_image, reference_image)
    return output_image


def min_max_normalizer(image: sitk.Image) -> sitk.Image:
    """
    Normalizes an image by scaling its values to the range [0, 1].

    Args:
        image (sitk.Image): Input image.

    Returns:
        sitk.Image: Normalized image.

    Raises:
        TypeError: If the input is not a SimpleITK.Image object.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError("The input must be a SimpleITK.Image object.")

    # Ensure the image is a float32 image
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Compute minimum and maximum values
    minimum_maximum_filter = sitk.MinimumMaximumImageFilter()
    minimum_maximum_filter.Execute(image)
    
    min_value = minimum_maximum_filter.GetMinimum()
    max_value = minimum_maximum_filter.GetMaximum()
    
    # Normalize the image
    if max_value > min_value:
        normalized_image = (image - min_value) / (max_value - min_value)
    else:
        # Handle the edge case where all voxel values are the same
        normalized_image = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        normalized_image.CopyInformation(image)
        normalized_image.Fill(0.0)  
    
    return normalized_image


def resize(image: sitk.Image, target_size: list[int]) -> sitk.Image:
    """
    Resizes an image to a specified target size.

    Args:
        image (sitk.Image): Input image.
        target_size (list[int]): Desired size of the output image.

    Returns:
        sitk.Image: Resized image.

    Raises:
        ValueError: If the target size is invalid (not a list of three positive integers).
    """
    if len(target_size) != 3 or not all(isinstance(size, int) and size > 0 for size in target_size):
        raise ValueError("Invalid target size. Must be a list of three positive integers.")

    # Compute the new spacing based on the target size
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [(original_size[i] * original_spacing[i]) / target_size[i] for i in range(3)]
    
    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    
    resized_image = resampler.Execute(image)
    return resized_image
    

def resize_and_adjust_slices(array: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resizes an array using bilinear interpolation and ensures the length of the array is 16.
    If the array length is greater than 16, it slices it to keep 16 elements.
    If the array length is less than 16, it pads the array with zeros.

    Parameters:
    - array (np.ndarray): Input array with shape (length, height, width).
    - target_size (tuple): Target size for resizing (height, width).

    Returns:
    - np.ndarray: Resized array with length of 16 and shape (16, target_size[0], target_size[1]).
    """
    
    # Convert numpy array to torch tensor
    array_torch = torch.tensor(array, dtype=torch.float32)
    
    # Interpolate the array to the target size
    resized_array = torch.nn.functional.interpolate(
        array_torch.unsqueeze(1),  # Add a channel dimension for interpolation
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # Remove the added channel dimension
    
    # Adjust the length of the array to 16 slices
    if resized_array.size(0) > 16:
        # If more than 16 slices, take the central 16 slices
        remainder = resized_array.size(0) - 16
        resized_array = resized_array[int(remainder / 2): 16 + int(remainder / 2)]
    elif resized_array.size(0) < 16:
        # If less than 16 slices, pad with zeros
        padding_size = 16 - resized_array.size(0)
        padding = (0, 0, 0, 0, 0, padding_size)  # (left, right, top, bottom, front, back)
        resized_array = torch.nn.functional.pad(resized_array, padding, mode='constant', value=0)
    
    # Convert the result back to numpy
    resized_array_np = resized_array.numpy()
    
    return resized_array_np


def bias_correction(image: sitk.Image) -> sitk.Image:
    """
    Performs N4 bias field correction on a given medical image, using a head mask to focus on relevant regions.
    
    Parameters:
    - image (sitk.Image): The input medical image.

    Returns:
    - sitk.Image: The bias-corrected image.
    """
    
    # Cast to float and rescale intensity to [0, 255]
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.RescaleIntensity(image, 0, 255)
    
    # Reorient the image to RPS (Right-Posterior-Superior) coordinate system
    image = sitk.DICOMOrient(image, 'RPS')
    
    # Create a binary head mask using Li thresholding
    head_mask = sitk.LiThreshold(image, 0, 1, 1)

    # Shrink the image and mask for bias correction to reduce computational cost
    shrink_factor = 4
    input_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    mask_image = sitk.Shrink(head_mask, [shrink_factor] * image.GetDimension())
    
    # Apply N4 bias field correction
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = bias_corrector.Execute(input_image, mask_image)

    # Recover full resolution by applying the computed bias field to the original image
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    
    return corrected_image_full_resolution


def bias_correct(
    input: sitk.Image,
    mask: sitk.Image = None,
    shrink_factor: int = 4,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
) -> sitk.Image:
    """Perform N4 bias correction on MRI

    Note:
    - if no mask is provided it will be generated using Otsu-thresholding
    """
    if not isinstance(mask, sitk.Image):
        mask = sitk.OtsuThreshold(input, 0, 1, 200)

    input = sitk.Cast(input, sitk.sitkFloat32)
    image = sitk.Shrink(input, [shrink_factor] * input.GetDimension())
    mask = sitk.Shrink(mask, [shrink_factor] * input.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)

    corrector.Execute(image, mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input)
    corrected_image_full_resolution = input / sitk.Exp(log_bias_field)
    return corrected_image_full_resolution


