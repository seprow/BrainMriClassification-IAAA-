import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import SimpleITK as sitk
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Optional, Tuple, List
import click




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def read_dicom_series(study_path: Path, series_instance_uid: Optional[str] = None) -> Optional[dict]:
    if series_instance_uid is None:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(study_path))
        if not series_ids:
            print(f"Warning: No DICOM series found in directory {study_path}")
            return None
        series_id = series_ids[0]
    else:
        series_id = series_instance_uid

    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(study_path), series_id)
    if not series_file_names:
        print(f"Warning: No DICOM files found for series ID {series_id} in directory {study_path}")
        return None

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
    
    
def apply_windowing(series: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    w_min = int(window_center - (window_width / 2))
    w_max = int(window_center + (window_width / 2))
    clipped = np.clip(series, w_min, w_max)
    return  clipped


def windowing(series: np.ndarray, headers: list[pydicom.FileDataset]) -> np.ndarray:
    windowed_series = []
    for i, header in enumerate(headers):
        window_center = header.get('WindowCenter')
        window_width = header.get('WindowWidth')
        windowed_series.append(apply_windowing(series[i], window_center, window_width))
    windowed_series = np.array(windowed_series)
    return windowed_series
    


def histogram_matcher(source_image: sitk.Image, reference_image_path: dict, headers: list) -> sitk.Image:
    series_description = None

    for header in headers:
        series_description = getattr(header, 'SeriesDescription', None)
        if series_description:
            break

    if series_description is None or series_description not in reference_image_path:
        raise ValueError(f"No matching series description found in reference image path for: {series_description}")

    reference_image_path = reference_image_path[series_description]

    # Read the reference image
    reader = sitk.ImageSeriesReader()
    reference_image_filenames = reader.GetGDCMSeriesFileNames(reference_image_path)
    reader.SetFileNames(reference_image_filenames)
    reference_image = reader.Execute()

    reference_image = sitk.Cast(reference_image, sitk.sitkFloat32)
    source_image = sitk.Cast(source_image, sitk.sitkFloat32)

    # Initialize histogram matcher
    histogram_matcher = sitk.HistogramMatchingImageFilter()
    histogram_matcher.SetNumberOfHistogramLevels(256)
    histogram_matcher.SetNumberOfMatchPoints(10)
    histogram_matcher.ThresholdAtMeanIntensityOn()

    output_image = histogram_matcher.Execute(source_image, reference_image)
    return output_image

def min_max_normalizer(image: sitk.Image) -> sitk.Image:
    # Ensure the image is a SimpleITK.Image
    if not isinstance(image, sitk.Image):
        raise TypeError("The input must be a SimpleITK.Image object.")
    
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    # Compute minimum and maximum values
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(image)
    
    min_value = min_max_filter.GetMinimum()
    max_value = min_max_filter.GetMaximum()
    
    # Normalize the image
    if max_value > min_value:
        normalized_image = (image - min_value) / (max_value - min_value)
    else:
        # Handle the edge case where all voxel values are the same
        normalized_image = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        normalized_image.CopyInformation(image)
        normalized_image.Fill(0.0)  
    
    return normalized_image

def resize(image: sitk.Image, new_size: list[int]) -> sitk.Image:
    # Compute the new spacing based on the new size
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [(original_size[i] * original_spacing[i]) / new_size[i] for i in range(3)]
    
    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    
    resized_image = resampler.Execute(image)
    return resized_image

def bias_correction(image: sitk.Image) -> sitk.Image:
    # Create head mask
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.RescaleIntensity(image,0,255)
    image = sitk.DICOMOrient(image, 'RPS')
    head_mask = sitk.LiThreshold(image, 0, 1)
    
    # Bias Correction
    shrinkFactor = 4 

    inputImage = sitk.Shrink( image, [ shrinkFactor ] * image.GetDimension() ) 
    maskImage = sitk.Shrink( head_mask, [ shrinkFactor ] * image.GetDimension() ) 
    
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    
    # Get corrected image
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp( log_bias_field )
    
    return corrected_image_full_resolution
    
def preprocess_data(series_path, reference_image_path=None, target_size=[288, 288, 16]):
        series = read_dicom_series(series_path)
        raw_image = series['array']
        metadata = series['headers']

        # Apply windowing
        windowed_sample =  windowing(raw_image, metadata)

        # Apply histogram matching
        windowed_sample = sitk.GetImageFromArray(windowed_sample)

        if reference_image_path:
            windowed_image = histogram_matcher(windowed_sample,reference_image_path, metadata)

        # Apply bias Field Correction
        corrected_image = bias_correction(windowed_image)

        # Normalize
        normalized_image = min_max_normalizer(corrected_image)

        # Resize images to the target size
        resized_image = resize(normalized_image, target_size)


        # output
        output = sitk.GetArrayFromImage(resized_image)
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return output
        


class SpatialAttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super(SpatialAttentionBlock3D, self).__init__()
       
        self.conv = nn.Conv3d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Create an attention map
        attention_map = torch.sigmoid(self.conv(x))  # Get attention map
        return attention_map


class SimpleVASNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleVASNet, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm3d(32)
        self.sab1 = SpatialAttentionBlock3D(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution Block 2
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.sab2 = SpatialAttentionBlock3D(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution Block 3
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.sab3 = SpatialAttentionBlock3D(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck Block
        self.conv_bottleneck = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.bn_bottleneck = nn.BatchNorm3d(256)

        # Global Average Pooling and FC Layer
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  
        self.fc = nn.Linear(256, num_classes)  
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))  
        attention1 = self.sab1(x) 
        x = x * attention1 
        x = self.pool1(x)  

        # Block 2
        x = F.relu(self.bn2(self.conv2(x))) 
        attention2 = self.sab2(x) 
        x = x * attention2  
        x = self.pool2(x)  

        # Block 3
        x = F.relu(self.bn3(self.conv3(x))) 
        attention3 = self.sab3(x)  
        x = x * attention3  
        x = self.pool3(x)  

        # Bottleneck
        x = F.relu(self.bn_bottleneck(self.conv_bottleneck(x))) 

        # Global Average Pooling
        x = self.global_avg_pool(x)  
        x = x.view(x.size(0), -1)  

        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)  

        return x  


    
HERE = Path(__file__).absolute().resolve().parent
@click.command()
@click.option('--data-dir', type=Path)
@click.option('--predictions-file-path', type=Path)



def main(data_dir: Path, predictions_file_path: Path):
    series_instance_uid_list = [i.name for i in data_dir.iterdir() if i.is_dir()]
    siuid_paths = [i for i in data_dir.iterdir() if i.is_dir()]

    reference_image_path = {
        'T1W_SE': (str(HERE / '1.3.46.670589.11.10042.5.0.1412.2024020321545257411')),
        'T2W_TSE' : (str(HERE / '1.3.46.670589.11.10042.5.0.1412.2024020313391873234')),
        'T2W_FLAIR' : (str(HERE / '1.3.46.670589.11.10042.5.0.1412.2024020410193570629'))
    }
    
    model = SimpleVASNet().to(device)
    state_dict = torch.load(str(HERE / 'model.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    
    predictions = list()

    for siuid in siuid_paths:
        arr = preprocess_data(siuid, reference_image_path=reference_image_path) 
        
        with torch.no_grad():
            output = model(arr)
            prediction = torch.sigmoid(output).cpu().numpy().squeeze()
            
        predictions.append(prediction)
    

    predictions = (np.array(predictions) >=0.3).astype(int)
    predictions_df = pd.DataFrame({'SeriesInstanceUID': series_instance_uid_list, 'prediction': predictions})
    predictions_df.to_csv(predictions_file_path)
    
if __name__ == "__main__":
    main()