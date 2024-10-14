import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
import random
import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from Preprocess import *


"""DICOMDataGenerator class :

    is designed to be memory-efficient by loading and processing data on-the-fly,
    only keeping the necessary data in RAM.
    
    which allows to work with larger datasets or use more resources for model training,
    larger batch/deeper models."""

class DICOMDataGenerator(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,  # DataFrame containing SeriesInstanceUID and predictions
        target_size: List[int],  # e.g., [255,255,16] for image resizing
        data_dir: str,  # Base directory where the data is stored
        reference_image_path: Optional[str] = None,  # Optional reference image path for histogram matching
        shuffle: bool = False,
        downsample: bool = False
    ):
        self.df = df
        self.target_size = target_size
        self.data_dir = data_dir
        self.reference_image_path = reference_image_path
        self.shuffle = shuffle
        self.downsample = downsample
        self.study_paths, self.labels = self._get_study_paths_and_labels(df)  # Initialize paths and labels
        
        # Optionally shuffle paths and labels
        if self.shuffle:
            self.study_paths, self.labels = self._shuffle_paths_and_labels()

    def __len__(self):
        return len(self.study_paths)

    def __getitem__(self, index):
        path = self.study_paths[index]
        label = self.labels[index]
        
        # Generate data
        X = self.__data_generation(path)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return torch.tensor(X, dtype=torch.float32).unsqueeze(0), label_tensor

    def _get_study_paths_and_labels(self, df):
        paths = [os.path.join(self.data_dir, uid) for uid in df['SeriesInstanceUID']]
        labels = list(df['prediction'])
        return paths, labels

    def _shuffle_paths_and_labels(self):
        indices = np.arange(len(self.study_paths))
        np.random.shuffle(indices)
        return np.array(self.study_paths)[indices], np.array(self.labels)[indices]

    def __data_generation(self, path):
        data = self.read_dicom_series(Path(path))
        if data is None:
            # Handle the case where no data is available
            return np.zeros((16, *self.target_size), dtype=np.float32)  
        
        images = self.preprocess_images(data)
        return images
    
    def downsample(self, paths, labels, upsample_fraction=1):
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]

        num_positive = len(positive_indices)
        num_negative = len(negative_indices)
    
        num_to_add = int(upsample_fraction * (num_negative - num_positive))
    
        if num_to_add > 0:
            upsampled_indices = random.choices(positive_indices, k=num_to_add)
            paths += [paths[i] for i in upsampled_indices]
            labels += [labels[i] for i in upsampled_indices]

        return paths, labels

    def read_dicom_series(self, study_path: Path, series_instance_uid: Optional[str] = None) -> Optional[dict]:
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

    def preprocess_images(self, series):

        raw_image = series['array']
        metadata = series['headers']

        # Apply windowing
        windowed_sample =  windowing(raw_image, metadata)

        # Apply histogram matching
        windowed_sample = sitk.GetImageFromArray(windowed_sample)

        if self.reference_image_path:
            windowed_image = histogram_matcher(windowed_sample, self.reference_image_path, metadata)

        # Apply bias Field Correction
        corrected_image = bias_correction(windowed_image)

        # Normalize
        normalized_image = min_max_normalizer(corrected_image)

        # Resize images to the target size
        resized_image = resize(normalized_image, self.target_size)


        # output
        output = sitk.GetArrayFromImage(resized_image)
        output = torch.tensor(output, dtype=torch.float32)
        return output

    
#######################################################################################################################

# Just For Fun :)

import tensorflow as tf

class DICOMDataGenerator_tf(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,  # The DataFrame containing SeriesInstanceUID and predictions
        batch_size: int,
        target_size: Tuple[int, int],  # e.g., (128, 128) for image resizing
        data_dir: str,  # Base directory where the data is stored
        shuffle: bool = True
    ):
        self.df = df
        self.batch_size = batch_size
        self.target_size = target_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.study_paths, self.labels = self._get_study_paths_and_labels(df)  # Initialize paths and labels
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.study_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_paths = self.study_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = np.array(self.labels[index * self.batch_size:(index + 1) * self.batch_size])
        
        # Generate data
        X = self.__data_generation(batch_paths)
        
        return X, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            self.study_paths, self.labels = self._shuffle_paths_and_labels()

    def _get_study_paths_and_labels(self, df):
        paths = [os.path.join(self.data_dir, uid) for uid in df['SeriesInstanceUID']]
        labels = list(df['prediction'])  
        return paths, labels

    def _shuffle_paths_and_labels(self):
        indices = np.arange(len(self.study_paths))
        np.random.shuffle(indices)
        return np.array(self.study_paths)[indices], np.array(self.labels)[indices]

    def __data_generation(self, batch_paths):
        X = []

        for path in batch_paths:
            data = self.read_dicom_series(Path(path))
            if data is None:
                continue  # Skip if no valid series is found
            images = self.preprocess_images(data)
            images = np.squeeze(images, axis=0)
            images = np.expand_dims(images, axis=-1)  
        
            X.append(images)  

        return np.array(X) 

    def read_dicom_series(self, study_path: Path, series_instance_uid: Optional[str] = None) -> Optional[dict]:
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
            sorted_indices = np.argsort([int(header.get(slice_number_tag)) for header in headers])
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

    def preprocess_images(self, series):
        preprocessed_volumes = []
    
        # Normalize/windowing image
        windowed_series = windowing(series['array'], series['headers'])

        # Resize image
        target_h, target_w = self.target_size
        if (windowed_series.shape[1] != target_h) or (windowed_series.shape[2] != target_w):
            arr = np.squeeze(tf.image.resize_with_pad(np.expand_dims(windowed_series, axis=-1), target_h, target_w))
        else:
            arr = windowed_series

        # Adjust the slice number
        if len(arr) > 16:
            remainder = len(arr) - 16
            arr = arr[int(remainder / 2): 16 + int(remainder / 2)]
        elif len(arr) < 16:
            arr = np.pad(arr, ((0, 16 - len(arr)), (0, 0), (0, 0)), mode='constant')

        preprocessed_volumes.append(arr)
             
        return np.array(preprocessed_volumes)

