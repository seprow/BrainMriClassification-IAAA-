import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



''' Volumetric Attention SkipNet (VASNet) '''

class SpatialAttentionBlock3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, 
                              stride=1, padding=kernel_size // 2, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        min_pool, _ = torch.min(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        min_pool = F.normalize(min_pool, p=2, dim=[2, 3, 4])
        max_pool = F.normalize(max_pool, p=2, dim=[2, 3, 4])
        sub = max_pool - min_pool
        attention_map = torch.sigmoid(self.conv(sub))
        return attention_map

class VASNet(nn.Module):
    def __init__(self):
        super(VASNet, self).__init__()

        # Convolution layers (equivalent to Keras' Conv2D but for 3D)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=25, stride=2, padding=12)  # Adjusted in_channels to 1
        self.conv1x1_1 = nn.Conv3d(64, 64, kernel_size=1)

        # Spatial Attention Block 1 (for 3D)
        self.sab1 = SpatialAttentionBlock3D(kernel_size=16)

        # Max Pooling layers (3D pooling)
        self.maxpool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution layers for SAB2 (3D)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=13, stride=2, padding=6)  # Equivalent to 'same' padding
        self.conv1x1_2 = nn.Conv3d(128, 128, kernel_size=1)
        self.sab2 = SpatialAttentionBlock3D(kernel_size=8)

        # Convolution layers for SAB3 (3D)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=9, stride=2, padding=4)  # Equivalent to 'same' padding
        self.conv1x1_3 = nn.Conv3d(256, 256, kernel_size=1)
        self.sab3 = SpatialAttentionBlock3D(kernel_size=4)

        # Resizing (Adaptive average pooling for 3D)
        self.resize1 = nn.AdaptiveAvgPool3d((29, 29, 29))
        self.resize2 = nn.AdaptiveAvgPool3d((7, 7, 7))

        # Classifier head (Linear layer size will be computed dynamically)
        self.flatten = nn.Flatten()
        self.fc1 = None  # We'll initialize this in forward()
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 3)

    def _get_flatten_size(self, x):
        # Utility to compute flatten size dynamically
        x = self.maxpool3d(x)  # After the last pooling operation
        flatten_size = self.flatten(x).shape[1]
        return flatten_size

    def forward(self, x):
        # Layer block 1
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv1x1_1(x1))
        K1 = self.sab1(x1)
        K1s = x1 * K1
        K1s = F.normalize(K1s, p=2, dim=[2, 3, 4])
        x1 = F.normalize(x1, p=2, dim=[2, 3, 4])
        K = K1s + x1
        K1s = self.maxpool3d(K)

        # Layer block 2
        conv2 = F.relu(self.conv2(K1s))
        conv2 = F.relu(self.conv1x1_2(conv2))
        K2s = self.sab2(conv2)
        K2 = conv2 * K2s

        # Skip connection A
        K1 = self.maxpool3d(K1)
        K12 = F.interpolate(K1, size=(29, 29, 29), mode='trilinear', align_corners=False)
        K12 = self.conv1x1_2(K12)
        K2 = K2 * K12

        K2 = F.normalize(K2, p=2, dim=[2, 3, 4])
        conv2 = F.normalize(conv2, p=2, dim=[2, 3, 4])
        K = K2 + conv2
        K2 = self.maxpool3d(K)

        # Layer block 3
        conv3 = F.relu(self.conv3(K2))
        conv3 = F.relu(self.conv1x1_3(conv3))
        K3 = self.sab3(conv3)
        K3 = conv3 * K3

        # Skip connection C
        K1 = self.maxpool3d(K1)
        K13 = F.interpolate(K1, size=(7, 7, 7), mode='trilinear', align_corners=False)
        K13 = self.conv1x1_3(K13)
        K3 = K3 * K13

        # Skip connection B
        K2s = self.maxpool3d(K2s)
        K23 = F.interpolate(K2s, size=(7, 7, 7), mode='trilinear', align_corners=False)
        K23 = self.conv1x1_3(K23)
        K3 = K3 * K23

        K3 = F.normalize(K3, p=2, dim=[2, 3, 4])
        conv3 = F.normalize(conv3, p=2, dim=[2, 3, 4])
        K = K3 + conv3
        K3 = self.maxpool3d(K)

        # Classifier head
        if self.fc1 is None:  # Initialize fc1 dynamically based on the flattened size
            flatten_size = self._get_flatten_size(K3)
            self.fc1 = nn.Linear(flatten_size, 1024).to(K3.device)

        x4 = self.flatten(K3)
        x4 = F.relu(x4)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        output = self.fc3(x6)
        output = F.softmax(output, dim=1)
        return output

##################################################################################################

''' SimpleVASNet '''

class SimpleSpatialAttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super(SimpleSpatialAttentionBlock3D, self).__init__()
        # Set in_channels to channels to match the expected number of channels in the input
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
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Input channels set to 1
        self.bn1 = nn.BatchNorm3d(32)
        self.sab1 = SimpleSpatialAttentionBlock3D(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution Block 2
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.sab2 = SimpleSpatialAttentionBlock3D(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution Block 3
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.sab3 = SimpleSpatialAttentionBlock3D(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck Block
        self.conv_bottleneck = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.bn_bottleneck = nn.BatchNorm3d(256)

        # Global Average Pooling and FC Layer
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Outputs (batch_size, 256, 1, 1, 1)
        self.fc = nn.Linear(256, num_classes)  # Output size set to num_classes
        self.dropout = nn.Dropout(p=0.45)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 32, D, H, W)
        attention1 = self.sab1(x)  # Apply spatial attention block
        x = x * attention1  # Element-wise multiplication
        x = self.pool1(x)  # (batch_size, 32, D//2, H//2, W//2)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, D//2, H//2, W//2)
        attention2 = self.sab2(x)  # Apply spatial attention block
        x = x * attention2  # Element-wise multiplication
        x = self.pool2(x)  # (batch_size, 64, D//4, H//4, W//4)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 128, D//4, H//4, W//4)
        attention3 = self.sab3(x)  # Apply spatial attention block
        x = x * attention3  # Element-wise multiplication
        x = self.pool3(x)  # (batch_size, 128, D//8, H//8, W//8)

        # Bottleneck
        x = F.relu(self.bn_bottleneck(self.conv_bottleneck(x)))  # (batch_size, 256, D//8, H//8, W//8)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch_size, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 256)

        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)

        return x  # Return raw logits
    
    def freeze_except_fc(self):
        # Freeze all layers except the fully connected layer
        for name, param in self.named_parameters():
            if "fc" not in name:  # Check if 'fc' is not in the layer name
                param.requires_grad = False
  

#model = SimpleVASNet(num_classes=1)
#model.freeze_except_fc()
    
##################################################################################################

''' Fine Tuning ResNet n last layers '''

class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()
        self.resnet10 = models.video.r3d_18(pretrained=True)

        # Modify the first conv layer to accept 1-channel input
        self.resnet10.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=self.resnet10.stem[0].out_channels,
            kernel_size=self.resnet10.stem[0].kernel_size,
            stride=self.resnet10.stem[0].stride,
            padding=self.resnet10.stem[0].padding,
            bias=False
        )
        
        # Freeze all layers except the last n layers (including fc)
        layers_to_unfreeze = 7
        total_layers = len(list(self.resnet10.named_parameters()))

        for i, (name, param) in enumerate(self.resnet10.named_parameters()):
            if i < total_layers - layers_to_unfreeze:
                param.requires_grad = False

        # Modify the fully connected layer for binary classification
        self.resnet10.fc = nn.Linear(self.resnet10.fc.in_features, 1)

        # The final fully connected layer should be trainable
        for param in self.resnet10.fc.parameters():
            param.requires_grad = True

    
    def forward(self, x):
        return self.resnet10(x)
    
##################################################################################################

'''
    Cerberus: fine-tuning the last layers of ResNet, with a separate classifier for each modality.

    ***Change Requirement in Data Generator:

    def __getitem__(self, index):
        path = self.study_paths[index]
        label = self.labels[index]

        # Generate data
        X, modality = self.__data_generation(path)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return torch.tensor(X, dtype=torch.float32).unsqueeze(0), label_tensor, modality

    def __data_generation(self, path):
        data = self.read_dicom_series(Path(path))
        if data is None:
            # Handle the case where no data is available
            return np.zeros((16, *self.target_size), dtype=np.float32)

        images = self.preprocess_images(data)

        for header in data['headers']:
          modality = getattr(header, 'SeriesDescription', None)
          if modality:
            break

        return images ,modality

'''

class Cerberus(nn.Module):
    def __init__(self):
        super(Cerberus, self).__init__()

        # Initialize the ResNet3D model
        self.resnet10 = models.video.r3d_18(pretrained=True)

        # Modify the first conv layer to accept 1-channel input
        self.resnet10.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=self.resnet10.stem[0].out_channels,
            kernel_size=self.resnet10.stem[0].kernel_size,
            stride=self.resnet10.stem[0].stride,
            padding=self.resnet10.stem[0].padding,
            bias=False
        )

        # Freeze all layers except the last n layers (including fc)
        layers_to_unfreeze = 7
        total_layers = len(list(self.resnet10.named_parameters()))

        for i, (name, param) in enumerate(self.resnet10.named_parameters()):
            if i < total_layers - layers_to_unfreeze:
                param.requires_grad = False

        # Classifiers for T1, T2, and FLAIR
        in_features = 400
        self.classifier_t1 = nn.Linear(in_features, 1)
        self.classifier_t2 = nn.Linear(in_features, 1)
        self.classifier_flair = nn.Linear(in_features, 1)

    def forward(self, x, modality):
        features = self.resnet10(x)

        if 'T1W_SE' in modality:
            output = self.classifier_t1(features)
        elif 'T2W_TSE' in modality:
            output = self.classifier_t2(features)
        elif 'T2W_FLAIR' in modality:
            output = self.classifier_flair(features)
        else:
            raise ValueError("Unknown modality type. Choose 'T1W_SE', 'T2W_TSE', or 'T2W_FLAIR'.")

        output = output.view(-1, 1)

        return output
    
