import torch
from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch.nn.functional as F

class EfficientNet_Embedding(nn.Module):
    def __init__(self, embedding_dim=1280, pretrained=True):
        super(EfficientNet_Embedding, self).__init__()
        
        # Load EfficientNet-B0 backbone
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        
        # image size 64x64
        self.efficientnet._conv_stem = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Change global pooling to fit small images
        self.efficientnet._avg_pooling = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer to reduce dimensionality
        self.fc = nn.Linear(1280, embedding_dim)  # Output of EfficientNet-B0 is 1280

    def forward(self, x):
        # Input x: [batch_size, channels, 64, 64]
        
        # Pass through EfficientNet backbone
        features = self.efficientnet.extract_features(x)  # [batch_size, 1280, height, width]
        
        # Global Average Pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # [batch_size, 1280, 1, 1]
        features = features.view(features.size(0), -1)      # Flatten to [batch_size, 1280]
        
        # Fully connected layer for embedding
        embedding = self.fc(features)  # [batch_size, embedding_dim]
        
        return embedding

class ConEfficientNet(nn.Module):
    def __init__(self, embedding_dim=1280, feat_dim=128, head='mlp', pretrained=False):
        super(ConEfficientNet, self).__init__()
        
        # Encoder with EfficientNet backbone
        self.encoder = EfficientNet_Embedding(embedding_dim=embedding_dim, pretrained=pretrained)
        
        # Projection head
        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, feat_dim)
            )
        elif head == 'linear':
            self.head = nn.Linear(embedding_dim, feat_dim)
        else:
            raise NotImplementedError(f"Projection head '{head}' not supported.")
    
    def forward(self, x):
        # Pass through EfficientNet embedding
        embedding = self.encoder(x)  # [batch_size, embedding_dim]
        
        # Pass through projection head
        feat = F.normalize(self.head(embedding), dim=1)  # Normalize to [batch_size, feat_dim]
        
        return feat
        
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features, return_embeddings=False):
        if return_embeddings:
            return features  # Return embeddings (features) before classification
        return self.fc(features)  # Perform classification
