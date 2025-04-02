import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCNN(nn.Module):
    def __init__(self, num_classes=10, num_lstm_hidden=128, num_lstm_layers=1):
        super(LSTMCNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling by a factor of 2
        
        self.lstm = nn.LSTM(input_size=256 * 8 * 8, hidden_size=num_lstm_hidden, 
                            num_layers=num_lstm_layers, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_lstm_hidden, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Input x shape: [batch_size, time_steps, channels, height, width]
        x = x.unsqueeze(1)  # Add channel dimension
        batch_size, time_steps, C, H, W = x.size()
        
        # Apply CNN layers on each time step
        cnn_features = []
        for t in range(time_steps):
            out = self.pool(F.relu(self.conv1(x[:, t, :, :, :])))  # 64x64 -> 32x32
            out = self.pool(F.relu(self.conv2(out)))              # 32x32 -> 16x16
            out = self.pool(F.relu(self.conv3(out)))              # 16x16 -> 8x8
            out = out.view(out.size(0), -1)  # Flatten: [batch_size, 256 * 8 * 8]
            cnn_features.append(out)
        
        # Stack CNN features along time axis
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, time_steps, 256 * 8 * 8]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # [batch_size, time_steps, num_lstm_hidden]
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        
        return x
    
class LSTMCNN_Embedding(nn.Module):
    def __init__(self, embedding_dim=512, num_lstm_hidden=128, num_lstm_layers=1):
        super(LSTMCNN_Embedding, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=num_lstm_hidden, 
                            num_layers=num_lstm_layers, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Linear(num_lstm_hidden, embedding_dim)

    def forward(self, x):
        # Input x: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        
        # CNN feature extraction
        out = self.pool(F.relu(self.conv1(x)))  # [batch_size, 64, height, width]
        out = self.pool(F.relu(self.conv2(out)))  # [batch_size, 128, height/2, width/2]
        out = self.pool(F.relu(self.conv3(out)))  # [batch_size, 256, height/4, width/4]

        # Flatten to [batch_size, feature_dim=256]
        out = out.view(batch_size, 256, -1).mean(dim=2)  # Reduce spatial dimensions (H, W) by averaging

        # Add time_steps dimension (treat batch as sequence)
        out = out.unsqueeze(1)  # [batch_size, 1, feature_dim=256]

        # Pass through LSTM
        lstm_out, _ = self.lstm(out)  # [batch_size, 1, lstm_hidden_dim]
        lstm_out = lstm_out.squeeze(1)  # Remove time_steps dimension: [batch_size, lstm_hidden_dim]

        # Fully connected layer
        embedding = self.fc(lstm_out)  # [batch_size, embedding_dim]
        
        return embedding

# class ConLSTMCNN(nn.Module):
#     def __init__(self, name='lstmcnn', head='mlp', feat_dim=128):
#         super(ConLSTMCNN, self).__init__()
#         self.encoder = LSTMCNN_Embedding(embedding_dim=512)
        
#         if head == 'linear':
#             self.head = nn.Linear(512, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(512, 512),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(512, feat_dim)
#             )
#         else:
#             raise NotImplementedError('head not supported: {}'.format(head))
    
#     def forward(self, x):
#         feat = self.encoder(x)  # [batch_size * 2, 512]
#         feat = F.normalize(self.head(feat), dim=1)  # Normalize: [batch_size * 2, feat_dim]
#         return feat
class ConLSTMCNN(nn.Module):
    def __init__(self, embedding_dim=512, feat_dim=128, head='mlp', lstm_hidden_dim=128, num_lstm_layers=1):
        super(ConLSTMCNN, self).__init__()
        
        # Encoder
        self.encoder = LSTMCNN_Embedding(embedding_dim=embedding_dim, 
                                         num_lstm_hidden=lstm_hidden_dim, 
                                         num_lstm_layers=num_lstm_layers)
        
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
            raise NotImplementedError('Head type "{}" not supported.'.format(head))

    def forward(self, x):
        # Pass through encoder
        embedding = self.encoder(x)  # [embedding_dim]

        # Pass through head
        feat = F.normalize(self.head(embedding), dim=0)  # Normalize to [feat_dim]
        
        return feat



class LinearClassifier(nn.Module):
    """Linear classifier with optional embedding return"""

    def __init__(self, input_dim=512, num_classes=5):
        """
        Args:
            input_dim (int): Kích thước của embedding đầu vào.
            num_classes (int): Số lượng lớp để phân loại.
        """
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features, return_embeddings=False):
        """
        Args:
            features (Tensor): Đầu vào là tensor đặc trưng (features) từ mô hình backbone.
            return_embeddings (bool): Nếu True, trả về embedding thay vì kết quả phân loại.

        Returns:
            Tensor: Embedding hoặc kết quả phân loại.
        """
        if return_embeddings:
            return features  # Return embeddings (features) before classification
        return self.fc(features)  # Perform classification
