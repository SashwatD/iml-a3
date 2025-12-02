import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, output_dim=512):
        super(CustomCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # 224 -> 112
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # 112 -> 56
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # 56 -> 28
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) # 28 -> 14
        
        # Block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2) # 14 -> 7
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Block 5
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten (B, 512)
        
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, embedding_matrix=None):
        super(CNNLSTMModel, self).__init__()
        
        # Visual Encoder
        self.cnn = CustomCNN(output_dim=512)
        
        # Fusion: Project CNN features to match Text Embedding dim
        self.visual_projection = nn.Linear(512, embed_dim)
        
        # Text Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False # Freeze if pre-trained
            
        # Sequence Model
        # Input: Previous word embedding + image feature vector (Concatenated)
        # Input size = embed_dim (word) + embed_dim (image) = 2 * embed_dim
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, num_layers=1, batch_first=True)
        
        # Output
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):
        # images: (B, 3, 224, 224)
        # captions: (B, SeqLen) - includes <start> ... <end>
        
        # 1. Extract Image Features
        img_features = self.cnn(images) # (B, 512)
        img_features = self.visual_projection(img_features) # (B, embed_dim)
        
        # 2. Embed Captions
        # We don't need the last token for input (it's the target)
        # But for teacher forcing, we feed the sequence.
        # Usually we feed <start>...<last_word> to predict <first_word>...<end>
        # Here we feed the whole caption sequence except the last token to predict the next token.
        
        embeds = self.embedding(captions) # (B, SeqLen, embed_dim)
        
        # 3. Concatenate Image Features to EACH word embedding
        # Expand image features to match sequence length
        seq_len = embeds.size(1)
        img_features_expanded = img_features.unsqueeze(1).repeat(1, seq_len, 1) # (B, SeqLen, embed_dim)
        
        # Concatenate: (B, SeqLen, 2 * embed_dim)
        lstm_input = torch.cat((embeds, img_features_expanded), dim=2)
        
        # 4. LSTM
        lstm_out, _ = self.lstm(lstm_input) # (B, SeqLen, hidden_dim)
        
        # 5. Output
        lstm_out = self.dropout(lstm_out)
        outputs = self.fc_out(lstm_out) # (B, SeqLen, vocab_size)
        
        return outputs
