import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, output_dim=512):
        super(CustomCNN, self).__init__()
        
        # 5 VGG-Style Blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # (B, 512)
        return x

class PowerfulCNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, embedding_matrix=None, num_emotions=9):
        super(PowerfulCNNLSTMModel, self).__init__()
        
        # A. Visual Encoder
        self.cnn = CustomCNN(output_dim=512)
        
        # B. Aux Branch (Emotion Head)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions)
        )
        
        # C. Main Branch
        # Visual Projector
        self.visual_projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Text Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
            
        # Stacked LSTM (2 Layers)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        
        # Caption Head
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):
        # images: (B, 3, 224, 224)
        # captions: (B, SeqLen) - includes <start> ... <end>
        
        # 1. Extract Features
        img_features = self.cnn(images) # (B, 512)
        
        # 2. Aux Branch: Emotion Prediction
        emotion_logits = self.emotion_head(img_features)
        
        # 3. Main Branch
        # Project Image Features
        img_embed = self.visual_projector(img_features) # (B, embed_dim)
        
        # Embed Captions
        word_embeds = self.embedding(captions) # (B, SeqLen, embed_dim)
        
        # Concatenate [Image, Words]
        # img_embed: (B, embed_dim) -> (B, 1, embed_dim)
        lstm_input = torch.cat((img_embed.unsqueeze(1), word_embeds), dim=1) # (B, SeqLen+1, embed_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input) # (B, SeqLen+1, hidden_dim)
        
        # Output
        lstm_out = self.dropout(lstm_out)
        outputs = self.fc_out(lstm_out) # (B, SeqLen+1, vocab_size)
        
        return outputs, emotion_logits
