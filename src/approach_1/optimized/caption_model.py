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

class OptimizedCNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, embedding_matrix=None, num_emotions=9, dropout_rate=0.5):
        super(OptimizedCNNLSTMModel, self).__init__()
        
        # Shared Encoder (The Backbone)
        self.cnn = CustomCNN(output_dim=512)
        
        # The "Caption" Path (For Captioning)
        # Deep projection to force the model to learn OBJECTS here
        self.caption_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embed_dim) # Down to embed_dim for LSTM
        )
        
        # The "Emotion" Path (For Classification)
        # Separate projection to keep SENTIMENT here
        self.emotion_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )
        
        # Sequence Decoder (Standard)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images, captions):
        # 1. Shared Features
        shared_feats = self.cnn(images) # (B, 512)
        
        # 2. Split Paths
        # Caption Vector: "Woman, Beach, Sand"
        caption_vec = self.caption_projector(shared_feats) # (B, embed_dim)
        
        # Emotion Logits: "Happy, Calm"
        emotion_logits = self.emotion_projector(shared_feats) # (B, num_emotions)
        
        # 3. Caption Generation (Using only Caption Vector)
        # We inject the caption vector as the first token (Context Primer)
        img_token = caption_vec.unsqueeze(1) # (B, 1, embed_dim)
        
        # Embed captions (exclude <end> as usual for input)
        text_embeds = self.embedding(captions) # (B, SeqLen, embed_dim)
        
        # Concatenate [Image, Words]
        lstm_input = torch.cat((img_token, text_embeds), dim=1) # (B, SeqLen+1, embed_dim)
        
        lstm_out, _ = self.lstm(lstm_input)
        
        # Output
        lstm_out = self.dropout(lstm_out)
        caption_logits = self.fc_out(lstm_out)
        
        return caption_logits, emotion_logits
