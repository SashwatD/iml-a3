import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Fix for "Too many open files" error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.dataset_torch import get_loader, save_vocab
from src.approach_1.powerful.caption_model import PowerfulCNNLSTMModel
from src.approach_1.powerful.embeddings import get_tfidf_embeddings

# Custom loss
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        # These are learnable parameters (log_vars)
        # We initialize them to 0.0 (which means sigma=1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_cap, loss_emo):
        # Task 1: Captioning
        precision1 = torch.exp(-self.log_vars[0])
        weighted_loss1 = precision1 * loss_cap + self.log_vars[0]

        # Task 2: Emotion
        precision2 = torch.exp(-self.log_vars[1])
        weighted_loss2 = precision2 * loss_emo + self.log_vars[1]

        return weighted_loss1 + weighted_loss2

def train_model(
    csv_path,
    image_dir,
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/approach-1-powerful")),
    epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    image_size=(224, 224),
    embedding_dim=512,
    embedding_type="tfidf"
):
    output_dir = os.path.join(output_dir, embedding_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Data
    print("Loading data...")
    num_workers = 0 if device.type == "mps" else 4
    
    train_loader, train_dataset = get_loader(
        image_dir, 
        csv_path, 
        transform, 
        batch_size=batch_size,
        max_length=50,
        num_workers=num_workers
    )
    
    vocab = train_dataset.vocab
    save_vocab(vocab, os.path.join(output_dir, "vocab.pkl"))
    
    # Embeddings
    embedding_matrix = None
    vocab_list = [vocab.itos[i] for i in range(len(vocab))]
    captions = train_dataset.captions
    
    if embedding_type == "tfidf":
        embedding_matrix = get_tfidf_embeddings(vocab_list, captions, embedding_dim=embedding_dim)
    else:
        print(f"Warning: Embedding type {embedding_type} not supported in powerful variant (TF-IDF only). Using random.")
    
    # Model
    model = PowerfulCNNLSTMModel(
        vocab_size=len(vocab),
        embed_dim=embedding_dim,
        hidden_dim=512,
        embedding_matrix=embedding_matrix
    ).to(device)
    
    # Uncertainty Loss
    mtl_loss = MultiTaskLoss(num_tasks=2).to(device)
    
    # Loss & Optimizer
    criterion_caption = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    criterion_emotion = nn.CrossEntropyLoss()
    
    # AdamW + Grad Clipping
    optimizer = optim.AdamW(
        list(model.parameters()) + list(mtl_loss.parameters()), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Training Loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Use sys.stdout for tqdm to work well with tee
        loop = tqdm(train_loader, total=len(train_loader), leave=True, file=sys.stdout)
        
        for imgs, captions, emotions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)
            emotions = emotions.to(device)
            
            optimizer.zero_grad()
            
            caption_inputs = captions[:, :-1]
            caption_targets = captions[:, 1:] # Input Feeding: Predict next word
            
            outputs, emotion_logits = model(imgs, caption_inputs)
            
            # Raw Losses
            loss_cap = criterion_caption(outputs.reshape(-1, outputs.shape[-1]), caption_targets.reshape(-1))
            loss_emo = criterion_emotion(emotion_logits, emotions)
            
            # Automatic Weighting
            loss = mtl_loss(loss_cap, loss_emo)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Get learned weights for display
            w1 = torch.exp(-mtl_loss.log_vars[0]).item()
            w2 = torch.exp(-mtl_loss.log_vars[1]).item()
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), cap=loss_cap.item(), emo=loss_emo.item(), w_cap=f"{w1:.2f}", w_emo=f"{w2:.2f}")
            
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))
        
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pth"))
    
    # Plot history
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "loss.png"))

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    IMG_DIR = os.path.join(BASE_DIR, "data/sampled_images/wikiart")
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        print("Starting Powerful Training on HPC...")
        train_model(CSV_PATH, IMG_DIR)
    else:
        print(f"Dataset not found at:\n{CSV_PATH}\n{IMG_DIR}")
