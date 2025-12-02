import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.dataset_torch import get_loader, save_vocab
from src.approach_1.basic.caption_model import CNNLSTMModel
from src.approach_1.embeddings import get_tfidf_embeddings, get_pretrained_embeddings

def train_model(
    csv_path,
    image_dir,
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/approach-1-basic")),
    epochs=20,
    batch_size=128,
    learning_rate=1e-3,
    image_size=(224, 224),
    embedding_dim=300,
    embedding_type="tfidf" # "tfidf", "glove-wiki-gigaword-100", "word2vec"
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
        # Pre-trained
        # Map simple names to gensim model names if needed, or pass directly
        model_name = embedding_type
        if embedding_type == "glove": model_name = "glove-wiki-gigaword-300" # Example
        
        embedding_matrix = get_pretrained_embeddings(
            vocab_list, 
            model_name=model_name, 
            embedding_dim=embedding_dim,
            captions=captions # Needed for word2vec training
        )
    
    # Model
    model = CNNLSTMModel(
        vocab_size=len(vocab),
        embed_dim=embedding_dim,
        hidden_dim=512,
        embedding_matrix=embedding_matrix
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=True, file=sys.stdout)
        
        for imgs, captions, _ in loop: # Ignore emotions for this model
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Input: captions[:, :-1] (all except last)
            # Target: captions[:, 1:] (all except first)
            outputs = model(imgs, captions[:, :-1])
            targets = captions[:, 1:]
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            
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
        train_model(CSV_PATH, IMG_DIR)
    else:
        print(f"Dataset not found at:\n{CSV_PATH}\n{IMG_DIR}")
