import sys
import os
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import resource

# Increase file descriptor limit
try:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except Exception as e:
    print(f"Could not increase file limit: {e}")

# Fix for "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')

from src.utils.dataset_torch import get_loader, save_vocab
from src.approach_2.scratch.caption_model import ViTCaptionModel
from src.approach_2.embeddings import get_tfidf_embeddings, get_pretrained_embeddings

def train_model(
    csv_path,
    image_dir,
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/approach-2-scratch")),
    epochs=50,
    batch_size=64,
    learning_rate=2e-4,
    image_size=(256, 256),
    vocab_size=15000,
    embedding_dim=256,
    embedding_type="tfidf"
):
    output_dir = os.path.join(output_dir, embedding_type)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), # [0, 1]
    ])

    # Load Data
    print("Loading data...")
    num_workers = 0 if device.type == "mps" else 8
    print(f"Using {num_workers} workers for DataLoader")

    train_loader, train_dataset = get_loader(
        image_dir, 
        csv_path, 
        transform, 
        batch_size=batch_size,
        max_length=50,
        num_workers=num_workers,
        pin_memory=(device.type != "mps")
    )
    
    vocab = train_dataset.vocab
    save_vocab(vocab, os.path.join(output_dir, "vocab.pkl"))
    
    # Embeddings
    embedding_matrix = None
    if embedding_type == "tfidf":
        captions = train_dataset.captions
        vocab_list = [vocab.itos[i] for i in range(len(vocab))]
        embedding_matrix = get_tfidf_embeddings(vocab_list, captions, embedding_dim=embedding_dim)
    elif embedding_type == "glove":
        vocab_list = [vocab.itos[i] for i in range(len(vocab))]
        embedding_matrix = get_pretrained_embeddings(vocab_list, "glove-wiki-gigaword-100", embedding_dim)
    elif embedding_type == "word2vec":
        vocab_list = [vocab.itos[i] for i in range(len(vocab))]
        embedding_matrix = get_pretrained_embeddings(vocab_list, "word2vec-google-news-300", embedding_dim)

    # Model
    model = ViTCaptionModel(
        image_size=image_size[0],
        vocab_size=len(vocab),
        embed_dim=embedding_dim,
        embedding_matrix=embedding_matrix
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    criterion_emotion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for imgs, captions, emotions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)
            emotions = emotions.to(device)

            # Forward
            # Input to decoder: <start> ... w_n
            # Target: w_1 ... <end>
            caption_logits, emotion_logits = model(imgs, captions[:, :-1]) # (B, SeqLen-1, Vocab)
            targets = captions[:, 1:] # (B, SeqLen-1)

            loss_caption = criterion(caption_logits.reshape(-1, caption_logits.shape[-1]), targets.reshape(-1))
            loss_emotion = criterion_emotion(emotion_logits, emotions)
            
            loss = loss_caption + loss_emotion

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), cap_loss=loss_caption.item(), emo_loss=loss_emotion.item())

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
    # Use absolute paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/images/artemis_dataset_release_v0.csv")
    IMG_DIR = os.path.join(BASE_DIR, "data/sampled_images/wikiart")
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        train_model(CSV_PATH, IMG_DIR, epochs=5)
    else:
        print(f"Dataset not found at:\n{CSV_PATH}\n{IMG_DIR}")
