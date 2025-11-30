import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.dataset_torch import get_loader, save_vocab
from src.approach_2.scratch.caption_model import ViTCaptionModel
from src.approach_2.embeddings import get_tfidf_embeddings, get_pretrained_embeddings

def train_model(
    csv_path,
    image_dir,
    output_dir="models/approach-2-scratch",
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    image_size=(256, 256),
    vocab_size=5000,
    embedding_dim=256,
    embedding_type="tfidf"
):
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
    train_loader, train_dataset = get_loader(
        image_dir, 
        csv_path, 
        transform, 
        batch_size=batch_size,
        max_length=50
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward
            # Input to decoder: <start> ... w_n
            # Target: w_1 ... <end>
            outputs = model(imgs, captions[:, :-1]) # (B, SeqLen-1, Vocab)
            targets = captions[:, 1:] # (B, SeqLen-1)

            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

            # Backward
            optimizer.zero_grad()
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

    # Plot history
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "loss.png"))

if __name__ == "__main__":
    CSV_PATH = "data/images/artemis_dataset_release_v0.csv"
    IMG_DIR = "data/sampled_images/wikiart"
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        train_model(CSV_PATH, IMG_DIR, epochs=5)
    else:
        print("Dataset not found.")
