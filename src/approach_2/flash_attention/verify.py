import torch
import os
from PIL import Image
from torchvision import transforms
from src.approach_2.flash_attention.caption_model import FlashViTCaptionModel
from src.utils.dataset_torch import load_vocab

def generate_caption(model, image_path, vocab, max_length=50, device="cpu", image_size=(256, 256)):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    result_caption = []
    
    with torch.no_grad():
        # Encoder
        enc_out = model.patch_embed(img)
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, is_causal=False)
            
        # Decoder
        inputs = torch.tensor([vocab.stoi["<start>"]]).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            B, SeqLen = inputs.shape
            positions = torch.arange(0, SeqLen).unsqueeze(0).to(device)
            
            tgt_emb = model.token_emb(inputs) + model.pos_emb(positions)
            
            dec_out = tgt_emb
            for layer in model.decoder_layers:
                dec_out = layer(dec_out, enc_out)
                
            outputs = model.fc_out(dec_out)
            predictions = outputs[:, -1, :]
            predicted_id = predictions.argmax(1).item()
            
            if predicted_id == vocab.stoi["<end>"]:
                break
                
            result_caption.append(vocab.itos[predicted_id])
            inputs = torch.cat([inputs, torch.tensor([[predicted_id]]).to(device)], dim=1)
            
    return " ".join(result_caption)

if __name__ == "__main__":
    MODEL_PATH = "models/approach-2-flash/model_epoch_5.pth"
    VOCAB_PATH = "models/approach-2-flash/vocab.pkl"
    IMG_PATH = "data/sampled_images/wikiart/Impressionism/claude-monet_water-lilies-1916.jpg"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        
        model = FlashViTCaptionModel(vocab_size=len(vocab)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        if os.path.exists(IMG_PATH):
            caption = generate_caption(model, IMG_PATH, vocab, device=device)
            print(f"Generated: {caption}")
        else:
            print("Image not found.")
    else:
        print("Model or vocab not found.")
