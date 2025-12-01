import sys
import os
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.utils.dataset_torch import load_vocab, EMOTION_MAP
from src.approach_2.scratch.caption_model import ViTCaptionModel
from src.approach_2.pretrained.caption_model import PretrainedViTCaptionModel
from src.approach_2.flash_attention.caption_model import FlashViTCaptionModel

def generate_caption(model, image_path, vocab, variant, max_length=50, device="cpu", image_size=(224, 224), beam_size=3, temperature=1.0):
    model.eval()
    
    # Preprocessing depends on variant
    if variant == "pretrained":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    # Extract true caption (for reference)
    painting_name = image_path.split("/")[-1].split(".")[0]
    try:
        df = pd.read_csv(CSV_PATH)
        row = df[df["painting"] == painting_name].iloc[0]
        true_caption = row["utterance"]
        true_emotion = row["emotion"]
    except Exception as e:
        true_caption = "Unknown (Could not find in CSV)"
        true_emotion = "Unknown"

    with torch.no_grad():
        # Encoder Pass
        if variant == "pretrained":
            vit_out = model.vit(img).last_hidden_state
            enc_out = model.visual_projection(vit_out)
            
        elif variant == "scratch":
            enc_out = model.patch_embed(img)
            for layer in model.encoder_layers:
                enc_out = layer(enc_out)
            
        elif variant == "flash":
            enc_out = model.patch_embed(img)
            for layer in model.encoder_layers:
                enc_out = layer(enc_out, is_causal=False)
        
        # Beam Search
        k = beam_size
        sequences = [[0.0, [vocab.stoi["<start>"]]]]
        
        for _ in range(max_length):
            all_candidates = []
            
            for score, seq in sequences:
                if seq[-1] == vocab.stoi["<end>"]:
                    all_candidates.append([score, seq])
                    continue
                
                inputs = torch.tensor([seq]).to(device)
                B, SeqLen = inputs.shape
                
                positions = torch.arange(0, SeqLen).unsqueeze(0).to(device)
                
                # Embedding
                tgt_emb = model.token_emb(inputs) + model.pos_emb(positions)

                # Mask
                tgt_mask = torch.triu(torch.ones(SeqLen, SeqLen) * float('-inf'), diagonal=1).to(device)
                
                # Decoder Pass
                dec_out = tgt_emb
                for layer in model.decoder_layers:
                    if variant == "pretrained":
                        dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
                    elif variant == "scratch":
                        dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
                    elif variant == "flash":
                        dec_out = layer(dec_out, enc_out) 

                outputs = model.fc_out(dec_out)
                predictions = outputs[:, -1, :] / temperature
                log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
                
                topk_probs, topk_ids = log_probs.topk(k, dim=1)
                
                for i in range(k):
                    token = topk_ids[0, i].item()
                    prob = topk_probs[0, i].item()
                    all_candidates.append([score + prob, seq + [token]])
            
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = ordered[:k]
            
            if all(seq[-1] == vocab.stoi["<end>"] for _, seq in sequences):
                break
        
        best_seq = sequences[0][1]
        
        result_caption = []
        for token in best_seq:
            if token == vocab.stoi["<start>"]: continue
            if token == vocab.stoi["<end>"]: break
            result_caption.append(vocab.itos[token])
            
    return " ".join(result_caption), true_caption, true_emotion

if __name__ == "__main__":
    VARIANT = "pretrained" # Options: "scratch", "pretrained", "flash"
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    
    if VARIANT == "pretrained":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-pretrained/model_final.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-pretrained/vocab.pkl")
        IMAGE_SIZE = (224, 224)
    elif VARIANT == "scratch":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-scratch/model_final.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-scratch/vocab.pkl")
        IMAGE_SIZE = (256, 256)
    elif VARIANT == "flash":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-flash/model_final.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-flash/vocab.pkl")
        IMAGE_SIZE = (256, 256)
        
    # Test Image
    IMG_PATH = os.path.join(BASE_DIR, "data/sampled_images/wikiart/Early_Renaissance/filippo-lippi_two-saints.jpg")
    
    # Generation Params
    BEAM_SIZE = 3
    TEMPERATURE = 1.0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Variant: {VARIANT}")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        
        # Initialize Model
        if VARIANT == "pretrained":
            model = PretrainedViTCaptionModel(vocab_size=len(vocab)).to(device)
        elif VARIANT == "scratch":
            model = ViTCaptionModel(image_size=IMAGE_SIZE[0], vocab_size=len(vocab)).to(device)
        elif VARIANT == "flash":
            model = FlashViTCaptionModel(image_size=IMAGE_SIZE[0], vocab_size=len(vocab)).to(device)
            
        # Load Weights
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Note: If model architecture changed (e.g. added emotion embeddings), you need to retrain.")
            sys.exit(1)
        
        if os.path.exists(IMG_PATH):
            caption, true_caption, emotion = generate_caption(
                model, 
                IMG_PATH, 
                vocab, 
                variant=VARIANT,
                device=device, 
                image_size=IMAGE_SIZE,
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE
            )
            print(f"Generated: {caption}")
            print(f"True Caption: {true_caption}")
            print(f"Emotion: {emotion}")
        else:
            print(f"Image not found: {IMG_PATH}")
    else:
        print(f"Model or vocab not found at:\n{MODEL_PATH}\n{VOCAB_PATH}")
