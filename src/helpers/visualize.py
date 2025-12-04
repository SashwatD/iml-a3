import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

def visualize_attention_map(model, image_path, device='cuda'):
    """
    Visualizes the attention map of the ViT Encoder's last layer for the CLS token.
    Returns the matplotlib figure object.
    """
    model.eval()
    
    # Load and Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    input_tensor = transform(raw_image).unsqueeze(0).to(device)
    
    # Forward pass through ViT only, requesting attentions
    with torch.no_grad():
        # Handle FlashViTCaptionModel wrapper vs raw ViT
        if hasattr(model, 'vit'):
             vit_model = model.vit
        else:
             vit_model = model
             
        # FORCE config to output attentions
        # Note: The model MUST be loaded with attn_implementation="eager" for this to work
        # if it was originally SDPA.
        if hasattr(vit_model, 'config'):
            vit_model.config.output_attentions = True
        
        # Access the .vit submodule directly
        try:
            outputs = vit_model(pixel_values=input_tensor, output_attentions=True, return_dict=True)
        except TypeError:
             # Fallback for models that might not accept pixel_values kwarg
             outputs = vit_model(input_tensor, output_attentions=True, return_dict=True)
        except ValueError as e:
            print(f"Visualization Error: {e}")
            print("Tip: Ensure the ViT model is loaded with attn_implementation='eager'.")
            return None
        
    # Extract Attention Weights
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        print("Error: Model did not return attentions.")
        return None

    # We take the last layer
    last_layer_attn = outputs.attentions[-1]
    
    # Average across all heads
    attn_avg = torch.mean(last_layer_attn, dim=1).squeeze(0) # (SeqLen, SeqLen)
    
    # 4. Get Attention of CLS token to all patches
    cls_attn_map = attn_avg[0, 1:] # (196,) for 14x14 patches
    
    # Reshape to grid (14x14)
    num_patches = cls_attn_map.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    if grid_size * grid_size != num_patches:
        print(f"Warning: Patch count {num_patches} is not a perfect square (grid_size={np.sqrt(num_patches)}). Cannot reshape.")
        return None

    attn_heatmap = cls_attn_map.reshape(grid_size, grid_size).cpu().numpy()
    
    # Visualization
    # Resize heatmap to original image size
    raw_image_np = np.array(raw_image)
    heatmap_resized = cv2.resize(attn_heatmap, (raw_image_np.shape[1], raw_image_np.shape[0]))
    
    # Normalize heatmap for better visualization
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(raw_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(heatmap_resized, cmap='jet')
    ax2.set_title("Attention Heatmap")
    ax2.axis('off')
    
    # Overlay
    ax3.imshow(raw_image)
    ax3.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    plt.tight_layout()
    return fig
