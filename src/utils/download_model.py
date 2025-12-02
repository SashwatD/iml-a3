import os
from transformers import ViTModel, ViTImageProcessor

# Make downloads directory if it doesn't exist
os.makedirs('../downloads', exist_ok=True)

# 1. Define the model ID and where you want to save it
model_id = 'google/vit-base-patch16-224'
save_path = '../downloads/google_vit_local'  # This will create a folder in your current directory

print(f"Downloading {model_id}...")

# 2. Download Model
model = ViTModel.from_pretrained(model_id)
model.save_pretrained(save_path)

# 3. Download Processor (Config/Feature Extractor)
# Even if you don't use the processor in the model class, 
# downloading the config is essential for 'from_pretrained' to work offline.
processor = ViTImageProcessor.from_pretrained(model_id)
processor.save_pretrained(save_path)

print(f"Model saved to {os.path.abspath(save_path)}")