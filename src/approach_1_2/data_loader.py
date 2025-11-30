# Data Loader for Improved Approach - Supports 224x224 images
# Reuses data loading logic but with updated image size

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image


class ImprovedArtEmisDataGenerator(Sequence):
    """
    Memory-efficient data generator for improved model.
    
    Key Difference from baseline:
    - Supports larger images (224x224 vs 128x128)
    - Otherwise same memory-efficient streaming
    
    Why this matters:
    - 224x224 = 3x more pixels than 128x128
    - Loads on-the-fly, so no extra RAM needed
    - Just slightly slower per batch due to more pixels
    """
    
    def __init__(self, metadata_path, batch_size=32, image_size=(224, 224),
                 shuffle=True, augment=False, **kwargs):
        """
        Initialize data generator.
        
        Parameters:
            metadata_path: Path to preprocessed metadata .pkl file
            batch_size: Batch size (reduced to 16-24 for larger images)
            image_size: (224, 224) for improved model
            shuffle: Shuffle data each epoch
            augment: Apply data augmentation (training only)
        """
        super().__init__(**kwargs)
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        self.image_paths = data['image_paths']
        self.captions_input = data['captions_input']
        self.captions_target = data['captions_target']
        
        self.n_samples = len(self.image_paths)
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        print(f"[DataLoader] Loaded {self.n_samples} samples")
        print(f"[DataLoader] Image size: {image_size}")
        print(f"[DataLoader] Batch size: {batch_size}")
        print(f"[DataLoader] Augmentation: {'ON' if augment else 'OFF'}")
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        
        return self._generate_batch(batch_indices)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_image(self, image_path):
        """
        Load and preprocess image.
        
        Steps:
        1. Load from disk
        2. Resize to 224x224 (larger than baseline)
        3. Normalize to [0, 1]
        4. Optional augmentation
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            img = img.resize(self.image_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Augmentation (training only)
            if self.augment:
                img_array = self._augment_image(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"[WARNING] Failed to load {image_path}: {e}")
            # Return black image as fallback
            return np.zeros((self.image_size[0], self.image_size[1], 3), 
                          dtype=np.float32)
    
    def _augment_image(self, image):
        """
        Apply data augmentation to reduce overfitting.
        
        Augmentations:
        - Random horizontal flip (50% chance)
        - Random brightness adjustment (±20%)
        - Random contrast adjustment (±20%)
        
        Why augmentation?
        - Creates more diverse training examples
        - Model learns robust features
        - Reduces overfitting
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        # Random brightness
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0.0, 1.0)
        
        # Random contrast
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast_factor + mean, 0.0, 1.0)
        
        return image
    
    def _generate_batch(self, batch_indices):
        images = []
        captions_input = []
        captions_target = []
        
        for idx in batch_indices:
            # Load image
            img = self._load_image(self.image_paths[idx])
            
            # Get captions
            cap_in = self.captions_input[idx]
            cap_out = self.captions_target[idx]
            
            images.append(img)
            captions_input.append(cap_in)
            captions_target.append(cap_out)
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        captions_input = np.array(captions_input, dtype=np.int32)
        captions_target = np.array(captions_target, dtype=np.int32)
        
        # Return in format expected by Keras (tuple of inputs, not list!)
        return (images, captions_input), captions_target


if __name__ == "__main__":
    # Test the data loader
    print("="*70)
    print("Testing Improved Data Loader")
    print("="*70)
    
    train_path = Path('data/processed/train_data_metadata.pkl')
    
    if train_path.exists():
        # Create generator
        train_gen = ImprovedArtEmisDataGenerator(
            metadata_path=train_path,
            batch_size=16,  # Smaller batch for 224x224
            image_size=(224, 224),
            shuffle=True,
            augment=True
        )
        
        print(f"\nTotal batches: {len(train_gen)}")
        
        # Load one batch
        (images, captions_in), captions_out = train_gen[0]
        
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Captions input: {captions_in.shape}")
        print(f"  Captions target: {captions_out.shape}")
        
        print("\n[SUCCESS] Data loader working correctly!")
    else:
        print(f"\n[WARNING] Metadata not found at {train_path}")
        print("Run preprocessing first!")

