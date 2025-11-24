# Approach 2: Vision Transformer for Image Captioning

This directory contains the implementation of a Vision Transformer (ViT) based image captioning model.

## Architecture Overview

The model follows an Encoder-Decoder architecture:

### 1. Encoder (Vision Transformer)
-   **Input**: Images resized to `(256, 256, 3)`.
-   **Patch Extraction**: The image is divided into fixed-size patches (16x16).
-   **Patch Embedding**: Each patch is linearly projected to a dense vector (`projection_dim=256`).
-   **Position Embedding**: Learnable position embeddings are added to retain spatial information.
-   **Transformer Blocks**: 4 layers of Transformer Encoder blocks, each containing:
    -   Multi-Head Self-Attention (4 heads).
    -   Feed-Forward Network (FFN).
    -   Layer Normalization and Dropout.

### 2. Decoder (Transformer Decoder)
-   **Input**: Tokenized caption sequences.
-   **Embedding**: Word embeddings + Position embeddings.
-   **Transformer Blocks**: 4 layers of Transformer Decoder blocks, each containing:
    -   **Masked Self-Attention**: Causal masking ensures the model only attends to past tokens.
    -   **Cross-Attention**: Attends to the Encoder's output (image features).
    -   Feed-Forward Network (FFN).
    -   Layer Normalization and Dropout.
-   **Output**: Dense layer projecting to vocabulary size (Softmax).

## Implementation Details

### File Structure
-   `caption_model.py`: Defines the Keras model, layers, and custom loss/metric functions.
-   `train.py`: Training script with mixed precision, callbacks, and data pipeline integration.
-   `verify.py`: Evaluation script for generating captions and calculating BLEU scores.
-   `tests/`: Contains automated tests for the pipeline and model.

### Key Features
-   **Causal Masking**: Explicitly implemented in the decoder to prevent information leakage from future tokens.
-   **Data Augmentation**: Random flip, rotation, and contrast adjustments during training.
-   **Vectorizer Persistence**: The `TextVectorization` vocabulary is saved to disk to ensure consistent tokenization between training and inference.
-   **Mixed Precision**: Supports `mixed_float16` for faster training on compatible GPUs.
-   **Embedding Strategies**: Supports multiple embedding types:
    -   **Learned**: Standard trainable Keras Embedding layer.
    -   **TF-IDF (LSA)**: Dense vectors reduced from TF-IDF matrices using SVD.
    -   **Pre-trained**: Supports loading GloVe (`glove-wiki-gigaword-100`) and Word2Vec (`word2vec-google-news-300`) via `gensim`.

## Usage

### Training
To train the model, run:
```bash
# Default (Learned Embeddings)
python -m src.approach_2.train

# TF-IDF (LSA)
python -m src.approach_2.train --embedding_type tfidf

# GloVe (Pre-trained)
python -m src.approach_2.train --embedding_type glove

# Word2Vec (Pre-trained)
python -m src.approach_2.train --embedding_type word2vec
```
This will:
1.  Load and preprocess the ArtEmis dataset.
2.  Create a `tf.data.Dataset` with augmentation.
3.  Train the model for the specified epochs.
4.  Save the best model weights and the vectorizer.

### Verification / Inference
To evaluate the model:
```bash
python -m src.approach_2.verify
```
This will:
1.  Load the saved model and vectorizer.
2.  Generate captions for a sample of the test set.
3.  Calculate BLEU-1 and BLEU-4 scores.

### Testing
To run the automated pipeline tests:
```bash
python src/approach_2/tests/test_pipeline.py
```
