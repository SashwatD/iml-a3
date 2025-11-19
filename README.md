# ArtEmis Image Caption Generation

Generate emotional, descriptive captions for artwork images using deep learning.

## Models

1. **CNN + LSTM**: Custom CNN encoder with LSTM decoder
2. **Vision Transformer**: Attention-based architecture (to be implemented)

## Dataset

- **ArtEmis Dataset**: 80,000 artworks with emotional captions
- **WikiArt Images**: Organized by art style
- **Subset**: Using 5-10k images for training

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Project Structure

```
iml-a3/
├── data/
│   ├── artemis_dataset_release_v0.csv
│   └── wikiart/
├── src/
│   ├── cnn_encoder.py
│   ├── lstm_decoder.py
│   ├── tokenizer.py
│   └── caption_model.py
├── notebooks/
│   └── 01_explore_artemis_data.ipynb
└── models/
```

## Usage

### Train Model
```python
from src.caption_model import build_caption_model, compile_model

encoder, decoder, model = build_caption_model(vocab_size=5000)
compile_model(model)
# Training code here
```

### Generate Captions
```python
from src.caption_model import generate_caption_greedy

caption = generate_caption_greedy(encoder, decoder, image, tokenizer)
print(caption)
```

## Architecture

### CNN Encoder
- 4 convolutional blocks (32→64→128→256 filters)
- BatchNorm + ReLU + MaxPool
- Output: 256D feature vector
- Parameters: ~26M

### LSTM Decoder
- 2-layer LSTM (256 units each)
- Word embeddings (256D)
- Dropout (0.3) for regularization
- Parameters: ~5-8M

## Evaluation Metrics

- BLEU (1-4 gram)
- ROUGE-L
- Qualitative analysis

## References

- ArtEmis Dataset: Achlioptas et al., CVPR 2021
- Word2Vec: Mikolov et al., 2013
- GloVe: Pennington et al., 2014
