# ArtEmis Image Caption Generation

This project implements image captioning models on the ArtEmis dataset using two approaches: CNN+LSTM and Vision Transformers. The models generate descriptive captions for artwork images by learning from emotion descriptions.

## Dataset

ArtEmis Dataset from WikiArt collection containing artwork images paired with human-written emotional captions. The dataset includes approximately 80,000 image-caption pairs. For this project, we use a subset of 10,000-20,000 images.

## Data Storage

## Prerequisites

- **Images**: Stored in `data/images/wikiart/`, organized by art style subdirectories (e.g., `Abstract_Expressionism`, `Cubism`).
- **Metadata**: The dataset CSV file is located at `data/images/artemis_dataset_release_v0.csv`.

## Setup Instructions

1. Create and activate virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

4. Download pre-trained embeddings (optional):

```bash
python src/utils/download_embeddings.py
```

## Dataset Preprocessing

The preprocessing pipeline performs the following steps:

### Image Preprocessing
1. Resize images to 224x224 pixels
2. Normalize pixel values to [0,1] range
3. Apply data augmentation during training (random flips, brightness, contrast)
4. Convert to tensor format

### Text Preprocessing
1. Convert captions to lowercase
2. Remove punctuation and special characters
3. Tokenize using whitespace splitting
4. Build vocabulary (5,000-15,000 tokens depending on configuration)
5. Add special tokens: `<start>`, `<end>`, `<pad>`, `<unk>`
6. Pad or truncate sequences to maximum length of 50 tokens



## Project Structure

```
iml-a3/
├── data/
│   ├── artemis_dataset_release_v0.csv
│   ├── wikiart/                       # Image files by art style
│   └── processed/                     # Preprocessed metadata
├── src/
│   ├── approach_1/                    # CNN + LSTM models
│   │   ├── basic/                     # Basic CNN+LSTM
│   │   └── powerful/                  # Enhanced CNN+LSTM with multi-task learning
│   ├── approach_2/                    # Vision Transformer models
│   │   ├── pretrained/                # Frozen ViT encoder
│   │   ├── finetuning/                # Trainable ViT encoder
│   │   └── flash_attention/           # Optimized transformer decoder
│   ├── helpers/
│   │   ├── metrics.py                 # BLEU, ROUGE, METEOR metrics
│   │   ├── validation.py              # Model evaluation
│   │   └── visualize.py               # Attention visualization
│   ├── preprocessing/
│   │   └── preprocess_unified.py      # Data preprocessing
│   └── utils/
│       └── dataset_torch.py           # DataLoader implementation
├── models/                            # Trained model checkpoints
├── notebooks/
│   ├── ArtEmis_Caption_Generation.ipynb  # Main report notebook
│   └── eda.ipynb                      # Exploratory data analysis
└── requirements.txt
```

## Training Models

### Approach 1: CNN + LSTM

Train with TF-IDF embeddings:
```bash
python src/approach_1/powerful/train.py
```

The script trains a model with:
- Custom 5-block CNN encoder
- 2-layer LSTM decoder with input feeding
- Multi-task learning (captioning + emotion prediction)
- Automatic task weighting

### Approach 2: Vision Transformer

Train with flash attention:
```bash
python src/approach_2/flash_attention/train.py
```

The script uses:
- Pre-trained ViT encoder (frozen)
- 6-layer transformer decoder
- Cross-attention between image patches and caption tokens

Training configurations can be modified in the respective train.py files. Models are saved to `models/` directory with training logs.

## Evaluation

Run evaluation on trained models:

```bash
python src/helpers/validation.py
```

Or use the validation functions in the Jupyter notebook. Metrics computed:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-1, ROUGE-2, ROUGE-L
- METEOR
- Emotion classification accuracy

## Text Embeddings

Three embedding strategies are implemented:

1. **TF-IDF**: Statistical embeddings based on term frequency-inverse document frequency. Computed from training captions.

2. **Word2Vec**: Pre-trained word embeddings from Google News corpus (300D). Loaded using gensim.

3. **GloVe**: Pre-trained global vectors from Wikipedia (100D). Loaded using gensim.

Embeddings are initialized in the model's embedding layer and can be frozen or fine-tuned during training.

## Generating Captions

The main notebook `notebooks/ArtEmis_Caption_Generation.ipynb` contains code for generating captions on test images. Models use beam search decoding with beam size of 5 for better quality captions.

## Model Checkpoints

Trained models are saved in the following format:
- `models/approach-1-powerful/{embedding_type}/model_final.pth`
- `models/approach-2-flash/{embedding_type}/model_final.pth`

Each model directory also contains:
- `vocab.pkl`: Vocabulary mapping
- Loss curves

