import os
import gensim.downloader as api
from gensim.models import KeyedVectors

def download_embeddings():
    # Define models to download
    # Only GloVe and Word2Vec as requested
    models_to_download = [
        'glove-wiki-gigaword-100',
        'word2vec-google-news-300'
    ]

    # Create downloads directory
    save_dir = './downloads/embeddings'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving embeddings to: {os.path.abspath(save_dir)}")

    for model_name in models_to_download:
        print(f"\nProcessing {model_name}...")
        
        # Define output path
        output_path = os.path.join(save_dir, f"{model_name}.bin")
        
        if os.path.exists(output_path):
            print(f"  File {output_path} already exists. Skipping.")
            continue

        try:
            print(f"  Downloading/Loading {model_name} via gensim...")
            # api.load() downloads the model to gensim-data and returns the model object
            model = api.load(model_name)
            
            print(f"  Saving to {output_path}...")
            # Save in Word2Vec binary format (compatible with KeyedVectors.load_word2vec_format)
            model.save_word2vec_format(output_path, binary=True)
            print("  Done.")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")

    print("\nAll requested embeddings processed.")

if __name__ == "__main__":
    download_embeddings()
