import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_sample_dataset(
    source_csv="data/images/artemis_dataset_release_v0.csv",
    source_img_dir="data/images/wikiart",
    target_base_dir="data/sampled_images_2",
    sample_size=20000,
    stratify_col="art_style"
):
    print(f"Loading dataset from {source_csv}...")
    df = pd.read_csv(source_csv)
    
    # Ensure image paths are correct/exist
    # We assume standard structure: source_img_dir/art_style/painting.jpg
    
    print(f"Performing stratified sampling (n={sample_size})...")
    try:
        sample_df, _ = train_test_split(
            df, 
            train_size=sample_size, 
            stratify=df[stratify_col], 
            random_state=42
        )
    except ValueError as e:
        print(f"Stratified sampling failed: {e}. Falling back to random sampling.")
        sample_df = df.sample(n=sample_size, random_state=42)
        
    print(f"Sampled {len(sample_df)} records.")
    
    # Create target directories
    target_csv_path = os.path.join(target_base_dir, "artemis_dataset_release_v0.csv")
    target_img_dir = os.path.join(target_base_dir, "wikiart")
    
    os.makedirs(target_base_dir, exist_ok=True)
    os.makedirs(target_img_dir, exist_ok=True)
    
    # Copy images
    print("Copying images...")
    success_count = 0
    missing_count = 0
    
    # We need to deduplicate images because one image might have multiple captions (rows)
    # We only need to copy the file once.
    unique_images = sample_df[['art_style', 'painting']].drop_duplicates()
    
    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images)):
        style = row['art_style']
        painting = row['painting']
        filename = f"{painting}.jpg"
        
        src_path = os.path.join(source_img_dir, style, filename)
        dst_dir = os.path.join(target_img_dir, style)
        dst_path = os.path.join(dst_dir, filename)
        
        if os.path.exists(src_path):
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst_path): # Don't overwrite if already copied
                shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            # print(f"Missing: {src_path}")
            missing_count += 1
            
    print(f"Copied {success_count} images. Missing: {missing_count}")
    
    # Save CSV
    print(f"Saving CSV to {target_csv_path}...")
    sample_df.to_csv(target_csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    create_sample_dataset()
