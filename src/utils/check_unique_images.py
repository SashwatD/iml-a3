import pandas as pd

csv_path = "data/sampled_images/artemis_dataset_release_v0.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"Total rows (captions): {len(df)}")
    
    # Construct full image path or just use painting/art_style columns to identify unique images
    # The dataset usually has 'art_style' and 'painting' columns.
    if 'painting' in df.columns and 'art_style' in df.columns:
        df['unique_id'] = df['art_style'] + "/" + df['painting']
        unique_images = df['unique_id'].nunique()
        print(f"Unique images referenced in CSV: {unique_images}")
        
        # Count captions per image
        counts = df['unique_id'].value_counts()
        print("\nTop 10 images with most captions:")
        print(counts.head(10))
        
        print(f"\nNumber of images with > 1 caption: {(counts > 1).sum()}")
    else:
        print("Columns 'painting' or 'art_style' not found. Checking available columns:", df.columns)

except Exception as e:
    print(f"Error reading CSV: {e}")
