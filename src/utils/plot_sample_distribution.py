import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(sample_dir="data/sampled_images/wikiart", output_file="data/sampled_images/distribution.png"):
    if not os.path.exists(sample_dir):
        print(f"Directory {sample_dir} does not exist.")
        return

    styles = []
    counts = []

    print("Counting images in subdirectories...")
    for entry in os.scandir(sample_dir):
        if entry.is_dir():
            style_name = entry.name
            # Count files in the directory
            num_files = len([f for f in os.listdir(entry.path) if os.path.isfile(os.path.join(entry.path, f))])
            styles.append(style_name)
            counts.append(num_files)

    # Sort by count descending
    data = sorted(zip(styles, counts), key=lambda x: x[1], reverse=True)
    styles, counts = zip(*data)

    print(f"Found {len(styles)} styles.")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(styles), palette="viridis")
    plt.xlabel("Number of Images")
    plt.ylabel("Art Style")
    plt.title("Distribution of Images per Art Style in Sampled Dataset")
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Check if seaborn is installed, otherwise use basic matplotlib
    try:
        import seaborn
    except ImportError:
        print("Seaborn not found, using matplotlib defaults.")
        sns = None
        
    print("Plotting sampled dataset distribution...")
    plot_distribution(sample_dir="data/sampled_images/wikiart", output_file="data/visualizations/distribution_sampled.png")
    
    print("\nPlotting original dataset distribution...")
    plot_distribution(sample_dir="data/images/wikiart", output_file="data/visualizations/distribution_original.png")
