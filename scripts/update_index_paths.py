import os
import sys
import pandas as pd

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config

def update_csv_paths():
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    for target in ["chirps", "oya"]:
        csv_path = os.path.join(metadata_dir, f"dataset_index_{target}.csv")
        if not os.path.exists(csv_path):
            print(f"Index file not found: {csv_path}")
            continue
            
        print(f"Updating paths in index: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Replace relative paths with absolute paths from config
        paths_updated = 0
        for col in ["era5_path", "era5_pl_path", "target_path", "npz_path"]:
            if col in df.columns:
                # Ensure values are strings before replacement
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace("./data/raw", Config.RAW_DATA_DIR, regex=False)
                df[col] = df[col].str.replace("./data/processed", Config.PROCESSED_DATA_DIR, regex=False)
                paths_updated += 1
                
        df.to_csv(csv_path, index=False)
        print(f"Successfully updated paths for {target} ({paths_updated} columns adjusted).")

if __name__ == "__main__":
    update_csv_paths()
