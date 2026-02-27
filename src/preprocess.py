import argparse
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(data_file: str, output_dir: str) -> None:
    """
    Preprocess raw protein sequence data for model training.

    This function loads the raw data, cleans it, encodes labels, and splits
    it into train/validation/test sets. The split strategy must handle the
    extreme class imbalance in the Pfam dataset.

    Parameters
    ----------
    data_file : str
        Path to the combined raw data CSV file.
    output_dir : str
        Directory where processed files will be saved.

    Steps
    -----
    1. Load the data with pandas
    2. Remove rows with missing values
    3. Encode the 'family_accession' column with LabelEncoder
    4. Design and implement a split strategy that handles class imbalance
    5. Save train.csv, val.csv, and test.csv to output_dir

    Notes
    -----
    sklearn's train_test_split with stratify will fail on this dataset
    because some classes have only one sample. You need to implement
    a custom strategy.
    """
    data_path = Path(data_file)
    output_path = Path(output_dir)

    df = pd.read_csv(data_path)
    print(len(df))
    
    df = df.dropna()
    print(len(df))
    
    #Encoder les colonnes categorielles
    
    le = LabelEncoder()
    df['family_id'] = le.fit_transform(df['family_accession'])
    
    # Split
    
    class_counts = df['family_id'].value_counts()
    
    classes_many = class_counts[class_counts >= 10].index
    classes_few = class_counts[(class_counts >= 3) & (class_counts < 10)].index
    classes_rare = class_counts[class_counts < 3].index
    
    print(f"Classes with ≥10 samples: {len(classes_many)}")
    print(f"Classes with 3-9 samples: {len(classes_few)}")
    print(f"Classes with <3 samples: {len(classes_rare)}")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # split stratifié pour les classes avec bcp d'échantillons
    df_many = df[df['family_id'].isin(classes_many)]
    if len(df_many) > 0:
        train_many, temp = train_test_split(df_many, test_size=0.2, stratify=df_many['family_id'], random_state=42)
        val_many, test_many = train_test_split(temp, test_size=0.5, stratify=temp['family_id'], random_state=42)
        train_dfs.append(train_many)
        val_dfs.append(val_many)
        test_dfs.append(test_many)
        print(f"Split {len(df_many)} samples from frequent classes")
    
    #1 echantillon pour test, reste dans train
    df_few = df[df['family_id'].isin(classes_few)]
    if len(df_few) > 0:
        for class_id in classes_few:
            class_data = df_few[df_few['family_id'] == class_id]
            test_sample = class_data.sample(n=1, random_state=42)
            train_sample = class_data.drop(test_sample.index)
            train_dfs.append(train_sample)
            test_dfs.append(test_sample)
        print(f"Split {len(df_few)} samples from medium classes")
    
    # pour les données rare, tout dans train
    df_rare = df[df['family_id'].isin(classes_rare)]
    if len(df_rare) > 0:
        train_dfs.append(df_rare)
        print(f"Put {len(df_rare)} samples from rare classes into train only")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Pfam data.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
