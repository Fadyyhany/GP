"""
Phase 2: Preprocess raw CSV files into ML-ready numpy arrays.
- Loads all CSVs
- Applies 5-second windowing
- Normalizes features
- Data augmentation (noise)
- Saves train/test split
"""

import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

def load_all_csvs(data_dir='data/raw'):
    all_data = []
    all_labels = []

    for folder in ['crashes', 'normal_driving']:
        path = os.path.join(data_dir, folder, '*.csv')
        files = sorted(glob.glob(path))
        print(f"Found {len(files)} files in {folder}/")

        for f in files:
            df = pd.read_csv(f)
            features = df[['accel_x', 'accel_y', 'accel_z',
                           'gyro_x', 'gyro_y', 'gyro_z']].values
            label = int(df['label'].iloc[0])
            all_data.append(features)
            all_labels.append(label)

    return all_data, np.array(all_labels)

def window_data(all_data, all_labels, window_size=500):
    X = []
    y = []

    for data, label in zip(all_data, all_labels):
        n = len(data)
        if n >= window_size:
            start = (n - window_size) // 2
            window = data[start:start + window_size]
        else:
            pad = np.zeros((window_size - n, data.shape[1]))
            window = np.vstack([data, pad])

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)

def normalize(X_train, X_test):
    n_train, seq_len, n_feat = X_train.shape
    flat = X_train.reshape(-1, n_feat)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std

def augment_data(X, y, noise_factor=0.05, num_augmented=2):
    X_aug = [X]
    y_aug = [y]

    for _ in range(num_augmented):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)

def main():
    print("=" * 50)
    print("PHASE 2: DATA PREPROCESSING")
    print("=" * 50)

    print("\n[1/5] Loading CSV files...")
    all_data, all_labels = load_all_csvs('data/raw')
    print(f"Total samples: {len(all_labels)}")
    print(f"Crashes: {sum(all_labels == 1)}, Normal: {sum(all_labels == 0)}")

    print("\n[2/5] Extracting 5-second windows...")
    X, y = window_data(all_data, all_labels, window_size=500)
    print(f"Shape: {X.shape} (samples, timesteps, features)")

    print("\n[3/5] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    print("\n[4/5] Augmenting training data...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"After augmentation: {len(y_train)} training samples")

    print("\n[5/5] Normalizing...")
    X_train, X_test, mean, std = normalize(X_train, X_test)

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    np.save('data/processed/norm_mean.npy', mean)
    np.save('data/processed/norm_std.npy', std)

    print(f"\nSaved to data/processed/")
    print("Done! Run 3_train_model.py next.")

if __name__ == '__main__':
    main()