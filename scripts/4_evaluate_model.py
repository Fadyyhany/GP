"""
Phase 4: Evaluate the trained model.
- Confusion matrix
- PR-AUC score
- False positive analysis
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score
)

def main():
    print("=" * 50)
    print("PHASE 4: MODEL EVALUATION")
    print("=" * 50)

    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = tf.keras.models.load_model('models/crash_detector.keras')

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Crash']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    ap = average_precision_score(y_test, y_pred_prob)
    print(f"\nAverage Precision (PR-AUC): {ap:.4f}")

    if cm[0][0] + cm[0][1] > 0:
        fpr = cm[0][1] / (cm[0][0] + cm[0][1])
        print(f"False Positive Rate: {fpr:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()