"""
Phase 3: Train a CNN+LSTM crash detection model.
- CNN extracts spatial features from sensor channels
- LSTM captures temporal patterns
- Exports to TFLite for mobile deployment
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(32, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='PR')]
    )

    return model

def convert_to_tflite(model, input_shape, output_path='models/crash_detector.tflite'):
    """Convert to TFLite using concrete function to handle LSTM properly."""
    # Create a concrete function with fixed input shape
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, input_shape[0], input_shape[1]], dtype=tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {output_path} ({size_mb:.2f} MB)")

def main():
    print("=" * 50)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 50)

    print("\nLoading data...")
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train labels: {sum(y_train==1)} crash, {sum(y_train==0)} normal")

    print("\nBuilding CNN+LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()

    print("\nTraining...")
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=5, mode='max', restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test PR-AUC: {results[2]:.4f}")

    os.makedirs('models', exist_ok=True)
    model.save('models/crash_detector.keras')
    print("Keras model saved: models/crash_detector.keras")

    print("\nConverting to TFLite...")
    convert_to_tflite(model, input_shape)

    print("\nDone! Run 4_evaluate_model.py next.")

if __name__ == '__main__':
    main()