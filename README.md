# AI-Powered Smartphone Crash Detection System

An Android-based crash detection system that uses smartphone sensors and artificial intelligence to recognize potential road accidents in real-time.

## System Architecture

```
Sensor Management Module --> Crash Detection Engine --> Emergency Response Module
[Accelerometer, Gyroscope, GPS] --> [Preprocessing → AI Model → Decision Logic] --> [Alert + SMS + GPS Location]
```

## Project Structure

```
GP/
├── scripts/
│   ├── 1_generate_data.py        # Phase 1: Generate synthetic sensor data
│   ├── 2_preprocess_data.py      # Phase 2: Clean, window, augment, split
│   ├── 3_train_model.py          # Phase 3: Train CNN+LSTM model + TFLite
│   └── 4_evaluate_model.py       # Phase 4: Evaluate (PR-AUC, confusion matrix)
├── data/
│   ├── raw/                      # Generated CSV files
│   │   ├── crashes/              # Frontal, side, rollover scenarios
│   │   └── normal_driving/       # Highway, hard brake, pothole scenarios
│   └── processed/                # Numpy arrays (train/test split)
├── models/                       # Saved Keras + TFLite models
├── requirements.txt
└── README.md
```

## How to Run (Step by Step)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (300 CSV files)
```bash
python scripts/1_generate_data.py
```
This creates 150 crash + 150 normal driving sensor files.

### 3. Preprocess the data
```bash
python scripts/2_preprocess_data.py
```
Loads CSVs, applies 5-second windowing, augments with noise, splits 80/20.

### 4. Train the AI model
```bash
python scripts/3_train_model.py
```
Trains a CNN+LSTM model and exports it to TFLite for mobile deployment.

### 5. Evaluate results
```bash
python scripts/4_evaluate_model.py
```
Shows accuracy, PR-AUC score, confusion matrix, and false positive rate.

## Crash Scenarios

| Scenario | Type | Peak G-Force | Description |
|----------|------|-------------|-------------|
| Frontal  | Crash | 20-50g | Head-on collision, sudden deceleration |
| Side     | Crash | 15-40g | T-bone impact, strong lateral force |
| Rollover | Crash | Variable | Vehicle rolling, oscillating forces |
| Highway  | Normal | <1g | Smooth driving, road vibrations |
| Hard Brake | Normal | 5-8g | Emergency stop, controlled deceleration |
| Pothole  | Normal | 3-6g | Sharp vertical spike only |

## AI Model

- **Architecture**: Conv1D → BatchNorm → MaxPool → Conv1D → BatchNorm → MaxPool → LSTM → Dense → Sigmoid
- **Input**: 5-second window (500 timesteps × 6 sensor channels)
- **Output**: Crash probability (0 to 1)
- **Metric**: PR-AUC (Average Precision)
- **Deployment**: TFLite with quantization for Android

## Key Design Decisions

| Problem | Solution |
|---------|----------|
| Battery drain from continuous sensing | Event-triggered 5-second windows |
| Noisy smartphone sensors | Data augmentation with Gaussian noise |
| Need for accurate detection | CNN+LSTM instead of simple thresholds |
| Must run on phone | TFLite export with quantization |
| False alarms | Hard brake + pothole false-positive testing |
| Imbalanced data | PR-AUC as primary metric |