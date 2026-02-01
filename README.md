# LSTM Weather Predictor

This project implements a univariate and multivariate LSTM-based model for next‑day mean temperature forecasting using historical daily weather observations. 

## 1. Project structure

- `LSTMWeatherPredictor.ipynb` – end‑to‑end notebook including:
  - Data loading and exploratory analysis
  - Feature selection and scaling
  - Sequence generation with a configurable lookback window
  - Baseline linear regression model
  - LSTM model definition, training, and evaluation
  - Plots of learning curves and predictions [file:1]
- `requirements.txt` – Python dependencies. [file:2]
- Weather data file (CSV) – referenced in the notebook via `DATA_PATH` (you must provide or adjust the path). [file:1]

## 2. Prerequisites

- Python 3.10+ (recommended)
- Optional: NVIDIA GPU with CUDA support for faster training
- Optional: VS.Code with the Jupyte notebook extension.

## 3. Installation

1. Unzip Summary of Weather.zip into the solution's folder as Summary of Weather.csv
2. Create and activate a virtual environment (example using `venv`):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the notebook 

If you are using VS.Code with the Jupyter notebook extension then simply open the notebook within your IDE otherwise.

jupyter notebook
