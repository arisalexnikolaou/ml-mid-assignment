# LSTM Weather Predictor

This project implements a univariate and multivariate LSTM-based model for next‑day mean temperature forecasting using historical daily weather observations. A linear regression model has also been included as a performance baseline.

## 1. Project structure

The notebook 'LSTMWeatherPredictor_2.ipynb' loads the Kaggle sourced 'Summary of Weather.csv' data sets run regression model using a Linear Regression Model, a 'naive' single layer LSTM implementation, an LSTM implementation with 2 layers focusing on a single station and an alternative multivariate implementation.

The `requirements.txt` describes the dependencies to run.
The 'Summary of Weather.csv' is loaded by the notebook to analyze and predict weather patterns. This data is derived from the datasets from Kaggle: <https://www.kaggle.com/datasets/smid80/weatherww2>

## 2. Prerequisites

- Python 3.10+ (recommended)
- Optional: NVIDIA GPU with CUDA support for faster training
- Optional: VS.Code with the Jupyter notebook extension.

## 3. Installation

1. Create and activate a virtual environment (example using `venv`):
2. Install the project dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the notebook

If you are using VS.Code with the Jupyter notebook extension then simply open the notebook within your IDE otherwise run it in Jupyter.
