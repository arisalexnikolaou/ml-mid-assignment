# LSTM Weather Predictor

This project implements a univariate and multivariate LSTM-based model for next‑day mean temperature forecasting using historical daily weather observations. A linear regression model has also been included as a performance baseline.

## 1. Project structure

The notebook 'LSTMWeatherPredictor_2.ipynb' loads the Kaggle sourced 'Summary of Weather.csv' data sets run regression model using a Linear Regression Model, a 'naive' single layer LSTM implementation, an LSTM implementation with 2 layers focusing on a single station and an alternative multivariate implementation.

The `requirements.txt` describes the dependencies to run.
The 'Summary of Weather.csv' is loaded by the notebook to analyze and predict weather patterns. This data is derived from the datasets from Kaggle: <https://www.kaggle.com/datasets/smid80/weatherww2>

The '/results' folder contains snapshots of the graphs used to analyze the performance of the models.
The '/output' folder stores the metrics and predicts values of the multi-variate LSTM model.
The '/models' folder contains the LSTM models trained.

## 2. Prerequisites

- Python 3.10+ (recommended)
- Optional: NVIDIA GPU with CUDA support for faster training
- Optional: VS.Code with the Jupyter notebook extension.

### Ubunto Installation Instructions and WSL 2 on Windows (Optional but highly recommended)

Install Python 3.13 and Pip on Linux or WSL 2

#### Installing CUDA Toolkit (Recommended)

```bash
sudo apt install nvidia-cuda-toolkit

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
```

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

#### Activate virtual environment

```bash
source .venv/bin/activate
```

#### Check CUDA version

```bash
nvcc --version
```

#### Watch GPU utilization

```bash
watch -n0.1 nvidia-smi
```

## 3. Installation

1. Create and activate a virtual environment (example using `venv`):
2. Install the project dependencies

Run the following commands:

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
