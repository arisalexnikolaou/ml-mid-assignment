# Ubunto Installation Instructions

Install Python 3.13 and Pip on Linux

## Installing CUDA Toolki

```bash
sudo apt install nvidia-cuda-toolkit
``

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
```

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

## Activate virtual environment

```bash
source .venv/bin/activate
```

### Check CUDA version

```bash
nvcc --version
```

### Check GPU utilization

```bash
watch -n0.1 nvidia-smi
```