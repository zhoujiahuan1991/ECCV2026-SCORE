# ECCV2026-SCORE

## Environment Installation

### 1. Create a conda environment from environment.yml

```bash
conda env create -f environment.yml
```

The environment is named `IRL` by default (derived from the original environment name), and uses Python version 3.7.16.

### 2. Activate the environment
```bash
conda activate IRL
```

### 3. Set dynamic library paths
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
You can also append this command to the end of `~/.bashrc` to avoid repeated configurations.

## Training
```bash
bash continual_train.sh
```