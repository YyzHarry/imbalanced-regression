## STS-B-DIR

### Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. (Optionally) We have provided both original STS-B dataset and our created balanced STS-B-DIR dataset in folder `./glue_data/STS-B`. To reproduce the results in the paper, please use our created STS-B-DIR dataset. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```

#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.1 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
```

### Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_sts.py`: download original STS-B dataset and create STS-B-DIR dataset with balanced val/test set 

#### Main Arguments

- `--lds`: LDS switch (whether to enable LDS)
- `--fds`: FDS switch (whether to enable FDS)
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--resume`: whether to resume training (only for training)
- `--evaluate`: evaluate only flag
- `--eval_model`: path to resume checkpoint (only for evaluation)
- `--retrain_fc`: whether to retrain regressor
- `--pretrained`: path to load backbone weights for regressor re-training (RRT)
- `--val_interval`: number of iterations between validation checks
- `--patience`: patience (number of validation checks) for early stopping

### Getting Started

#### Train a vanilla model

```bash
python train.py --cuda <gpuid> --reweight none
```

Always specify `--cuda <gpuid>` for the GPU ID (single GPU) to be used. We will omit this argument in the following for simplicity.

#### Train a model using re-weighting

To perform inverse re-weighting

```bash
python train.py --reweight inverse
```

To perform square-root inverse re-weighting

```bash
python train.py --reweight sqrt_inv
```

#### Train a model with different losses

To use Focal-R loss

```bash
python train.py --loss focal_mse
```

To use huber loss

```bash
python train.py --loss huber --huber_beta 0.3
```

#### Train a model using RRT

```bash
python train.py [...retrained model arguments...] --retrain_fc --pretrained <path_to_pretrained_ckpt>
```

#### Train a model using LDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
```

#### Train a model using FDS

To use Gaussian kernel (kernel size: 5, sigma: 2)

```bash
python train.py --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

#### Train a model using LDS + FDS

```bash
python train.py --reweight inverse --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
```

#### Evaluate only 

```bash
python train.py [...evaluation model arguments...] --evaluate --eval_model <path_to_evaluation_ckpt>
```

### Reproduced Benchmarks and Model Zoo

Metric: MSE

|   Model   | Overall | Many-Shot | Medium-Shot | Few-Shot | Download |
| :-------: | :-----: | :-------: | :---------: | :------: | :------: |
|    LDS    |  0.914  |   0.819   |    1.319    |   0.955  |  model   |
|    FDS    |  0.916  |   0.875   |    1.027    |   1.086  |  model   |
| LDS + FDS |  0.907  |   0.802   |    1.363    |   0.942  |  model   |