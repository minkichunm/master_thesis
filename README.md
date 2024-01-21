# Neural Network Parameter Compression

This tool allows you to compress neural network parameters with various options for customization.

## Usage

```bash
python3 main.py -dir "test" -ds "cifar" -e 1 -m 1 -type entropy -so 3 -coeff 0.0 1.0
```

## Options

- `-dir`, `--directory_path`: Path to the output directory (default: "results_").
- `-e`, `--epoch`: Number of training epochs.
- `-m`, `--model`: Choose a model between [1, 2, 3, 4] (default: 1).
  - 1: `get_mobilenetv3s`
  - 2: `get_densenet121`
  - 3: `get_resnet50`
  - 4: `get_3d_model`
- `-type`, `--regularization_type`: Choose regularization type between "sparsity" and "entropy" (default: "entropy").
- `-coeff`, `--coefficients`: Coefficients for a custom option (default: [1.0]).
- `-b`, `--batch_size`: Batch size for training (default: 256).
- `-so`, `--scale_outlier`: Standard deviation for outlier setting (default: 3.0).
- `-ds`, `--dataset`: Choose dataset "mnist", "cifar", "celeba" (default: "cifar").
- `-L`, `--load_model`: Load a saved model.
- `-pq`, `--post_quantization`: Quantize weights after training.

