import argparse

def __create_options():
    options = {
        "directory_path": "results_",
        "epoch": 10,
        "model": "1",
        "regularization_type" : "entropy",
        "coefficients": [1.0],
        "batch_size": 64,
        "scale_outlier": 3,
        "dataset": "cifar",
        "load_model": False, 
        "post_quantization": False,
    }

    p = argparse.ArgumentParser(description='Compressing neural network parameter')
    p.add_argument('-dir', '--directory_path', type=str,
                   help='Path to the output directory (default: "results_")')
    p.add_argument('-e', '--epoch', type=int,
                   help='Number of training epochs (default: 1)')
    p.add_argument('-m', '--model', type=int,
                   help='Choose a model between [1, 2, 3, 4] (default: 1)\n'
                        '1: get_model()\n'
                        '2: get_3_model()\n'
                        '3: get_32_model()\n'
                        '4: get_simple_model()')
    p.add_argument('-type', '--regularization_type', type=str,
                   help='Choose regularization type between "sparsity" and "entropy" (default: "entropy")')
    p.add_argument('-coeff', '--coefficients', type=float, nargs='*', default=[1.0],
                   help='Coefficients for a custom option (default: [1.0])')
    p.add_argument('-b', '--batch_size', type=int,
                   help='Batch size for training (default: 256)')
    p.add_argument('-so', '--scale_outlier', type=float,
                   help='Standard deviation for outlier setting (default: 3)')
    p.add_argument('-ds', '--dataset', type=str,
                   help='Choose dataset "mnist", "cifar","celeba","3d" (default: "cifar")')
    p.add_argument('-L', '--load_model', action='store_true', help='Load a saved model')
    p.add_argument('-pq', '--post_quantization', action='store_true', help='Quantize weights after training')
              

    args = p.parse_args()

    if args.directory_path:
        options["directory_path"] = args.directory_path
    if args.epoch:
        options["epoch"] = args.epoch
    if args.model:
        options["model"] = args.model
    if args.regularization_type:
        options["regularization_type"] = args.regularization_type
    if args.coefficients:
        options["coefficients"] = args.coefficients
    if args.batch_size:
        options["batch_size"] = args.batch_size
    if args.scale_outlier:
        options["scale_outlier"] = args.scale_outlier
    if args.dataset:
        options["dataset"] = args.dataset
    if args.load_model:
        options["load_model"] = args.load_model
    if args.post_quantization:
        options["post_quantization"] = args.post_quantization
        
    print(options)

    return options
