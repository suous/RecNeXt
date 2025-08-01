import torch
import argparse
from timm import create_model
from fvcore.nn import FlopCountAnalysis

import lsnet.model
import model
import utils


def analyze_model(model_name, resolution, verbose=False):
    print(f"Analyzing model: {model_name}")
    print(f"Resolution: {resolution}x{resolution}")

    inputs = torch.randn(1, 3, resolution, resolution)
    model = create_model(model_name, num_classes=1000)
    utils.replace_batchnorm(model)

    if verbose:
        print(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters (M): {n_parameters / 1e6:.2f}')

    flops = FlopCountAnalysis(model, inputs)
    print(f"FLOPs (G): {flops.total() / 1e9:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze a timm model.')
    parser.add_argument('-m', '--model-name', type=str, required=True, help='The model name to analyze.')
    parser.add_argument('-r', '--resolution', type=int, default=224, help='Input resolution for the model.')
    parser.add_argument('-p', '--print-model', action='store_true', default=False, help='Print model.')

    args = parser.parse_args()
    analyze_model(args.model_name, args.resolution, args.print_model)


if __name__ == '__main__':
    main()
