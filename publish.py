import sys
import shutil
import argparse
from pathlib import Path

import torch
from timm import create_model

import model
import utils


def main(args):
    path = Path(args.checkpoint_path)
    assert path.is_file(), f"{args.checkpoint_path} is not a file"
    assert path.suffix == ".pth", f"{args.checkpoint_path} is not a pth file"

    distill_suffix = "distill" if args.distillation else "without_distill"
    name = f"{args.model_name}_{distill_suffix}_{args.epochs}e"

    output_dir = Path(args.checkpoint_path).parent / name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = path.parent / 'log.txt'
    if log_file.exists():
        shutil.copy(log_file, output_dir / f"{name}.txt")

    model = create_model(args.model_name)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    if args.fused:
        utils.replace_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters / 1e6)
        model = torch.jit.script(model)
        model.save(output_dir / f"{name}.pt")
        sys.exit(0)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters / 1e6)
    torch.save({"model": model.state_dict()}, output_dir / f"{name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--distillation", action="store_true")
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    main(args)
