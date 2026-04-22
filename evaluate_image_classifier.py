import argparse
from pathlib import Path

import torch
import torch.nn as nn

from train_image_classifier import build_model, checkpoint_name, prepare_qat_model
from utils.dataset_loaders import read_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["resnet18", "vgg16"], default="resnet18")
    parser.add_argument("--qat", action="store_true", help="Load a QAT checkpoint and convert before testing.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output, 1)
            seen = target.size(0)
            total_loss += loss.item() * seen
            total_correct += pred.eq(target).sum().item()
            total_seen += seen

    return total_loss / total_seen, 100.0 * total_correct / total_seen


def resolve_checkpoint(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(args.checkpoint_dir) / checkpoint_name(args.model, args.dataset, args.qat)


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_name = checkpoint.get("model", args.model)
    dataset_name = checkpoint.get("dataset", args.dataset)
    is_qat = bool(checkpoint.get("qat", args.qat))

    device = torch.device("cpu" if is_qat else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, dataset_name, qat=is_qat)

    if checkpoint.get("converted", False):
        prepare_qat_model(model)
        model.eval()
        torch.quantization.convert(model, inplace=True)
        model.load_state_dict(checkpoint["model_state"])
    elif is_qat:
        prepare_qat_model(model)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        torch.quantization.convert(model, inplace=True)
    else:
        model.load_state_dict(checkpoint["model_state"])

    model = model.to(device)
    _, _, test_loader, _ = read_dataset(
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        augment=False,
        cutout=False,
        seed=args.seed,
    )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run: model={model_name}, dataset={dataset_name}, mode={'QAT/int8' if is_qat else 'FP32'}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
