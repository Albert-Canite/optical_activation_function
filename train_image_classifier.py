import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.dataset_loaders import get_dataset_spec, read_dataset
from utils.resnet18_classifier import ResNet18
from utils.vgg16_classifier import vgg16


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 or VGG16 on MNIST or CIFAR10, with optional QAT.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["resnet18", "vgg16"], default="resnet18")
    parser.add_argument("--qat", action="store_true", help="Enable quantization-aware training.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.05)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-cutout", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def build_model(model_name, dataset_name, qat=False):
    spec = get_dataset_spec(dataset_name)
    kwargs = {
        "in_channels": spec["in_channels"],
        "num_classes": spec["num_classes"],
        "quantize": qat,
    }
    if model_name == "resnet18":
        return ResNet18(**kwargs)
    if model_name == "vgg16":
        return vgg16(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")


def quant_backend():
    supported = torch.backends.quantized.supported_engines
    for backend in ("fbgemm", "x86", "qnnpack"):
        if backend in supported:
            return backend
    return supported[0]


def prepare_qat_model(model):
    model.eval()
    model.fuse_model()
    model.train()
    backend = quant_backend()
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(model, inplace=True)
    return backend


def accuracy_from_logits(logits, targets):
    _, preds = torch.max(logits, 1)
    return preds.eq(targets).sum().item(), targets.size(0)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            if is_train:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            if is_train:
                loss.backward()
                optimizer.step()

            correct, seen = accuracy_from_logits(output, target)
            total_loss += loss.item() * seen
            total_correct += correct
            total_seen += seen

    return total_loss / total_seen, 100.0 * total_correct / total_seen


def checkpoint_name(model_name, dataset_name, qat):
    mode = "qat" if qat else "fp32"
    return f"{model_name}_{dataset_name}_{mode}.pt"


def save_checkpoint(path, model, args, best_val_loss, best_val_acc, backend=None):
    checkpoint = {
        "model_state": model.state_dict(),
        "model": args.model,
        "dataset": args.dataset,
        "qat": args.qat,
        "backend": backend,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    project = args.wandb_project or f"{args.dataset}-{args.model}"
    run_name = f"{args.model}-{args.dataset}-{'qat' if args.qat else 'fp32'}"
    wandb.init(project=project, name=run_name, config=vars(args))
    return wandb


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = read_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        calibration_size=args.calibration_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        augment=not args.no_augment,
        cutout=not args.no_cutout,
        seed=args.seed,
    )
    train_loader, valid_loader, _, _ = loaders

    model = build_model(args.model, args.dataset, qat=args.qat).to(device)
    backend = prepare_qat_model(model) if args.qat else None

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name(args.model, args.dataset, args.qat)

    wandb = maybe_init_wandb(args)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    print(f"Device: {device}")
    print(f"Run: model={args.model}, dataset={args.dataset}, mode={'QAT' if args.qat else 'FP32'}")
    if args.qat:
        print(f"Quantization backend: {backend}")

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        if wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(checkpoint_path, model, args, best_val_loss, best_val_acc, backend)
            print(f"Saved best checkpoint: {checkpoint_path}")

    if args.qat and best_state is not None:
        model.load_state_dict(best_state)
        model.cpu().eval()
        torch.quantization.convert(model, inplace=True)
        int8_path = checkpoint_path.with_name(checkpoint_path.stem + "_int8.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model": args.model,
            "dataset": args.dataset,
            "qat": args.qat,
            "backend": backend,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "converted": True,
            "args": vars(args),
        }, int8_path)
        print(f"Saved converted int8 checkpoint: {int8_path}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
