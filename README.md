# MNIST/CIFAR10 FP32 and QAT Classifier Training

This directory supports the KAUST Integrated Photonics Laboratory (IPL) article "Integrated van der Waals Waveguides for All-Optical Nonlinear Photonic Circuits" (accepted by Nature Communications). It provides the image-classification training framework used for the optical activation function experiments associated with the work.

## Supported Tests

Datasets:

- `mnist`
- `cifar10`

Models:

- `resnet18`
- `vgg16`

Training modes:

- normal FP32 training
- quantization-aware training with `--qat`

This gives these main combinations:

```text
resnet18 + mnist   + FP32
resnet18 + mnist   + QAT
resnet18 + cifar10 + FP32
resnet18 + cifar10 + QAT
vgg16    + mnist   + FP32
vgg16    + mnist   + QAT
vgg16    + cifar10 + FP32
vgg16    + cifar10 + QAT
```

## Train

Train ResNet18 on CIFAR10 in normal FP32 mode:

```bash
python train_image_classifier.py --dataset cifar10 --model resnet18
```

Train ResNet18 on CIFAR10 with QAT:

```bash
python train_image_classifier.py --dataset cifar10 --model resnet18 --qat
```

Train VGG16 on MNIST in normal FP32 mode:

```bash
python train_image_classifier.py --dataset mnist --model vgg16
```

Train VGG16 on MNIST with QAT:

```bash
python train_image_classifier.py --dataset mnist --model vgg16 --qat
```

Short smoke-test runs:

```bash
python train_image_classifier.py --dataset mnist --model resnet18 --epochs 3
python train_image_classifier.py --dataset cifar10 --model vgg16 --qat --epochs 3
```

## Evaluate

Evaluate a normal FP32 checkpoint:

```bash
python evaluate_image_classifier.py --dataset cifar10 --model resnet18
```

Evaluate a QAT checkpoint:

```bash
python evaluate_image_classifier.py --dataset cifar10 --model resnet18 --qat
```

Evaluate a specific checkpoint file:

```bash
python evaluate_image_classifier.py --checkpoint checkpoint/vgg16_mnist_qat.pt
```

## Checkpoints

By default, checkpoints are saved under `checkpoint/` using this pattern:

```text
{model}_{dataset}_{mode}.pt
```

Examples:

```text
checkpoint/resnet18_cifar10_fp32.pt
checkpoint/resnet18_cifar10_qat.pt
checkpoint/resnet18_cifar10_qat_int8.pt
checkpoint/vgg16_mnist_fp32.pt
```

For QAT runs, training saves the best fake-quantized QAT checkpoint and also saves a converted int8 checkpoint after training finishes.

The `dataset/` and `checkpoint/` directories are intentionally present in the release even when empty. `dataset/` is where MNIST/CIFAR10 will be downloaded, and `checkpoint/` is where training outputs will be saved. Their generated contents are ignored by Git.

## Main Training Options

```bash
python train_image_classifier.py --help
```

Important arguments:

- `--dataset {mnist,cifar10}` selects the dataset.
- `--model {resnet18,vgg16}` selects the model architecture.
- `--qat` enables quantization-aware training.
- `--epochs` controls training length.
- `--batch-size` controls batch size.
- `--lr` controls the initial learning rate.
- `--valid-size` controls the training split used for validation.
- `--calibration-size` controls the held-out calibration split.
- `--data-dir` controls where datasets are stored.
- `--checkpoint-dir` controls where checkpoints are saved.
- `--wandb` enables Weights & Biases logging.

## Main Evaluation Options

```bash
python evaluate_image_classifier.py --help
```

Important arguments:

- `--dataset {mnist,cifar10}` selects the dataset when `--checkpoint` is not explicit.
- `--model {resnet18,vgg16}` selects the model when `--checkpoint` is not explicit.
- `--qat` loads the QAT checkpoint naming pattern and converts it before evaluation.
- `--checkpoint` evaluates a specific checkpoint path.
- `--batch-size` controls evaluation batch size.
- `--data-dir` controls where datasets are loaded from.

## Notes

- CIFAR10 uses random crop, horizontal flip, normalization, and Cutout by default during training.
- MNIST uses random crop and normalization by default during training.
- Evaluation disables training augmentation.
- QAT evaluation runs on CPU because PyTorch quantized inference is CPU-oriented in this setup.
- If the system default `python` cannot import PyTorch, use the conda Python/environment that has `torch` installed.
