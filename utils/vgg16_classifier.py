import torch
import torch.nn as nn


VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M"]


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    channels = in_channels
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        conv = nn.Conv2d(channels, value, kernel_size=3, padding=1, bias=not batch_norm)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(value), nn.ReLU(inplace=False)])
        else:
            layers.extend([conv, nn.ReLU(inplace=False)])
        channels = value
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, quantize=False, dropout=0.5):
        super().__init__()
        self.quantize = quantize
        if quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        self.features = make_layers(VGG16_CFG, in_channels=in_channels, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.quantize:
            x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def fuse_model(self):
        names = list(self.features._modules.keys())
        for index in range(len(names) - 2):
            first = self.features._modules[names[index]]
            second = self.features._modules[names[index + 1]]
            third = self.features._modules[names[index + 2]]
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d) and isinstance(third, nn.ReLU):
                torch.quantization.fuse_modules(self.features, [names[index], names[index + 1], names[index + 2]], inplace=True)


def vgg16(in_channels=3, num_classes=10, quantize=False, **kwargs):
    return VGG(in_channels=in_channels, num_classes=num_classes, quantize=quantize, **kwargs)


def vgg16_cifar10(quantize=False, **kwargs):
    return vgg16(in_channels=3, num_classes=10, quantize=quantize, **kwargs)
