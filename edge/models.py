"""Model definitions and the intermediate representations used by KD."""
import torch
from torch import nn
from torchvision import models

STUDENTS = ("mobilenetv2", "squeezenet", "efficientnet_lite0")


class Classifier(nn.Module):
    def __init__(self, name, classes=100):
        super().__init__()
        self.name = name
        if name == "resnet50":
            net = models.resnet50(num_classes=classes)
            # Preserve spatial information in 32 px images.
            net.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            net.maxpool = nn.Identity()
            self.feature_channels = 1024
        elif name == "mobilenetv2":
            net = models.mobilenet_v2(num_classes=classes)
            net.features[0][0].stride = (1, 1)
            self.feature_channels = 96
        elif name == "squeezenet":
            net = models.squeezenet1_1(num_classes=classes)
            net.features[0].stride = (1, 1)
            # Cross-entropy expects unrestricted class logits.
            net.classifier[2] = nn.Identity()
            self.feature_channels = 384
        elif name == "efficientnet_lite0":
            import timm
            net = timm.create_model("tf_efficientnet_lite0", pretrained=False,
                                    num_classes=classes, exportable=True)
            net.conv_stem.stride = (1, 1)
            self.feature_channels = 112
        else:
            raise ValueError(f"Unknown architecture: {name}")
        self.net = net

    def forward(self, image):
        return self.net(image)

    def logits_and_features(self, image):
        net = self.net
        if self.name == "resnet50":
            x = net.maxpool(net.relu(net.bn1(net.conv1(image))))
            x = net.layer3(net.layer2(net.layer1(x)))
            scores = net.fc(net.avgpool(net.layer4(x)).flatten(1))
            return scores, x

        if self.name == "efficientnet_lite0":
            x = net.bn1(net.conv_stem(image))
            for block in net.blocks[:5]:
                x = block(x)
            last = net.blocks[5:](x)
            scores = net.forward_head(net.bn2(net.conv_head(last)))
            return scores, x

        # Tap an intermediate block; reuse each model's original classification head.
        tap = 13 if self.name == "mobilenetv2" else 10
        x = net.features[:tap + 1](image)
        last = net.features[tap + 1:](x)
        if self.name == "mobilenetv2":
            scores = net.classifier(last.mean(dim=(2, 3)))
        else:
            scores = net.classifier(last).flatten(1)
        return scores, x


def build_model(spec):
    if spec["architecture"] == "nas":
        from .space import SearchCNN
        return SearchCNN(spec["config"])
    return Classifier(spec["architecture"])
