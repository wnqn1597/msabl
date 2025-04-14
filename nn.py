# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28, 1)):
        super(LeNet5, self).__init__()
        self.size = 16 * ((image_size[0] // 2 - 6) // 2) * ((image_size[1] // 2 - 6) // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.neck = nn.Sequential(
            nn.Linear(self.size, 120),
            nn.ReLU(),
            nn.Linear(120, 84)
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward_flatten(self, x, return_feat=False):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        feat = self.neck(x)
        logits = self.head(feat)
        if not return_feat:
            return logits
        feat_ret = feat.detach().clone()
        return logits, feat_ret

    def forward(self, bag, return_feat=False):
        assert bag.ndim == 5
        n_inst = bag.shape[1]
        x_list = torch.cat(torch.unbind(bag, dim=1), dim=0)  # inst  B x img
        logits, feat = self.forward_flatten(x_list, return_feat=True)
        logits = torch.stack(torch.chunk(logits, n_inst, dim=0), dim=1)  # B x inst x out_d
        feat = torch.stack(torch.chunk(feat, n_inst, dim=0), dim=1)  # B x inst x out_d
        if not return_feat:
            return logits
        return logits, feat


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()
        self.res = resnet18(weights=None)
        self.res.fc = nn.Identity()
        out_d = 512
        self.head = nn.Linear(out_d, num_classes)

    def forward_flatten(self, x, return_feat=False):
        assert x.ndim == 4  # B x img
        feat = self.res(x)
        logits = self.head(feat)
        if not return_feat:
            return logits  # B x inst x 4
        return logits, feat

    def forward(self, bag, return_feat=False):
        assert bag.ndim == 5
        n_inst = bag.shape[1]
        x_list = torch.cat(torch.unbind(bag, dim=1), dim=0)  # inst  B x img
        logits, feat = self.forward_flatten(x_list, return_feat=True)
        logits = torch.stack(torch.chunk(logits, n_inst, dim=0), dim=1)  # B x inst x out_d
        feat = torch.stack(torch.chunk(feat, n_inst, dim=0), dim=1)  # B x inst x out_d
        if not return_feat:
            return logits
        return logits, feat


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()
        self.res = resnet50(weights=None)
        self.res.fc = nn.Identity()
        out_d = 2048
        self.head = nn.Linear(out_d, num_classes)

    def forward_flatten(self, x, return_feat=False):
        assert x.ndim == 4  # B x img
        feat = self.res(x)
        logits = self.head(feat)
        if not return_feat:
            return logits  # B x inst x 4
        return logits, feat

    def forward(self, bag, return_feat=False):
        assert bag.ndim == 5
        n_inst = bag.shape[1]
        x_list = torch.cat(torch.unbind(bag, dim=1), dim=0)  # inst  B x img
        logits, feat = self.forward_flatten(x_list, return_feat=True)
        logits = torch.stack(torch.chunk(logits, n_inst, dim=0), dim=1)  # B x inst x out_d
        feat = torch.stack(torch.chunk(feat, n_inst, dim=0), dim=1)  # B x inst x out_d
        if not return_feat:
            return logits
        return logits, feat


if __name__ == "__main__":
    model = ResNet18Classifier(10)
    X = torch.randn(2, 3, 32, 32)
    y, f = model.forward_flatten(X, True)
    print(y.shape)
    print(f.shape)
