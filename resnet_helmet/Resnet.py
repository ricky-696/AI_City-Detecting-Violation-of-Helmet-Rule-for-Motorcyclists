import torch.nn as nn
import torch
from torchvision import models


class resnet_50(nn.Module):
    def __init__(self, num_classes = 2):
        super(resnet_50, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)	
        self.resnet50.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x


if __name__ == "__main__":
    model = resnet_50(2)
    print(model)