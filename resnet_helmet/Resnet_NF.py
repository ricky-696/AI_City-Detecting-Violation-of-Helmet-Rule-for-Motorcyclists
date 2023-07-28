import torch.nn as nn
import torch
from torchvision import models


class resnet_50(nn.Module):
    def __init__(self, num_classes = 2):
        super(resnet_50, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)
        
        for param in self.resnet50.parameters():
            param.requires_grad = True	
            
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class resnet_101(nn.Module):
    def __init__(self, num_classes = 2):
        super(resnet_101, self).__init__()
        self.resnet101 = models.resnet101(pretrained = True)
        
        for param in self.resnet101.parameters():
            param.requires_grad = True	
            
        self.resnet101.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet101(x)
        return x


class resnet_152(nn.Module):
    def __init__(self, num_classes = 2):
        super(resnet_152, self).__init__()
        self.resnet152 = models.resnet152(pretrained = True)
        
        for param in self.resnet152.parameters():
            param.requires_grad = True	
            
        self.resnet152.fc = nn.Sequential(nn.Linear(2048, num_classes),
                                          nn.Dropout())

    def forward(self, x):
        x = self.resnet152(x)
        return x


if __name__ == "__main__":
    model = resnet_50(2)
    print(model)