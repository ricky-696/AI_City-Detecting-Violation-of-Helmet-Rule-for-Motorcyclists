import load_data
from Resnet_NF import resnet_152

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

device_num = 0
train_loss = []
train_acc = []
val_loss = []
val_acc = []
train_epoch = []
epoch_n = 100
val_best_loss = 999
model_name = 'resnet152_dropout'

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        gt = get_2_class_gt(target)
        data, target, gt = data.to(device), target.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, gt)
        loss.backward()
        optimizer.step()
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end = '')

        pred = output.argmax(dim = 1, keepdim=True) # predicted answer
        correct += pred.eq(target.view_as(pred)).sum().item() # currect answer

    acc = round(correct / len(train_loader.dataset), 4)
    train_acc.append(acc)
    train_loss.append(loss.item())

    print('\nTrain acc: ', acc, 'Train loss: ', loss.item())
    

def get_2_class_gt(target):
    gt = []
    for t in target:
        if t == 0:
            gt.append([1., 0.])
        else:
            gt.append([0., 1.])

    return torch.tensor(gt)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            gt = get_2_class_gt(target)
            data, target, gt = data.to(device), target.to(device), gt.to(device)
            output = model(data)           
            test_loss += F.cross_entropy(output, gt, reduction = 'sum').item()       
            pred = output.argmax(dim = 1, keepdim = True) # predicted answer
            correct += pred.eq(target.view_as(pred)).sum().item() # currect answer

    test_loss /= len(test_loader.dataset)
    acc = round(correct / len(test_loader.dataset), 4)

    if test_loss < val_best_loss:
        torch.save(model.state_dict(), model_name + '.pt')

    val_loss.append(test_loss)
    val_acc.append(acc)
    print('\r\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc), end='')

def draw(title, xlabel, ylabel, x, y1, y2, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['train', 'valid'], loc = 'upper left')
    plt.savefig(filename)

def main():
    img_w, img_h = 224, 224
    train_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/Helmet_Datasets/train'
    valid_path = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_5/Helmet_Datasets/valid'

    train_dict, train_info_dict = load_data.read_image_folder((img_w, img_h), train_path, 'Train Img')
    train_set = load_data.ImageDataset(train_dict["data"], train_dict["labels"])
    
    valid_dict, valid_info_dict = load_data.read_image_folder((img_w, img_h), valid_path, 'Test Img')
    valid_set = load_data.ImageDataset(valid_dict["data"], valid_dict["labels"])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 64)

    with torch.cuda.device(device_num):
        model = resnet_152(num_classes = 2).cuda()
        adam = torch.optim.Adam(model.parameters(), lr = 1e-4)

        for epoch in range(1, epoch_n + 1):
            train_epoch.append(epoch)
            train(model, device_num, train_loader, adam, epoch)
            test(model, device_num, valid_loader)

        #print loss graph
        draw('train loss', 'epoch', 'loss', train_epoch, train_loss, val_loss, model_name + '_train_loss.jpg')
        draw('train accuracy', 'epoch', 'accuracy', train_epoch, train_acc, val_acc, model_name + '_train_acc.jpg')
        torch.save(model.state_dict(), model_name + '.pt')

if __name__ == "__main__":
    main()
