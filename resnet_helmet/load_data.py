import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torchvision import utils as vutils

# Define a dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img = self.to_tensor(self.img[index])
        angle = random.choice([90, 180, 270])
        img = TF.rotate(img, angle)
        return img, self.label[index]

    def __len__(self):
        return len(self.img)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img):
        self.img = img
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        return self.to_tensor(self.img[index])

    def __len__(self):
        return len(self.img)

def read_image_folder(resize_shape, image_folder):
    resize = torchvision.transforms.Resize(resize_shape)
    image_folder = ImageFolder(image_folder, transform=resize)

    idx_to_class = {value: key for key, value in image_folder.class_to_idx.items()}
    image_paths = [item[0] for item in image_folder.imgs]

    image_shape = np.array(image_folder[0][0]).shape
    data_length = len(image_folder)

    data_shape = list(image_shape)
    data_shape.insert(0, data_length)

    data = np.zeros(data_shape, dtype=np.uint8)
    labels = np.zeros([data_length], dtype=np.int64)

    i = 0
    for image, label in tqdm(image_folder, desc="Reading Images"):
        data[i] = np.array(image)
        labels[i] = label
        i += 1

    data_dict = {"data": data, "labels": labels, 'data_shape': image_shape}
    info_dict = {"label_names": idx_to_class, "file_paths": image_paths}

    return data_dict, info_dict


def read_test_img(resize_shape, path):
    resize = torchvision.transforms.Resize(resize_shape)
    data_img = []
    img_name = []
    for filename in tqdm(os.listdir(path)):
        img_name.append(filename)
        img = Image.open("./test/{}".format(filename)).convert('RGB')
        img = resize(img)
        img = np.array(img)
        data_img.append(img)

    return data_img, img_name

if __name__ == "__main__":
    train_dict, train_info_dict = read_image_folder((224, 224), "./train")
    print(train_dict)