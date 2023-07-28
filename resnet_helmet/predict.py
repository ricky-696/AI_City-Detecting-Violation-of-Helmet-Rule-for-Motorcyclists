import load_data
from Resnet import resnet_50

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
import cv2
import os


ans = []
device_num = 0
label = ['Helmet', 'NoHelmet']

# data_img, img_name = load_data.read_test_img((224, 224), "/home/Ricky/AI_City_Track5/resnet_helmet/test")
# dataset = load_data.TestDataset(data_img)
# loader = torch.utils.data.DataLoader(dataset, batch_size = 1)

path = '/home/Ricky/AI_City_Track5/resnet_helmet/test/NoHelmet'
file_names = os.listdir(path)

model = resnet_50(num_classes = 2).cuda()
model.load_state_dict(torch.load('resnet50.pt'))
model.eval()
        
for img in file_names:
    crop_img = cv2.imread(os.path.join(path, img))
    dataset = load_data.TestDataset(crop_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    
    with torch.cuda.device(device_num):
        with torch.no_grad():
            for data in loader:
                data = data.to(device_num)
                output = model(data)
                res = output.argmax(dim = 1, keepdim = True)
                ans.append(label[res])

print(ans)
# dict = {'file' : img_name, 'species' : ans}
# df = pd.DataFrame(dict)
# df.to_csv('predict.csv', index = False)