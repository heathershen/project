from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import operator
import cv2

def checkAccuracy(model, criterion, optimizer, scheduler):
    best_model_wts = model.state_dict()
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for data in dataloaders['test']:
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dataset_sizes['test']
    acc = running_corrects / dataset_sizes['test']

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', loss, acc))
    return acc


data_transforms = {
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'facesData'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

model_ft = models.resnet18(pretrained=True)
# for param in model_ft.parameters():
#     param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft.load_state_dict(torch.load('savedInitialModel.pt'))

# model_ft.load_state_dict(torch.load('savedModelCNN-finetune.pt'))

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

accFT = checkAccuracy(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
print('Finetune CNN accuracy: ', accFT)

# model_conv = models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)

# model_conv.load_state_dict(torch.load('savedModelCNN-freeze.pt'))

# criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# accFixed = checkAccuracy(model_conv, criterion, optimizer_ft, exp_lr_scheduler)
# print('Fixed CNN accuracy: ', accFixed)
