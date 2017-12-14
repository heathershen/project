from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import operator
import cv2
import csv
import ntpath
import shutil

######################################################################
# Load data
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),

        # transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'unlabeled': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'facesData'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'unlabeled', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'unlabeled', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'unlabeled', 'test']}
class_names = image_datasets['train'].classes

######################################################################
# Run model on all images

def getConfidence(model):
    # Iterate over data.
    confidence = {}
    model.train(False)  # Set model to evaluate mode

    for i, data in enumerate(dataloaders['unlabeled']):
        # get the inputs
        inputs, __ = data

        # wrap them in Variable
        inputs = Variable(inputs)

        # Output of trained model
        outputs = model(inputs)

        # Find confidence (probability) of classification
        softmaxModel = nn.Softmax()
        outputProb = softmaxModel(Variable(outputs.data))
        # print(outputProb)

        scores, preds_softmax = torch.max(outputProb.data, 1)
        # print("PROB: ", scores)
        for j in range(inputs.size()[0]):
            # print("SCORE:", scores[i])
            # confidence[inputs[i]] = (scores[i], preds_softmax[i]) # store each input's (confidence, prediction)
            # Each key is a tuple (image, label)

            # isLabeled = False
            # for image in labeledImages:
            #     if torch.equal(inputs[i].data, image): 
            #         isLabeled = True
            #         break
            # if isLabeled: continue

            # Image path tuple
            imgPathTuple = image_datasets['unlabeled'].imgs[i*4+j]
            # print((imgPathTuple[0], preds_softmax[j]))
            # confidence[(imgPathTuple[0], preds_softmax[j])] = scores[j]

            confidence[(imgPathTuple[0], inputs[j].data, preds_softmax[j])] = scores[j]

        ## END TEST
    return confidence

######################################################################
# Set up model

model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft.load_state_dict(torch.load('savedInitialModel.pt'))

confidenceScores = getConfidence(model_ft)
for (imgPath, image, pred) in confidenceScores.keys():
	fileName = ntpath.basename(imgPath)
	if pred == 1:
		destPath = "facesData/trainAll/attractive"
	else: 
		destPath = "facesData/trainAll/unattractive"
	shutil.copyfile(imgPath, os.path.join(destPath, fileName))

