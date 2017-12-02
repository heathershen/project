# -*- coding: utf-8 -*-
"""
Transfer Learning from PyTorch tutorial by `Sasank Chilamkurthy <https://chsasank.github.io>`_

**Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.

"""

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

plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# We have about 120 training images each for hot or not.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),

        # transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'unlabeled': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
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
                  for x in ['train', 'val', 'unlabeled', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'unlabeled', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'unlabeled', 'test']}
print("TRAIN SIZE = %d" % dataset_sizes['train'])
print("VAL SIZE = %d" % dataset_sizes['val'])

class_names = image_datasets['train'].classes

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # print(outputs.size())
                # print(labels.size())
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j][0]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


######################################################################
# Run current model over unlabeled data and get confidence of each classification

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
        for i in range(inputs.size()[0]):
            # print("SCORE:", scores[i])
            # confidence[inputs[i]] = (scores[i], preds_softmax[i]) # store each input's (confidence, prediction)
            # Each key is a tuple (image, label)

            # isLabeled = False
            # for image in labeledImages:
            #     if torch.equal(inputs[i].data, image): 
            #         isLabeled = True
            #         break
            # if isLabeled: continue

            # Image path
            imgPathTuple = image_datasets['unlabeled'].imgs[i]

            confidence[(imgPathTuple, inputs[i].data, preds_softmax[i])] = scores[i].numpy() 

        ## END TEST
    return confidence

######################################################################
# Sort the confidence scores to find the images the model is least sure of its classification
def sortScores(scores, k):
    # Returns a list of tuples (ie images, preds) based on ascending confidence scores    
    sortedInputs = sorted(scores, key=scores.get) 

    # Select number of least confident images (ie difficult images for the model to classify) 
    leastConfident = sortedInputs[0:k]
    return leastConfident

######################################################################
def removeFromDataset(leastConfident):
    for imgPathTuple, image, pred in leastConfident:
        image_datasets['unlabeled'].imgs.remove(imgPathTuple)

######################################################################
# Visualize the least confident images and show the model's prediction for them
def collectData():
    cv2.waitKey(0)
    
    while True:
        try:
            attractiveRating = int(raw_input("Attractive? (type 1) else (type 0)"))
        except ValueError:
            print("Please type either 1 for attractive or 0 for unattractive.")
            continue
        else:
            if attractiveRating != 0 and attractiveRating != 1:
                print("Please type either 1 for attractive or 0 for unattractive.")
            else:
                return attractiveRating


def visualizeConfidence(leastConfident, num_images):
    corrected = []
    for data in leastConfident:
        imgPathTuple, image, prediction = data
        out = torchvision.utils.make_grid(image)
        fig = plt.figure()
        imshow(out, title=[class_names[prediction.numpy()[0].astype(int)]])
        correctLabel = collectData()
        correctLabelTensor = torch.LongTensor(1)
        correctLabelTensor.fill_(correctLabel)
        imgPathTuple = (imgPathTuple[0], correctLabelTensor)
        corrected.append(imgPathTuple)
        plt.close()
    return corrected

######################################################################
# Run on test set
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

######################################################################
# Add ambiguous inputs to the training set
def addDataExamples(model_ft, criterion, optimizer_ft, exp_lr_scheduler):
    # Find confidence of scores
    confidenceScores = getConfidence(model_ft)
    print('Model run on unlabeled set')
    print('-' * 10)
    k = 20

    # Select k number of images that the model is least confident about and remove them 
    # from the unlabeled dataset
    leastConfident = sortScores(confidenceScores, k)
    # removeFromDataset(leastConfident)
    print('Confidence sorted')
    print('-' * 10)

    # Correct any misclassifications and add the new images to the training set
    correctedExamples = visualizeConfidence(leastConfident, k)
    for example in correctedExamples:
        # dataloaders['train'].dataset.imgs is the list of (image path, class_index) tuples
        dataloaders['train'].dataset.imgs.append(example)

    # Retrain the model on the dataset
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'unlabeled', 'test']}
    print("TRAIN SIZE = %d" % dataset_sizes['train'])

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)
    # model_ft = retrain(model_ft, criterion, optimizer_ft, exp_lr_scheduler, correctedExamples, numEpochs = 5)
    print('Finished retraining')
    print('-' * 10)
    
    # Check accuracy on test set
    accuracy = checkAccuracy(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    return labeledImages, accuracy

######################################################################
# MAIN


# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
print('==========INITIAL TRAINING==========')
print()
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
                       # num_epochs=25)

visualize_model(model_ft)
plt.ioff()
plt.show()
plt.savefig('preds_initial.png')
print()
print('==========INITIAL TRAINING FINISHED==========')
print()
######################################################################
# Data mining until we are satisfied with our accuracy
accuracyHistory = []
acc = checkAccuracy(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
accuracyHistory.append(acc)

labeledImages = []
while acc < 0.90:
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    labeledImages, acc = addDataExamples(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    accuracyHistory.append(acc)

plt.plot(range(len(accuracyHistory)), accuracyHistory)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('acc_iters.jpg')
