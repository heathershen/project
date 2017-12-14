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

def visualize_model(model, num_images, phase):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders[phase]):
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
        for j in range(inputs.size()[0]):

            # Image path tuple
            imgPathTuple = image_datasets['unlabeled'].imgs[i*4+j]
            # print((imgPathTuple[0], preds_softmax[j]))
            # confidence[(imgPathTuple[0], preds_softmax[j])] = scores[j]

            confidence[(imgPathTuple[0], inputs[j].data, preds_softmax[j])] = scores[j]

        ## END TEST
    return confidence

######################################################################
# Sort the confidence scores to find the images the model is least sure of its classification
def sortScores(scores, k):
    # Returns a list of tuples (ie images, preds) based on ascending confidence scores    
    sortedInputs = sorted(scores, key=scores.get, reverse=True) 

    # Select number of least confident images (ie difficult images for the model to classify) 
    leastConfident = sortedInputs[0:k]
    # print (leastConfident)
    return leastConfident

######################################################################
def removeFromDataset(leastConfident):
    for imgPathTuple, pred in leastConfident:
        if imgPathTuple in image_datasets['unlabeled'].imgs:
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
        imgPath, image, prediction = data
        out = torchvision.utils.make_grid(image)
        fig = plt.figure()
        imshow(out, title=[class_names[prediction]])
        correctLabel = collectData()
        # correctLabelTensor = torch.LongTensor(1)
        # correctLabelTensor.fill_(correctLabel)
        imgPathTuple = (imgPath, correctLabel)
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
    # print(dataloaders['train'].dataset.imgs)
    for example in correctedExamples:
        # dataloaders['train'].dataset.imgs is the list of (image path, class_index) tuples
        # print(example)
        dataloaders['train'].dataset.imgs.append(example)

    # Retrain the model on the dataset
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'unlabeled', 'test']}
    print("TRAIN SIZE = %d" % dataset_sizes['train'])

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

######################################################################
# Data mining until we are satisfied with our accuracy
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

# Load model
model_ft = models.resnet18(pretrained=True)
# for param in model_ft.parameters():
#     param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft.load_state_dict(torch.load('savedInitialModel.pt'))

# Data mining until we are satisfied with our accuracy
accuracyHistory = []

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

acc = checkAccuracy(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
accuracyHistory.append(acc)

cv2.waitKey(0)
dataMine = 1
while True:
    try:
        dataMine = int(raw_input("Continue data mining? (type 1) else (type 0)"))
    except ValueError:
        print("Please type either 1 to continue or 0 to exit.")
        continue
    else:
        if dataMine != 0 and dataMine != 1:
            print("Please type either 1 to continue or 0 to exit.")
    
    if dataMine == 0: break    

    addDataExamples(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

    # Train from scratch
    model_ft = models.resnet18(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
    # model_ft = retrain(model_ft, criterion, optimizer_ft, exp_lr_scheduler, correctedExamples, numEpochs = 5)
    print('Finished retraining')
    print('-' * 10)
    torch.save(model_ft.state_dict(), 'savedModel.pt')

    # Check accuracy on test set
    acc = checkAccuracy(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    accuracyHistory.append(acc)

# Save model
torch.save(model_ft.state_dict(), 'savedModel.pt')

plt.plot(range(len(accuracyHistory)), accuracyHistory)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.savefig('acc_iters.jpg')