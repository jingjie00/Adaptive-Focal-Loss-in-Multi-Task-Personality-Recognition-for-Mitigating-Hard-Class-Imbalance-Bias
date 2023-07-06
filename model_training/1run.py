# bow

# # Part 1: Environment Setup

import os
os.getcwd()

# import general_module which locate at my parent directory's child
import sys
sys.path.append("..")
from general_module.evaluation import *
from general_module.training import *
from model_training.loss import *

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle


torch.manual_seed(42)

from transformers import logging
logging.set_verbosity_error()


# # Part 2: Load dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, dimension, feature):
        self.dataframe = dataframe
        self.dimension = dimension
        self.feature = feature

    def __getitem__(self, index):
        if self.feature == "bow":
            fea = torch.tensor(self.dataframe['text'].values[index])
        elif self.feature == "psycho":
            fea = torch.tensor(self.dataframe['psychofeature'].values[index])
        elif self.feature == "bow+psycho":
            #merge two features
            fea = torch.tensor(np.concatenate((self.dataframe['text'].values[0], self.dataframe['psychofeature'].values[0])))

        fea = fea.type(torch.FloatTensor)

        label = torch.tensor(float(str(self.dataframe[self.dimension].values[index])))
        label = label.type(torch.FloatTensor)

        return fea, label

    def __len__(self):
        return len(self.dataframe)


# # Part 3 Model Training
class CustomNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    


def train(loss_function, epochs, trainloader, validationloader, testloader):
    # get the input_size from trainloader
    in_size= trainloader.dataset[0][0].shape[0]

    network = CustomNetwork(input_size=in_size)
    network = best_device(network)

    model = CustomModel(network)
        
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    for e in range(epochs):
        # a dictionary that store the training loss, validation loss, train_size, validation_size, TP, FP, TN, FN
        running_info = {'train_loss':0, 'validation_loss':0, 'train_size':0, 'validation_size':0, 'TP':0, 'FP':0, 'TN':0, 'FN':0}

        # set to training mode
        network.train(True)

        # per epoch training activity
        for inputs, labels in trainloader:

            # clear all the gradient to 0
            optimizer.zero_grad()

            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)
            
            # compute loss
            loss = loss_function.forward(inputs=outs, targets=labels)
            
            # backpropagation
            loss.backward()

            # update w
            optimizer.step()

            # update running_info
            running_info['train_loss'] += loss.item()*labels.size(0)
            running_info['train_size'] += labels.size(0)


        # Turn off training mode for reporting validation loss
        network.train(False)

        # per epoch validation activity
        for inputs, labels in validationloader:

 
            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)

            # update running_info
            running_info['validation_loss'] += loss.item()*labels.size(0)
            running_info['validation_size'] += labels.size(0)

            preds = (outs > 0.5).type(torch.FloatTensor)
            running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN'] = e_confusion_matrix(preds,labels)

        
        train_loss = running_info['train_loss']/running_info['train_size']
        validation_loss = running_info['validation_loss']/running_info['validation_size']

        confusion_matrix = running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN']
        regular_accuracy,balanced_accuracy = e_accuracy(confusion_matrix)

        print(f'[Epoch {e + 1:2d}/{epochs:d}]: train_loss = {train_loss:.4f}, validation_loss = {validation_loss:.4f}, RA = {regular_accuracy:.4f}, BA: {balanced_accuracy:.4f}, CM:{confusion_matrix}')

        model.update(network, epochs = e+1, ba = balanced_accuracy, ra=regular_accuracy)

     # per epoch test activity
    for inputs, labels in testloader:


        inputs,labels = best_device(inputs, labels)

        # forward propagation
        outs = network(inputs)
        outs = outs.view(-1)

        # update running_info
        running_info['test_loss'] += loss.item()*labels.size(0)
        running_info['test_size'] += labels.size(0)

        preds = (outs > 0.5).type(torch.FloatTensor)
        running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN'] = e_confusion_matrix(preds,labels)
    
    
    return model



# # Execution

def run(config):
    trainset_dataframe = extract("../dataset/merged/"+config["dataset"]+"_train.pickle")
    validationset_dataframe = extract("../dataset/merged/"+config["dataset"]+"_validation.pickle")
    testset_dataframe = extract("../dataset/merged/"+config["dataset"]+"_test.pickle")
    
    # a dictionary of model on different personality dimension, "O", "C", "E", "A", "N"
    if config["dataset"] =="essays":
        models = {"O":None, "C":None, "E":None, "A":None, "N":None}
    elif config["dataset"] =="mbti":
        models = {"O":None, "C":None, "E":None, "A":None}
    else:
        raise Exception("dataset name not found")
    

    if config["feature"] != "bow+psycho" and config["feature"] != "bow" and config["feature"] != "psycho":
        raise ValueError("feature must be one of bow, psycho, bow+psycho")
    


    # train the model on different personality dimension using for loop on the key of the dictionary
    for dimension in models.keys():

        if config["loss"] == "bce":
            loss_function = BCE()

        # weighted
        elif config["loss"] == "wbces":
            loss_function = WBCEs(trainset_dataframe[dimension])
        elif config["loss"] == "wbceb":
            loss_function = WBCEb()
        elif config["loss"] == "bbces":
            loss_function = BBCEs(trainset_dataframe[dimension])
        elif config["loss"] == "bbceb":
            loss_function = BBCEb()
        
        # focal loss
        elif config["loss"] == "cfbce":
            loss_function = CFBCE()
        elif config["loss"] == "wfbces":
            loss_function = WFBCEb()
        else:
            raise Exception("loss function not found")

        print("P_"+dimension)

        custom_trainset = CustomDataset(dataframe=trainset_dataframe,dimension=dimension, feature=config["feature"])
        custom_validationset = CustomDataset(dataframe=validationset_dataframe,dimension=dimension, feature=config["feature"])
        custom_textset = CustomDataset(dataframe=testset_dataframe,dimension=dimension, feature=config["feature"])

        batch_size = 256
        trainloader = DataLoader(custom_trainset, batch_size=batch_size, shuffle=False)
        validationloader = DataLoader(custom_validationset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(custom_textset,batch_size=batch_size, shuffle=False)

        model = train(loss_function = loss_function, epochs= config["epochs"], trainloader=trainloader,testloader = testloader, validationloader=validationloader)
        
        models[dimension] = model

    return models

config ={
    "dataset":"mbti", #mbti, essays
    "feature":"bow+psycho", #bow, psycho, bow+psycho
    "loss":"bce", #bce, wbces, wbceb, bbces, bbceb, cfbce, wfbceb
    "epochs":200 #any
}

print_results(run(config=config))



