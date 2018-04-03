
# coding: utf-8

# # Package

# In[1]:

#Torch related package
from __future__ import print_function
import argparse
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from torch.utils import data


# In[2]:

#cuda related package
import torch.cuda
import torch.backends.cudnn as cudnn
print(torch.backends.cudnn.version())


# In[3]:

# Other package
import time
#from ggplot import *
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from os.path import exists
#import csv
#import os
from PIL import Image
#import seaborn as sns; sns.set()
import sys
import random
import math



# In[5]:

import GridNet_structure
import Save_import
import Loss_Error

txt_path = "/home_expes/kt82128h/GridNet/Python_Files/Python_print.txt"
txt_path = "Python_print.txt"

with open(txt_path, 'w') as txtfile:
            txtfile.write('\n               Start of the program \n')

# # Commentaire pour la suite (TODO)

# On peut enlever le biais dans les convolutions avant la normalisation
# 
# Le diminution de la taille des images pas clair cette division par deux....
# 
# Change la fonction de cout, lien entre fonction de cout et de perte ?
# 
# 
# Il faut enlever les biais quand il y a une batch normalisation apres
# 
# Verifier ce qu est la mesure de test IoU
# 
# Mettre des commentaire
# 
# Ne pas sauvgarder n importe quand le réseau. Peut ere garder un truc qui a en memoire le meilleur IuO sur 
# la validation
# 
# TODO Quel modification sont a faire sur les donnees  genre pour les augmenter
# 
# Parallelisation, comment ca marche 
# 
# COMMENT VERIFIER QUE TOUT FONCTIONNE Regarder le graph peut être.
# Regarder le temps de calcul quand 
# 
# Idee: reregarder comment on fait de la segmentation avec les champs de Markov, ca peut donner des idees
# 
# Afficher le nom de la courbe avec la courbe et pas juste a coter
# 
# 
# Indice qui indique si une fonction est derivable informatiquement, et prendre uen fonction qui se rapproche de l IoU
# 
# Corrélation entre IoU et le logsoftmax
# 
# distance entre deux fonctions 
# 
# Faire les tests sur la base de tests
# 
# Taille en octet de ce que j'importe et du réseau 
# # Code

# In[7]:

class Parameters():
    def __init__(self,
                 # Number of columns of the grid
                 nColumns = 2,
                 # Number of features map at each rows
                 nFeatMaps = [3,6],
                 # Number of feature map of the input image
                 nFeatureMaps_init = 3,
                 #Number of classes (19(usefull classes) + 1(all other classes together))
                 number_classes = 19,
                 # DataFrame with the name of each label associated with their number
                 label_DF = None,

                 # Size of initial image
                 width_image_initial = 2048, height_image_initial = 1024,
                 # Size after the crop # 353 the perfect size !
                 width_image_crop = 19, height_image_crop = 19,

                 # Probability of a Blockwise dropout
                 dropFactor = 0.1,
                 learning_rate=0.01,
                 weight_decay = 5*10**(-6),
                 #Parameter of the Adam Optimizer (beta1 beta2 and epsilon)
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 1*10**(-8),
                 # Size of the mini batch
                 batch_size = 2,
                 # Size of the mini batch to compute error (if the entire validation set cannot be in loaded)
                 batch_size_val = 10,
                 # Maximum value of epoch iteration
                 epoch_total = 10,
                 # The actual epoch is not null if we train the network which has already been train
                 actual_epoch = 0,

                 # File where all the parameter model can be store
                 path_save_net = "../Model/",
                 #Name of the network, used for store (name_network and train_number)
                 name_network = "test",
                 train_number = 0,
                 # File where the error will be stored
                 path_CSV = "../CSV/",
                 # Path of the Data
                 path_data = "../Cityscapes_Copy/",
                 # Number of process that will load the Data
                 num_workers = 0):
        
        super(Parameters, self).__init__()
        # Image
        self.number_classes = number_classes
        self.label_DF = label_DF
        self.width_image_initial = width_image_initial
        self.height_image_initial = height_image_initial
        self.width_image_crop = width_image_crop
        self.height_image_crop = height_image_crop
        # Number of feature map at the begining, if RGB image it would be 3
        self.nFeatureMaps_init = nFeatureMaps_init
        self.path_data = path_data
        
        # GridNet
        self.nColumns = nColumns
        self.nFeatMaps = nFeatMaps
        self.name_network = name_network
        self.train_number = train_number
        self.num_workers = num_workers
        
        #Save
        self.path_CSV = path_CSV
        self.path_save_net = path_save_net
        
        
        # Learning
        self.dropFactor = dropFactor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.epoch_total = epoch_total
        self.actual_epoch = actual_epoch


# In[8]:

"""train return nothing but modify the weight of the network and call save_error to store the error.
    (0) = parameters : list of parameters of the network
    (1) = network : network that will be learned
    (2) = train_loader : Dataloader which contains input and target of the train dataset
    (3) = val_loader : Dataloader which contains input and target of the validation dataset
"""
def train(parameters,network,train_loader,val_loader):
    #Store the time at the begining of the training
    timer_init = time.time()
     
    # create your optimizer
    optimizer = optim.Adam(params = network.parameters(), lr = parameters.learning_rate,
                           betas = (parameters.beta1, parameters.beta2),
                           eps = parameters.epsilon, weight_decay = parameters.weight_decay)
    
    #High value just to initialize this variable
    # validation error min will store the lowest validation result
    validation_error_min = 9999
    
    # Store the indice of the next checkpoint. This value is 0 or 1. We always keep one checkpoint untouched
    # while the other one is changed.
    index_save_regular = 0
    index_save_best = 0
    
    
    for epoch in range(parameters.actual_epoch,parameters.epoch_total):
        #Store the time at the begining of each epoch
        timer_epoch = time.time()
        timer_batch = time.time()

        for i,(x_batch, y_batch) in enumerate(train_loader):
            # zero the gradient buffers
            optimizer.zero_grad()
            
            #Transform into Variable
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)

            # Compute the forward function
            y_batch_estimated = network(x_batch)
            
            #Get the error
            loss = Loss_Error.criterion(y_batch_estimated, y_batch, parameters)
            
            #Compute the backward function
            loss.backward()
            
            # Does the update according to the optimizer define above
            optimizer.step()

            #Save error of the training Dataset
            Save_import.save_error(x = x_batch,y = y_batch, network = network,epoch = epoch, set_type = "train",
                                   parameters = parameters)
            with open(txt_path, 'a') as txtfile:
                txtfile.write("\nEpoch : "+str(epoch)+". Batch : "+str(i)+".\nLast loss : "+str(loss.data[0])+ "\n" +
                              "Time batch : " + str(time.time() - timer_batch) +
                              ". Time total batch : " + str(time.time() - timer_epoch)+"\n \n")

            timer_batch = time.time()
            break
        # Validation_error contains the error on the validation set
        validation_error = 0

        # Save the error of the validation Dataset
        for i,(x_val_batch, y_val_batch) in enumerate(val_loader):
            x_val_batch, y_val_batch = Variable(x_val_batch), Variable(y_val_batch)
            validation_error += Save_import.save_error(x = x_val_batch,y = y_val_batch,network = network,epoch = epoch,
                                                       set_type = "validation", parameters = parameters)

        # Divise by the the number of element in the entire batch
        validation_error = validation_error/(i+1)
        with open(txt_path,'a') as txtfile:
            txtfile.write("temps necessaire pour effectuer la validation sur toutes les donne : "+str(time.time-timer_batch))
        time_before_save = time.time()
        # checkpoint will save the network if needed
        validation_error_min,index_save_best,index_save_regular = Save_import.checkpoint(validation_error,
                                                                                         validation_error_min,
                                                                                         index_save_best,
                                                                                         index_save_regular,
                                                                                         epoch,network,parameters,
                                                                                         txt_path)
        with open(txt_path,'a') as txtfile:
            txtfile.write("temps necessaire pour effectuer la sauvegarde sur toutes les donne : "+str(time.time-time_before_save))
                    
        with open(txt_path, 'a') as txtfile:
            txtfile.write("\n              End of Epoch :" + str(epoch) + "/" + str(parameters.epoch_total - 1) +
                          ". Validation Loss : " + str(validation_error) + ".\nTime Epoch :" +
                          str(time.time() - timer_epoch) + ". Time total : " + str(time.time() - timer_init)+"\n \n")


    with open(txt_path, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n")
    
    return()


# In[13]:

label_name = {'Real_name' : ["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
                             "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle",
                             "bicycle","autre"],
              'Class_name' : ["class"+str(i) for i in range(19)] + ["class19"]}

label_DF = pd.DataFrame(data=label_name)
 

parameters = Parameters(nColumns = 2,
                            nFeatMaps = [3,6],
                            nFeatureMaps_init = 3,
                            number_classes = 20-1,
                            label_DF = label_DF,

                            width_image_initial = 2048, height_image_initial = 1024,
                            width_image_crop = 353, height_image_crop = 353,

                            dropFactor = 0.1,
                            learning_rate=0.01,
                            weight_decay = 5*10**(-6),
                            beta1 = 0.9,
                            beta2 = 0.999,
                            epsilon = 1*10**(-8),
                            # taille memoire : a = 155000 b = 1810000
                            batch_size = 5,
                            batch_size_val = 5,
                            epoch_total = 4,
                            actual_epoch = 0,

                            path_save_net = "/home_expes/kt82128h/GridNet/Python_Files/Model/",
                            name_network = "test",
                            train_number = 0,
                            path_CSV = "/home_expes/kt82128h/GridNet/Python_Files/CSV/",
                            path_data = "/home_expes/collections/Cityscapes/",
                            num_workers = 0)


# In[10]:

def main(path_continue_learning = None, total_epoch = 0, parameters = parameters):
    
    #Transformation that will be apply on the data just after the import
    transform = transforms.Compose([
        #transforms.CenterCrop(parameters.width_image_crop),
        # We keep ratio that are given by default
        # And put scale in order to always have an image smaller than 1024 for the crop.
        # With 0.2 and 0.37 for scale value we can always crop into the image
        #transforms.RandomResizedCrop(5, scale=(0.2, 0.37), ratio=(0.75, 1.3333333333333333)),
        # TODO choisir laquel des deux solution
        # Autre option, pas de ratio car cela n a pas de sens de deformer l image
        #transforms.RandomResizedCrop(5, scale=(0.763, 0.5), ratio=(1,1)),
        transforms.RandomSizedCrop(parameters.width_image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    #Transformation that will be apply on the data just after the import
    transform_target = transforms.Compose([
        #transforms.CenterCrop(parameters.width_image_crop),
        #transforms.RandomResizedCrop(5, 5, scale=(0.2, 0.37), ratio=(0.75, 1.3333333333333333),
        #transforms.RandomResizedCrop(5, scale=(0.763, 0.5), ratio=(1, 1),
        #                             interpolation= Image.NEAREST),
        transforms.RandomSizedCrop(parameters.width_image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    
    #Import both dataset with the transformation
    train_dataset = Save_import.CityScapes_final('fine', 'train',transform = transform,
                                                 transform_target = transform_target,parameters = parameters)
    val_dataset = Save_import.CityScapes_final('fine', 'val',transform = transform,
                                               transform_target=transform_target, parameters=parameters)
    # Creat the DataSet for pytorch used
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True,
                                               num_workers=parameters.num_workers,drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=parameters.batch_size_val, shuffle = True,
                                             num_workers=parameters.num_workers,drop_last=False)

    # Define the GridNet
    network = GridNet_structure.gridNet(nInputs = parameters.nFeatureMaps_init,nOutputs = parameters.number_classes,
                      nColumns = parameters.nColumns,nFeatMaps = parameters.nFeatMaps, dropFactor = parameters.dropFactor)

    if(path_continue_learning is not None):
        # Load the trained Network
        parameters = Save_import.load_from_checkpoint(
            path_checkpoint=parameters.path_save_net + path_continue_learning,
            network=network, txt_path=txt_path)

        # Here we can change some parameters
        parameters.epoch_total = total_epoch

    else:
        #Init the csv file that will store the error
        Save_import.init_csv(name_network = parameters.name_network, train_number = parameters.train_number,
                 path_CSV = parameters.path_CSV,txt_path = txt_path)

    #Train the network
    train(network = network, parameters = parameters, train_loader = train_loader, val_loader = val_loader)


if(len(sys.argv) == 3):
    main(path_continue_learning = sys.argv[1], total_epoch = sys.argv[2], parameters = parameters)
if(len(sys.argv) == 1):
    main()
