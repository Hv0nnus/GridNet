
# coding: utf-8

from os.path import exists
import csv
import os
from torch.utils import data
import torch
from PIL import Image
import numpy as np

import Loss_Error
# # Save Loss and Error

# In[18]:

""" Creat two CSV files in wich the data of the training/validation will be store.
    (0) = path : path to store the CSV files
    (1) = name_network : name of the network associated with the CSV files
    (2) = train_number : number of train associated with the CSV files
"""
def init_csv(path_CSV,name_network,train_number):
    
    # Stop the program if the CSV file already exist !
    if(exists(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv")):
        print("TODO ici il faudra faire arreter le programme ! On ecrase les fichiers !")
        #TODO ici il faudra faire arreter le programme !
    
    # header of the futur Panda"TODOs DataFrames
    header_confMat = ["Set","Value","Target","Prediction","Epoch"]
    header_loss = ["Value","Set","Epoch"]
    
    # Try to open the file and write the header
    with open(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv", 'w') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(header_confMat)

    # Try to open the file and write the header
    with open(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv", 'w') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(header_loss)


# In[ ]:

""" Save the error in CSV : Confusion matrix and loss for training and validation
    (0) = x : Input data of validation or training set
    (1) = y : Output data expected of validation or training set
    (2) = network : network that will be used to compute the y_estimated
    (3) = epoch : Actual epoch of the training
    (4) = name_network : name of the network associated with the CSV files
    (5) = train_number : number of train associated with the CSV files
    (6) = path_CSV : path to store the CSV files
    (7) = set_type : validation or train dataset
"""
def save_error(x,y,network,epoch,set_type,parameters):

    # Result of the network
    y_estimated = network(x)
    
    # Compare the real result and the one of the network
    conf_mat = Loss_Error.IoU_pd_format(y_estimated, y, set_type = set_type, epoch = epoch,parameters = parameters)
    # Store the IoU confusion matrix into a CSV file
    with open(parameters.path_CSV + "CSV_confMat_" + parameters.name_network +
                      str(parameters.train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(conf_mat)

    # Same with the loss
    loss = Loss_Error.criterion_pd_format(y_estimated = y_estimated,y = y,epoch=epoch,set_type = set_type,
                                          parameters = parameters)
    with open(parameters.path_CSV + "CSV_loss_" + parameters.name_network +
                      str(parameters.train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([loss])
    return(loss[0])


# In[19]:

def save_error_copy_useless_for_the_moment(x_train,y_train,x_validation,y_validation,network,epoch,name_network,
                                           train_number,path_CSV,parameters):
    
    #Compute the forward function for the train and the validation
    y_train_estimated = network(x_train)
    y_validation_estimated = network(x_validation)
    
    # compute the matrice of confusion between classes
    conf_mat_train = Loss_Error.IoU_pd_format(y_train_estimated, y_train, set_type = "train", epoch = epoch,
                                              parameters = parameters)
    conf_mat_validation = Loss_Error.IoU_pd_format(y_validation_estimated, y_validation, set_type = "validation",
                                                   epoch = epoch)
    
    with open(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(conf_mat_train)
        writer.writerows(conf_mat_validation)

    loss_train = Loss_Error.criterion_pd_format(y_estimated = y_train_estimated,y = y_train,epoch=epoch,
                                                set_type = "train")
    loss_validation = Loss_Error.criterion_pd_format(y_estimated = y_validation_estimated,y = y_validation,
                                          epoch=epoch, set_type = "validation")

    with open(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([loss_train])
        writer.writerows([loss_validation])


# # Save and import entire network

# In[93]:

""" Save_checkpoint save the network with weight and parameters.
    (0) = state : state contains the network, the epoch and the parameter
    (1) = filename : filname is the name of the file where the checkpoint should be saved
"""
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    
    
""" 

"""
def load_from_checkpoint(path_checkpoint,network):
    if (os.path.isfile(path_checkpoint)):
        print("=> loading checkpoint '{}'".format(path_checkpoint))
        
        checkpoint = torch.load(path_checkpoint)
        parameters = checkpoint['parameters']
        parameters.actual_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['state_dict'])
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path_checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path_checkpoint))
    return(parameters)


# # Import Data

# In[ ]:

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)


# In[2]:

def make_dataset(quality, mode,parameters):
    assert (quality == 'fine' and mode in ['train', 'val']) \
    or (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = '/leftImg8bit'
        mask_path = os.path.join(parameters.path_data, 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit'
        mask_path = os.path.join(parameters.path_data, 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'

    img_path = os.path.join(parameters.path_data, img_dir_name, mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items


class CityScapes_final(data.Dataset):
    def __init__(self, quality, mode,parameters, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(quality, mode, parameters)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        ignore_label = parameters.number_classes - 1
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)

        
        mask_copy = mask.copy()
        
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
            
        mask = Image.fromarray(mask_copy.astype(np.uint8))


        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
                 
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            mask = torch.squeeze(mask)
            return img, mask.long(), torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:

                mask = self.target_transform(mask)*255

            mask = torch.squeeze(mask)
            #mask_copy = torch.rand(19,19,19)            
            #for k in range(19):
            #    mask_copy[k,:,:] = (mask == k)*1

            return img, mask.long()
        
    def __len__(self):
        return len(self.imgs)


# In[ ]:

""" checkpoint save the network if this network is better than the previous one or just save it regulary
    (0) = validation_error : The error on the validation DataSet 
    (1) = validation_error_min : The last best validation error
    (2) = index_save_best : Index is 0 or 1, there is always a checkpoint untouched (best0 or best1)
    (3) = index_save_regular : Index is 0 or 1, there is always a checkpoint untouched(regular0 or regular1)
    (4) = epoch :
    (5) = network : 
    (6) = parameters : 
"""
def checkpoint(validation_error,validation_error_min,index_save_best,index_save_regular,epoch,network,parameters):
    
    if(validation_error < validation_error_min):    
        #Save the entire model with parameter, network and optimizer
        save_checkpoint({'epoch': epoch + 1, # +1 because we start to count at 0
                         'parameters': parameters,
                         'state_dict': network.state_dict(),
                        },
                        filename = parameters.path_save_net + "best" +str(index_save_best)+ parameters.name_network +
                        str(parameters.train_number) + "checkpoint.pth.tar")
        
        validation_error_min = validation_error
        
        print("The network as been saved at the epoch " + str(epoch) + "(best score)" + str(index_save_best))
        
        index_save_best = (index_save_best+1)%2
        
    else:
        #Maybe useless to save the network in this way
        if(False):
            
            #Save the entire model with parameter, network and optimizer
            save_checkpoint({'epoch': epoch + 1, # +1 because we start to count at 0
                             'parameters': parameters,
                             'state_dict': network.state_dict(),
                            },
                            filename = parameters.path_save_net + "save" + str(index_save_regular) +
                            parameters.name_network + str(parameters.train_number) + "checkpoint.pth.tar")
            validation_error_min = validation_error
            

            print("The network as been saved at the epoch " + str(epoch) + "(regular save)" + str(index_save_regular))
            
            index_save_regular = (index_save_regular+1)%2
        
    return(validation_error_min,index_save_best,index_save_regular)

