# coding: utf-8

# Torch related package
from torch.utils import data
import torch

# Other package
from os.path import exists
import csv
import os
from PIL import Image
import numpy as np
import random

# Own python program
import Loss_Error


def init_csv(path_CSV, name_network, train_number, path_print):
    """

    :param path_CSV: path to store the CSV files
    :param name_network: name of the network associated with the CSV files
    :param train_number: number of train associated with the CSV files
    :param path_print: path were the print are display
    :return: Nothing but create two CSV files in which the data of the training/validation will be store.
    """

    # Stop the program if the CSV file already exist
    if exists(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv"):
        with open(path_print, 'a') as txtfile:
            txtfile.write("Ici il faudra faire arreter le programme ! On ecrase les fichiers ! \n")
        # TODO ici il faudra faire arreter le programme !
    
    # header of the futur Pandas DataFrames
    header_confMat = ["Set", "Value", "Target", "Prediction", "Epoch"]
    header_loss = ["Value", "Set", "Epoch"]
    
    # Try to open the file and write the header
    with open(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv", 'w') as csvfile:
        cwriter = csv.writer(csvfile,
                             delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(header_confMat)

    # Try to open the file and write the header
    with open(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv", 'w') as csvfile:
        cwriter = csv.writer(csvfile,
                             delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(header_loss)



def save_error(x,y,network,epoch,set_type,parameters):
    """
    :param x: Input data of validation or training set
    :param y: Output data expected of validation or training set
    :param network: network that will be used to compute the y_estimated
    :param epoch: Actual epoch of the training
    :param set_type: Validation or train DataSet
    :param parameters: List of all the parameters
    :return: Nothing but save the error in CSV : Confusion matrix and loss for training and validation
    """

    # Result of the network
    y_estimated = network(x)
    
    # Compare the real result and the one of the network
    conf_mat = Loss_Error.IoU_pd_format(y_estimated=y_estimated,
                                        y=y,
                                        set_type=set_type,
                                        epoch=epoch,
                                        parameters=parameters)

    # Store the IoU confusion matrix into a CSV file
    with open(parameters.path_CSV + "CSV_confMat_" + parameters.name_network +
              str(parameters.train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(conf_mat)

    # Same with the loss
    loss = Loss_Error.criterion_pd_format(y_estimated=y_estimated,
                                          y=y,
                                          epoch=epoch,
                                          set_type=set_type,
                                          parameters=parameters)

    with open(parameters.path_CSV + "CSV_loss_" + parameters.name_network +
              str(parameters.train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([loss])

    return loss[0]


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """ Save_checkpoint save the network with weight and parameters.
        (0) = state : state contains the network, the epoch and the parameter
        (1) = filename : filname is the name of the file where the checkpoint should be saved
    """
    torch.save(state, filename)


def load_from_checkpoint(path_checkpoint, network, path_print):
    """
    :param path_checkpoint: path to load the network and the parameters
    :param network: network architecture that will will be updated with all the weights
    :param path_print: path and file were the print are display
    :return: Parameters that contain all the parameter of the previous run and
    change Network that that all the weight are set to the saved one.
    """

    # If the file exist we import the Data
    if os.path.isfile(path_checkpoint):

        # Show that the file will be loaded
        with open(path_print, 'a') as txtfile:
            txtfile.write("=> loading checkpoint "+str(format(path_checkpoint)) + "\n")

        # Load the structure
        checkpoint = torch.load(path_checkpoint)

        # Set the parameters
        parameters = checkpoint['parameters']

        # Change the actual epoch to the last iteration done
        parameters.actual_epoch = checkpoint['epoch']

        # Load the weight
        network.load_state_dict(checkpoint['state_dict'])

        # Show that the file as been loaded correctly
        with open(path_print, 'a') as txtfile:
            txtfile.write("=> loaded checkpoint '{}' (epoch {})" +
                          str(format(path_checkpoint, str(checkpoint['epoch']))) + "\n")
        return parameters

    # If there is no file, print an error
    else:
        with open(path_print, 'a') as txtfile:
            txtfile.write("=> no checkpoint found at '{}'" + str(format(path_checkpoint)) + "\n")
        return ()


def colorize_mask(mask):
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)

    for i in range(zero_pad):
        palette.append(0)

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)


# In[2]:

def make_dataset(quality, mode,parameters):
    assert (quality == 'fine' and mode in ['train', 'val','test']) \
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
    assert (len(set(os.listdir(img_path))-set(os.listdir(mask_path))) +
            len(set(os.listdir(mask_path))-set(os.listdir(img_path))))== 0
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items


class CityScapes_final(data.Dataset):
    def __init__(self, quality, mode,parameters, joint_transform=None,
                 sliding_crop=None, transform=None, transform_target = None):
        self.imgs = make_dataset(quality, mode, parameters)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.transform_target = transform_target
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

            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            if self.transform is not None:
                random.seed(seed)  # apply this seed to img transforms
                img_slices = [self.transform(e) for e in img_slices]
            if self.transform_target is not None:
                random.seed(seed)  # apply this seed to mask transforms
                mask_slices = [self.transform_target(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            mask = torch.squeeze(mask)
            return img, mask.long()
        else:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            if self.transform is not None:
                random.seed(seed)  # apply this seed to img transforms
                img = self.transform(img)
            if self.transform_target is not None:
                random.seed(seed)  # apply this seed to mask transforms
                mask = self.transform_target(mask)
                mask = (mask * 255).round()
            mask = torch.squeeze(mask)
            return img, mask.long()
        
    def __len__(self):
        return len(self.imgs)


def checkpoint(validation_error,validation_error_min,index_save_best,
               index_save_regular,epoch,network,parameters):
    """
    :param validation_error: The error on the validation DataSet
    :param validation_error_min: The last best validation error
    :param index_save_best: Index is 0 or 1, there is always a checkpoint untouched (best0 or best1)
    :param index_save_regular: Index is 0 or 1, there is always a checkpoint untouched(regular0 or regular1)
    :param epoch: Epoch of the algorithm
    :param network: GridNet with all parameter that will be saved
    :param parameters: instance of the parameter class that store many usefull information of the network
    :return: The new validation_error_min, the next index_save_best and index_save_regular.
    It also save the network if there is a better validation error
    """
    
    if validation_error < validation_error_min:

        # Save the entire model with parameter, network and optimizer
        save_checkpoint({'epoch': epoch + 1, # +1 because we start to count at 0
                         'parameters': parameters,
                         'state_dict': network.state_dict(),
                        },
                        filename = parameters.path_save_net + "best" +str(index_save_best)+ parameters.name_network +
                        str(parameters.train_number) + "checkpoint.pth.tar")
        
        validation_error_min = validation_error
        
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("The network as been saved at the epoch " + str(epoch) + " (best score) " +
                          str(index_save_best) + '\n')

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
        
    return validation_error_min,index_save_best,index_save_regular


def time_to_string(time):
    """
    :param time: Duration in second (float)
    :return: A string with the time in day, hour, minute, second
    """

    # Divmod is the euclidian division
    q, s = divmod(time, 60)
    q, s = round(q), round(s, 2)

    if q == 0:
        return "Seconds:" + str(s)

    q, m = divmod(q, 60)
    if q == 0:
        return "Minutes:" + str(m) + " Seconds:" + str(s)

    j, h = divmod(q, 24)
    return "Days:" + str(j) + " Hours:" + str(h) + " Minutes:" + str(m) + " Seconds:" + str(s)