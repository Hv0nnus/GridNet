# coding: utf-8

# Torch related package
from torch.utils import data
import torch
from collections import OrderedDict

# Other package
from os.path import exists
import csv
import os
from PIL import Image
import numpy as np
import random
import pandas as pd

# Own python program
import Loss_Error
import GridNet_structure


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
            txtfile.write("We don t want to delete the files that already exist \n")
            # raise RuntimeError('We don t want to delete the files that already exist')
            # TODO ici il faudra faire arreter le programme !

    # header of the futur Pandas DataFrames
    header_confMat = ['Target', 'Prediction', 'Epoch', 'Set', 'Value']
    header_loss = ["Set", "Epoch", "Value"]

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


def duplicated_csv(path_CSV, old_name_network, new_name_network, train_number):
    """
    :param path_CSV: path to store the CSV files
    :param old_name_network: name of the network associated with the CSV files
    :param new_name_network: name of the new network
    :param train_number: number of train associated with the CSV files
    :return: Nothing but copy the actual CSV with the the name of the network
    """
    # Import the CSV file into pandas DataFrame
    loss_DF = pd.read_csv(path_CSV + "CSV_loss_" + old_name_network + str(train_number) + ".csv")
    # This Groupby will regroupe all line that have the same "Set" and "Epoch" and compute the mean over the "Values"
    loss_DF = loss_DF.groupby(['Set', 'Epoch'])['Value'].mean().reset_index()
    # Recreate the CSV file
    loss_DF.to_csv(path_CSV + "CSV_loss_" + new_name_network + str(train_number) + ".csv", index=False)

    # Import the CSV file into pandas DataFrame
    conf_DF = pd.read_csv(path_CSV + "CSV_confMat_" + old_name_network + str(train_number) + ".csv")
    # This Groupby will regroupe all line that have the same 'Target','Prediction','Epoch','Set'
    # and compute the mean over the "Values"
    conf_DF = conf_DF.groupby(['Target', 'Prediction', 'Epoch', 'Set'])['Value'].mean().reset_index()
    # Recreate the CSV file
    conf_DF.to_csv(path_CSV + "CSV_confMat_" + new_name_network + str(train_number) + ".csv", index=False)


def save_error(x, y, network, epoch, set_type, parameters, loss=None, y_estimated=None):
    """
    :param x: Input data of validation or training set
    :param y: Output data expected of validation or training set
    :param network: network that will be used to compute the y_estimated
    :param epoch: Actual epoch of the training
    :param set_type: Validation or train DataSet
    :param parameters: List of all the parameters
    :param loss: Value of loss if we are on the train set
    :param y_estimated: Value of y_predicted by the network
    :return: the loss and save the error in CSV : Confusion matrix and loss for the set_type
    """
    if y_estimated is None:
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

    # If we don't compute the loss before (we are on validation set). We compute it.
    if loss is None:
        # Same with the loss
        loss = Loss_Error.criterion_pd_format(y_estimated=y_estimated,
                                              y=y,
                                              epoch=epoch,
                                              set_type=set_type,
                                              parameters=parameters)

    # If the loss was already compute, we transform it to a better format
    else:
        loss = [set_type, epoch, loss.data[0]]

    # Save the loss
    with open(parameters.path_CSV + "CSV_loss_" + parameters.name_network +
              str(parameters.train_number) + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows([loss])

    return loss[2]


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: state contains the network, the epoch and the parameter
    :param filename: filname is the name of the file where the checkpoint should be saved
    :return: Save_checkpoint save the network with weight and parameters.
    """
    torch.save(state, filename)


def load_from_checkpoint(path_checkpoint):
    """
    :param path_checkpoint: path to load the network and the parameters
    :return: Parameters that contain all the parameters of the previous run and
     the network with all the weight set to the saved one.
    """

    # If the file exist we import the Data
    if os.path.isfile(path_checkpoint):
        is_GPU = False
        if is_GPU:
            # Load the structure in GPU
            checkpoint = torch.load(path_checkpoint)
        else:
            # If it failed try in CPU
            checkpoint = torch.load(path_checkpoint, map_location=lambda storage, loc: storage)

        # Set the parameters
        parameters = checkpoint['parameters']

        # Change the actual epoch to the last iteration done
        parameters.actual_epoch = checkpoint['epoch']

        # Recreate the same network
        network = GridNet_structure.gridNet(nInputs=parameters.nFeatureMaps_init, nOutputs=parameters.number_classes,
                                            nColumns=parameters.nColumns, nFeatMaps=parameters.nFeatMaps,
                                            dropFactor=parameters.dropFactor)

        if torch.cuda.device_count() > 1:
            with open('Python_print_test.txt', 'a') as txtfile:
                txtfile.write("\nLet's use " + str(torch.cuda.device_count()) + " GPUs! \n")
            network = torch.nn.DataParallel(network)
        else:
            with open('Python_print_test.txt', 'a') as txtfile:
                txtfile.write("\nWe can t use more than 1 GPU \n")

        if torch.cuda.is_available():
            network.cuda()
        else:
            with open('Python_print_test.txt', 'a') as txtfile:
                txtfile.write("\nAccording to torch Cuda is not available \n")

        # Load the parameters
        if torch.cuda.device_count() <= -100:
            network.load_state_dict(checkpoint['state_dict'])
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            network.load_state_dict(new_state_dict)

        # Show that the file as been loaded correctly
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("=> loaded checkpoint '{}' (epoch {})" +
                          str(format(path_checkpoint, str(checkpoint['epoch']))) + "\n")
        return parameters, network

    # If there is no file, print an error
    else:
        raise ValueError('No file to load here')


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


def make_dataset(quality, mode, path_data, only_image):
    assert (quality == 'fine' and mode in ['train', 'val', 'test']) \
           or (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])
    img_dir_name = 'leftImg8bit'

    if quality == 'coarse':
        mask_path = os.path.join(path_data, 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        if not only_image:
            mask_path = os.path.join(path_data, 'gtFine', mode)
            mask_postfix = '_gtFine_labelIds.png'

    img_path = os.path.join(path_data, img_dir_name, mode)

    if not only_image:
        assert (len(set(os.listdir(img_path)) - set(os.listdir(mask_path))) +
                len(set(os.listdir(mask_path)) - set(os.listdir(img_path)))) == 0

    items = []
    image_names = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            if not only_image:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                        os.path.join(mask_path, c, it + mask_postfix))
            else:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), None)
            image_names.append(os.path.join(c, it + '_leftImg8bit.png'))
            items.append(item)
    return items, image_names


class cityscapes_create_dataset(data.Dataset):
    def __init__(self,
                 quality,
                 mode,
                 parameters,
                 joint_transform=None,
                 sliding_crop=None,
                 transform=None,
                 transform_target=None,
                 only_image=False):

        self.imgs, self.image_names = make_dataset(quality, mode, parameters.path_data, only_image)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.transform_target = transform_target
        ignore_label = parameters.number_classes
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path).convert('RGB')

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        if self.transform is not None:
            random.seed(seed)  # apply this seed to img transforms
            img = self.transform(img)

        if mask_path is not None:
            mask = Image.open(mask_path)

            mask = np.array(mask)

            mask_copy = mask.copy()

            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v

            mask = Image.fromarray(mask_copy.astype(np.uint8))

            if self.transform_target is not None:
                random.seed(seed)  # apply this seed to mask transforms
                mask = self.transform_target(mask)
                mask = (mask * 255).round()

            mask = torch.squeeze(mask)

            return img, mask.long(), self.image_names
        else:
            return img, self.image_names[index]

    def __len__(self):
        return len(self.imgs)


def make_dataset_pretrain(mode, path_data):
    assert (mode in ['train', 'val'])

    assert len(set(os.listdir(path_data))) != 0

    items = []
    image_names = []

    fichier_classes_imagesnet = open("./image_net_only_classes_small.txt").read()
    list_classes_imagesnet = []
    for i, ligne in enumerate(fichier_classes_imagesnet.split('\n')):
        list_classes_imagesnet.append(ligne)
    list_classes_imagesnet = list_classes_imagesnet[:-1]

    for i, c in enumerate(list_classes_imagesnet):
        c_items = os.listdir(os.path.join(path_data, c))
        if mode == 'train':
            for it in range(0, int(0.7 * len(c_items))):
                items.append((os.path.join(path_data, c, c_items[it]), i))
        if mode == 'val':
            for it in range(int(0.7 * len(c_items)), len(c_items)):
                items.append((os.path.join(path_data, c, c_items[it]), i))
    return items


class cityscapes_create_dataset_pretrain(data.Dataset):
    def __init__(self,
                 mode,
                 parameters,
                 sliding_crop=None,
                 transform=None):

        self.imgs = make_dataset_pretrain(mode, parameters.path_data)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.sliding_crop = sliding_crop
        self.transform = transform

    def __getitem__(self, index):

        img_path, img_class = self.imgs[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_class

    def __len__(self):
        return len(self.imgs)


def checkpoint(validation_error, validation_error_min, index_save_best,
               index_save_regular, epoch, network, parameters, network_final = None):
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
        save_checkpoint({'epoch': epoch + 1,  # +1 because we start to count at 0
                         'parameters': parameters,
                         'state_dict': network.state_dict(),
                         },
                        filename=parameters.path_save_net + "best" + str(index_save_best) + parameters.name_network +
                                 str(parameters.train_number) + "checkpoint.pth.tar")
        if network_final is not None:
            save_checkpoint({'state_dict': network_final.state_dict(),
                             },
                            filename=parameters.path_save_net + "best" + str(
                                index_save_best) + parameters.name_network +
                                     str(parameters.train_number) + "final_part.pth.tar")

        validation_error_min = validation_error

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("The network as been saved at the epoch " + str(epoch) + " (best score) " +
                          str(index_save_best) + '\n')

        index_save_best = (index_save_best + 1) % 2

    else:
        # Maybe useless to save the network in this way
        # Save the entire model with parameter, network and optimizer
        save_checkpoint({'epoch': epoch + 1,  # +1 because we start to count at 0
                         'parameters': parameters,
                         'state_dict': network.state_dict(),
                         },
                        filename=parameters.path_save_net + "save" + str(index_save_regular) +
                                 parameters.name_network + str(parameters.train_number) + "checkpoint.pth.tar")

        if network_final is not None:
            save_checkpoint({'state_dict': network_final.state_dict(),
                             },
                            filename=parameters.path_save_net + "save" + str(
                                index_save_regular) + parameters.name_network +
                                     str(parameters.train_number) + "final_part.pth.tar")

        validation_error_min = validation_error

        print("The network as been saved at the epoch " + str(epoch) + "(regular save)" + str(index_save_regular))
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("The network as been saved at the epoch " + str(epoch) + "(regular save)" +
                          str(index_save_regular) +  '\n')

        index_save_regular = (index_save_regular + 1) % 2

    return validation_error_min, index_save_best, index_save_regular



def organise_CSV(path_CSV, name_network, train_number, both = True):
    """
    :param path_CSV: Path to the CSV files that will be used
    :param name_network: name of the network associated with the CSV file
    :param train_number: number of the network associated with the CSV file
    :return: organise_CSV import two CSV files and delete all duplicate row. Because the algorithme work with
        mini_batch there is many value for the loss for one epoch and one data set. We compute here the mean
        of all this loss that have the same epoch and data set. We did the same with the confusion matrix.
    """

    # Import the CSV file into pandas DataFrame
    loss_DF = pd.read_csv(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv")
    # This Groupby will regroupe all line that have the same "Set" and "Epoch" and compute the mean over the "Values"
    loss_DF = loss_DF.groupby(['Set', 'Epoch'])['Value'].mean().reset_index()
    # Recreate the CSV file
    loss_DF.to_csv(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv", index=False)
    
    if both:
        # Import the CSV file into pandas DataFrame
        conf_DF = pd.read_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv")
        # This Groupby will regroupe all line that have the same 'Target','Prediction','Epoch','Set'
        # and compute the mean over the "Values"
        conf_DF = conf_DF.groupby(['Target', 'Prediction', 'Epoch', 'Set'])['Value'].mean().reset_index()
        # Recreate the CSV file
        conf_DF.to_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv", index=False)


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
