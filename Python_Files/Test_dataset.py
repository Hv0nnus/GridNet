# coding: utf-8

# Torch related package
from __future__ import print_function
import torch
from torchvision import transforms
from torch.autograd import Variable

# cuda related package
import torch.cuda

# Other package
import time
import sys
import numpy as np

# Other python files

import GridNet_structure
import Save_import
import Loss_Error
import Parameters
import Label


def test_loop(parameters, network, test_loader, dataset):
    """
    :param parameters: list of parameters of the network
    :param network: network that will be learned
    :param test_loader: Dataloader which contains input of the test dataset
    :return: Nothing but modify the weight of the network and call save_error to store the error.
    """

    # Store the time at the begining of the test
    timer_init = time.time()

    for i, (x_batch,img_path) in enumerate(test_loader):

        # Store the time at the begining of each batch
        timer_batch = time.time()

        # Transform into Variable
        x_batch = Variable(x_batch)

        # Compute the forward function
        y_batch_estimated = network(x_batch)

        # Keep only the highest probability, that will be the predicted class
        y_batch_estimated = torch.max(y_batch_estimated, dim=1)[1]

        # Convert into numpy array
        y_batch_estimated = y_batch_estimated.numpy()

        # Save numpy array
        for index_image in range(y_batch_estimated.size(0)):
            np.savetxt("Result/" + dataset + img_path[index_image], y_batch_estimated, fmt='%d')

        # Change the color
        size_y_batch = y_batch_estimated.size()
        numpy_image = np.zeros((3,size_y_batch[1],size_y_batch[2]))
        for index_image in range(size_y_batch(0)):
            numpy_image[:,] = [index_image]

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("Batch time : " + str(time.time() - timer_batch) + "\n")

    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n")

    return ()


parameters = Parameters.Parameters(nColumns=2,
                                    nFeatMaps=[3, 6],
                                    nFeatureMaps_init=3,
                                    number_classes=20 - 1,
                                    label_DF=Label.create_label(),

                                    width_image_initial=2048, height_image_initial=1024,

                                    dropFactor=0.1,
                                    learning_rate=0.01,
                                    weight_decay=5 * 10 ** (-6),
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1 * 10 ** (-8),

                                    batch_size=5,
                                    batch_size_val=5,
                                    epoch_total=4,
                                    actual_epoch=0,

                                    #path_save_net="/home_expes/kt82128h/GridNet/Python_Files/Model/",
                                    name_network="test",
                                    train_number=0,
                                    #path_CSV="/home_expes/kt82128h/GridNet/Python_Files/CSV/",
                                    #path_data="/home_expes/collections/Cityscapes/",
                                    num_workers=0)


def main_test_dataset(path_learning=None, parameters=parameters, dataset='test' ):

    # Import both dataset with the transformation
    test_dataset = Save_import.CityScapes_final('fine', dataset,
                                                transform=parameters.transforms_test,
                                                parameters=parameters, only_image = True)

    # Creat the DataSet for pytorch used
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=parameters.batch_size,
                                              shuffle=True,
                                              num_workers=parameters.num_workers,
                                              drop_last=False)

    # Define the GridNet
    network = GridNet_structure.gridNet(nInputs=parameters.nFeatureMaps_init,
                                        nOutputs=parameters.number_classes,
                                        nColumns=parameters.nColumns,
                                        nFeatMaps=parameters.nFeatMaps,
                                        dropFactor=parameters.dropFactor)

    # Load the trained Network
    parameters = Save_import.load_from_checkpoint(
        path_checkpoint=parameters.path_save_net + path_learning,
        network=network, path_print=parameters.path_print)

    # How many image per loop of test
    parameters.batch_size = 10

    # Train the network
    test_loop(network=network, parameters=parameters, test_loader=test_loader, dataset = dataset)


main_test_dataset(path_learning="best0Test0checkpoint.pth.tar", parameters=parameters)
#main_test_dataset(path_learning=sys.argv[1], parameters=parameters)

