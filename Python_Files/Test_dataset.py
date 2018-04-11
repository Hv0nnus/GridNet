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
from PIL import Image

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

    for i, (x_batch, img_path) in enumerate(test_loader):

        # Store the time at the begining of each batch
        timer_batch = time.time()

        # Transform into Variable
        x_batch = Variable(x_batch)

        # Compute the forward function
        y_batch_estimated = network(x_batch)

        # Keep only the highest probability, that will be the predicted class
        y_batch_estimated = torch.max(y_batch_estimated, dim=1)[1]

        # Convert into numpy array
        y_batch_estimated = y_batch_estimated.data.numpy()

        # Save numpy array
        for index_image in range(y_batch_estimated.shape[0]):
            im = Image.fromarray(y_batch_estimated[index_image,:,:].astype('uint8'))
            im.save("./Result/" + dataset + "/" + img_path[index_image])

        # Change the color
        #size_y_batch = y_batch_estimated.size()
        #numpy_image = np.zeros((3, size_y_batch[1], size_y_batch[2]))
        #for index_image in range(size_y_batch(0)):
        #    numpy_image[:, ] = [index_image]

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("Batch time : " + str(time.time() - timer_batch) + "\n")

    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n")

    return ()


def main_test_dataset(path_learning=None, dataset='val'):

    
    # Load the trained Network
    parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_learning)

    # How many image per loop of test
    parameters.batch_size = 1
    
    # Import both dataset with the transformation
    test_dataset = Save_import.CityScapes_final('fine', dataset,
                                                transform=parameters.transforms_test,
                                                parameters=parameters, only_image=True)

    # Creat the DataSet for pytorch used
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=parameters.batch_size,
                                              shuffle=True,
                                              num_workers=parameters.num_workers,
                                              drop_last=False)

    # Train the network
    test_loop(network=network,
              parameters=parameters,
              test_loader=test_loader,
              dataset=dataset)

#main_test_dataset(path_learning="best0Test0checkpoint.pth.tar", dataset="test")
main_test_dataset(path_learning=sys.argv[1])
