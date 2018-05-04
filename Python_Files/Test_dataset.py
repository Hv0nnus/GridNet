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
from PIL import Image
import numpy as np

# Other python files
import Save_import


def test_loop(parameters, network, dataset, test_dataset, position_crop):
    """
    :param position_crop: List of all position that the image will be crop. format : [(i , j, w, h),...]
    Which are the first pixel of width and height and the width and height of the crop
    :param test_dataset: test_dataset is a class that will be used to load alls the paths of the images
    :param dataset: Type of Dataset (train, val, test)
    :param parameters: list of parameters of the network
    :param network: network that will be learned
    :return: Nothing but save every image that have been tested
    """

    # Store the time at the begining of the test
    timer_init = time.time()

    for k in range(len(test_dataset.imgs)):

        # Store the time at the begining of each image
        timer_image = time.time()

        # Load the path k
        x_batch_PIL_path, _ = test_dataset.imgs[k]

        # Load the image in RGB format
        x_batch_PIL = Image.open(x_batch_PIL_path).convert('RGB')

        # Create a numpy array that have the size of the entire image and parameters.number_classes channel
        x_batch_np = np.zeros((parameters.number_classes,
                               x_batch_PIL.size[1],
                               x_batch_PIL.size[0]))

        for i, j, w, h in position_crop:

            assert (i+w) <= parameters.width_image_initial
            assert (j+h) <= parameters.height_image_initial

            # Crop the PIL image
            x_batch = x_batch_PIL.crop((i, j, i + w, j + h))

            # Transform it into Tensor
            x_batch = parameters.transforms_test(x_batch)

            # Transform into Variable
            if torch.cuda.is_available():
                x_batch = Variable(x_batch.cuda(), requires_grad=False)
            else:
                x_batch = Variable(x_batch, requires_grad=False)

            # Add a dimension because we only have a batch size of 1
            x_batch = x_batch.unsqueeze(dim=0)

            # Compute the forward function
            y_batch_estimated = network(x_batch)

            # Delete this new dimension after the forward
            y_batch_estimated = y_batch_estimated.squeeze()
            # Convert into numpy array
            y_batch_estimated = y_batch_estimated.cpu().data.numpy()

            # Add every value to the image. i and j or exchange compare to PIL image
            x_batch_np[:, j:j + h, i:i + w] += y_batch_estimated

        # Keep only the highest probability, that will be the predicted class
        x_batch_np = x_batch_np.argmax(axis=0)

        # Add one to each value to have the right format (from 1 to 19)
        x_batch_np = x_batch_np + 1

        # Transform numpy array to PIL
        x_batch_np = Image.fromarray(x_batch_np.astype('uint8'))

        # Get the name of the image and change it to have the new name
        image_name = test_dataset.image_names[k]
        end_name = 'prediction.png'
        image_name = image_name.replace("leftImg8bit.png", end_name)

        print("./Result/" + dataset + "/" + image_name)
        # Save the image in png
        x_batch_np.save("./Result/" + dataset + "/" + image_name)

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("Time for one image : " + str(time.time() - timer_image) + "\n")

        if k == 9:
            break

    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n")

    return end_name


def main_test_dataset(parameters, network, position_crop, path_learning=None, dataset='val'):

    #parameters.path_data = "../Cityscapes_Copy/"

    # transforms_test
    parameters.transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Import both dataset with the transformation
    test_dataset = Save_import.cityscapes_create_dataset(quality='fine',
                                                         mode=dataset,
                                                         parameters=parameters,
                                                         transform=parameters.transforms_test,
                                                         only_image=True)

    test_loop(network=network,
              parameters=parameters,
              test_dataset=test_dataset,
              dataset=dataset,
              position_crop=position_crop)

