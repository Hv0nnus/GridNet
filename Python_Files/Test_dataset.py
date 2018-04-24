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
import PIL
import numpy as np

# Other python files
import Save_import
import Parameters


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


def test_loop(parameters, network, dataset, test_dataset, position_crop):
    """
    :param position_crop:
    :param test_dataset:
    :param dataset:
    :param parameters: list of parameters of the network
    :param network: network that will be learned
    :return: Nothing but modify the weight of the network and call save_error to store the error.
    """

    # Store the time at the begining of the test
    timer_init = time.time()

    for k in range(len(test_dataset.imgs)):

        # Store the time at the begining of each batch
        timer_batch = time.time()

        x_batch_PIL, _ = test_dataset.imgs[k]

        x_batch_PIL = Image.open(x_batch_PIL).convert('RGB')

        x_batch_np = np.zeros((parameters.number_classes,
                               x_batch_PIL.size[0],
                               x_batch_PIL.size[1]))

        for i, j, h, w in position_crop:

            # Crop the PIL image
            x_batch = crop(x_batch_PIL, i, j, h, w)

            # Transform it into Tensor
            x_batch = parameters.transforms_test(x_batch)

            # Transform into Variable
            if torch.cuda.is_available():
                x_batch = Variable(x_batch.cuda(), requires_grad=False)
            else:
                x_batch = Variable(x_batch, requires_grad=False)

            # Compute the forward function
            print(x_batch.shape)
            y_batch_estimated = network(x_batch)

            # Convert into numpy array
            y_batch_estimated = y_batch_estimated.data.numpy()

            x_batch_np[i:i - h, j:j + w, :] += y_batch_estimated

        # Keep only the highest probability, that will be the predicted class
        x_batch_np = x_batch_np.max(axis=2)

        # Save numpy array
        x_batch_np = Image.fromarray(x_batch_np.astype('uint8'))
        print("./Result/" + dataset + "/" + test_dataset.img[k])
        x_batch_np.save("./Result/" + dataset + "/" + test_dataset.img[k])

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("Batch time : " + str(time.time() - timer_batch) + "\n")

        if k == 1:
            break

    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n")

    return ()


def main_test_dataset(path_learning=None, dataset='val'):
    # Load the trained Network
    parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_learning)

    # How many image per loop of test
    parameters.batch_size = 1
    parameters.path_data = "../Cityscapes_Copy/"

    # transforms_test
    Parameters.transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Import both dataset with the transformation
    test_dataset = Save_import.cityscapes_create_dataset(quality='fine',
                                                         mode=dataset,
                                                         parameters=parameters,
                                                         transform=parameters.transforms_test,
                                                         only_image=True)

    # Train the network
    test_loop(network=network,
              parameters=parameters,
              test_dataset=test_dataset,
              dataset=dataset,
              position_crop=[(30, 45, 20, 10)])


main_test_dataset(path_learning="./Model/best1test_cpu0checkpoint.pth.tar", dataset="val")
if len(sys.argv) == 3:
    main_test_dataset(path_learning=sys.argv[1], dataset=sys.argv[2])
if len(sys.argv) == 2:
    main_test_dataset(path_learning=sys.argv[1])
else:
    raise ValueError('No path define to load the network')
