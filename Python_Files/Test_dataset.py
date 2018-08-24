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


def final_value(i, j, w, h, parameters, y_batch_estimated):
    """
    :param i: position in x of the beginning for the crop
    :param j: position in y of the beginning for the crop
    :param w: width of the crop
    :param h: height of the crop
    :param parameters: List of parameter of the network
    :param y_batch_estimated: Prediction of the network
    :return: the new position and the prediction reduce by pixel_edge on the edge.
    The edge of an image is often badly predicted.
    """
    pixel_edge = 5
    if i >= pixel_edge:
        i += pixel_edge
        w -= pixel_edge
        y_batch_estimated = y_batch_estimated[:, :, pixel_edge:]

    if j >= pixel_edge:
        j += pixel_edge
        h -= pixel_edge
        y_batch_estimated = y_batch_estimated[:, pixel_edge:, :]

    if w + i < (parameters.width_image_initial - pixel_edge):
        w -= pixel_edge
        y_batch_estimated = y_batch_estimated[:, :, :-pixel_edge]

    if h + j < (parameters.height_image_initial - pixel_edge):
        h -= pixel_edge
        y_batch_estimated = y_batch_estimated[:, :-pixel_edge, :]

    return i, j, w, h, y_batch_estimated


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
    with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
        txtfile.write("\n " + str(test_dataset.imgs) + "\n")

    # The three next line are usefull to force one image to be run.
    #test_dataset.imgs[0] = ('/home_expes/collections/Cityscapes/leftImg8bit/val/munster/munster_000076_000019_leftImg8
    # bit.png', '/home_expes/collections/Cityscapes/gtFine/val/munster/munster_000076_000019_gtFine_labelIds.png')
    #test_dataset.image_names[0] = 'munster/munster_000076_000019_leftImg8bit.png'

    # Loop over the images
    for iteration in range(len(test_dataset.imgs)):
        with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
            txtfile.write("\n " + "First loop" + str(iteration) + "\n")
        
        np.random.seed(iteration*20)
        k = iteration

        # Store the time at the begining of each image
        timer_image = time.time()

        # Load the path k
        x_batch_PIL_path, y_batch_PIL_path = test_dataset.imgs[k]

        # Load the image in RGB format
        x_batch_PIL = Image.open(x_batch_PIL_path).convert('RGB')

        # Get the name of the image and change it to have the new name
        image_name = test_dataset.image_names[k]
        # Save the image in png
        x_batch_PIL.save("./Result/" + dataset + "/" + image_name)

        # Load the image in RGB format
        y_batch_PIL_color = Image.open(y_batch_PIL_path.replace("labelIds", "color")).convert('RGB')
        # Save the image in png
        y_batch_PIL_color.save("./Result/" + dataset + "/" + image_name.replace("leftImg8bit", "color"))

        mask = np.array(Image.open(y_batch_PIL_path))
        mask_copy = mask.copy()

        # Create a mask with value 1 and 0 to simply multiply to apply the mask over the image
        for j, v in test_dataset.id_to_trainid.items():

            if v == parameters.number_classes:
                mask_copy[mask == j] = 0

            if v != parameters.number_classes:
                mask_copy[mask == j] = 1

        # Create a numpy array that have the size of the entire image and parameters.number_classes channel
        x_batch_np = np.zeros((parameters.number_classes,
                               x_batch_PIL.size[1],
                               x_batch_PIL.size[0]))

        # loop over all position of the crop
        for i, j, w, h in position_crop:
            with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
                txtfile.write("\n " "Before i" + str(i) + " j "+ str(j) + " w " + str(w) + " h " + str(h) + "\n")
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
            with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
                txtfile.write("\n " "Before test the network" + "\n")
            # Compute the forward function
            y_batch_estimated = network(x_batch)

            # Delete this new dimension after the forward
            y_batch_estimated = y_batch_estimated.squeeze()
            # Convert into numpy array
            y_batch_estimated = y_batch_estimated.cpu().data.numpy()

            # Delete the border if possible.
            i_used, j_used, w_used, h_used, y_batch_estimated = final_value(i=i, j=j, w=w, h=h,
                                                                            parameters=parameters,
                                                                            y_batch_estimated=y_batch_estimated)

            # Add every value to the image. i and j or exchange compare to PIL image
            x_batch_np[:, j_used:j_used + h_used, i_used:i_used + w_used] = x_batch_np[:,j_used:j_used + h_used, i_used:i_used + w_used] +  y_batch_estimated

        with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
            txtfile.write("\n " "After the loop" + str(iteration))

        # Keep only the highest probability, that will be the predicted class
        x_batch_np = x_batch_np.argmax(axis=0)

        # Add one to each value to have the right format (from 1 to 19)
        x_batch_np = x_batch_np + 1

        # Apply mask over the data to put 0 to the area where we don't care about the prediction
        x_batch_np = x_batch_np * mask_copy

        # Transform numpy array to PIL
        x_batch_np = Image.fromarray(x_batch_np.astype('uint8'))

        # Get the name of the image and change it to have the new name
        image_name = test_dataset.image_names[k]
        end_name = 'prediction.png'
        image_name = image_name.replace("leftImg8bit.png", end_name)

        # Save the image in png
        x_batch_np.save("./Result/" + dataset + "/" + image_name)

        with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
            txtfile.write("Time for one image : " + str(time.time() - timer_image) + "\n")

        # Not a very beautiful way of stopping the program... #TODO this could be change
        if iteration == 10:
            return end_name

    with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
        txtfile.write("Finish. Total time : " + str((time.time() - timer_init)) + "\n" + str(end_name))

    return end_name


def main_test_dataset(parameters, network, position_crop, dataset='val'):
    """
    :param position_crop: List of all position that the image will be crop. format : [(i , j, w, h),...]
    Which are the first pixel of width and height and the width and height of the crop
    :param dataset: Type of Dataset (train, val, test)
    :param parameters: list of parameters of the network
    :param network: network that will be learned
    :return: Nothing but save the predicted images of the chosen dataset.
    """

    # transforms_test
    parameters.transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Import both dataset with the transformation
    test_dataset = Save_import.cityscapes_create_dataset(quality='fine',
                                                         mode=dataset,
                                                         parameters=parameters,
                                                         transform=parameters.transforms_test,
                                                         only_image=False)

    with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'w') as txtfile:
        txtfile.write(str(test_dataset.imgs[0]) + "\n" + str(test_dataset.imgs[1]))

    test_loop(network=network,
              parameters=parameters,
              test_dataset=test_dataset,
              dataset=dataset,
              position_crop=position_crop)
