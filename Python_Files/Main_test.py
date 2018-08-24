# coding: utf-8

# Torch related package
from __future__ import print_function

# cuda related package

# Other package
import sys
import random

# Other python files
import Save_import
import Test_dataset
import Manage_image


def main_test(path_learning, dataset="train"):
    """
    :param path_learning:
    :param dataset:
    :return: Nothing but save the predicted images of the chosen dataset.
    """
    # Load the trained Network
    parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_learning)

    w = 801
    h = 801

    # We cannot put the entire image in one batch.
    # position_crop is a list with the position of each crop that will be made
    position_crop = []
    for k in range(2*(int(parameters.width_image_initial/w)) + 1):
        for l in range(2*(int(parameters.height_image_initial/h)) + 1):
            # i and w are associated with the x axis
            i = k * w//2
            j = l * h//2
            if i+w > parameters.width_image_initial:
                i = parameters.width_image_initial - w
            if j+h > parameters.height_image_initial:
                j = parameters.height_image_initial - h
            position_crop.append((i, j, w, h))

    for r in range(20):
        i = random.randint(0, parameters.width_image_initial - w)
        j = random.randint(0, parameters.height_image_initial - h)
        position_crop.append((i, j, w, h))

    # Compute and save the predictions and the original images
    end_name = Test_dataset.main_test_dataset(parameters=parameters,
                                              network=network,
                                              position_crop=position_crop,
                                              dataset=dataset)

    # Take the image that have been generate and make a copy with a color to make it easier to understand
    Manage_image.transform_image_to_RGB(path_data="./Result/",
                                        mode=dataset,
                                        from_picture=0,
                                        to_picture=11,
                                        end_name="prediction.png")

if len(sys.argv) == 3:
    main_test(path_learning=sys.argv[1], dataset=sys.argv[2])
elif len(sys.argv) == 2:
    main_test(path_learning=sys.argv[1])
else:
    raise ValueError('No path define to load the network or too many argument')
