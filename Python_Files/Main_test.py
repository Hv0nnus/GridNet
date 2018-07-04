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
    # Load the trained Network
    parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_learning)

    i = 0
    j = 0
    w = 2017
    h = 993
    w = 801
    h = 801
    position_crop = []
    for k in range(5):
        for l in range(2):
            # i est l axe des x
            i = k * w//2
            j = l * h//2
            if i+w > parameters.width_image_initial:
                i = parameters.width_image_initial - w
            if j+h > parameters.height_image_initial:
                j = parameters.height_image_initial - h
            position_crop.append((i, j, w, h))

    for r in range(10):
        i = random.randint(0, parameters.width_image_initial - w)
        j = random.randint(0, parameters.height_image_initial - h)
        position_crop.append((i, j, w, h))

    end_name = Test_dataset.main_test_dataset(parameters=parameters,
                                              network=network,
                                              position_crop=position_crop,
                                              path_learning=path_learning,
                                              dataset=dataset)

    Manage_image.transform_image_to_RGB(path_data="./Result/",
                                        mode=dataset,
                                        from_picture=0,
                                        to_picture=11,
                                        end_name=end_name)

    # end_name="labelTrainIds.png")

if len(sys.argv) == 3:
    main_test(path_learning=sys.argv[1], dataset=sys.argv[2])
if len(sys.argv) == 2:
    main_test(path_learning=sys.argv[1])
else:
    raise ValueError('No path define to load the network or too many argument')
