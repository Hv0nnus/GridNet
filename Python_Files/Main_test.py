# coding: utf-8

# Torch related package
from __future__ import print_function

# cuda related package

# Other package
import sys

# Other python files
import Save_import
import Test_dataset
import Manage_image


def main_test(path_learning, dataset="val"):
    # Load the trained Network
    parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_learning)

    i = 0
    j = 0
    w = 2017
    h = 993
    position_crop = []
    for k in range(1):
        for l in range(1):
            # i est l axe des x
            i = k * w//2
            j = l * h//2
            if i+w > 2048:
                i = 2048 - w
            if j+h > 1024:
                j = 1024 - h
            position_crop.append((i, j, w, h))

    end_name = Test_dataset.main_test_dataset(parameters=parameters,
                                              network=network,
                                              position_crop=position_crop,
                                              path_learning=path_learning,
                                              dataset=dataset)

    Manage_image.transform_image_to_RGB(path_data="./Result/",
                                        mode=dataset,
                                        from_picture=0,
                                        to_picture=10,
                                        end_name="prediction.png")

    # end_name="labelTrainIds.png")

if len(sys.argv) == 3:
    main_test(path_learning=sys.argv[1], dataset=sys.argv[2])
if len(sys.argv) == 2:
    main_test(path_learning=sys.argv[1])
else:
    raise ValueError('No path define to load the network or too many argument')
