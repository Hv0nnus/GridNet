# Some standard imports
import io
import numpy as np
import torch.onnx

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

import GridNet_structure
import Grid_Net_copy
import Parameters
import Label

parameters = Parameters.Parameters(nColumns=2,
                                   nFeatMaps=[5,7],
                                   nFeatureMaps_init=3,
                                   number_classes=20 - 1,
                                   label_DF=Label.create_label(),

                                   width_image_initial=2048, height_image_initial=1024,
                                   size_image_crop=353,

                                   dropFactor=0.05,
                                   learning_rate=0.01,
                                   weight_decay=5 * 10 ** (-6),
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1 * 10 ** (-8),
                                   batch_size=4,
                                   batch_size_val=6,
                                   epoch_total=200,
                                   actual_epoch=0,
                                   scale=(0.39, 0.5),
                                   ratio=(1, 1),

                                   path_save_net="./Model/",
                                   name_network="architecture_gpu",
                                   train_number=0,
                                   path_CSV="./CSV/",
                                   path_data="/home_expes/collections/Cityscapes/",
                                   path_print="./Python_print.txt",
                                   path_result="./Result",
                                   num_workers=2)

# Define the GridNet
network = Grid_Net_copy.gridNet(nInputs=parameters.nFeatureMaps_init,
                                nOutputs=parameters.number_classes,
                                nColumns=parameters.nColumns,
                                nFeatMaps=parameters.nFeatMaps,
                                dropFactor=parameters.dropFactor)
print(network)
torch_model = network

from torch.autograd import Variable
batch_size = 2   # just a random number

# Input to the model
x = Variable(torch.randn(batch_size, 3, 5, 5), requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "GridNet.onnx",          # where to save the model (can be a file or file-like object)
                               export_params=True,    # store the trained parameter weights inside the model file
                               verbose=True)
