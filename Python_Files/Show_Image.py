# coding: utf-8

# Torch related package
from __future__ import print_function
import torch
from torchvision import transforms
from torch.autograd import Variable

# Other package
import time
import sys

# Other python files

import GridNet_structure
import Save_import
import Loss_Error
import Parameters
import Label

class Image():

    def init(self,
             # Size of initial image
             width_image_initial = 2048, height_image_initial = 1024,
             path_image = "./"):

        self.width_image_initial = width_image_initial
        self.height_image_initial = height_image_initial
        self.path_image = path_image


    def import_one_image(self):
        return()

    def classes_to_color(self):
        return()






