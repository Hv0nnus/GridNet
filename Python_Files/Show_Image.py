# coding: utf-8

# Torch related package
from __future__ import print_function

# Other package
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

# Other python files
import Label


class Picture:
    """
    Picture as two image that correspond to
    """
    def __init__(self,
                 path_image_predicted="./",
                 path_image_target="./",
                 image_name=None):
        self.path_image_predicted = path_image_predicted
        self.path_image_target = path_image_target
        self.image_name = image_name
        if os.path.exists(self.path_image_predicted):
            self.image_predicted = np.array(Image.open(self.path_image_predicted))
            self.height_image_predicted, self.width_image_predicted = self.image_predicted.shape
        if os.path.exists(self.path_image_target):
            self.image_target = np.array(Image.open(self.path_image_target))
            self.height_image_target, self.width_image_target = self.image_target.shape

    def classes_to_color(self, id_to_label):
        # Change the color
        RGB_prediction = np.zeros((self.height_image_predicted, self.width_image_predicted, 3))
        for i, row in enumerate(self.image_predicted):
            for j, pixel in enumerate(row):
                RGB_prediction[i, j, :] = id_to_label[pixel].color

        if os.path.exists(self.path_image_target):
            RGB_target = np.zeros((self.height_image_target, self.width_image_target,3))
            for i, row in enumerate(self.image_target):
                for j, pixel in enumerate(row):
                    RGB_target[i, j, :] = id_to_label[pixel].color
        else:
            RGB_target = None

        return RGB_prediction, RGB_target


def make_dataset(mode, path_data, is_prediction):
    img_path = os.path.join(path_data, mode)
    items = []
    image_names = []
    categories = os.listdir(img_path)
    for c in categories:
        for name in os.listdir(os.path.join(img_path, c)):
            if is_prediction:
                if name.endswith('_leftImg8bit.png'):
                #if name.endswith('_gtFine_labelTrainIds.png'):
                    items.append(os.path.join(img_path, c, name))
                    image_names.append(os.path.join(c, name))

            else:
                if name.endswith('_gtFine_labelTrainIds.png'):
                    items.append(os.path.join(img_path, c, name))
                    image_names.append(os.path.join(c, name))

    return items, image_names


def show_picture(img, is_RGB=False):
    if type(img) is np.ndarray:
        if is_RGB:
            img = Image.fromarray(np.uint8(img), 'RGB')
        else:
            img = Image.fromarray(np.uint8(img))
    img.show()


def main(path_data_prediction, path_data_target=None, from_picture=0, to_picture=2, mode="test"):
    if path_data_target is not None:
        path_target, name_target = make_dataset(mode=mode, path_data=path_data_target, is_prediction=False)
        path_target = path_target[from_picture:to_picture]
        name_target = name_target[from_picture:to_picture]

    path_prediction, name_prediction = make_dataset(mode=mode, path_data=path_data_prediction, is_prediction=True)
    path_prediction = path_prediction[from_picture:to_picture]
    name_prediction = name_prediction[from_picture:to_picture]

    labels = Label.create_label_plot()
    train_id2label = Label.train_id2label(labels)
    path_target.sort()
    path_prediction.sort()
    name_prediction.sort()
    print(path_target)
    print(path_prediction)
    for i in range(len(path_prediction)):
        if path_data_target is not None:
            picture = Picture(path_image_predicted=path_prediction[i],
                              path_image_target=path_target[i],
                              image_name=name_prediction[i])
        else:
            picture = Picture(path_image_predicted=path_prediction[i],
                              path_image_target="nothing",
                              image_name=name_prediction[i])
        RGB_prediction, RGB_target= picture.classes_to_color(train_id2label)

        Image.fromarray(np.uint8(RGB_prediction)).show("Prediction")
        #import time
        #time.sleep(5)
        if path_data_target is not None:
            a = np.uint8(RGB_target)
            Image.fromarray(a).show("Target")


main(path_data_prediction='./Result/', path_data_target="../Cityscapes_Copy/gtFine/", mode="train", from_picture=1,
     to_picture=2)
