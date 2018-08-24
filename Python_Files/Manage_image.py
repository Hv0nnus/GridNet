# coding: utf-8

# Torch related package
from __future__ import print_function

# Other package
from PIL import Image
import numpy as np
import os

# Other python files
import Label


class Picture:
    """
    Picture is associated with one image (19 or 20 classes)
    """

    def __init__(self,
                 path_image="./"):
        """
        Initialize the image black and white image (19 or 20 classes)
        :param path_image: path to the image
        """
        # Path of the image
        self.path_image = path_image
        # Import the image and transform it into array
        self.image_array = np.array(Image.open(self.path_image))
        # Height and width of the image
        self.height_image_array, self.width_image_array = self.image_array.shape

    def classes_to_color(self, id_to_label):
        """
        :param id_to_label: name of the dictionary that will transform 1D image into RGB
        :return: the RGB image
        """
        # Initialize the RGb image
        RGB = np.zeros((self.height_image_array, self.width_image_array, 3))

        # For each pixel add the write color according to some rules given by id_to_label
        for i, row in enumerate(self.image_array):
            for j, pixel in enumerate(row):
                RGB[i, j, :] = id_to_label[pixel].color

        return RGB


def make_dataset(mode, path_data, end_name):
    """
    :param mode: train val or test
    :param path_data: path to the folder that contain all the data
    :param end_name: end of the name of image that you want to keep
    :return: The entire DataSet with the right name (according to end_name) which is at the location path_data and mode
    The result is an array with all the files. The names of each pictures is also store.
    """
    # Creat the complete path
    img_path = os.path.join(path_data, mode)
    # Item will contains the paths
    items = []
    # image_names will contains all the image name associated with the paths
    image_names = []
    # list of all the directory in the folder (it will be the name of german city)
    categories = os.listdir(img_path)
    # We look at every City
    for c in categories:
        # Get the name of all file in the directory
        for name in os.listdir(os.path.join(img_path, c)):
            # Keep only the files names that we wanted
            if name.endswith(end_name):
                items.append(os.path.join(img_path, c, name))
                image_names.append(os.path.join(c, name))

    return items, image_names



def transform_image_to_RGB(path_data,
                           from_picture=0, to_picture=2,
                           mode="test",
                           end_name='leftImg8bit.png'):
    """
    :param path_data: path to the folder that contain all the data
    :param from_picture: first picture that will be transform
    :param to_picture: last picture that will be transform
    :param mode: train val or test
    :param end_name: end of the name of image that you want to keep
    :return: Nothing but save to_picture-from_picture images into RGB format that will be in the path_data/mode
    location and have the exact end_name
    """
    # Make the DataSet and the names
    paths, names = make_dataset(mode=mode,
                                path_data=path_data,
                                end_name=end_name)

    # sort the path and name so that it is always organized in the same way
    paths.sort()
    names.sort()

    # Reduce the number of image
    paths = paths#[from_picture:to_picture]
    names = names#[from_picture:to_picture]

    # Create the dictionary that will transform 1D to RBG image
    labels = Label.create_label_plot()
    train_id2label = Label.train_id2label(labels)

    # Loop over each picture
    for i in range(len(paths)):
        # Create the Picture type
        picture = Picture(path_image=paths[i])

        # Transform into RGB
        RGB = picture.classes_to_color(id_to_label=train_id2label)

        # Change the name
        path_export_color = paths[i].replace(end_name, 'prediction_color.png')

        # transform into PIL image
        RGB = np.uint8(RGB)
        RGB = Image.fromarray(RGB)

        # Save the file
        RGB.save(path_export_color)
        with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print_test.txt", 'a') as txtfile:
            txtfile.write("image saved " + path_export_color + "\n")
