# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F
import torch


def criterion(y_estimated, y, parameters):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :return: difference between y_estimated and y, according to some function (most of the time, NLLLoss)
    """
    if False:
        #http://pytorch.org/docs/master/nn.html : torch.nn.NLLLoss
        nllcrit = nn.NLLLoss2d(reduce=True)

        # Apply softmax then the log on the result
        y_estimated = F.log_softmax(input=y_estimated, dim=1)

        # (y!= parameters.number_classes) is a matrix with 0 on position were the target is class parameters.number_classes
        # unsqueeze ad a dimension to allowed multiplication and float transform the Variable to a float Variable
        y_estimated = y_estimated * (y != parameters.number_classes).unsqueeze(1).float()

        # Set all target value of number_classes to 0 (we could have choose another class.
        # The nllcrit will do y_estimated[k,0,i,j]*y[k,i,j]
        # It will be 0 if the class is parameters.number_classes : which is exactly what is expect for this class
        # The other classes remain unchanged
        y = y * (y != parameters.number_classes).long()

        # Apply the criterion define in the first line
        return nllcrit(y_estimated, y)

    if True:
        # Apply softmax then the log on the result
        y_estimated = F.softmax(input=y_estimated, dim=1)

        IoU = 0

        for k in range(parameters.number_classes):
            y_only_k = (y == k).float()

            intersection = torch.sum(y_estimated[:, k, :, :] * y_only_k)
            union = torch.sum(y_only_k + y_estimated[:, k, :, :] - y_estimated[:, k, :, :] * y_only_k)

            IoU += intersection / union

        return 1 - (IoU / parameters.number_classes)


def criterion_pd_format(y_estimated, y, epoch, set_type, parameters):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param epoch: The actual epoch of the learning
    :param set_type: String which is train of validation
    :param parameters: List of parameters of the network
    :return: Define the loss function between the y_estimated and y and return a vector with the epoch the value
    and if the criterion is used on training or validation set.
    """

    loss = criterion(y_estimated, y, parameters)

    # Return a vector  usefull to copy to CSV 
    return [loss.data[0], set_type, epoch]


def IoU_pd_format(y_estimated, y, set_type, epoch, parameters):
    """
    :param y_estimated: Result of train(x) which is the forward action
    :param y: Label associated with x
    :param set_type: String which is train of validation
    :param epoch: The actual epoch of the learning
    :param parameters: List of parameters of the network
    :return: Return a matrix of confusion in a good format to creat a pandas DataFrame
    """
    # We keep only the higest value, which is the prediction
    pred = torch.max(y_estimated, dim=1)[1]

    # Creat the confusion matrix, only the second column will have the value
    confusion_matrix = [[0] * 5 for i in range(parameters.number_classes ** 2)]

    pred = pred.view(-1)
    target = y.view(-1)

    # We will normalise the value in the matrix. We don t want to be influence by the size of the batch
    normalisation_value = y.size()

    # It is also possible to add the size of the image to normalise
    normalisation_value = normalisation_value[0]  # * normalisation_value[1] * normalisation_value[2]

    # Double loop over the number of classes at each iteration we add the intersection
    # i will be the number of iteration
    i = 0
    for cls1 in range(parameters.number_classes):

        pred_inds = pred == cls1

        for cls2 in range(parameters.number_classes):
            # i increase by one at each iteration
            i = cls1 * parameters.number_classes + cls2

            target_inds = target == cls2

            # Intersection is the value predicted of class cls2 and are in reality class cls1
            intersection = (pred_inds * target_inds).long().sum().data[0]

            # Traning or validation set
            confusion_matrix[i][0] = set_type
            # The value normalised
            confusion_matrix[i][1] = intersection / normalisation_value
            # Associated with this value we keep the two classes
            confusion_matrix[i][2] = "class" + str(cls1)
            confusion_matrix[i][3] = "class" + str(cls2)
            # The epoch associated
            confusion_matrix[i][4] = epoch

    return confusion_matrix
