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

    mask = (y != parameters.number_classes).unsqueeze(1).float()

    if parameters.loss == "cross_entropy":
        # http://pytorch.org/docs/master/nn.html : torch.nn.NLLLoss
        nllcrit = nn.NLLLoss2d(weight=parameters.weight_grad, reduce=True)

        # Apply softmax then the log on the result
        y_estimated = F.log_softmax(input=y_estimated, dim=1)

        # (y!= parameters.number_classes) is a matrix with 0 on position were the target is class
        # parameters.number_classes
        # unsqueeze add a dimension to allowed multiplication and float transform the Variable to a float Variable
        y_estimated = y_estimated * mask

        # Set all target value of number_classes to 0 (we could have choose another class.
        # The nllcrit will do y_estimated[k,0,i,j]*y[k,i,j]
        # It will be 0 if the class is parameters.number_classes : which is exactly what is expect for this class
        # The other classes remain unchanged
        y = y * (y != parameters.number_classes).long()

        # Apply the criterion define in the first line
        return nllcrit(y_estimated, y)

    if parameters.loss == "IoU":

        # Apply softmax then the log on the result
        y_estimated = F.softmax(input=y_estimated, dim=1)

        IoU = 0

        # Compute the IoU per classes
        for k in range(parameters.number_classes):
            # Keep only the classes k.
            y_only_k = (y == k).float()

            # Definition of intersection and union
            intersection = torch.sum(y_estimated[:, k, :, :] * y_only_k)
            union = torch.sum(y_only_k + y_estimated[:, k, :, :] - y_estimated[:, k, :, :] * y_only_k)

            IoU += intersection / union

        # Divide by the number of class to have IoU between 0 and 1. we add "1 -" to have a loss to minimize and
        # to stay between 0 and 1.
        return 1 - (IoU / parameters.number_classes)

    if parameters.loss == "hinge":
        hinge_loss = torch.nn.MultiLabelMarginLoss(size_average=True)

        # Apply the mask to avoid any back propagation on the value of the class number_classes
        y_estimated = y_estimated * mask

        # Their is no back prop, but the value of the error is still influenced
        # we force the network to predict the class 0 for the point that are class number_classes.
        y_estimated[:, 0, :, :] = y_estimated[:, 0, :, :] + (y == parameters.number_classes).float()
        # Set all target value of number_classes to 0, the hinge loss will be max(0, 1 - (1 - 0))
        # Because the network predict the class 0 to 1 and the other to 0
        y = y * (y != parameters.number_classes).long()

        y_estimated = y_estimated.transpose(0, 2, 3, 1)
        y_estimated_reshape = y_estimated.view(-1, parameters.number_classes)
        y = y.transpose(0, 2, 3)
        y_reshape = y.view(-1)

        return hinge_loss(input=y_estimated_reshape,
                          target=y_reshape)


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
    return [set_type, epoch, loss.data[0]]


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



            # Associated with this value we keep the two classes
            confusion_matrix[i][0] = "class" + str(cls2)
            confusion_matrix[i][1] = "class" + str(cls1)
            # The epoch associated
            confusion_matrix[i][2] = epoch
            # Traning or validation set
            confusion_matrix[i][3] = set_type
            # The value normalised
            confusion_matrix[i][4] = intersection / normalisation_value

    return confusion_matrix
