# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math
import time


def IoU_loss(y_estimated, y, parameters, mask=None, global_IoU_modif=False):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :param mask: Image with 1 were there is a class to predict, and 0 if not.
    :return: difference between y_estimated and y, according to the continuous IoU loss
    """
    # Apply softmax the log on the result
    y_estimated = F.softmax(input=y_estimated, dim=1)

    # Apply the mask
    y_estimated = y_estimated * mask

    IoU = 0

    if parameters.momentum_IoU != 0 and global_IoU_modif:

        # Ugly hack that check if we are at the first epoch and first iteration of batch
        if torch.sum(parameters.inter_union.cpu().data != torch.zeros((2, parameters.number_classes))) == 0:
            momentum = 0
        else:
            momentum = parameters.momentum_IoU

        # Compute the IoU per classes
        for k in range(parameters.number_classes):
            # Keep only the classes k.
            y_only_k = (y == k).float()

            # Definition of intersection and union
            inter = momentum * parameters.inter_union[0, k] + \
                    (1 - momentum) * torch.sum(y_estimated[:, k, :, :] * y_only_k)

            union = momentum * parameters.inter_union[1, k] + \
                    (1 - momentum) * torch.sum(
                        y_only_k + y_estimated[:, k, :, :] - y_estimated[:, k, :, :] * y_only_k)

            IoU += parameters.weight_grad[k] * inter / union

            parameters.inter_union[0, k] = inter.data
            parameters.inter_union[1, k] = union.data
    else:
        # Compute the IoU per classes
        for k in range(parameters.number_classes):
            # Keep only the classes k.
            y_only_k = (y == k).float()

            # Definition of intersection and union
            inter = torch.sum(y_estimated[:, k, :, :] * y_only_k)

            union = torch.sum(y_only_k + y_estimated[:, k, :, :] - y_estimated[:, k, :, :] * y_only_k)

            IoU += parameters.weight_grad[k] * inter / union

    # Divide by the number of class to have IoU between 0 and 1. we add "1 -" to have a loss to minimize and
    # to stay between 0 and 1.
    return 1 - (IoU / parameters.number_classes)


def IoU_Lovasz(y_estimated, y, parameters, mask=None, global_IoU_modif=False):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :param mask: Image with 1 were there is a class to predict, and 0 if not.
    :return: difference between y_estimated and y, according to the approximation of IoU : Lovasz function
    """
    # Apply softmax the log on the result
    y_estimated = F.softmax(input=y_estimated, dim=1)

    # Apply the mask
    y_estimated = y_estimated * mask

    IoU_loss = 0

    if parameters.momentum_IoU != 0 and global_IoU_modif:

        # Ugly hack that check if we are at the first epoch and first iteration of batch
        if torch.sum(parameters.inter_union.cpu().data != torch.zeros((2, parameters.number_classes))) == 0:
            momentum = 0
        else:
            momentum = parameters.momentum_IoU

    else:
        # Permute the prediction and the ground truth so that there is only 1 dimension for the groudn truth
        # and parameters.number_class for the estimated
        y_estimated = y_estimated.permute(0, 2, 3, 1).contiguous()
        y_estimated = y_estimated.view(-1, parameters.number_classes)
        y = y.contiguous().view(-1)

        # We will divide by the the number of classes present in the dataset.
        number_class_used = 0
        # Compute the IoU for each class
        for k in range(parameters.number_classes):

            # Keep only the classes k.
            y_only_k = (y == k).float()

            # If this class is not present we don t use it (it avoid division by 0 and speed up the computation)
            if not (torch.sum(y_only_k).data.numpy == 0):
                number_class_used += 1

                # Sort y_estimated so that the highest probability is first
                y_estimated_k, y_order = torch.sort(y_estimated[:, k],
                                                    dim=0,
                                                    descending=True)

                y_only_k = y_only_k[y_order, ]

                # Compute the intersection and union according to the algorithm :
                # The Lovasz-Softmax loss: A tractable surrogate for the optimization of the
                # intersection-over-union measure in neural networks
                inter = torch.cumsum(y_only_k, dim=0)
                union = torch.sum(y_only_k) + torch.cumsum(1 - y_only_k, dim=0)
                grad = - (inter / union)

                # We fixe the IoU of the empty set = 0
                IoU_loss += y_estimated_k[0] * (grad[0] + 0)
                # Again follow the algorithm given in the paper to understand this line.
                IoU_loss += torch.sum(y_estimated_k[1:] * (grad[1:] - grad[:-1]))
            else:
                print("There is no such class : ", k, " in this image")

    # Divide by the number of class used to normalize
    return IoU_loss / number_class_used


def cross_entropy_loss(y_estimated, y, parameters, mask=None, number_of_used_pixel=None):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :param mask: Image with 1 were there is a class to predict, and 0 if not.
    :param number_of_used_pixel: Number of pixel with 1 in the mask. Useful to normalize the loss
    :return: difference between y_estimated and y, according to the cross entropy
    """
    # http://pytorch.org/docs/master/nn.html : torch.nn.NLLLoss
    nllcrit = nn.NLLLoss2d(weight=parameters.weight_grad, size_average=False)

    # Apply softmax then the log on the result
    y_estimated = F.log_softmax(input=y_estimated, dim=1)

    # Apply the mask
    y_estimated = y_estimated * mask

    # Set all target value of number_classes to 0 (we could have choose another class.
    # The nllcrit will do y_estimated[k,0,i,j]*y[k,i,j]
    # It will be 0 if the class is parameters.number_classes : which is exactly what is expect for this class
    # The other classes remain unchanged
    y = y * (y != parameters.number_classes).long()

    # Apply the criterion define in the first line
    return nllcrit(y_estimated, y) / number_of_used_pixel


def hinge_multidimensional_loss(y_estimated, y, parameters, mask=None, number_of_used_pixel=None):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :param mask: Image with 1 were there is a class to predict, and 0 if not.
    :param number_of_used_pixel: Number of pixel with 1 in the mask. Useful to normalize the loss
    :return: difference between y_estimated and y, according to the hinge loss
    """
    hinge_loss = torch.nn.MultiMarginLoss(p=1,
                                          margin=0.5,
                                          size_average=False)

    # Apply softmax then the log on the result
    y_estimated = F.softmax(input=y_estimated, dim=1)

    # Apply the mask to avoid any back propagation on the value of the class number_classes
    y_estimated = y_estimated * mask

    # There is no back prop, but the value of the error is still influenced
    # we force the network to predict the class 0 for the point that are class number_classes.
    y_estimated[:, 0, :, :] = y_estimated[:, 0, :, :] + (y == parameters.number_classes).float()
    # Set all target value of number_classes to 0, the hinge loss will be max(0, 1 - (1 - 0))
    # Because the network predict the class 0 to 1 and the other to 0
    y = y * (y != parameters.number_classes).long()

    y_estimated = y_estimated.permute(0, 2, 3, 1).contiguous()
    y_estimated_reshape = y_estimated.view(-1, parameters.number_classes)
    y_reshape = y.contiguous().view(-1)

    return hinge_loss(input=y_estimated_reshape,
                      target=y_reshape) / number_of_used_pixel


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def criterion_pretrain(y_estimated, y, parameters):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param parameters: List of parameters of the network
    :return: difference between y_estimated and y, according to some function
    """

    y_estimated = F.log_softmax(input=y_estimated, dim=1)

    # Set all target value of number_classes to 0 (we could have choose another class.
    # The nllcrit will do y_estimated[k,0,i,j]*y[k,i,j]
    # It will be 0 if the class is parameters.number_classes : which is exactly what is expect for this class
    # The other classes remain unchanged
    y = y * (y != parameters.number_classes).long()

    # Apply the criterion define in the first line
    return nllcrit(y_estimated, y) / number_of_used_pixel


def criterion(y_estimated, y, parameters, global_IoU_modif=False):
    """
    :param y_estimated: result of train(x) which is the forward action
    :param y: Label associated with x
    :param inter_union: List of the sum of Intersection and Union for each classes
    :param parameters: List of parameters of the network
    :return: difference between y_estimated and y, according to some function
    """
    # (y!= parameters.number_classes) is a matrix with 0 on position were the target is class
    # parameters.number_classes
    # unsqueeze add a dimension to allowed multiplication and float transform the Variable to a float Variable
    mask = (y != parameters.number_classes).unsqueeze(1).float()
    number_of_used_pixel = torch.sum(mask)

    if parameters.loss == "cross_entropy":
        return cross_entropy_loss(y_estimated=y_estimated,
                                  y=y,
                                  parameters=parameters,
                                  mask=mask,
                                  number_of_used_pixel=number_of_used_pixel)

    if parameters.loss == "IoU":
        return IoU_loss(y_estimated=y_estimated,
                        y=y,
                        parameters=parameters,
                        mask=mask,
                        global_IoU_modif=global_IoU_modif)

    if parameters.loss == "hinge":
        return hinge_multidimensional_loss(y_estimated=y_estimated,
                                           y=y,
                                           parameters=parameters,
                                           mask=mask)

    if parameters.loss == "IoU_Lovasz":
        return IoU_Lovasz(y_estimated=y_estimated,
                          y=y,
                          parameters=parameters,
                          mask=mask)

    if parameters.loss == "cross_entropy_to_IoU":
        if parameters.actual_epoch < 470:
            return cross_entropy_loss(y_estimated=y_estimated,
                                      y=y,
                                      parameters=parameters,
                                      mask=mask,
                                      number_of_used_pixel=number_of_used_pixel)
        elif parameters.actual_epoch > 670:
            return IoU_loss(y_estimated=y_estimated,
                            y=y,
                            parameters=parameters,
                            mask=mask)
        else:
            balance_between_loss = sigmoid(
                (parameters.actual_epoch - 570) * (15 / 200))

            return (1 - balance_between_loss) * cross_entropy_loss(y_estimated=y_estimated,
                                                                   y=y,
                                                                   parameters=parameters,
                                                                   mask=mask,
                                                                   number_of_used_pixel=number_of_used_pixel) + \
                   balance_between_loss * IoU_loss(y_estimated=y_estimated,
                                                   y=y,
                                                   parameters=parameters,
                                                   mask=mask)


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

    loss = criterion(y_estimated=y_estimated,
                     y=y,
                     parameters=parameters)

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
