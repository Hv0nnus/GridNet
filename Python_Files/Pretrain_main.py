# coding: utf-8

# Package

# Torch related package
from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as datasets

# cuda related package
import torch.cuda
# import torch.backends.cudnn as cudnn

# Other package
import time
import sys
import csv
import os

# Other python program
import GridNet_structure
import Save_import
import Loss_Error
import Parameters
import Label


def batch_loop(optimizer, train_loader, network, epoch, parameters, timer_batch, timer_epoch):
    """
    :param optimizer: The optimiser that containt parameter of Adam optimizer
    :param train_loader: Dataloader which contains input and target of the train dataset
    :param network: Network that will be learned
    :param epoch: Actual epoch of the program
    :param parameters: List of parameters of the network
    :param timer_batch: The time since the beginning of the batch
    :param timer_epoch: The time since the beginning of the epoch
    :return: Nothing but update the network and save the train error
    """

    # Loop over the mini-batch, the size of the mini match is define in the train_loader
    for i, (x_batch, y_batch) in enumerate(train_loader):

        # zero the gradient buffers
        optimizer.zero_grad()

        # Transform into Variable
        if torch.cuda.is_available():
            x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda())
        else:
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)

        # Compute the forward function
        y_batch_estimated = network(x_batch)

        #count = 0
        #for child in network.children():
            #if count == 0:
                #for param in child.parameters():
                    #with open(parameters.path_print, 'a') as txtfile:
                        #txtfile.write("param of linear1"+str(param)+"\n")
                    #break
            #count += 1

        # Get the error
        loss = Loss_Error.criterion_pretrain(y_estimated=y_batch_estimated,
                                             y=y_batch,
                                             parameters=parameters)

        # Compute the backward function
        loss.backward()

        # Does the update according to the optimizer define above
        optimizer.step()

        # Update the optimizer
        # optimizer.param_groups[0]['lr'] = parameters.learning_rate/(1 + (epoch-390) * parameters.learning_rate_decay)

        # Save error of the training DataSet
        loss = ["train", epoch, loss.data[0]]

        # Save the loss
        with open(parameters.path_CSV + "CSV_loss_" + parameters.name_network +
                          str(parameters.train_number) + ".csv", 'a') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows([loss])

        # Similar to a "print" but in a textfile
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write(
                "\nEpoch : " + str(epoch) + ". Batch : " + str(i) +
                ".\nTrain_Error : " + str(loss[2]) +
                "\n" + "Time batch : " + Save_import.time_to_string(time.time() - timer_batch) +
                ".\nTime total batch : " + Save_import.time_to_string(time.time() - timer_epoch) + "\n \n")

        timer_batch = time.time()

    return ()


def validation_loop(val_loader, network, epoch, parameters, timer_epoch):
    """
    validation_loop do a loop over the validation set
    :param val_loader: Dataloader which contains input and target of the validation dataset
    :param network: Network that will be learned
    :param epoch: Actual epoch of the program
    :param parameters: List of parameters of the network
    :param timer_epoch: The time since the beginning of the epoch
    :return: The mean validation_error over the entire validation set. This function also save this error.
    """

    # Save the error of the validation DataSet
    for i, (x_val_batch, y_val_batch) in enumerate(val_loader):

        if torch.cuda.is_available():
            x_val_batch, y_val_batch = Variable(x_val_batch.cuda()), Variable(y_val_batch.cuda())
        else:
            x_val_batch, y_val_batch = Variable(x_val_batch), Variable(y_val_batch)

        loss = Loss_Error.criterion_pretrain(y_estimated=network(x_val_batch),
                                             y=y_val_batch,
                                             parameters=parameters)

        loss = ["validation", epoch, loss.data[0]]

        # Save the loss
        with open(parameters.path_CSV + "CSV_loss_" + parameters.name_network +
                          str(parameters.train_number) + ".csv", 'a') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows([loss])

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write(
                "\nEpoch : " + str(epoch) + ". Batch : " + str(i) + ".\nValidation error : " + str(
                    loss[2]) +
                ".\nTime total batch : " + Save_import.time_to_string(time.time() - timer_epoch) + "\n \n")


def train(parameters, network, train_loader, val_loader):
    """
    :param parameters: List of parameters of the network
    :param network: Network that will be learned
    :param train_loader: Dataloader which contains input and target of the train dataset
    :param val_loader: Dataloader which contains input and target of the validation dataset
    :return: Nothing but modify the weight of the network and call save_error to store the error.
    """

    # Store the time at the beginning of the training
    timer_init = time.time()

    # create your optimizer
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,network.parameters()),
                           lr=parameters.learning_rate,
                           betas=(parameters.beta1, parameters.beta2),
                           eps=parameters.epsilon,
                           weight_decay=parameters.weight_decay)

    # Store the index of the next checkpoint. This value is 0 or 1. We always keep one checkpoint untouched
    # while the other one is changed.
    index_save_regular = 0

    # Loop from the actual epoch (not 0 if we already train) to the last epoch
    initial_epoch = parameters.actual_epoch
    for epoch in range(initial_epoch, parameters.epoch_total):
        # Store the time at the begining of each epoch
        timer_epoch = time.time()
        timer_batch = time.time()

        batch_loop(optimizer=optimizer,
                   train_loader=train_loader,
                   network=network,
                   epoch=epoch,
                   parameters=parameters,
                   timer_batch=timer_batch,
                   timer_epoch=timer_epoch)

        validation_loop(val_loader=val_loader,
                        network=network,
                        epoch=epoch,
                        parameters=parameters,
                        timer_epoch=timer_epoch)

        # checkpoint will save the network if needed
        validation_error_min, index_save_regular = \
            Save_import.checkpoint(validation_error=10,  # useless value just to save with regular at each time
                                   validation_error_min=0,  # useless value just to save with regular at each time
                                   index_save_regular=index_save_regular,
                                   epoch=epoch,
                                   network=network,
                                   parameters=parameters,
                                   optimizer=optimizer)

        # Similar to a "print" but in a text file
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\n              End of Epoch :" + str(epoch) + "/" + str(parameters.epoch_total - 1) +
                          ".\nTime Epoch :" + Save_import.time_to_string(time.time() - timer_epoch) +
                          ".\nTime total : " + Save_import.time_to_string(time.time() - timer_init) +
                          ".\n \n")
        if (epoch % 10) == 0:
            Save_import.organise_CSV(path_CSV=parameters.path_CSV,
                                     name_network=parameters.name_network,
                                     train_number=parameters.train_number,
                                     both=False)

        # Increase the actual epoch
        parameters.actual_epoch += 1

    # Similar to a "print" but in a text file
    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + Save_import.time_to_string(time.time() - timer_init) +
                      "\n")
    return ()


def main(path_continue_learning=None, total_epoch=0, new_name=None):
    """
    :param path_continue_learning: Path were the network is already saved
                                   (don t use if it is the beginning of the training)
    :param total_epoch: Number of epoch needed don t use if it is the beginning of the training)
    :param new_name: New name of the network, if we want to use again a network already train.
    :return: Nothing but train the network and save CSV files for the error and also save the network regularly
    """
    # Manual seed of the network to have reproducible experiment
    torch.manual_seed(945682461)

    # If the network was already train we import it
    if path_continue_learning is not None:
        # Load the trained Network
        parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_continue_learning)

        # Here we can change some parameters, the only one necessary is the total_epoch
        parameters.epoch_total = total_epoch
        # parameters.learning_rate_decay = 0.5 * (10 ** (-2))
        # parameters.batch_size = 5
        # parameters.batch_size_val = 5
        parameters.learning_rate = 0.001
        # parameters.momentum_IoU = 0.9

        # If a new name is define, we create new CSV files associated and change the name of the network
        if new_name is not None:
            # Init the csv file that will store the error, this time we make a copy of the existing error
            Save_import.duplicated_csv(path_CSV=parameters.path_CSV,
                                       old_name_network=parameters.name_network,
                                       new_name_network=new_name,
                                       train_number=parameters.train_number)
            parameters.name_network = new_name

        with open(parameters.path_print, 'w') as txtfile:
            txtfile.write('\n               The program will continue \n')

    # If the network was not train, we start from scratch
    else:
        # Define all the parameters
        parameters = Parameters.Parameters(nColumns=6,
                                           nFeatMaps=[16, 32, 64, 128],
                                           nFeatureMaps_init=3,
                                           number_classes=20 - 1,
                                           label_DF=Label.create_label(),

                                           width_image_initial=2048, height_image_initial=1024,
                                           size_image_crop=401,

                                           dropFactor=0.1,
                                           learning_rate=0.0000001,
                                           learning_rate_decay=1 * (10 ** (-2)),
                                           weight_decay=0,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1 * 10 ** (-8),
                                           batch_size=40,
                                           batch_size_val=40,
                                           epoch_total=100,
                                           actual_epoch=0,
                                           ratio=(1, 1),
                                           weight_grad=torch.FloatTensor([1 for i in range(19)]),
                                           loss="cross_entropy_pretrain",
                                           momentum_IoU=0,
                                           pretrain=True,
                                           path_save_net="./Model/",
                                           name_network="resnet18_1000classes",
                                           train_number=0,
                                           path_CSV="./CSV/",
                                           # path_data="/home_expes/collections/Cityscapes/",
                                           path_data="/home_expes/collections/imagenet_1000_classes/",
                                           path_print="./Python_print_resnet18_1000classes.txt",
                                           path_result="./Result",
                                           num_workers=2)
        # Define the GridNet
        network = GridNet_structure.gridNet_imagenet(nInputs=parameters.nFeatureMaps_init,
                                                     nOutputs=parameters.number_classes,
                                                     nColumns=parameters.nColumns,
                                                     nFeatMaps=parameters.nFeatMaps,
                                                     dropFactor=parameters.dropFactor)
        network = GridNet_structure.ResNet18(nOutputs=len(Label.create_imagenet_class()))

        with open(parameters.path_print, 'w') as txtfile:
            txtfile.write('\n               Start of the program \n')

        # Init the csv file that will store the error
        Save_import.init_csv(name_network=parameters.name_network,
                             train_number=parameters.train_number,
                             path_CSV=parameters.path_CSV,
                             path_print=parameters.path_print)

    # Import both DataSets with the transformation
    train_dataset = Save_import.cityscapes_create_dataset_pretrain(mode="train",
                                                                   parameters=parameters,
                                                                   sliding_crop=None,
                                                                   transform=transforms.Compose([
                                                                       transforms.RandomResizedCrop(224),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                   ]))
    val_dataset = Save_import.cityscapes_create_dataset_pretrain(mode="val",
                                                                 parameters=parameters,
                                                                 sliding_crop=None,
                                                                 transform=transforms.Compose([
                                                                     transforms.RandomResizedCrop(224),
                                                                     transforms.ToTensor(),
                                                                 ]))

    # Create the DataSets for Pytorch used
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=parameters.batch_size,
                                               shuffle=True,
                                               num_workers=parameters.num_workers,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=parameters.batch_size_val,
                                             shuffle=True,
                                             num_workers=parameters.num_workers,
                                             drop_last=False)

    # If there is more than one GPU we can sue them
    if torch.cuda.device_count() > 1:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nLet's use " + str(torch.cuda.device_count()) + " GPUs! \n")
        network = torch.nn.DataParallel(network)

    else:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nWe don t have more than one GPU \n")
            # ... But we still use it in this case ? ... TODO try without to check if it is still working
            # network = torch.nn.DataParallel(network)

    # Put the network on GPU if possible
    if torch.cuda.is_available():
        network.cuda()

    else:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nAccording to torch Cuda is not even available \n")

    # Train the network

    train(network=network,
          parameters=parameters,
          train_loader=train_loader,
          val_loader=val_loader)


# Check if there is argument and run the main program with good argument, load previous Data or not.
if len(sys.argv) == 4:
    print("4 arguments")
    main(path_continue_learning=sys.argv[1], total_epoch=int(sys.argv[2]), new_name=sys.argv[3])
elif len(sys.argv) == 3:
    main(path_continue_learning=sys.argv[1], total_epoch=int(sys.argv[2]))
elif len(sys.argv) == 1:
    main()
