# coding: utf-8

# Package

# Torch related package
from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

# cuda related package
import torch.cuda
import torch.backends.cudnn as cudnn

# Other package
import time
import sys

# Other python program
import GridNet_structure
import Save_import
import Loss_Error
import Parameters
import Label

# # Commentaire pour la suite (TODO)

#
# Change la fonction de cout, lien entre fonction de cout et de perte ?
#
# Verifier ce qu est la mesure de test IoU
# 
# Mettre des commentaire
# 
# Ne pas sauvgarder n importe quand le réseau. Peut ere garder un truc qui a en memoire le meilleur IuO sur 
# la validation
# 
# TODO Quel modification sont a faire sur les donnees  genre pour les augmenter
# 
# Parallelisation, comment ca marche 
# 
# COMMENT VERIFIER QUE TOUT FONCTIONNE Regarder le graph peut être.
# Regarder le temps de calcul quand 
# 
# Idee: reregarder comment on fait de la segmentation avec les champs de Markov, ca peut donner des idees
# 
# Afficher le nom de la courbe avec la courbe et pas juste a coter
# 
# 
# Indice qui indique si une fonction est derivable informatiquement, et prendre uen fonction qui se rapproche de l IoU
# 
# Corrélation entre IoU et le logsoftmax
# 
# distance entre deux fonctions 
# 
# Faire les tests sur la base de tests
# 
# Taille en octet de ce que j'importe et du réseau


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
    optimizer = optim.Adam(params=network.parameters(),
                           lr=parameters.learning_rate,
                           betas=(parameters.beta1, parameters.beta2),
                           eps=parameters.epsilon,
                           weight_decay=parameters.weight_decay)

    # High value just to initialize this variable
    # validation error min will store the lowest validation result
    validation_error_min = 9999

    # Store the index of the next checkpoint. This value is 0 or 1. We always keep one checkpoint untouched
    # while the other one is changed.
    index_save_regular = 0
    index_save_best = 0

    # Loop from the actual epoch (not 0 if we already train) to the last epoch
    for epoch in range(parameters.actual_epoch, parameters.epoch_total):
        # Store the time at the begining of each epoch
        timer_epoch = time.time()
        timer_batch = time.time()

        # Loop over the mini-batch, the size of the mini match is define in the train_loader
        for i, (x_batch, y_batch) in enumerate(train_loader):

            # zero the gradient buffers
            optimizer.zero_grad()

            # Transform into Variable
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)

            # Compute the forward function
            y_batch_estimated = network(x_batch)

            # Get the error
            loss = Loss_Error.criterion(y_batch_estimated, y_batch, parameters)

            # Compute the backward function
            loss.backward()

            # Does the update according to the optimizer define above
            optimizer.step()

            # Save error of the training DataSet
            Save_import.save_error(x=x_batch, y=y_batch,
                                   network=network,
                                   epoch=epoch,
                                   set_type="train",
                                   parameters=parameters)

            # Similar to a "print" but in a textfile
            with open(parameters.path_print, 'a') as txtfile:
                txtfile.write(
                    "\nEpoch : " + str(epoch) + ". Batch : " + str(i) + ".\nLast loss : " + str(loss.data[0]) + "\n" +
                    "Time batch : " + Save_import.time_to_string(time.time() - timer_batch) +
                    ".\n Time total batch : " + Save_import.time_to_string(time.time() - timer_epoch) + "\n \n")

            timer_batch = time.time()
            break

        # Validation_error contains the error on the validation set
        validation_error = 0

        # Save the error of the validation DataSet
        for i, (x_val_batch, y_val_batch) in enumerate(val_loader):

            x_val_batch, y_val_batch = Variable(x_val_batch), Variable(y_val_batch)

            validation_error += Save_import.save_error(x=x_val_batch, y=y_val_batch,
                                                       network=network,
                                                       epoch=epoch,
                                                       set_type="validation",
                                                       parameters=parameters)
            break

        # Divide by the the number of element in the entire batch
        validation_error /= i + 1

        # checkpoint will save the network if needed
        index = Save_import.checkpoint(validation_error=validation_error,
                                       validation_error_min=validation_error_min,
                                       index_save_best=index_save_best,
                                       index_save_regular=index_save_regular,
                                       epoch=epoch,
                                       network=network,
                                       parameters=parameters)
        validation_error_min, index_save_best, index_save_regular = index

        # Similar to a "print" but in a text file
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\n              End of Epoch :" + str(epoch) + "/" + str(parameters.epoch_total - 1) +
                          ". Validation Loss : " + str(validation_error) +
                          ".\nTime Epoch :" + Save_import.time_to_string(time.time() - timer_epoch) +
                          ".\n Time total : " + Save_import.time_to_string(time.time() - timer_init) +
                          ".\n \n")

    # Similar to a "print" but in a text file
    with open(parameters.path_print, 'a') as txtfile:
        txtfile.write("Finish. Total time : " + Save_import.time_to_string(time.time() - timer_init) +
                      "\n")
    return ()


def main(path_continue_learning=None, total_epoch=0):
    """
    :param path_continue_learning: Path were the network is already saved
                                   (don t use if it is the beginning of the training)
    :param total_epoch: Number of epoch needed don t use if it is the beginning of the training)
    :return: Nothing but train the network and save CSV files for the error and also save the network regularly
    """

    # Define all the parameters
    parameters = Parameters.Parameters(nColumns=2,
                                       nFeatMaps=[3, 6],
                                       nFeatureMaps_init=3,
                                       number_classes=20 - 1,
                                       label_DF=Label.creat_label(),

                                       width_image_initial=2048, height_image_initial=1024,
                                       size_image_crop=7,

                                       dropFactor=0.1,
                                       learning_rate=0.01,
                                       weight_decay=5 * 10 ** (-6),
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1 * 10 ** (-8),
                                       batch_size=5,
                                       batch_size_val=10,
                                       epoch_total=1,
                                       actual_epoch=0,
                                       scale=(0.39, 0.5),
                                       ratio=(1, 1),

                                       # path_save_net = "/home_expes/kt82128h/GridNet/Python_Files/Model/",
                                       name_network="test",
                                       train_number=0,
                                       # path_CSV = "/home_expes/kt82128h/GridNet/Python_Files/CSV/",
                                       # path_data = "/home_expes/collections/Cityscapes/",
                                       # path_print = "/home_expes/kt82128h/GridNet/Python_Files/Python_print.txt"
                                       num_workers=0,
                                       )

    with open(parameters.path_print, 'w') as txtfile:
        txtfile.write('\n               Start of the program \n')

    # Import both DataSets with the transformation
    train_dataset = Save_import.CityScapes_final('fine', 'train',
                                                 transform=parameters.transforms_input,
                                                 transform_target=parameters.transforms_output,
                                                 parameters=parameters)
    val_dataset = Save_import.CityScapes_final('fine', 'val',
                                               transform=parameters.transforms_input,
                                               transform_target=parameters.transforms_output,
                                               parameters=parameters)

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

    # Define the GridNet
    network = GridNet_structure.gridNet(nInputs=parameters.nFeatureMaps_init,
                                        nOutputs=parameters.number_classes,
                                        nColumns=parameters.nColumns,
                                        nFeatMaps=parameters.nFeatMaps,
                                        dropFactor=parameters.dropFactor)

    # If the network was already train we import it
    if path_continue_learning is not None:
        # Load the trained Network
        parameters = Save_import.load_from_checkpoint(
            path_checkpoint=parameters.path_save_net + path_continue_learning,
            network=network,
            path_print=parameters.path_print)

        # Here we can change some parameters
        parameters.epoch_total = total_epoch

    # If the network was not train we start from scratch
    else:
        # Init the csv file that will store the error
        Save_import.init_csv(name_network=parameters.name_network,
                             train_number=parameters.train_number,
                             path_CSV=parameters.path_CSV,
                             path_print=parameters.path_print)

    # Train the network
    train(network=network,
          parameters=parameters,
          train_loader=train_loader,
          val_loader=val_loader)


# Check if there is argument and run the main program with good argument, load previous Data or not.
if len(sys.argv) == 3:
    main(path_continue_learning=sys.argv[1], total_epoch=int(sys.argv[2]))
elif len(sys.argv) == 1:
    main()
