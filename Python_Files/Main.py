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
#
# Dans les fully connected, pourquoi prendre des images de tailles 400
# *400 plutot que 200*200. Difference...

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

    train_error = 0

    # Loop over the mini-batch, the size of the mini match is define in the train_loader
    for i, (x_batch, y_batch, _) in enumerate(train_loader):

        # zero the gradient buffers
        optimizer.zero_grad()

        # Transform into Variable
        if torch.cuda.is_available():
            x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda())
        else:
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)

        # Compute the forward function
        y_batch_estimated = network(x_batch)

        if torch.cuda.is_available():
            with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print.txt", 'a') as txtfile:
                txtfile.write("\n " "Outside: input size" + str(x_batch.size()) +
                              "output_size" + str(y_batch_estimated.size()) + "\n")

        # Get the error
        loss = Loss_Error.criterion(y_batch_estimated, y_batch, parameters)

        # Compute the backward function
        loss.backward()

        # Does the update according to the optimizer define above
        optimizer.step()

        # Save error of the training DataSet
        train_error += Save_import.save_error(x=x_batch, y=y_batch,
                                              network=network,
                                              epoch=epoch,
                                              set_type="train",
                                              parameters=parameters)

        # Similar to a "print" but in a textfile
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write(
                "\nEpoch : " + str(epoch) + ". Batch : " + str(i) + ".\nTrain_Error : " + str(
                    train_error / (i + 1)) + "\n" +
                "Time batch : " + Save_import.time_to_string(time.time() - timer_batch) +
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
    # Validation_error contains the error on the validation set
    validation_error = 0

    # Save the error of the validation DataSet
    for i, (x_val_batch, y_val_batch, _) in enumerate(val_loader):

        if torch.cuda.is_available():
            x_val_batch, y_val_batch = Variable(x_val_batch.cuda()), Variable(y_val_batch.cuda())
        else:
            x_val_batch, y_val_batch = Variable(x_val_batch), Variable(y_val_batch)

        validation_error += Save_import.save_error(x=x_val_batch, y=y_val_batch,
                                                   network=network,
                                                   epoch=epoch,
                                                   set_type="validation",
                                                   parameters=parameters)

        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write(
                "\nEpoch : " + str(epoch) + ". Batch : " + str(i) + ".\nValidation error : " + str(
                    validation_error / (i + 1)) +
                ".\nTime total batch : " + Save_import.time_to_string(time.time() - timer_epoch) + "\n \n")

    # Divide by the the number of element in the entire batch
    return validation_error / (i + 1)


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

        batch_loop(optimizer=optimizer,
                   train_loader=train_loader,
                   network=network,
                   epoch=epoch,
                   parameters=parameters,
                   timer_batch=timer_batch,
                   timer_epoch=timer_epoch)

        validation_error = validation_loop(val_loader=val_loader,
                                           network=network,
                                           epoch=epoch,
                                           parameters=parameters,
                                           timer_epoch=timer_epoch)

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
                          ".\nTime total : " + Save_import.time_to_string(time.time() - timer_init) +
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
    torch.manual_seed(26542461)
    # If the network was already train we import it
    if path_continue_learning is not None:
        # Load the trained Network
        parameters, network = Save_import.load_from_checkpoint(path_checkpoint=path_continue_learning)

        # Here we can change some parameters
        parameters.epoch_total = total_epoch

        with open(parameters.path_print, 'w') as txtfile:
            txtfile.write('\n               The program will continue \n')


    # If the network was not train we start from scratch
    else:
        #Define the weight
        weight_grad = torch.FloatTensor([2.381681e+09, 3.856594e+08,
                                           1.461642e+09, 4.291781e+07, 5.597591e+07, 8.135516e+07, 1.328548e+07,
                                           3.654657e+07, 1.038652e+09, 7.157456e+07, 2.527450e+08, 7.923985e+07,
                                           9.438758e+06, 4.460595e+08, 1.753254e+07, 1.655341e+07, 1.389560e+07,
                                           6.178567e+06, 2.936571e+07])

        sum = weight_grad.sum()
        # normalize and then take the invert
        for i in range(weight_grad.size(0)):
            weight_grad[i] = sum / weight_grad[i]
        # Normalize again and mult by the number of classes
        weight_grad = (weight_grad/weight_grad.sum())*weight_grad.size(0)

        # Define all the parameters
        parameters = Parameters.Parameters(nColumns=8,
                                           nFeatMaps=[16, 32, 64, 128, 256],
                                           nFeatureMaps_init=3,
                                           number_classes=20 - 1,
                                           label_DF=Label.create_label(),

                                           width_image_initial=2048, height_image_initial=1024,
                                           size_image_crop=401,

                                           dropFactor=0.1,
                                           learning_rate=0.01,
                                           weight_decay=5 * 10 ** (-6),
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1 * 10 ** (-8),
                                           batch_size=10,
                                           batch_size_val=10,
                                           epoch_total=800,
                                           actual_epoch=0,
                                           ratio=(1, 1),
                                           weight_grad=weight_grad,
                                           loss='cross_entropy',

                                           path_save_net="./Model/",
                                           name_network="big_no_weight",
                                           train_number=0,
                                           path_CSV="./CSV/",
                                           path_data="/home_expes/collections/Cityscapes/",
                                           path_print="./Python_print_big.txt",
                                           path_result="./Result",
                                           num_workers=2)
        # Define the GridNet
        network = GridNet_structure.gridNet(nInputs=parameters.nFeatureMaps_init,
                                            nOutputs=parameters.number_classes,
                                            nColumns=parameters.nColumns,
                                            nFeatMaps=parameters.nFeatMaps,
                                            dropFactor=parameters.dropFactor)

        with open(parameters.path_print, 'w') as txtfile:
            txtfile.write('\n               Start of the program \n')

        # Init the csv file that will store the error
        Save_import.init_csv(name_network=parameters.name_network,
                             train_number=parameters.train_number,
                             path_CSV=parameters.path_CSV,
                             path_print=parameters.path_print)

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

    if torch.cuda.device_count() > 1:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nLet's use " + str(torch.cuda.device_count()) + " GPUs! \n")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        network = torch.nn.DataParallel(network)
    else:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nWe can t use Cuda here \n")
        network = torch.nn.DataParallel(network)
    if torch.cuda.is_available():
        network.cuda()
    else:
        with open(parameters.path_print, 'a') as txtfile:
            txtfile.write("\nAccording to torch Cuda is not available \n")

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
