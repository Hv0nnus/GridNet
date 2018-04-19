import GridNet_structure
import Parameters
import Label
import Save_import

from torchvision import transforms
import numpy as np
import torch


def count_classes(y_batch, parameters):
    count = np.array([0 for i in range(parameters.number_classes)])
    for i in range(parameters.number_classes):
        count[i] = torch.sum(y_batch == i)
    return (count)


def f():
    # Define all the parameters
    parameters = Parameters.Parameters(nColumns=2,
                                       nFeatMaps=[8, 16],
                                       nFeatureMaps_init=3,
                                       number_classes=20 - 1,
                                       label_DF=Label.create_label(),

                                       width_image_initial=2048, height_image_initial=1024,
                                       size_image_crop=50,

                                       dropFactor=0.1,
                                       learning_rate=0.01,
                                       weight_decay=5 * 10 ** (-6),
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1 * 10 ** (-8),
                                       batch_size=1,
                                       batch_size_val=8,
                                       epoch_total=200,
                                       actual_epoch=0,
                                       scale=(0.39, 0.5),
                                       ratio=(1, 1),

                                       path_save_net="./Model/",
                                       name_network="IoU_loss8",
                                       train_number=0,
                                       path_CSV="./CSV/",
                                       path_data="/home_expes/collections/Cityscapes/",
                                       # path_data="../Cityscapes_Copy/",
                                       path_print="./Python_print8.txt",
                                       path_result="./Result",
                                       num_workers=2)

    parameters.transforms_output = transforms.Compose([transforms.ToTensor()])
    parameters.transforms_input = transforms.Compose([transforms.ToTensor()])

    # Define the GridNet
    network = GridNet_structure.gridNet(nInputs=parameters.nFeatureMaps_init,
                                        nOutputs=parameters.number_classes,
                                        nColumns=parameters.nColumns,
                                        nFeatMaps=parameters.nFeatMaps,
                                        dropFactor=parameters.dropFactor)

    # Import both DataSets with the transformation
    train_dataset = Save_import.CityScapes_final('fine', 'train',
                                                 transform=parameters.transforms_input,
                                                 transform_target=parameters.transforms_output,
                                                 parameters=parameters)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=parameters.batch_size,
                                               shuffle=True,
                                               num_workers=parameters.num_workers,
                                               drop_last=False)

    count = np.array([0 for i in range(parameters.number_classes)])

    for i, (x_batch, y_batch, _) in enumerate(train_loader):
        count += count_classes(y_batch, parameters)
    count = count / sum(count)
    print(count)
    return (1 / count)


a = f()
print(a)