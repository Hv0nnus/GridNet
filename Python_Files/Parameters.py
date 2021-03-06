from torchvision import transforms
from PIL import Image
import torch
from torch.autograd import Variable
import torch.cuda


class Parameters:
    """
    Parameters contains many information useful for the training
    It is really useful to just pass Parameters as argument to the other function
    """
    def __init__(self,
                 # If we are on pretrain or not
                 pretrain=False,
                 # Number of columns of the grid
                 nColumns=2,
                 # Number of features map at each rows
                 nFeatMaps=[3,6],
                 # Number of feature map of the input image
                 nFeatureMaps_init=3,
                 # Number of classes (19(usefull classes) + 1(all other classes together))
                 number_classes=19,
                 # DataFrame with the name of each label associated with their number
                 label_DF=None,

                 # Size of initial image
                 width_image_initial=2048, height_image_initial=1024,
                 # Size after the crop # 353 the perfect size !
                 size_image_crop=5,

                 # Probability of a Blockwise dropout
                 dropFactor=0.1,
                 learning_rate=0.01,
                 learning_rate_decay=5 * (10 ** (-6)),
                 weight_decay=0,
                 # Parameter of the Adam Optimizer (beta1 beta2 and epsilon)
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1 * 10 ** (-8),
                 # Size of the mini batch
                 batch_size=2,
                 # Size of the mini batch to compute error (if the entire validation set cannot be in loaded)
                 batch_size_val=5,
                 # Maximum value of epoch iteration
                 epoch_total=10,
                 # The actual epoch is not null if we train the network which has already been train
                 actual_epoch=0,
                 # The loss that will be used to learn
                 loss="cross_entropy",
                 # The weight that are associated with the loss, most of the time it change the gradient
                 weight_grad=torch.FloatTensor([1 for i in range(19)]),
                 # Momentum of the IoU smooth. To better approximated the IoU
                 momentum_IoU = 0.9,

                 # File where all the parameter model can be store
                 path_save_net="../Model/",
                 # Name of the network, used for store (name_network and train_number)
                 name_network="test",
                 train_number=0,
                 # File where the error will be stored
                 path_CSV="../CSV/",
                 # Path of the Data
                 path_data="../Cityscapes_Copy/",
                 # Path were the output (print) is store at each iteration
                 # The information about the actual state of the execution is store in Python_print.txt
                 path_print="Python_print.txt",
                 # Path were the pictures are stored
                 path_result="./Result",
                 # Number of process that will load the Data
                 num_workers=0,

                 # Ratio is the deformation between width and height (increase width and reduce height for example)
                 ratio=(1, 1),
                 ):

        super(Parameters, self).__init__()
        # pretrain or train
        self.pretrain = pretrain
        # Image
        self.number_classes = number_classes
        self.label_DF = label_DF
        self.width_image_initial = width_image_initial
        self.height_image_initial = height_image_initial
        self.size_image_crop = size_image_crop
        self.scale = (((size_image_crop+1)**2)/(width_image_initial*height_image_initial),
                      (height_image_initial**2)/(width_image_initial*height_image_initial))
        # Number of feature map at the begining, if RGB image it would be 3
        self.nFeatureMaps_init = nFeatureMaps_init
        self.path_data = path_data

        # GridNet
        self.nColumns = nColumns
        self.nFeatMaps = nFeatMaps
        self.name_network = name_network
        self.train_number = train_number
        self.num_workers = num_workers

        # Save
        self.path_CSV = path_CSV
        self.path_save_net = path_save_net
        self.path_print = path_print
        self.path_result = path_result

        # Learning
        self.dropFactor = dropFactor
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.epoch_total = epoch_total
        self.actual_epoch = actual_epoch
        if torch.cuda.is_available():
            self.weight_grad = Variable(weight_grad.cuda(), requires_grad=False)
        else:
            self.weight_grad = Variable(weight_grad, requires_grad=False)
        self.momentum_IoU = momentum_IoU
        if torch.cuda.is_available():
            # List of the sum of Intersection and Union for each classes
            self.inter_union = Variable(torch.zeros(2, number_classes).cuda(), requires_grad=False)
        else:
            # List of the sum of Intersection and Union for each classes
            self.inter_union = Variable(torch.zeros(2, number_classes), requires_grad=False)

        self.loss = loss

        # Transformation that will be apply on the input just after the import
        self.transforms_input = transforms.Compose([
            transforms.RandomResizedCrop(size_image_crop, scale=self.scale, ratio=ratio),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        # Transformation that will be apply on the output just after the import
        self.transforms_output = transforms.Compose([
            transforms.RandomResizedCrop(size_image_crop, scale=self.scale, ratio=ratio, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
        ])



