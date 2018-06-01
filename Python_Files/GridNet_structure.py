# coding: utf-8

import torch.nn as nn
import torch
import random


class firstConv(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        This is the first convolution used to enter into the grid.
        :param nInputs: number of features map for the input which is also the same for the output
        """
        super(firstConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nInputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        return x


class convSequence(nn.Module):
    def __init__(self, nInputs, nOutputs, dropFactor):
        """

        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        :param dropFactor: Total Dropout on the entire Sequence, there is a probability p = dropFactor that
        the residual is deleted
        This class represent a residual bloc that doesn't change the number nor the size of the features maps
        """
        super(convSequence, self).__init__()

        self.dropFactor = dropFactor

        self.batch1 = nn.BatchNorm2d(num_features=nInputs, eps=1e-05, momentum=0.1, affine=True)

        self.ReLU1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x_init):
        x = self.batch1(x_init)
        x = self.ReLU1(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.ReLU2(x)
        x = self.conv2(x)
        # *1 is a small trick that transform boolean into integer
        x = ((random.random() > self.dropFactor) * 1) * x
        x += x_init
        return x


class subSamplingSequence(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente a bloc that reduce the resolution of each feature map (factor2)
        """
        super(subSamplingSequence, self).__init__()

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x):
        x = self.batch1(x)
        x = self.ReLU1(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.ReLU2(x)
        x = self.conv2(x)
        return x


class upSamplingSequence(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente a bloc that increase the resolution of each feature map(factor2)
        """

        super(upSamplingSequence, self).__init__()

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU1 = nn.ReLU()

        self.convTranspose1 = nn.ConvTranspose2d(in_channels=nInputs, out_channels=nOutputs,
                                                 kernel_size=(3, 3),
                                                 stride=(2, 2),
                                                 padding=(1, 1),
                                                 dilation=1,
                                                 groups=1,
                                                 bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x):
        x = self.batch1(x)
        x = self.ReLU1(x)
        x = self.convTranspose1(x)
        x = self.batch2(x)
        x = self.ReLU2(x)
        x = self.conv2(x)
        return x


class lastConv(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente the last Convolution of the network before the prediction
        """

        super(lastConv, self).__init__()

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.ReLU1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

    def forward(self, x):
        x = self.batch1(x)
        x = self.ReLU1(x)
        x = self.conv1(x)
        x = self.batch2(x)
        return x


class gridNet(nn.Module):
    def __init__(self, nInputs, nOutputs, nColumns, nFeatMaps, dropFactor):
        """
        :param nInputs: number of features maps for the input
        :param nOutputs: number of features maps for the output
        :param nColumns: number of columns of the gridNet, this number should be divisible by two.
        It count the number of bloc +1
        :param nFeatMaps: number of feature at each row of the gridNet
        :param dropFactor: factor witch control the dropout of an entire bloc
        This is the final network : called GridNet
        """

        super(gridNet, self).__init__()

        # Define some parameters as an attribut of the class
        len_nfeatureMaps = len(nFeatMaps)
        self.nColumns = nColumns
        self.nFeatMaps = nFeatMaps
        self.len_nfeatureMaps = len_nfeatureMaps

        # A normalisation before any computation
        self.batchNormInitial = nn.BatchNorm2d(num_features=nInputs,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True)

        # The first convolution before entering into the grid.
        self.firstConv = firstConv(nInputs=nInputs, nOutputs=nFeatMaps[0])

        # We create the Grid. We will creat conv and sub/up sequences with different name.
        # The name is : "sequenceName" + starting position of the sequence(i,j) + "to" + ending position (k,l)
        for i in range(len(nFeatMaps)):

            for j in range(nColumns):

                # We don t creat a residual bloc on the last column
                if j < (nColumns - 1):
                    setattr(self, "convSequence" + str(i) + "_" + str(j) + "to" + str(i) + "_" + str(j + 1),
                            convSequence(nFeatMaps[i], nFeatMaps[i], dropFactor))

                # We creat subSampling only on half of the grid and not in the last row
                if j < (nColumns // 2) and i < (len(nFeatMaps) - 1):
                    setattr(self, "subSamplingSequence" + str(i) + "_" + str(j) + "to" + str(i + 1) + "_" + str(j),
                            subSamplingSequence(nFeatMaps[i], nFeatMaps[i + 1]))

                # Welook a the other half but not the first row
                if j >= (nColumns // 2) and i > 0:
                    setattr(self, "upSamplingSequence" + str(i) + "_" + str(j) + "to" + str(i - 1) + "_" + str(j),
                            upSamplingSequence(nFeatMaps[i], nFeatMaps[i - 1]))

        # The last convolution before the result.
        self.lastConv = lastConv(nInputs=nFeatMaps[0],
                                 nOutputs=nOutputs)

        # self.batchNormFinal = nn.BatchNorm2d(num_features = nInputs, eps=1e-05, momentum=0,affine=True)

        self.logsoftmax1 = nn.LogSoftmax()

    def addTransform(self, X_i_j, SamplingSequence):
        """
        :param X_i_j: The value on the grid a the position (i,j)
        :param SamplingSequence: The sampling that should be added to the point (i,j)
        :return: The fusion of the actual value on (i,j) and the new data which come from the sampling
        """
        return X_i_j + SamplingSequence

    def forward(self, x):

        # A normalisation before any computation

        x = self.batchNormInitial(x)

        # The first convolution before entering into the grid.
        x = self.firstConv(x)

        # X is the matrix that represente the values of the features maps at the point (i,j) in the grid.
        X = [[0 for i in range(self.nColumns)] for j in range(self.len_nfeatureMaps)]
        # The input of the grid is on (0,0)
        X[0][0] = x

        # Looking on half of the grid, with sumsampling and convolution sequence
        for j in range(self.nColumns // 2):

            for i in range(self.len_nfeatureMaps):

                # For the first column, there is only subsampling
                if j > 0:
                    # This syntaxe call self.conSequencei_(j-1)toi_j(X[i][j-1])
                    X[i][j] = getattr(self, "convSequence"
                                      + str(i) + "_" + str(j - 1) + "to" + str(i) + "_" + str(j))(X[i][j - 1])

                    # For the first row, there is only ConvSequence (residual bloc)
                    if i > 0:
                        X[i][j] = self.addTransform(X[i][j], getattr(self, "subSamplingSequence"
                                                                     + str(i - 1) + "_" + str(j) + "to" + str(i) +
                                                                     "_" + str(j))(X[i - 1][j]))
                else:
                    # For the first row, there is only ConvSequence (residual bloc)
                    if i > 0:
                        X[i][j] = getattr(self, "subSamplingSequence"
                                          + str(i - 1) + "_" + str(j) + "to" + str(i) +
                                          "_" + str(j))(X[i - 1][j])

        # Looking on the other half of the grid
        for j in range(self.nColumns // 2, self.nColumns):

            for i in range(self.len_nfeatureMaps - 1, -1, -1):

                X[i][j] = getattr(self, "convSequence" +
                                  str(i) + "_" + str(j - 1) + "to" + str(i) + "_" + str(j))(X[i][j - 1])

                # There is no upSampling on the last row
                if i < (self.len_nfeatureMaps - 1):
                    X[i][j] = self.addTransform(X[i][j], getattr(self, "upSamplingSequence"
                                                                 + str(i + 1) + "_" + str(j) + "to" + str(i) +
                                                                 "_" + str(j))(X[i + 1][j]))

        x_final = self.lastConv(X[0][self.nColumns - 1])

        if torch.cuda.is_available():
            with open("/home_expes/kt82128h/GridNet/Python_Files/Python_print.txt", 'a') as txtfile:
                txtfile.write("\n  In Model: input size" + str(x.size()) + "output size" + str(x_final.size()) + "\n")

        return x_final


class pretrain_end_network(nn.Module):

    def __init__(self, nInputs=19, nOutputs=1000):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of classses
        This class represent the last operation of the network before the prediction for the ImageNet Dataset
        """

        super(pretrain_end_network, self).__init__()

        self.MaxPool1 = torch.nn.MaxPool2d(29, # padding de 2 : (257 + 4) / 29 = 9
                                           stride=None,
                                           padding=2,
                                           dilation=1,
                                           return_indices=False,
                                           ceil_mode=False)

        self.Linear1 = nn.Linear(in_features=19*9*9,
                                 out_features=1000,
                                 bias=True)

        self.Sigmoid1 = nn.Sigmoid()

        self.Linear2 = nn.Linear(in_features=1000,
                                 out_features=1000,
                                 bias=True)

    def forward(self, x):
        x = self.MaxPool1(x)
        x = self.Linear1(x)
        x = self.Sigmoid1(x)
        x = self.Linear2(x)
        return x
