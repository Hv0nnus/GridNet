
# coding: utf-8

import pandas as pd
import seaborn as sns; sns.set()
from ggplot import *
import numpy as np
import matplotlib.pyplot as plt

# In[6]:

"""organise_CSV import two CSV files and delete all duplicate row. Because the algorithme work with
    mini_batch there is many value for the loss for one epoch and one data set. We compute here the mean
    of all this loss that have the same epoch and data set. We did the same with the confusion matrix
    (0) = name_network : name of the network associated with the CSV file
    (1) = train_number : number of the network associated with the CSV file
"""
def organise_CSV(name_network,train_number,path_CSV):
    # Import the CSV file into pandas DataFrame
    loss_DF = pd.read_csv(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv")
    # This Groupby will regroupe all line that have the same "Set" and "Epoch" and compute the mean over the "Values"
    loss_DF = loss_DF.groupby(['Set','Epoch'])['Value'].mean().reset_index()
    # Recreate the CSV file
    loss_DF.to_csv(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv",index = False)
    
    # Import the CSV file into pandas DataFrame
    conf_DF = pd.read_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv")
    # This Groupby will regroupe all line that have the same 'Target','Prediction','Epoch','Set'
    # and compute the mean over the "Values"
    conf_DF = conf_DF.groupby(['Target','Prediction','Epoch','Set'])['Value'].mean().reset_index()
    # Recreate the CSV file
    conf_DF.to_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv",index = False)


# In[2]:

"""plot_loss will plot the loss against the epoch
    (0) = name_network : name of the network associated with the CSV file
    (1) = train_number : number of the network associated with the CSV file
"""
def plot_loss(name_network,train_number,path_CSV):
    # Import the CSV file into pandas DataFrame
    loss_DF = pd.read_csv(path_CSV + "CSV_loss_" + name_network + str(train_number) + ".csv")
    
    # Epoch against Value with a different color for train and validation
    my_plot_loss = ggplot(aes(x="Epoch", y="Value",color = "Set"),data = loss_DF)
    my_plot_loss = my_plot_loss + geom_line() 
    my_plot_loss = my_plot_loss + ggtitle("Loss per epoch")
    my_plot_loss = my_plot_loss + ylab("Loss")
    print(my_plot_loss)


# In[1]:

"""plot_IuO will plot the loss against the epoch
    (0) = name_network : name of the network associated with the CSV file
    (1) = train_number : number of the network associated with the CSV file
"""
def plot_IuO(name_network,train_number,parameters,path_CSV):

    # Import the CSV of the conv matrix
    conf_DF = pd.read_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv")
    
    # Delete the class 19
    conf_DF_uselfull_classes = conf_DF[(conf_DF["Prediction"] != "class"+str(parameters.number_classes - 1)) 
                                       & (conf_DF["Target"] != "class"+str(parameters.number_classes - 1))]
    # In the column "Value" there is the TP information
    conf_DF_TP = conf_DF_uselfull_classes[conf_DF_uselfull_classes["Prediction"] == conf_DF_uselfull_classes["Target"]]

    # In the column "Value" there is the FN information
    conf_DF_FN = conf_DF_uselfull_classes.groupby(["Set","Epoch","Target"])["Value"].sum().reset_index()
    # In the column "Value" there is the FP information
    conf_DF_FP = conf_DF_uselfull_classes.groupby(["Set","Epoch","Prediction"])["Value"].sum().reset_index()

    # Change the name
    conf_DF_FN.rename(columns={"Value": "FN"}, inplace=True)
    conf_DF_FP.rename(columns={"Value": "FP"}, inplace=True)

    # Merge the dataset together according to certain column
    conf_DF_TP_FN = conf_DF_TP.merge(conf_DF_FN, on=["Epoch","Set","Target"])
    conf_DF_TP_FN_FP = conf_DF_TP_FN.merge(conf_DF_FP, on=["Epoch","Set","Prediction"])
    
    # We compute the realFP and FN value, we have to substract the TP values
    conf_DF_TP_FN_FP["FP"] = conf_DF_TP_FN_FP["FP"] - conf_DF_TP_FN_FP["Value"]
    conf_DF_TP_FN_FP["FN"] = conf_DF_TP_FN_FP["FN"] - conf_DF_TP_FN_FP["Value"]
    
    # Compute the IoU each class
    conf_DF_TP_FN_FP["IoU"] = conf_DF_TP_FN_FP["Value"]/(conf_DF_TP_FN_FP["Value"] + 
                                                         conf_DF_TP_FN_FP["FP"] + conf_DF_TP_FN_FP["FN"])

    # Compute the mean IoU
    plot_DF = conf_DF_TP_FN_FP.groupby(["Set","Epoch"])["IoU"].mean().reset_index()

    #Add the real name of Data.
    
    # Plot the IuO mean
    my_plot_IoU = ggplot(aes(x="Epoch", y="IoU",color = "Set"),data = plot_DF) + geom_line()
    my_plot_IoU = my_plot_IoU + ggtitle("Mean IoU per epoch")
    my_plot_IoU = my_plot_IoU + ylab("Mean IoU")
    print(my_plot_IoU)
    
    #Change change the name of class for the plot
    conf_DF_TP_FN_FP = conf_DF_TP_FN_FP.merge(parameters.label_DF, left_on='Target', right_on='Class_name')

    # Plot the IuO per class only train
    my_plot_IoU = ggplot(aes(x="Epoch", y="IoU",color = "Real_name"),
                          data = conf_DF_TP_FN_FP[conf_DF_TP_FN_FP["Set"] == "train"]) + geom_line()
    my_plot_IoU = my_plot_IoU + ggtitle("Train IoU per epoch")
    my_plot_IoU = my_plot_IoU + ylab("Train IoU")
    print(my_plot_IoU)
    
    # Plot the IuO per class only validation
    my_plot_IoU = ggplot(aes(x="Epoch", y="IoU",color = "Real_name"),
                          data = conf_DF_TP_FN_FP[conf_DF_TP_FN_FP["Set"] == "validation"]) + geom_line()
    my_plot_IoU = my_plot_IoU + ggtitle("Validation IoU per epoch")
    my_plot_IoU = my_plot_IoU + ylab("Validation IoU")
    print(my_plot_IoU)


# In[2]:

"""plot_IuO will plot confusion matrix
    (0) = name_network : name of the network associated with the CSV file
    (1) = train_number : number of the network associated with the CSV file
    (2) = epoch : Value of the epoch were we want to display the confusion matrix
    (3) = data_set : 
"""
def plot_mat_confusion(name_network,train_number,epoch,data_set,path_CSV):
    
    # Import the CSV as DataFrame
    conf_DF = pd.read_csv(path_CSV + "CSV_confMat_" + name_network + str(train_number) + ".csv")
    # Select only the usefull epoch
    conf_DF = conf_DF[conf_DF["Epoch"]==epoch]
    # Keep only the DataSet to display
    conf_DF = conf_DF[conf_DF["Set"]==data_set]

    # Confusion Matrix
    conf_mat_for_plot = np.zeros((20,20))
    #Double loop over the confusion matrix the ad each value
    for i in range(20):
        for j in range(20):
            conf_mat_for_plot[i,j] = conf_DF.loc[(conf_DF["Prediction"]==("class"+str(i))) &
                                                 (conf_DF["Target"]==("class"+str(j))),"Value"].values
    
    #Just change column and row name to make good plot and make the confusion matrix a dataframe
    conf_mat_for_plot=pd.DataFrame(conf_mat_for_plot, columns = ["road","sidewalk","building","wall","fence",
                                                                        "pole","traffic light","traffic sign",
                                                                        "vegetation","terrain","sky","person","rider",
                                                                        "car","truck","bus","train","motorcycle",
                                                                        "bicycle","autre"])
    # Rename the rows
    conf_mat_for_plot = conf_mat_for_plot.rename({0 : "road", 1 : "sidewalk", 2 : "building", 3 : "wall",4 : "fence",
                                                  5 : "pole",6 : "traffic light",7 : "traffic sign", 8 : "vegetation",
                                                  9 : "terrain",10 : "sky" ,11 : "person" ,12 : "rider", 13 :"car",
                                                  14 : "truck",15 : "bus",16 : "train",17 : "motorcycle",
                                                  18 : "bicycle",19 : "autre"},axis='index')
    # Set the size
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    sns.heatmap(conf_mat_for_plot,annot=True,ax=ax,square = True,cmap = "Reds")
    plt.xlabel(r'Prediction',fontsize = 20)
    plt.ylabel(r'Real class',fontsize = 20)
    plt.title("Confusion matrix of " + data_set + "set for epoch " + str(epoch), fontsize = 20)
    plt.show()

