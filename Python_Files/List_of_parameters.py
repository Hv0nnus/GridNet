import Save_import
import sys

parameters, _ = Save_import.load_from_checkpoint(sys.argv[1])
parameters.momentum_IoU = 10
print("Parameters :\n")
print("\nnumber of Columns : ", parameters.nColumns)
print("\nsize image crop\n",parameters.size_image_crop)
print("\nnumber of number_classes : ", parameters.number_classes)
print("\nnumber of actual_epoch : ", parameters.actual_epoch)
print("\nnumber of batch_size : ", parameters.batch_size)
print("\nnumber of batch_size_val : ", parameters.batch_size_val)
print("\nnumber of beta1 : ", parameters.beta1)
print("\nnumber of beta2 : ", parameters.beta2)
print("\nnumber of dropFactor : ", parameters.dropFactor)
print("\nnumber of epsilon : ", parameters.epsilon)
print("\nnumber of epoch_total : ", parameters.epoch_total)
print("\nnumber of learning_rate : ", parameters.learning_rate)
print("\nnumber of name_network : ", parameters.name_network)
print("\nname of the loss",parameters.loss)
print("\nnumber of nFeatMaps : ", parameters.nFeatMaps)
print("\nnumber of num_workers : ", parameters.num_workers)
print("\nnumber of weight_decay : ", parameters.weight_decay)
print("\nnumber of size_image_crop : ", parameters.size_image_crop)
print("\nlearning_rate_decay :", parameters.learning_rate_decay)
print("\ntest : ",parameters.momentum_IoU)
print("\nnumber of weight_grad : ", parameters.weight_grad)
print("\nnumber of train_number : ", parameters.train_number)
