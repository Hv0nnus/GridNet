import Save_import

parameters, _ = Save_import.load_from_checkpoint("./Model/best1cross_weight0checkpoint.pth.tar")

print("Parameters :\n")
print("number of Columns : ",parameters.nColumns)
