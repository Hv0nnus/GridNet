import csv
import sys
a = 4

print(a)
b = a+1
print(b)

header_confMat = ["Set","Value","Target","Prediction","Epoch"]
header_loss = ["Value","Set","Epoch"]
    
# Try to open the file and write the header
print(sys.argv[1], sys.argv[2])
