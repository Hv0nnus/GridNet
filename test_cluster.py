import csv
a = 4

print(a)
b = a+1
print(b)

header_confMat = ["Set","Value","Target","Prediction","Epoch"]
header_loss = ["Value","Set","Epoch"]
    
# Try to open the file and write the header
with open("CSV/CSV_confMat.csv", 'w') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(header_confMat)
