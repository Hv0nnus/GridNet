import Save_import

parameters, _ = Save_import.load_from_checkpoint("Model/best1LongTest0checkpoint.pth.tar")

print(parameters)