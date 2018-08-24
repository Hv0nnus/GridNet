# GridNet
Code for my 6 months intership about the a neural network called GridNet

Launch the program "Main.py" without any argument. All the parameter should be change directly in the fonction "Main.py". This will creat a txt files with all the information about the learning (error, time, number of epoch). This will also save regulary a CSV files with the loss and the data to recreate the confusion matrix (pandas format). This will also creat the Model regulary. You can later import it to retrain or use it to test on the image. All the path can be change in the "Main.py" program in the parameter classe

To continue the learning (it has been stop before the end or you want to continue to learn) use Main.py with the right argument describe at the end of the "Main.py" files.

To test and save the test (images) run the program "Main_test.py". This program need the path of model already saved and the string "train" "test" or "val" to find the right dataset. Again, all the path can be change in the the parameter classe. The number of images compute can be change manually at the end of the fonction "Test dataset.test loop".

The program Plot.ipnb show all the result with graphs.

"Pretrain main.py" doesn't work well. It try to pretrain on imagenet.
"export onnx.py" and "net drawer.py" have work during my internship but I didn't spend too much time in it so it's messy and doesn't work everytime. It should use "Grid Net copy.py", "net drawer.py", "export onnx.py".

For any questions you can contact me : tanguy.kerdoncuff@laposte.net
