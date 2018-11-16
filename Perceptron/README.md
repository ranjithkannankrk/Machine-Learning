# Perceptron

This an implementation of multiclass perceptron over MNIST dataset for prediction of hand written digits

The program takes two inputs

1. The zipped MNIST image file consisting of samples for 10000 digit representations

2. The zipped MNIST lable file for the respective digit representation in the image file.

The design followed the same design as that of perceptron’s binary classification, but since we have a multiclass problem in hand, there were few tweaks made in the data. The algorithm was trained with respect the One Versus All methodology where each digit was trained against the rest, i.e. while training there are only two classes either a class which denotes digit 1 for example or another class which represents the rest of the digits. Now the algorithm has to be trained 10 different times to obtain a weight which for each class, hence now we would obtain 10 different weights. For prediction the incoming sample with input features is multiplied with all the 10 weights and we obtain 10 results, the highest value of the results of multiplication is the digit in the sample.
