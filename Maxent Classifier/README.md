# Maxent Classifier

This is an implementation of maxent classifier and the algorithm used was logistic regression with the solver as limited memory BFGS and a multinomial scheme is used where the cross-entropy loss function is applied, these were the basic configurations that were set.

The program takes two inputs

1. The zipped MNIST image file consisting of samples for 10000 digit representations

2. The zipped MNIST lable file for the respective digit representation in the image file.

The implementation used is scikit-learn’s sklearn.linear_model.LogisticRegression, and the best configuration that could be used were the following parameters that were set to the classifier.

* multi_class --> multinomial
* solver --> lbfgs
* fit_intercept --> True
* intercept_scaling --> 1
* penalty --> l2

Here the fit_intercept and intercept_scaling parameters of the classifier are the bias setting and penalty is the regularization term used.
