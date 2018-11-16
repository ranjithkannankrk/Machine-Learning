import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from struct import unpack
import gzip
import sys
from numpy import zeros, uint8, float32
from sklearn.datasets import fetch_mldata

#sigmoid activation function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

#derivative of sigmoid function
def sigmGrad(x):
	return sigmoid(x)*(1-sigmoid(x))

#function to retrieve the gradient
def getgrad(data, label, theta1, theta2, regconst):
		# Feedforwarding
		(n, m) = data.shape
		value1 = np.vstack((np.ones((1,m)),data))
		dotprob = np.dot(theta1.T, value1)
		value2 = np.vstack((np.ones((1,m)),sigmoid(dotprob)))
		value3 = sigmoid(np.dot(theta2.T, value2))
		theta1In = theta1[1:,:]
		theta2In = theta2[1:,:]

		# Logistic Regression Cost function
		lrcost = np.sum(np.sum( -label*np.log(value3) - (1-label)*np.log(1-value3) ))/m + (regconst/(2*m))*(sum(sum(theta1In*theta1In))+sum(sum(theta2In*theta2In)))
		
		# Backpropagating
		d3 = value3 - label
		d2 = (theta2In.dot(d3))*sigmGrad(dotprob)
		tri2 = d3.dot(value2.T)
		tri1 = d2.dot(value1.T)
		theta2Grades = tri2.T/m + (regconst/m)*np.vstack((np.zeros((1,ol)),theta2In))
		theta1Grades = tri1.T/m + (regconst/m)*np.vstack((np.zeros((1,hl)),theta1In))

		return [theta1Grades, theta2Grades, lrcost]

# gradient Descent
def gradDescent(data, label, theta1, theta2, regconst, maxiter, alpha):
	cost = np.zeros(maxiter)
	for i in range(maxiter):
		[theta1Grad, theta2Grad, cost[i]] = getgrad(data, label, theta1, theta2, regconst)
		theta1 = theta1 - alpha*theta1Grad
		theta2 = theta2 - alpha*theta2Grad
		# printing the current status
		if (i+1)%(maxiter*0.1) == 0:
			per = float(i+1)/maxiter*100
			print(str(per)+"% Completed")
	return [theta1, theta2, cost]


# The pediction function
def predict(data, theta1, theta2):	
	(i, j) = data.shape
	value1 = np.vstack((np.ones((1,j)),data))
	dotprod = np.dot(theta1.T, value1)
	value2 = np.vstack((np.ones((1,j)),sigmoid(dotprod)))
	value3 = sigmoid(np.dot(theta2.T, value2))
	return value3

def initialize(in1, in2):
	e = 0.25
	return np.random.random((in1+1, in2))*2*e - e
	
hl = 50		# number of neurons in hidden layer
ol = 10		# number of neurons in outer layer

# function used to read the data from file
def get_labeled_data(imagefile, labelfile):
   
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')
    images.read(4) 
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    labels.read(4) 
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

 
    x = zeros((N, rows, cols), dtype=float32)
    y = zeros((N, 1), dtype=uint8)
    for i in range(N):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

#function used for splitting the data set
def split_training_data(data, label):
        datalen = 0.8 * len(data)
        intdatalen = int(datalen)
        int_train = data[:intdatalen]
        int_dev = data[intdatalen:]
        int_train_label = label[:intdatalen]
        int_dev_label = label[intdatalen:]
        return (int_train, int_dev, int_train_label, int_dev_label)



def main():

	imagefilename = sys.argv[1]
	labelfilename = sys.argv[2]
	

	data, label = get_labeled_data(imagefilename,labelfilename)
	int_train,int_dev,int_train_label,int_dev_label = split_training_data(data,label)

	resultArray = []
	for i in range(len(int_train)):
			resultArray.append(int_train[i].flatten())
			
	int_train_t = np.asarray(resultArray)
	int_train_label_t = int_train_label

	int_train_data = int_train_t.T
	temp_train_label = int_train_label_t.T
	(n, m) = int_train_data.shape

	temp_train_label = temp_train_label*(temp_train_label!=10)

	label_int_train = np.zeros((ol,m))


	###############################################
	someArray = []
	for i in range(len(int_dev)):
                someArray.append(int_dev[i].flatten())
                
	int_dev_t = np.asarray(someArray)
	int_dev_label_t = int_dev_label

	int_dev_data = int_dev_t.T
	temp_int_dev_label = int_dev_label_t.T
	(k,j) = int_dev_data.shape

	temp_int_dev_label = temp_int_dev_label*(temp_int_dev_label!=10)
	################################################
	
	
	for i in range(0,m):	
		label_int_train[temp_train_label[0,i], i]= 1


	alpha = 1.2	#learning rate 
	regconst = 1.3	#regularization constant
	maxIter = 500	

	initial_theta1 =  initialize(n, hl)
	initial_theta2 = initialize(hl, ol)


	gradout = gradDescent(int_train_data, label_int_train , initial_theta1, initial_theta2, regconst, maxIter, alpha)

	with open('output.pickle', 'wb') as f:
                pickle.dump(gradout, f)
	
	[theta1, theta2, cost] = gradout
	pediction = predict(int_dev_data, theta1, theta2)
	max = np.empty((1,j))
	for i in range(0,j):
		max[0,i] = np.argmax(pediction[:,i])
		

	accuracy_of_prediction = np.mean((max==temp_int_dev_label)*np.ones(max.shape))*100
	print("Accuracy:"+str(accuracy_of_prediction))

if __name__ == "__main__":
    main()
