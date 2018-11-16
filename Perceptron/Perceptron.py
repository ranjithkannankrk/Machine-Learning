from struct import unpack
import gzip
import sys
import numpy as np
from numpy import zeros, uint8, float32
from pylab import imshow, show, cm
from sklearn.preprocessing import StandardScaler

#Function for reading the data from the file
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
	
#Function for splitting the data set
def split_training_data(data, label):
        datalen = 0.8 * len(data)
        intdatalen = int(datalen)
        int_train = data[:intdatalen]
        int_dev = data[intdatalen:]
        int_train_label = label[:intdatalen]
        int_dev_label = label[intdatalen:]
        return (int_train, int_dev, int_train_label, int_dev_label)

#Function for making the data set to a binary classification problem		
def manipulate_train_data(label, labelvalue):
        labelarray = []
        for i in range(len(label)):
                if label[i] == labelvalue:
                        labelarray.append(1)
                else:
                        labelarray.append(0)
        return labelarray

# The function for training
def train_dataset(data, bias, wieightVector, label):
        maxaccuracy = 0
        for i in range(len(data)):
                y = 0
                flattened = data[i]#.reshape(1,784)
                yresult = np.dot(flattened, wieightVector.T)
                if yresult <= 0:
                        y = 0
                else:
                        y = 1
                if y != label[i]:
                         wieightVector = wieightVector + label[i] * flattened
                         bias = bias + label[i]
        return (bias, wieightVector)

#The prediction function        
def predict_function(data, weight, label):
        correctpred = 0
        for row in range(len(data)):
                resultArray = np.array(0, dtype=float32)
                for weightvector in range(len(weight)):
                        result = np.dot(data[row], weight[weightvector].T)
                        resultArray = np.append(resultArray,result)
                value = np.argmax(resultArray) - 1
                if value == label[row]:
                        correctpred = correctpred + 1
        return correctpred
		

def main():

	imagefilename = sys.argv[1]
	labelfilename = sys.argv[2]
	trainData,trainLabel = get_labeled_data(imagefilename,labelfilename)
	trainData=trainData.reshape(-1,784)
	std_scaler=StandardScaler()
	trainData_scaled=std_scaler.fit_transform(trainData.astype(np.float32))
	int_train, int_dev, int_train_label, int_dev_label = split_training_data(trainData_scaled,trainLabel)
	wieightVector = np.zeros((784,), dtype=float32)[np.newaxis]
	weightOfTenClasses = []
	bais = []
	
	for i in range (0,10):
		label = manipulate_train_data(int_train_label,i)
		bias, weight = train_dataset(int_train, 0, wieightVector, label)
		bais.append(bias)
		weightOfTenClasses.append(weight)
	
	correct_pred_count = predict_function(int_dev, weightOfTenClasses, int_dev_label)
	accuracy = correct_pred_count/len(int_dev)
	print("Accuracy is")
	print(accuracy*100)
	
if __name__ == "__main__":
    main()
